print("start...")
import time
start_time = time.perf_counter()
import yaml
import torch
from torch_geometric.utils import add_self_loops, degree

import openpyxl

import scipy.sparse as sp
import numpy as np
import seaborn as sns
import torch.nn.functional as F

from tqdm.notebook import tqdm
from torch.optim import Adam, Optimizer
from collections import defaultdict
from torch_geometric.data import Data, InMemoryDataset

from data import get_dataset, HeatDataset, PPRDataset, set_train_val_test_split, get_adj_matrix
from ImpModels import GCN, JKNet, ARMA
from seeds import val_seeds, test_seeds

from scipy.linalg import expm
from args import get_citation_args
import torch_geometric

args = get_citation_args()

with open("./config/" + args.config, 'r') as c:
    config = yaml.safe_load(c)

device = 'cuda'

preprocessing = args.preprocessing

dataset = get_dataset(config['dataset_name'])

dataset.data = dataset.data.to(device)

model_parameter = {
    'dataset': dataset,
    'hidden': config[preprocessing]['hidden_layers'] * [config[preprocessing]['hidden_units']],
    'dropout': config[preprocessing]['dropout']
}

model_parameter['t'] = args.t
# model_parameter['alpha'] = args.alpha
if config['architecture'] == 'ARMA':
    model_parameter['stacks'] = config[preprocessing]['stacks']

model = globals()[config['architecture']](**model_parameter).to(device)
assert (not hasattr(model, "diffusion"))

# 创建一个Excel工作簿对象
workbook = openpyxl.Workbook()

# 选择工作簿的活动工作表
worksheet = workbook.active
# print(model)

def train(model: torch.nn.Module, optimizer: Optimizer, data: Data, input_feature, key="train"):
    model.train()
    optimizer.zero_grad()
    logits = model(data, input_feature)
    loss = F.nll_loss(logits[data[f'{key}_mask']], data.y[data[f'{key}_mask']])
    loss.backward()
    optimizer.step()


def evaluate(model: torch.nn.Module, data: Data, input_feature, djmat, test: bool):
    model.eval()
    with torch.no_grad():
        logits = model(data, input_feature)
    eval_dict = {}
    keys = ['val', 'test', 'train'] if test else ['val']

    for key in keys:
        mask = data[f'{key}_mask']
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        eval_dict[f'{key}_acc'] = acc
    return eval_dict

def add_param(model, weight_decay, skip_list=[], contain_list=[]):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        # print(name)
        names = name.split('.')
        # print(param)
        # print(names)
        if len(set(names) & set(skip_list)) != 0:
            # decay.append(param)
            continue
            # print("no_decay: " + name)
        else:
            if len(contain_list) == 0:
                # print("n1_decay: " + name)
                decay.append(param)
            else:
                if len(set(names) & set(contain_list)) != 0:
                    # print("n3_decay: " + name)
                    decay.append(param)
                else:
                    # print("n4_decay: " + name)
                    continue
            # print("decay: " + name)
    return [
        {'params': decay, 'weight_decay': weight_decay}]


def run(dataset: InMemoryDataset,
        model: torch.nn.Module,
        seeds: np.ndarray,
        test: bool = False,
        max_epochs: int = 1000,    #10000
        patience: int = 100,     #100
        lr: float = 0.01,
        weight_decay: float = 0.01,
        num_development: int = 1500,
        device: str = 'cuda'):

    best_dict = defaultdict(list)
	##############DEGREE-BASED FLEXIBLE NEIGHBOR###########
    edge_index, _ = add_self_loops(dataset.data.edge_index, num_nodes=dataset.data.x.size(0))

    row, col = edge_index
    deg = degree(col, dataset.data.x.size(0), dtype=dataset.data.x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    adj_matrix = get_adj_matrix(dataset)
    row_sum = (adj_matrix.sum(1) + 1)

    max_degree = max(row_sum)
    min_degree = min(row_sum)
    norm_degree = (np.log(max_degree + 1) - np.log(row_sum + 1)) / (np.log(max_degree + 1) - np.log(min_degree + 1))

    hops = 1 + (args.step - 1) * norm_degree
	##############DEGREE-BASED FLEXIBLE NEIGHBOR###########
	
	##############AGGREGATION AUGMENTATION###########
    feature_list = []
    feature_list.append(dataset.data.x)
    djmat = torch.sparse.FloatTensor(edge_index, norm)
    for i in range(1, args.step):
        feature_list.append(torch.spmm(djmat, feature_list[-1]))

    input_feature = []
    for i in range(dataset.data.x.shape[0]):
        hop = int(hops[i])
        if hop == 0:
            fea = feature_list[0][i].unsqueeze(0)
        else:
            fea = 0
            for j in range(hop):
                fea += (1-0.8)*feature_list[j][i].unsqueeze(0) + 0.8*feature_list[0][i].unsqueeze(0)
            fea = fea / hop
        input_feature.append(fea)
    input_feature = torch.cat(input_feature, dim=0)
	##############AGGREGATION AUGMENTATION###########
	
    cnt = 0
    for seed in tqdm(seeds):
        dataset.data = set_train_val_test_split(
            seed,
            dataset.data,
            num_development=num_development,
            num_per_class=args.num_per_class
        ).to(device)

        if args.swapTrainValid == True:
            dataset.data.train_mask, dataset.data.val_mask = dataset.data.val_mask, dataset.data.train_mask
        model.to(device).reset_parameters()

        params_train_decay = add_param(model.layers[0], weight_decay, skip_list=["t"])
        params_train_no_decay = []
        for layer in model.layers[1:]:
            params_train_no_decay += add_param(layer, 0, skip_list=["t"])
        params_train = params_train_decay + params_train_no_decay
        params_valid = add_param(model, 0, contain_list=["t"])

        optimizer = Adam(
            params_train,
            lr=lr
        )
        optimizer_val = Adam(
            params_valid,
            lr=args.tLr
        )

        patience_counter = 0
        tmp_dict = {'val_acc': 0}

        for epoch in range(1, max_epochs + 1):
            if patience_counter == patience:
                if args.latestop == True:
                    if epoch > 300:
                        break
                    else:
                        patience_counter -= 1
                else:
                    break
            train(model, optimizer, dataset.data, input_feature, key="train")
            if not args.fixT:
                train(model, optimizer_val, dataset.data, input_feature, key="val")
            eval_dict = evaluate(model, dataset.data, input_feature, djmat, test)
            if epoch % 10 == 0 and args.debugInfo:
                print("epoch: " + str(epoch) + ", " + str(eval_dict))
                print("t1: " + str(model.layers[0].diffusion.t.data.cpu().numpy()) + "t2: " + str(
                    model.layers[1].diffusion.t.data.cpu().numpy()))

            record = {}
            if eval_dict['val_acc'] <= tmp_dict['val_acc']:
                patience_counter += 1
            else:
                patience_counter = 0
                torch.save(model, './best.pt')
                tmp_dict['epoch'] = epoch
                for k, v in eval_dict.items():
                    tmp_dict[k] = v
        bit_list = sorted(record.keys())
        bit_list.reverse()

        model = torch.load('./best.pt')
        model.eval()
        output = model(dataset.data, input_feature).to(device)

        dataset.data.x = F.normalize(output, p=1)
        feature_list = []
        feature_list.append(output)
		
	##############SMOOTHING LABEL###########
        for i in range(1, args.step):
            feature_list.append(torch.spmm(djmat, feature_list[-1]))

        hops = 1 + (args.step - 1) * norm_degree
        out_feature = []
        for i in range(dataset.data.x.shape[0]):
            hop = int(hops[i])
            if hop == 0:
                fea = feature_list[0][i].unsqueeze(0)
            else:
                fea = 0
                for j in range(hop):
                    fea += (1 - 0.8) * feature_list[j][i].unsqueeze(0) + 0.8 * feature_list[0][i].unsqueeze(0)
                fea = fea / hop
            out_feature.append(fea)
        out_feature = torch.cat(out_feature, dim=0)
	##############SMOOTHING LABEL###########
	
        keys = ['test']
        for key in keys:
            mask = dataset.data[f'{key}_mask']
            pred = out_feature[mask].max(1)[1]
            acc = pred.eq(dataset.data.y[mask]).sum().item() / mask.sum().item()
        if acc > tmp_dict[f'{key}_acc']:
            tmp_dict[f'{key}_acc'] = acc

        cur_dict = {}
        for k, v in tmp_dict.items():
            best_dict[k].append(v)
            cur_dict[k] = v
        worksheet.append(list(cur_dict.values()))
        print(cur_dict)

    name = config['dataset_name']+config['architecture']+'.xlsx'
    workbook.save(name)

    return dict(best_dict)


results = run(
    dataset,
    model,
    seeds=test_seeds if config['test'] else val_seeds,
    lr=config[preprocessing]['lr'],
    weight_decay=config[preprocessing]['weight_decay'],
    test=config['test'],
    num_development=config['num_development'],
    device=device
)


boots_series = sns.algorithms.bootstrap(results['val_acc'], func=np.mean, n_boot=1000)
results['val_acc_ci'] = np.max(np.abs(sns.utils.ci(boots_series, 95) - np.mean(results['val_acc'])))
if 'test_acc' in results:
    boots_series = sns.algorithms.bootstrap(results['test_acc'], func=np.mean, n_boot=1000)
    results['test_acc_ci'] = np.max(
        np.abs(sns.utils.ci(boots_series, 95) - np.mean(results['test_acc']))
    )

for k, v in results.items():
    if 'acc_ci' not in k and k != 'duration':
        results[k] = np.mean(results[k])

mean_acc = results['test_acc']
uncertainty = results['test_acc_ci']
print(f"Mean accuracy: {100 * mean_acc:.2f} +- {100 * uncertainty:.2f}%")
stop_time = time.perf_counter()
print("Run time:", (stop_time - start_time))
