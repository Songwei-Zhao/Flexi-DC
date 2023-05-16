import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from scipy.special import factorial
from args import get_citation_args
import numpy as np



class TDConv(MessagePassing):
    def __init__(self, in_channels, init_t):
        super(TDConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        args = get_citation_args()
        self.init_t = init_t
        self.step = args.step
        # self.init_alpha = init_alpha
        if not args.denseT:
            self.t = Parameter(torch.Tensor(in_channels))
            self.alpha = Parameter(torch.Tensor(in_channels))
            # self.alpha = Parameter(torch.Tensor(1))
        else:
            self.t = Parameter(torch.Tensor(1))
            # self.alpha = Parameter(torch.Tensor(1))
        # self.alpha = Parameter(torch.Tensor(1))
        # self.t.data.fill_(2)
        self.reset_parameters()
        # self.t.requires_grad = False


    def forward(self, x, edge_index, edge_weight=None):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        feature_list = []
        feature_list.append(x)

        djmat = torch.sparse.FloatTensor(edge_index, norm)

        for i in range(1, self.step):
            feature_list.append(torch.spmm(djmat, feature_list[-1]))

        for k in range(self.step):

            feature_list[k] = torch.pow(torch.exp(-torch.from_numpy(np.array(1)).to('cuda')), k/2) * torch.pow(self.t, k) / factorial(k) * feature_list[k]  ## important! DIFFUSION AGGREGATION HIDDEN FEATURE 

            if k != 0:
                y += feature_list[k]
            else:
                y = feature_list[k]
        return y
    
    def reset_parameters(self):
        torch.nn.init.constant_(self.t, self.init_t)
        # torch.nn.init.constant_(self.alpha, self.init_alpha)
        #self.t.requires_grad = False
    

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j