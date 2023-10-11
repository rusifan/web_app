from __future__ import absolute_import
from functools import reduce
import torch.nn as nn
from .modulated_gcn_conv import ModulatedGraphConv
from .graph_non_local import GraphNonLocal
from .non_local_embedded_gaussian import NONLocalBlock2D


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()
        # import pdb;pdb.set_trace() input_dim =2 output_dim =384
        self.gconv =  ModulatedGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = self.gconv(x).transpose(1, 2) # [256, 17, 384]
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.non_local = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.non_local(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class ModulatedGCN(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(ModulatedGCN, self).__init__()
        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        else:
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break

            _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = ModulatedGraphConv(hid_dim, coords_dim[1], adj) 
        self.non_local = NONLocalBlock2D(in_channels=hid_dim, sub_sample=False)
        # self.quant = torch.quantization.QuantStub() #quant stubs added
        # self.dequant = torch.quantization.DeQuantStub() #quant stubs added
    def forward(self, x):
        # import pdb; pdb.set_trace()
        # x = self.quant(x)
        #change made to make one data to work and batch
        # x = x.squeeze() 

        # x = x.unsqueeze(0) # change made to make one data work
        x = x.squeeze(2) 
        x = x.squeeze(3) 
        #31st jan to make all types of data to work batch and single

        
        x = x.permute(0,2,1)
        out = self.gconv_input(x)
        out = self.gconv_layers(out) #[256, 17, 384]
        # import pdb;pdb.set_trace()
        out = out.unsqueeze(2) #[256, 17, 1, 384]
        out = out.permute(0,3,2,1) #[256, 384, 1, 17]
        out = self.non_local(out) #[256, 384, 1, 17]
        out = out.permute(0,3,1,2) #[256, 17, 384, 1]
        out = out.squeeze() #[256, 17, 384]
        out = self.gconv_output(out) #[256, 17, 3]
        # import pdb;pdb.set_trace()
        
        out = out.permute(0,2,1) #[256, 3, 17]
        out = out.unsqueeze(2) #[256, 3, 1, 17]
        out = out.unsqueeze(4) #[256, 3, 1, 17, 1]
        # out = self.dequant(out)
        return out
