import argparse
import os.path as osp
import time
import sys
import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import ARMAConv, global_mean_pool
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn import ModuleList

class GraphSST2Net(torch.nn.Module):
    def __init__(self, in_channels, hid_channels=128, num_classes=2, num_layers=1):
        super(GraphSST2Net, self).__init__()

        self.convs = ModuleList([
            ARMAConv(in_channels, hid_channels),
            ARMAConv(hid_channels, hid_channels, num_layers=num_layers)])
        self.causal_mlp = torch.nn.Sequential(
            Linear(hid_channels, 2*hid_channels),
            ReLU(),
            Linear(2*hid_channels, num_classes)
        )
        self.conf_mlp = torch.nn.Sequential(
            Linear(hid_channels, 2*hid_channels),
            ReLU(),
            Linear(2*hid_channels, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_causal_pred(graph_x)

    def get_node_reps(self, x, edge_index, edge_attr, batch):
        edge_weight = edge_attr.view(-1)
        x = F.relu(self.convs[0](x, edge_index, edge_weight))
        node_x = self.convs[1](x, edge_index, edge_weight)
        return node_x

    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_causal_pred(self, graph_x):
        return self.causal_mlp(graph_x)

    def get_conf_pred(self, graph_x):
        return self.conf_mlp(graph_x)

    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        causal_pred = self.causal_mlp(causal_graph_x)
        conf_pred = self.conf_mlp(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

