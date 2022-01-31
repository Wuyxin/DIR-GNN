import os.path as osp

import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F
from torch.nn import Sequential as Seq, ReLU, Tanh, Linear as Lin, Softmax
from torch_geometric.nn import GraphConv, BatchNorm, global_max_pool

class MNISTSPNet(torch.nn.Module):

    def __init__(self, in_channels, hid_channels=32, num_classes=10, conv_unit=2):
        super(MNISTSPNet, self).__init__()

        self.node_emb = Lin(in_channels, hid_channels)

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        for i in range(conv_unit):
            conv = GraphConv(in_channels=hid_channels, out_channels=hid_channels)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hid_channels))
            self.relus.append(ReLU())

        self.causal_mlp = nn.Sequential(
            nn.Linear(hid_channels, 2*hid_channels),
            nn.ReLU(),
            nn.Linear(2*hid_channels, num_classes)
        )
        
        self.conf_mlp = torch.nn.Sequential(
            nn.Linear(hid_channels, 2*hid_channels),
            ReLU(),
            nn.Linear(2*hid_channels, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_max_pool(node_x, batch)
        return self.get_causal_pred(graph_x)
    
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        edge_weight = edge_attr.view(-1)
        x = self.node_emb(x)
        for conv, batch_norm, ReLU in \
                zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = ReLU(batch_norm(x))
        node_x = x
        return node_x
    
    def get_graph_rep(self, x, edge_index, edge_attr, batch):

        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_max_pool(node_x, batch)
        return graph_x

    def get_causal_pred(self, causal_graph_x):
        pred = self.causal_mlp(causal_graph_x)
        return pred

    def get_conf_pred(self, conf_graph_x):
        pred = self.conf_mlp(conf_graph_x)
        return pred

    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        causal_pred = self.causal_mlp(causal_graph_x)
        conf_pred = self.conf_mlp(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred