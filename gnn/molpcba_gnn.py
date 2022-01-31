"""adapt from https://github.com/p-lambda/wilds/blob/6d96cff360018bdb7c0863ba3976f7fa646aaaab/examples/models/gnn.py"""

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch.nn.functional as F

from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder

class MolPCBANet(torch.nn.Module):
    """
    Graph Isomorphism Network augmented with virtual node for multi-task binary graph classification
    Input:
        - batched Pytorch Geometric graph object
    Output:
        - prediction (Tensor): float torch tensor of shape (num_graphs, num_tasks)
    """

    def __init__(self, num_tasks=128, num_layers = 3, emb_dim = 300, dropout = 0.5, encode_node = True):
        """
        Args:
            - num_tasks (int): number of binary label tasks. default to 128 (number of tasks of ogbg-molpcba)
            - num_layers (int): number of message passing layers of GNN
            - emb_dim (int): dimensionality of hidden channels
            - dropout (float): dropout ratio applied to hidden channels
        """

        super(MolPCBANet, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.d_out = self.num_tasks

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        self.gnn_node = GINVirtual_node(num_layers, emb_dim, dropout = dropout, encode_node = encode_node)

        # Pooling function to generate whole-graph embeddings
        self.pool = global_mean_pool
        self.causal_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
        self.conf_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
        self.cq = torch.nn.Linear(self.num_tasks, self.num_tasks)
        self.conf_fw = torch.nn.Sequential(
            self.conf_linear,
            self.cq
        )
    def forward(self, x, edge_index, edge_attr, batch):

        h_graph = self.get_graph_rep(x, edge_index, edge_attr, batch)
        return self.get_causal_pred(h_graph)

    def get_graph_rep(self, x, edge_index, edge_attr, batch):

        h_node = self.gnn_node(x, edge_index, edge_attr, batch)
        h_graph = self.pool(h_node, batch)
        return h_graph
    
    def get_causal_pred(self, h_graph):
        return self.causal_linear(h_graph)
    
    def get_conf_pred(self, conf_graph_x):
        return self.conf_fw(conf_graph_x)
    
    def get_comb_pred(self, causal_graph_x, conf_graph_x):

        causal_pred = self.causal_linear(causal_graph_x)
        conf_pred = self.conf_linear(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred


class GINVirtual_node(torch.nn.Module):
    """
    Helper function of Graph Isomorphism Network augmented with virtual node for multi-task binary graph classification
    This will generate node embeddings
    Input:
        - batched Pytorch Geometric graph object
    Output:
        - node_embedding (Tensor): float torch tensor of shape (num_nodes, emb_dim)
    """
    def __init__(self, num_layers, emb_dim, dropout = 0.5, encode_node = False):
        '''
        Args:
            - num_tasks (int): number of binary label tasks. default to 128 (number of tasks of ogbg-molpcba)
            - num_layers (int): number of message passing layers of GNN
            - emb_dim (int): dimensionality of hidden channels
            - dropout (float): dropout ratio applied to hidden channels
        '''

        super(GINVirtual_node, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encode_node = encode_node
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        
        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, x, edge_index, edge_attr, batch):

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        if self.encode_node:
            h_list = [self.atom_encoder(x)]
        else:
            h_list = [x]
        for layer in range(self.num_layers):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.dropout, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training = self.training)

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP
                virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.dropout, training = self.training)

        node_embedding = h_list[-1]

        return node_embedding


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    """
    Graph Isomorphism Network message passing
    Input:
        - x (Tensor): node embedding
        - edge_index (Tensor): edge connectivity information
        - edge_attr (Tensor): edge feature
    Output:
        - prediction (Tensor): output node emebedding
    """
    def __init__(self, emb_dim):
        """
        Args:
            - emb_dim (int): node embedding dimensionality
        """

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out