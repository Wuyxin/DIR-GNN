import torch
import math
import numpy as np
from torch_geometric.utils import (negative_sampling, remove_self_loops, degree, 
                                   add_self_loops, batched_negative_sampling)
from torch_geometric.utils.num_nodes import maybe_num_nodes
MAX_DIAM=100


def get_neg_edge_index(g):
    neg_edge_index = batched_negative_sampling(edge_index=g.edge_index,
                                               batch=g.batch,
                                               num_neg_samples=None,
                                               force_undirected=False)
    neg_edge_index, _ = remove_self_loops(neg_edge_index)
    return neg_edge_index


def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges

import math
def bool_vec(length, r_True, shuffle=True):
    n_True = math.ceil(length * r_True)
    n_False = length - n_True
    vec = np.concatenate([np.zeros(n_False, dtype=np.bool), np.ones(n_True, dtype=np.bool)])
    if shuffle:
        np.random.shuffle(vec)

    return vec


def sample(dataset, ratio):
    reserve = bool_vec(len(dataset), ratio)
    reserve = torch.tensor(reserve).bool()
    return dataset[reserve]


def relabel(x, edge_index, batch, pos=None):
        
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


def get_broken_graph(g, broken_ratio, connectivity=True):

    edge_indices, num_nodes, cum_nodes, num_edges, _ = split_batch(g)
    out_edge_ratio = []
    broken_masks = []
    for edge_index, N, C, E in zip(edge_indices, num_nodes.tolist(),
                                cum_nodes.tolist(), num_edges.tolist()):
        if connectivity:
            flag = 0
            node_idx = np.random.choice([i for i in range(N)])
            node_idx = torch.tensor([node_idx])
            num_edges = int(broken_ratio * E)
            for num_hops in range(1, MAX_DIAM):
                _, _, _, broken_mask = bid_k_hop_subgraph(
                    node_idx=node_idx, 
                    num_hops=num_hops, 
                    edge_index=edge_index-C,
                    num_nodes=N)
                if broken_mask.sum() >= num_edges:
                    flag = 1
                    break
            if flag == 0:
                print("ERROR!")
        else:
            broken_mask = bool_vec(E, r_True=broken_ratio, shuffle=True)
            broken_mask = torch.tensor(broken_mask, dtype=torch.float)
        
        broken_masks.append(broken_mask)
        out_edge_ratio.append(broken_mask.sum().float()/E)
    broken_masks = torch.cat(broken_masks, dim=0).bool()
    broken_edge_index = g.edge_index[:, broken_masks]
    broken_edge_attr = g.edge_attr[broken_masks]
    out_edge_ratio = torch.tensor(out_edge_ratio).to(g.x.device)

    return broken_edge_index, broken_edge_attr, out_edge_ratio


# Bidirectional k-hop subgraph
# modified from torch-geometric.utils.subgraph
def bid_k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        if len(subsets) > 1:
            node_mask[subsets[-2]] = True
        edge_mask1 = torch.index_select(node_mask, 0, row)
        edge_mask2 = torch.index_select(node_mask, 0, col)
        subsets.append(col[edge_mask1])
        subsets.append(row[edge_mask2])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


def get_syn_ground_truth_graph(g):

    _, _, _, num_edges, cum_edges = split_batch(g)
    
    nodel_label = np.concatenate(g.z, axis=0)
    row, col = g.edge_index.detach().cpu().numpy()
    broken_mask = torch.tensor(nodel_label[row] * nodel_label[col] > 0, dtype=torch.bool)
    broken_edge_index = g.edge_index[:, broken_mask]
    broken_edge_attr = g.edge_attr[broken_mask]
    out_edge_ratio = []
    for E, C in zip(num_edges.tolist(), cum_edges.tolist()):
        out_edge_ratio.append(broken_mask[C: C + E].sum().float()/E)
    
    out_edge_ratio = torch.tensor(out_edge_ratio).to(g.x.device)
    return broken_edge_index, broken_edge_attr, out_edge_ratio


def get_single_ground_truth_graph(g):

    _, _, _, num_edges, cum_edges = split_batch(g)
    nodel_label = np.concatenate(g.z, axis=0)
    row, col = g.edge_index.detach().cpu().numpy()
    broken_mask = torch.tensor(nodel_label[row] * nodel_label[col] > 0, dtype=torch.bool)
    
    broken_edge_indices = torch.LongTensor([[],[]]).to(g.x.device)
    broken_edge_attrs = torch.LongTensor([]).to(g.x.device)
    out_edge_ratio = []
    for E, C in zip(num_edges.tolist(), cum_edges.tolist()):
        edge_idx = torch.nonzero(broken_mask[C: C + E]).view(-1) + C
        edge_index = g.edge_index[:, edge_idx]
        node_idx = np.random.choice(np.unique(edge_index.detach().cpu().numpy()))
        node_idx = torch.tensor([node_idx]).to(g.x.device)
        _, broken_edge_index, _, edge_mask = bid_k_hop_subgraph(node_idx, num_hops=5, edge_index=edge_index)
        broken_edge_attr = g.edge_attr[C: C + E][edge_idx - C][edge_mask]
        broken_edge_indices = torch.cat([broken_edge_indices, broken_edge_index], dim=1)
        broken_edge_attrs = torch.cat([broken_edge_attrs, broken_edge_attr], dim=0)
        out_edge_ratio.append(float(broken_edge_index.size(1)) / E)
        
    out_edge_ratio = torch.tensor(out_edge_ratio).to(g.x.device)
    return broken_edge_indices, broken_edge_attrs, out_edge_ratio


def get_mnist_ground_truth_graph(g):
    
    _, _, _, num_edges, cum_edges = split_batch(g)
    
    nodel_label = torch.tensor(g.x.view(-1) > 0, dtype=torch.bool)
   
    row, col = g.edge_index.detach().cpu().numpy()
    broken_mask = torch.tensor(nodel_label[row] * nodel_label[col] > 0, dtype=torch.bool)
    broken_edge_index = g.edge_index[:, broken_mask]
    broken_edge_attr = g.edge_attr[broken_mask]
    out_edge_ratio = []
    for E, C in zip(num_edges.tolist(), cum_edges.tolist()):
        out_edge_ratio.append(broken_mask[C: C + E].sum().float()/E)
    
    out_edge_ratio = torch.tensor(out_edge_ratio).to(g.x.device)
    return broken_edge_index, broken_edge_attr, out_edge_ratio


def get_ground_truth_graph(args, g):
    if args.dataset == 'ba3':
        return get_single_ground_truth_graph(g)
    elif args.dataset == 'tr3':
        return get_syn_ground_truth_graph(g)
    elif args.dataset == 'mnist':
        return get_mnist_ground_truth_graph(g)