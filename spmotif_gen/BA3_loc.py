"""synthetic_structsim.py
Copied from [gnn-model-explainer](https://github.com/RexYing/gnn-model-explainer)
"""
import os

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as colors

import networkx as nx

import numpy as np

import synthetic_structsim
import featgen

figsize = (8, 6)

def find_gd(edge_index, ids):
    row, col = edge_index
    gd = np.array(ids[row] > 0, dtype=np.float) * np.array(ids[col] > 0, dtype=np.float)
    
    return gd
####################################
#
# Experiment utilities
#
####################################
def perturb(graph_list, p, id=None):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            if (not id == None) and (id[u]==0 or id[v]==0): 
                G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def join_graph(G1, G2, n_pert_edges):
    """ Join two graphs along matching nodes, then perturb the resulting graph.
    Args:
        G1, G2: Networkx graphs to be joined.
        n_pert_edges: number of perturbed edges.
    Returns:
        A new graph, result of merging and perturbing G1 and G2.
    """
    assert n_pert_edges > 0
    F = nx.compose(G1, G2)
    edge_cnt = 0
    while edge_cnt < n_pert_edges:
        node_1 = np.random.choice(G1.nodes())
        node_2 = np.random.choice(G2.nodes())
        F.add_edge(node_1, node_2)
        edge_cnt += 1
    return F


def preprocess_input_graph(G, labels, normalize_adj=False):
    """ Load an existing graph to be converted for the experiments.
    Args:
        G: Networkx graph to be loaded.
        labels: Associated node labels.
        normalize_adj: Should the method return a normalized adjacency matrix.
    Returns:
        A dictionary containing adjacency, node features and labels
    """
    adj = np.array(nx.to_numpy_matrix(G))
    if normalize_adj:
        sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].shape[0]
    f = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        f[i, :] = G.nodes[u]["feat"]

    # add batch dim
    adj = np.expand_dims(adj, axis=0)
    f = np.expand_dims(f, axis=0)
    labels = np.expand_dims(labels, axis=0)
    return {"adj": adj, "feat": f, "labels": labels}


####################################
#
# Generating synthetic graphs
#
###################################
def gen_syn1(nb_shapes=80, width_basis=300, feature_generator=None, m=5, draw=True):
    """ Synthetic Graph #1:

    Start with Barabasi-Albert graph and attach house-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  number of edges to attach to existing node (for BA graph)

    Returns:
        G                 :  A networkx graph
        role_id           :  A list with length equal to number of nodes in the entire graph (basis
                          :  + shapes). role_id[i] is the ID of the role of node i. It is the label.
        name              :  A graph identifier
    """
    basis_type = "ba"
    list_shapes = [["house"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, m=m, rdm_basis_plugins=True
    )
    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
    return G, role_id, name


def gen_syn2(nb_shapes=100, width_basis=350, draw=True):
    """ Synthetic Graph #2:

    Start with Barabasi-Albert graph and add node features indicative of a community label.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).

    Returns:
        G                 :  A networkx graph
        label             :  Label of the nodes (determined by role_id and community)
        name              :  A graph identifier
    """
    basis_type = "ba"

    random_mu = [0.0] * 8
    random_sigma = [1.0] * 8

    # Create two grids
    mu_1, sigma_1 = np.array([-1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
    mu_2, sigma_2 = np.array([1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
    feat_gen_G1 = featgen.GaussianFeatureGen(mu=mu_1, sigma=sigma_1)
    feat_gen_G2 = featgen.GaussianFeatureGen(mu=mu_2, sigma=sigma_2)
    G1, role_id1, name = gen_syn1(feature_generator=feat_gen_G1, m=4)
    G2, role_id2, name = gen_syn1(feature_generator=feat_gen_G2, m=4)
    G1_size = G1.number_of_nodes()
    num_roles = max(role_id1) + 1
    role_id2 = [r + num_roles for r in role_id2]
    label = role_id1 + role_id2

    # Edit node ids to avoid collisions on join
    g1_map = {n: i for i, n in enumerate(G1.nodes())}
    G1 = nx.relabel_nodes(G1, g1_map)
    g2_map = {n: i + G1_size for i, n in enumerate(G2.nodes())}
    G2 = nx.relabel_nodes(G2, g2_map)

    # Join
    n_pert_edges = width_basis
    G = join_graph(G1, G2, n_pert_edges)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes) + "_2comm"

    return G, label, name


def gen_syn3(nb_shapes=80, width_basis=300, feature_generator=None, m=5, draw=True):
    """ Synthetic Graph #3:

    Start with Barabasi-Albert graph and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'grid') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  number of edges to attach to existing node (for BA graph)

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph.
        name              :  A graph identifier
    """
    basis_type = "ba"
    list_shapes = [["grid", 3]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, m=5
    )
    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
    return G, role_id, name


def gen_syn4(nb_shapes=60, width_basis=8, feature_generator=None, m=4, draw=True):
    """ Synthetic Graph #4:

    Start with a tree and attach cycle-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'Tree').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    basis_type = "tree"
    list_shapes = [["cycle", m]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, plugins = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.1, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


def gen_syn5(nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    basis_type = "tree"
    list_shapes = [["grid", m]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.1, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


def gen_syn6(nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #6:

    Start with a tree and attach house-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    basis_type = "tree"
    list_shapes = [["house"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.1, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


def gen_syn7(nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #7:

    Start with a tree and attach clique-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    basis_type = "tree"
    list_shapes = [["clique", m]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], p=0.1, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


def gen_syn8(nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    basis_type = "tree"
    list_shapes = [["crossgrid"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.1, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


