# graph_utils.py
# This module contains utility functions for generating and manipulating graphs, including assigning random weights, generating Erdos-Renyi and regular graphs, and computing adjacency matrices.

import networkx as nx
import numpy as np

def assign_random_weights(
    G,
    low=10,
    high=100,
    seed=2023,
):
    rng = np.random.default_rng(seed)

    for u, v in G.edges():
        G[u][v]["weight"] = rng.integers(low, high + 1)

    return G


def generate_erdos_graph(
    n,
    p,
    weighted=True,
    low=10,
    high=100,
    seed=2023,
):
    G = nx.fast_gnp_random_graph(n=n, p=p, seed=seed)

    if weighted:
        assign_random_weights(G, low, high, seed)

    return nx.convert_node_labels_to_integers(G)


def generate_regular_graph(
    n,
    degree,
    weighted=True,
    low=10,
    high=100,
    seed=2023,
):
    G = nx.random_regular_graph(degree, n, seed)

    if weighted:
        assign_random_weights(G, low, high, seed)

    return nx.convert_node_labels_to_integers(G)


def adjacency_matrix(G):

    A = nx.to_numpy_array(G, weight="weight", dtype=np.float32)

    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1

    return A / row_sum