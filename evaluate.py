# evaluate.py
# Script to evaluate the cut value of a binary assignment on a given graph.

import numpy as np

def compute_weighted_cut(G, assignment):
    """
    Computes the weighted cut value of a binary assignment.
    """

    cut = 0

    for u, v, data in G.edges(data=True):
        if assignment[u] != assignment[v]:
            cut += data.get("weight", 1)

    return cut


def sample_to_array(sample, n):
    """
    Convert OpenJij sample dict to a NumPy array.

    Example:
    {'X[0]':1,'X[1]':0}
        ->
    [1,0]
    """

    assignment = np.zeros(n, dtype=int)

    for key, value in sample.items():
        idx = int(key[2:-1])
        assignment[idx] = value

    return assignment