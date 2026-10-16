# qubo.py
# Script to define the QUBO model for the Max-Cut problem and compute metrics.

from pyqubo import Array
import numpy as np
import jax.numpy as jnp

def create_max_cut_model(graph):
    N = graph.number_of_nodes()
    X = Array.create("X", shape=(N,), vartype="BINARY")

    hamiltonian = 0
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1.0)
        hamiltonian -= w * (X[u] - X[v]) ** 2

    return hamiltonian.compile()

def create_Q_matrix(graph):
    model = create_max_cut_model(graph)

    N = graph.number_of_nodes()
    extract_val = lambda x: int(x[2:-1])
    Q_matrix = np.zeros((N, N))

    qubo_dict, _ = model.to_qubo()

    for (a, b), quv in qubo_dict.items():
        u = min(extract_val(a), extract_val(b))
        v = max(extract_val(a), extract_val(b))
        Q_matrix[u, v] = quv

    return jnp.array(Q_matrix)

def qubo_approx_cost(probs, Q):
    cost = jnp.sum(jnp.matmul(jnp.matmul(jnp.transpose(probs), Q), probs))
    return cost

def compute_metrics(*, probs, q_matrix):
    energy = qubo_approx_cost(probs=probs, Q=q_matrix)
    metrics = {
        "energy": energy,
        # Support potential extension
    }
    return metrics