# model.py
# Script to define the GNN model and training functions for solving QUBO problems.

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax import linen as nn
from qubo import qubo_approx_cost, compute_metrics

class GraphConvLayer(nn.Module):
    c_out: int  # Output feature size

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        node_feats_w1 = nn.Dense(features=self.c_out, use_bias=False)(node_feats)
        node_feats_w2 = jax.lax.batch_matmul(adj_matrix, node_feats)
        node_feats_w2 = nn.Dense(features=self.c_out)(node_feats_w2)
        return node_feats_w1 + node_feats_w2

class CombGNN(nn.Module):
    hidden_size: int
    num_classes: int
    dropout_frac: float

    @nn.compact
    def __call__(self, node_feats, adj_matrix, train=False):
        # First convolution
        h = GraphConvLayer(c_out=self.hidden_size)(
            node_feats=node_feats, adj_matrix=adj_matrix
        )
        h = nn.relu(h)
        h = nn.Dropout(rate=self.dropout_frac, deterministic=not train)(h)
        # Second convolution
        h = GraphConvLayer(c_out=self.num_classes)(node_feats=h, adj_matrix=adj_matrix)
        probs = nn.sigmoid(h)

        return probs


def create_train_state(
    n_vertices, embedding_size, hidden_size, rng, learning_rate, dropout_frac=0.0
):
    gnn = CombGNN(hidden_size=hidden_size, num_classes=1, dropout_frac=dropout_frac)
    dropout_rng = jax.random.PRNGKey(0)
    params = gnn.init(
        rngs={"params": rng, "dropout": dropout_rng},
        node_feats=jnp.ones([n_vertices, embedding_size]),
        adj_matrix=jnp.ones([n_vertices, n_vertices]),
        train=True,
    )["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=gnn.apply, params=params, tx=tx)


@jax.jit
def train_step(state, node_embeddings, adj_matrix, q_matrix, dropout_rng):
    """Train for a single step."""

    def cost_fn(params):
        probs = state.apply_fn(
            {"params": params},
            node_embeddings,
            adj_matrix,
            rngs={"dropout": dropout_rng},
            train=True,
        )
        cost = qubo_approx_cost(probs=probs, Q=q_matrix)
        return cost, probs

    grad_fn = jax.grad(cost_fn, has_aux=True)
    grads, probs = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(probs=probs, q_matrix=q_matrix)
    return state, metrics

def get_classification(apply_fn, params, node_embeddings, adj_matrix):
    pred_probs = apply_fn({"params": params}, node_embeddings, adj_matrix)
    classification = jnp.where(pred_probs >= 0.5, 1, 0)
    return np.ravel(classification)
