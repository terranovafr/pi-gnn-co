"""
    train.py
    Train PI-GNN models for weighted Max-Cut.

    Supports:
    - train_gnn: independent optimization on one graph (original behavior).
    - train_gnn_domain_randomized: shared GNN parameters optimized across multiple
    graphs. Node embeddings are graph-specific, while model parameters are shared.
    - predict_gnn: inference on a graph using a trained parameter state.
"""

import jax
import jax.numpy as jnp
import numpy as np

from graph_utils import adjacency_matrix
from evaluate import compute_weighted_cut
from model import create_train_state, train_step, get_classification
from qubo import create_Q_matrix


def create_node_embeddings(num_nodes, embedding_size, seed):
    """Create deterministic random node features for one graph."""
    rng = jax.random.PRNGKey(int(seed))
    return jax.random.uniform(rng, (num_nodes, embedding_size))


def _run_inference(G, state, node_embeddings, energy):
    """Run inference using an already-trained parameter state."""
    A = jnp.asarray(adjacency_matrix(G))
    probs = state.apply_fn({"params": state.params}, node_embeddings, A)
    classification = np.asarray(
        get_classification(state.apply_fn, state.params, node_embeddings, A)
    )
    cut = compute_weighted_cut(G, classification)

    return {
        "classification": classification,
        "probs": np.asarray(probs),
        "energy": float(energy),
        "cut": float(cut),
        "solution": {
            f"X[{i}]": int(classification[i])
            for i in range(G.number_of_nodes())
        },
        "state": state,
    }


def _prepare_graph(G, embedding_size, seed):
    """Prepare graph tensors and fixed node embeddings."""
    return {
        "graph": G,
        "adjacency": jnp.asarray(adjacency_matrix(G)),
        "q_matrix": create_Q_matrix(G),
        "embeddings": create_node_embeddings(
            G.number_of_nodes(), embedding_size, seed
        ),
        "embedding_seed": int(seed),
    }


def _train_steps_on_graph(state, item, rng, epochs, verbose=False, label=""):
    metrics = None
    for epoch in range(epochs):
        rng, dropout_rng = jax.random.split(rng)
        state, metrics = train_step(
            state=state,
            node_embeddings=item["embeddings"],
            adj_matrix=item["adjacency"],
            q_matrix=item["q_matrix"],
            dropout_rng=dropout_rng,
        )
        if verbose and (epoch + 1) % 100 == 0:
            energy = float(metrics["energy"])
            print(f"{label}Epoch {epoch + 1} | Energy={energy:.4f}")
    return state, metrics, rng


def train_gnn(G, config, verbose=False):
    """Train an independent GNN on a single graph; return the best shot."""
    if config.shots < 1:
        raise ValueError("config.shots must be >= 1")
    if config.epochs < 1:
        raise ValueError("config.epochs must be >= 1")

    best_solution = None
    best_cut = -np.inf

    for shot in range(config.shots):
        seed = int(config.seed + shot)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, _ = jax.random.split(rng, 3)

        state = create_train_state(
            n_vertices=G.number_of_nodes(),
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            rng=init_rng,
            learning_rate=config.learning_rate,
            dropout_frac=config.dropout,
        )
        item = _prepare_graph(G, config.embedding_size, seed)
        state, metrics, _ = _train_steps_on_graph(
            state, item, rng, config.epochs, verbose,
            label=f"Shot {shot + 1} | ",
        )
        result = _run_inference(
            G, state, item["embeddings"], metrics["energy"]
        )
        result["embedding_seed"] = seed
        result["shots_trained"] = shot + 1

        if result["cut"] > best_cut:
            best_cut = result["cut"]
            best_solution = result

    return best_solution


def train_gnn_domain_randomized(graphs, config, verbose=False, shuffle_graphs=True):
    """Train one shared GNN parameter state across multiple graphs.

    Parameters
    ----------
    graphs : sequence
        A sequence of NetworkX graphs, or sample dictionaries containing a
        ``"graph"`` entry.
    config : TrainConfig
        Shared model/training configuration.
    verbose : bool
        Print progress every 100 epochs.
    shuffle_graphs : bool
        Shuffle graph order on each epoch using a deterministic Python RNG.

    Notes
    -----
    Each graph has its own fixed random node embeddings. The GNN parameters
    and optimizer state are shared across all graphs. Different graph sizes
    are supported because the model parameters are dense-layer weights, not
    node-index-specific parameters.
    """
    graphs = [
        item["graph"] if isinstance(item, dict) and "graph" in item else item
        for item in graphs
    ]
    if not graphs:
        raise ValueError("Domain-randomized training requires at least one graph.")
    if config.shots < 1 or config.epochs < 1:
        raise ValueError("config.shots and config.epochs must be >= 1.")

    best_model = None
    best_score = -np.inf

    for shot in range(config.shots):
        shot_seed = int(config.seed + shot)
        rng = jax.random.PRNGKey(shot_seed)
        rng, init_rng, _ = jax.random.split(rng, 3)

        # All graph inputs are prepared once per shot; only the shared model
        # parameters and optimizer state are updated during training.
        prepared = [
            _prepare_graph(G, config.embedding_size, shot_seed + 100_003 * i)
            for i, G in enumerate(graphs)
        ]
        first_graph = graphs[0]
        state = create_train_state(
            n_vertices=first_graph.number_of_nodes(),
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            rng=init_rng,
            learning_rate=config.learning_rate,
            dropout_frac=config.dropout,
        )
        order_rng = np.random.default_rng(shot_seed)
        last_metrics = None

        for epoch in range(config.epochs):
            order = np.arange(len(prepared))
            if shuffle_graphs:
                order_rng.shuffle(order)

            for graph_index in order:
                item = prepared[int(graph_index)]
                rng, dropout_rng = jax.random.split(rng)
                state, last_metrics = train_step(
                    state=state,
                    node_embeddings=item["embeddings"],
                    adj_matrix=item["adjacency"],
                    q_matrix=item["q_matrix"],
                    dropout_rng=dropout_rng,
                )

            if verbose and (epoch + 1) % 100 == 0:
                print(
                    f"Shared shot {shot + 1} | Epoch {epoch + 1} | "
                    f"Last energy={float(last_metrics['energy']):.4f}"
                )

        results = []
        normalized_scores = []
        for G, item in zip(graphs, prepared):
            result = _run_inference(
                G, state, item["embeddings"], last_metrics["energy"]
            )
            result["embedding_seed"] = item["embedding_seed"]
            results.append(result)
            total_weight = sum(
                float(data.get("weight", 1.0))
                for _, _, data in G.edges(data=True)
            )
            normalized_scores.append(
                result["cut"] / total_weight if total_weight > 0 else 0.0
            )

        score = float(np.mean(normalized_scores))
        if score > best_score:
            best_score = score
            # Keep one shared state and per-graph inference records.
            best_model = {
                "state": state,
                "embedding_seed": shot_seed,
                "cut": float(np.mean([r["cut"] for r in results])),
                "energy": float(last_metrics["energy"]),
                "score": score,
                "graph_results": results,
                "shots_trained": shot + 1,
                "training_strategy": "domain_randomization",
            }

    return best_model


def predict_gnn(G, trained_model, config, trials=1):
    """Evaluate a trained parameter state using several embedding seeds."""
    if trials < 1:
        raise ValueError("trials must be >= 1")

    best_result = None
    best_cut = -np.inf
    base_seed = int(trained_model["embedding_seed"])

    for trial in range(trials):
        seed = base_seed + trial
        embeddings = create_node_embeddings(
            G.number_of_nodes(), config.embedding_size, seed
        )
        result = _run_inference(
            G, trained_model["state"], embeddings, np.nan
        )
        result["embedding_seed"] = seed
        if result["cut"] > best_cut:
            best_cut = result["cut"]
            best_result = result

    return best_result