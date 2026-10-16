"""
    hyperopt.py
    Hyperparameter optimization for PI-GNN weighted Max-Cut.
"""

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import itertools
import random
import yaml
from benchmark import run_benchmark
from config import SEARCH_SPACE
from dataset import load_dataset
from train import train_gnn, train_gnn_domain_randomized, predict_gnn

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"


def sample_random_config(base_config, rng=None):
    """Sample one configuration reproducibly when an RNG is provided."""
    rng = rng or random
    params = {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}
    return replace(base_config, **params)


def grid_configs(base_config):
    """Yield every combination in SEARCH_SPACE."""
    keys = list(SEARCH_SPACE)
    for combination in itertools.product(*(SEARCH_SPACE[key] for key in keys)):
        yield replace(base_config, **dict(zip(keys, combination)))


def _config_record(cfg):
    return {
        "embedding_size": cfg.embedding_size,
        "hidden_size": cfg.hidden_size,
        "learning_rate": cfg.learning_rate,
        "dropout": cfg.dropout,
        "shots": cfg.shots,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
    }


def _graph_from_sample(sample):
    return sample["graph"] if isinstance(sample, dict) else sample


def _cut_fraction(graph, cut):
    total_weight = sum(
        float(data.get("weight", 1.0))
        for _, _, data in graph.edges(data=True)
    )
    return float(cut) / total_weight if total_weight > 0 else 0.0


def score_generated_graphs(graphs, config, training_strategy="single", seed=None):
    """Score a configuration on a generated graph collection.

    Uses a deterministic train/validation split. The objective is the mean
    fraction of total edge weight cut on held-out graphs, making graph sizes
    and weight scales more comparable than raw cut values.
    """
    graphs = list(graphs)
    if len(graphs) < 2:
        raise ValueError(
            "Generated-graph hyperopt needs at least two graphs so one can be "
            "held out for validation. Use --num-graphs >= 2."
        )

    rng = random.Random(config.seed if seed is None else seed)
    indices = list(range(len(graphs)))
    rng.shuffle(indices)
    n_validation = max(1, int(round(0.2 * len(graphs))))
    n_validation = min(n_validation, len(graphs) - 1)
    validation_indices = set(indices[:n_validation])
    train_graphs = [graphs[i] for i in range(len(graphs)) if i not in validation_indices]
    validation_graphs = [graphs[i] for i in range(len(graphs)) if i in validation_indices]

    if training_strategy == "domain_randomization":
        model = train_gnn_domain_randomized(train_graphs, config)
        predictions = [
            predict_gnn(_graph_from_sample(graph), model, config, trials=5)
            for graph in validation_graphs
        ]
    elif training_strategy == "single":
        # Train on each training graph independently and evaluate on held-out
        # graphs using the existing cross-graph prediction method.
        model = train_gnn(_graph_from_sample(train_graphs[0]), config)
        predictions = [
            predict_gnn(_graph_from_sample(sample), model, config, trials=5)
            for sample in validation_graphs
        ]
        validation_graphs = [_graph_from_sample(g) for g in validation_graphs]
    else:
        raise ValueError(f"Unknown training strategy: {training_strategy}")

    if training_strategy == "domain_randomization":
        validation_graphs = [_graph_from_sample(g) for g in validation_graphs]

    fractions = [
        _cut_fraction(graph, prediction["cut"])
        for graph, prediction in zip(validation_graphs, predictions)
    ]
    return float(sum(fractions) / len(fractions))


def hyperopt(
    dataset_name=None,
    base_config=None,
    sa_reads=2000,
    search="random",
    trials=20,
    *,
    graph_samples=None,
    training_strategy="single",
    seed=None,
):
    """Run hyperparameter search on a dataset or generated graph samples.

    Dataset mode preserves the existing benchmark-based objective. Generated
    mode uses a held-out validation split and mean weighted-cut fraction.
    ``graph_samples`` may contain NetworkX graphs or dictionaries with a
    ``"graph"`` field.
    """
    if base_config is None:
        raise ValueError("base_config is required.")
    if search not in ("random", "grid"):
        raise ValueError(f"Unknown search strategy {search!r}.")
    if search == "random" and trials < 1:
        raise ValueError("trials must be >= 1.")
    if graph_samples is None and not dataset_name:
        raise ValueError("Provide dataset_name or graph_samples.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    search_seed = int(base_config.seed if seed is None else seed)
    rng = random.Random(search_seed)

    log = {
        "dataset": dataset_name,
        "source": "generated" if graph_samples is not None else "dataset",
        "timestamp": datetime.now().isoformat(),
        "search": search,
        "training_strategy": training_strategy,
        "seed": search_seed,
        "trials": [],
    }

    if search == "grid":
        configs = grid_configs(base_config)
        if trials is not None:
            # Grid search deliberately traverses the full search space.
            pass
    else:
        configs = (
            sample_random_config(base_config, rng)
            for _ in range(trials)
        )

    best_score = -float("inf")
    best_record = None
    source_name = dataset_name or "generated"
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in source_name
    )
    logfile = LOG_DIR / f"hyperopt_{safe_name}_{datetime.now():%Y%m%d_%H%M%S}.yaml"

    def persist_progress(status):
        log["status"] = status
        log["best"] = best_record
        with logfile.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(log, stream, sort_keys=False)

    for trial, cfg in enumerate(configs, start=1):
        print("=" * 60)
        print(f"HYPEROPT TRIAL {trial}")
        print("=" * 60)
        print(_config_record(cfg))

        if graph_samples is not None:
            score = score_generated_graphs(
                graph_samples,
                cfg,
                training_strategy=training_strategy,
                seed=search_seed,
            )
        elif training_strategy == "domain_randomization":
            dataset = load_dataset(dataset_name)
            dataset_samples = (
                dataset.get("train", [])
                + dataset.get("validation", [])
                + dataset.get("test", [])
            )
            score = score_generated_graphs(
                dataset_samples,
                cfg,
                training_strategy="domain_randomization",
                seed=search_seed,
            )
        else:
            score = run_benchmark(
                dataset_name=dataset_name,
                config=cfg,
                sa_reads=sa_reads,
            )
            if score is None:
                raise RuntimeError(
                    "run_benchmark returned None. Ensure benchmark.py returns "
                    "a numeric objective score after all sampling strategies."
                )

        record = {
            "trial": trial,
            "score": float(score),
            "config": _config_record(cfg),
        }
        log["trials"].append(record)
        persist_progress("running")

        if float(score) > best_score:
            best_score = float(score)
            best_record = record
            print(f"New best score: {best_score:.6f}")

        if search == "random" and trial >= trials:
            break

    persist_progress("completed")

    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(yaml.safe_dump(best_record, sort_keys=False))
    print(f"Saved to {logfile}")
    return log