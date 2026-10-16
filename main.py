"""
    main.py
    CLI for PI-GNN weighted Max-Cut experiments.
"""

import argparse
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import yaml

from config import TrainConfig
from graph_utils import generate_regular_graph
from train import train_gnn, train_gnn_domain_randomized, predict_gnn
from annealing import solve_with_sa
from dataset import load_dataset
from benchmark import select_training_graph
from hyperopt import hyperopt


SAMPLING_CHOICES = ("smallest", "largest", "mean", "random")
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"


def save_run_log(mode, args, payload):
    """Save a YAML record for a completed train/SA run."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    source = args.dataset if args.source == "dataset" else "generated"
    path = LOG_DIR / f"{mode}_{source}_{stamp}.yaml"
    document = {
        "mode": mode,
        "source": args.source,
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "epochs": args.epochs,
        "reads": args.reads,
        "training_strategy": getattr(args, "training_strategy", None),
        "results": payload,
    }
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False)
    print(f"Saved results to {path}")
    return path


def add_source_arguments(parser, *, include_training_strategy=False):
    parser.add_argument(
        "--source", choices=("generated", "dataset"), default="generated",
        help="Use newly generated graphs or load graphs from envs/<dataset>/.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset directory name inside envs/ (required for --source dataset).",
    )
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--num-graphs", type=int, default=1)
    parser.add_argument("--min-nodes", type=int, default=50)
    parser.add_argument("--max-nodes", type=int, default=150)
    parser.add_argument("--min-degree", type=int, default=2)
    parser.add_argument("--max-degree", type=int, default=6)
    parser.add_argument("--weight-low", type=int, default=10)
    parser.add_argument("--weight-high", type=int, default=100)
    parser.add_argument(
        "--sampling", choices=SAMPLING_CHOICES, nargs="+", default=["smallest"]
    )
    parser.add_argument("--num-runs", type=int, default=1)
    if include_training_strategy:
        parser.add_argument(
            "--training-strategy",
            choices=("single", "domain_randomization"),
            default="single",
            help="Train each graph independently or share GNN parameters across graphs.",
        )


def build_parser():
    parser = argparse.ArgumentParser(description="PI-GNN weighted Max-Cut experiments.")
    sub = parser.add_subparsers(dest="mode", required=True)

    train_parser = sub.add_parser("train", help="Train/evaluate PI-GNN.")
    add_source_arguments(train_parser, include_training_strategy=True)

    hyperopt_parser = sub.add_parser("hyperopt", help="Optimize hyperparameters.")
    add_source_arguments(hyperopt_parser, include_training_strategy=True)
    hyperopt_parser.add_argument("--search", choices=("random", "grid"), default="random")
    hyperopt_parser.add_argument("--trials", type=int, default=20)

    sa_parser = sub.add_parser("sa", help="Run simulated annealing.")
    add_source_arguments(sa_parser)

    for command in (train_parser, hyperopt_parser, sa_parser):
        command.add_argument("--epochs", type=int, default=10000)
        command.add_argument("--seed", type=int, default=2023)
        command.add_argument("--reads", type=int, default=2000)

    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def validate_args(args):
    if args.source == "dataset" and not args.dataset:
        raise ValueError("--dataset is required when --source dataset.")
    if args.epochs < 1 or args.reads < 1 or args.num_runs < 1:
        raise ValueError("--epochs, --reads, and --num-runs must be positive.")
    if args.num_graphs < 1:
        raise ValueError("--num-graphs must be >= 1.")
    if args.weight_low > args.weight_high:
        raise ValueError("--weight-low must be <= --weight-high.")

    if args.source == "generated":
        if args.num_graphs == 1:
            pairs = [(args.nodes, args.degree)]
        else:
            if args.min_nodes < 1 or args.max_nodes < args.min_nodes:
                raise ValueError("Require 1 <= --min-nodes <= --max-nodes.")
            if args.min_degree < 0 or args.max_degree < args.min_degree:
                raise ValueError("Require 0 <= --min-degree <= --max-degree.")
            pairs = [
                (n, d)
                for n in range(args.min_nodes, args.max_nodes + 1)
                for d in range(args.min_degree, args.max_degree + 1)
            ]
        if not any(n > 0 and 0 <= d < n and (n * d) % 2 == 0 for n, d in pairs):
            raise ValueError("No valid regular graph exists for the specified node/degree settings.")
        if args.num_graphs == 1:
            n, d = args.nodes, args.degree
            if not (n > 0 and 0 <= d < n and (n * d) % 2 == 0):
                raise ValueError("A regular graph requires degree < nodes and nodes * degree even.")


def make_config(args):
    return TrainConfig(epochs=args.epochs, seed=args.seed)


def generate_graph_samples(args):
    if args.num_graphs == 1:
        specs = [(args.nodes, args.degree)]
    else:
        valid_pairs = [
            (n, d)
            for n in range(args.min_nodes, args.max_nodes + 1)
            for d in range(args.min_degree, args.max_degree + 1)
            if d < n and (n * d) % 2 == 0
        ]
        rng = random.Random(args.seed)
        specs = [rng.choice(valid_pairs) for _ in range(args.num_graphs)]

    samples = []
    for i, (n, degree) in enumerate(specs):
        graph_seed = args.seed + i
        graph = generate_regular_graph(
            n=n, degree=degree, weighted=True,
            low=args.weight_low, high=args.weight_high, seed=graph_seed,
        )
        samples.append({
            "id": f"generated_{i + 1:04d}",
            "graph": graph,
            "metadata": {"nodes": n, "degree": degree, "seed": graph_seed},
        })
    return samples


def load_dataset_samples(name):
    dataset = load_dataset(name)
    samples = dataset.get("train", []) + dataset.get("test", [])
    if not samples:
        raise ValueError(f"Dataset {name!r} contains no graph samples.")
    return dataset, samples


def print_result(result, graph, name, config=None, method="PI-GNN"):
    print("\n" + "=" * 64)
    print(f"{method} RESULTS — {name}")
    print("=" * 64)
    print(f"Nodes          : {graph.number_of_nodes()}")
    print(f"Edges          : {graph.number_of_edges()}")
    if config is not None:
        print(f"Seed           : {config.seed}")
        print(f"Epochs         : {config.epochs}")
    print(f"Weighted cut   : {result['cut']}")
    print(f"Energy         : {result['energy']}")
    print("=" * 64)


def _total_weight(graph):
    return float(sum(float(data.get("weight", 1.0)) for _, _, data in graph.edges(data=True)))


def _score_fraction(graph, cut):
    weight = _total_weight(graph)
    return float(cut) / weight if weight > 0 else 0.0


def _iqm(values):
    """Interquartile mean: mean after trimming 25% from each tail."""
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return float("nan")
    trim = int(np.floor(len(values) * 0.25))
    trimmed = values[trim:len(values)-trim] if trim and len(values) - 2 * trim > 0 else values
    return float(np.mean(trimmed))


def _report_comparison(records, title="Aggregate comparison"):
    if not records:
        print("No graph results available for aggregate statistics.")
        return
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
    for method in ("GNN", "SA"):
        scores = [r[f"{method.lower()}_fraction"] for r in records]
        cuts = [r[f"{method.lower()}_cut"] for r in records]
        print(f"{method}: mean score={np.mean(scores):.4f} | IQM={_iqm(scores):.4f} | max={np.max(scores):.4f} | mean cut={np.mean(cuts):.3f}")
    ratios = [r["ratio"] for r in records if np.isfinite(r["ratio"])]
    if ratios:
        print(f"GNN/SA cut ratio: mean={np.mean(ratios):.4f} | IQM={_iqm(ratios):.4f} | max={np.max(ratios):.4f}")
    print(f"Graphs evaluated: {len(records)}")


def _compare_graph(graph, gnn_result, sample_id, config, *, reads=2000):
    sa_result = solve_with_sa(graph, num_reads=reads)
    gnn_cut = float(gnn_result["cut"])
    sa_cut = float(sa_result["cut"])
    ratio = gnn_cut / sa_cut if sa_cut != 0 else float("nan")
    record = {
        "id": sample_id,
        "gnn_cut": gnn_cut,
        "sa_cut": sa_cut,
        "gnn_fraction": _score_fraction(graph, gnn_cut),
        "sa_fraction": _score_fraction(graph, sa_cut),
        "ratio": ratio,
    }
    print(f"Compare {sample_id} | GNN cut={gnn_cut:.3f}, score={record['gnn_fraction']:.4f} | SA cut={sa_cut:.3f}, score={record['sa_fraction']:.4f} | GNN/SA={ratio:.4f}")
    return record


def run_train(args, config):
    if args.source == "generated":
        samples = generate_graph_samples(args)
        if args.training_strategy == "domain_randomization" and len(samples) >= 2:
            shuffled = list(samples)
            random.Random(args.seed).shuffle(shuffled)
            n_train = max(1, min(len(shuffled) - 1, int(round(0.8 * len(shuffled)))))
            train_samples, test_samples = shuffled[:n_train], shuffled[n_train:]
        else:
            train_samples, test_samples = samples, []
    else:
        dataset, _ = load_dataset_samples(args.dataset)
        train_samples = dataset.get("train", [])
        test_samples = dataset.get("test", [])
        if not train_samples:
            raise ValueError("Dataset has no training_set samples.")

    records = []
    if args.training_strategy == "domain_randomization":
        model = train_gnn_domain_randomized(train_samples, config, verbose=False)
        print(f"Shared GNN trained on {len(train_samples)} graphs.")
        if not test_samples:
            print("Warning: no held-out graphs available; reporting training graphs instead.")
            test_samples = train_samples
        for sample in test_samples:
            prediction = predict_gnn(sample["graph"], model, config, trials=5)
            print_result(prediction, sample["graph"], sample["id"], config, method="PI-GNN (domain-randomized)")
            records.append(_compare_graph(sample["graph"], prediction, sample["id"], config, reads=args.reads))
        _report_comparison(records, "Held-out domain-randomization comparison (GNN vs SA)")
        save_run_log("train", args, {"evaluation_split": "test" if test_samples != train_samples else "training_fallback", "comparisons": records})
        return

    # Single strategy: fit one model per graph and compare it against SA on that graph.
    if args.source == "generated":
        eval_samples = samples
    else:
        eval_samples = train_samples
    for sample in eval_samples:
        graph = sample["graph"]
        prediction = train_gnn(graph, config)
        print_result(prediction, graph, sample["id"], config)
        records.append(_compare_graph(graph, prediction, sample["id"], config, reads=args.reads))
    _report_comparison(records, "Per-graph PI-GNN vs SA comparison")

    if args.source == "dataset" and test_samples:
        print("\nCross-graph inference on dataset test_set:")
        cross_records = []
        # Evaluate each test graph with the model trained on the first training graph.
        model = train_gnn(train_samples[0]["graph"], config)
        for sample in test_samples:
            prediction = predict_gnn(sample["graph"], model, config, trials=5)
            cross_records.append(_compare_graph(sample["graph"], prediction, sample["id"], config, reads=args.reads))
        _report_comparison(cross_records, "Dataset held-out test comparison (GNN vs SA)")
        save_run_log("train_test", args, {"evaluation_split": "test", "comparisons": cross_records})
    save_run_log("train", args, {"evaluation_split": "training_or_generated", "comparisons": records})

def run_sa(args):
    if args.source == "generated":
        sample = generate_graph_samples(args)[0]
    else:
        _, samples = load_dataset_samples(args.dataset)
        sample = select_training_graph(samples, args.sampling[0], set())[0]
    graph = sample["graph"]
    result = solve_with_sa(graph, num_reads=args.reads)
    print_result(result, graph, sample["id"], method="SA")
    cut = float(result["cut"])
    total_weight = _total_weight(graph)
    save_run_log("sa", args, [{
        "id": sample["id"],
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "cut": cut,
        "energy": float(result["energy"]),
        "score_fraction": cut / total_weight if total_weight > 0 else 0.0,
    }])


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    config = make_config(args)

    if args.mode == "train":
        run_train(args, config)
    elif args.mode == "sa":
        run_sa(args)
    elif args.mode == "hyperopt":
        if args.source == "generated":
            samples = generate_graph_samples(args)
            if len(samples) < 2:
                raise ValueError("Generated-graph hyperopt requires --num-graphs >= 2.")
            hyperopt(
                base_config=config,
                search=args.search,
                trials=args.trials,
                graph_samples=samples,
                training_strategy=args.training_strategy,
                seed=args.seed,
            )
        else:
            hyperopt(
                dataset_name=args.dataset,
                base_config=config,
                sa_reads=args.reads,
                search=args.search,
                trials=args.trials,
                training_strategy=args.training_strategy,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()