# benchmark.py
# Script to benchmark the PI-GNN against Simulated Annealing (SA) and the best known cut for a given dataset. It trains the GNN on one graph and tests it on the remaining graphs, comparing the results with SA and the best known cut.

from pathlib import Path
from datetime import datetime
import yaml
from dataset import load_dataset
from train import train_gnn, predict_gnn
from annealing import solve_with_sa
import random
import numpy as np
from utils import summarize, print_summary

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"

def select_training_graph(
    envs,
    strategy,
    already_selected,
):
    """
    Select one training graph according to the requested strategy.

    Previously selected graphs are never selected again and are instead
    forced into the test set.

    Parameters
    ----------
    envs : list
        List of dataset samples.
    strategy : str
        One of:
            smallest
            largest
            mean
            random
    already_selected : set
        Environment ids already used for training.

    Returns
    -------
    train_sample : dict
    test_samples : list
    """

    available = [
        env
        for env in envs
        if env["id"] not in already_selected
    ]

    if len(available) == 0:
        raise ValueError("No environments left to select.")

    if strategy == "smallest":

        available.sort(
            key=lambda x: x["metadata"]["score"]
        )

    elif strategy == "largest":

        available.sort(
            key=lambda x: x["metadata"]["score"],
            reverse=True,
        )

    elif strategy == "mean":

        mean_score = np.mean(
            [
                env["metadata"]["score"]
                for env in available
            ]
        )

        available.sort(
            key=lambda x: abs(
                x["metadata"]["score"] - mean_score
            )
        )

    elif strategy == "random":

        random.shuffle(available)

    else:

        raise ValueError(
            f"Unknown sampling strategy '{strategy}'."
        )

    train_sample = available[0]

    test_samples = [
        env
        for env in envs
        if env["id"] != train_sample["id"]
    ]

    return train_sample, test_samples

def run_benchmark(
    dataset_name,
    config,
    sa_reads=2000,
    num_runs=1,
    sampling="smallest",
):

    dataset = load_dataset(dataset_name)

    LOG_DIR.mkdir(exist_ok=True)

    if not isinstance(sampling, list):
        sampling = [sampling]

    for sampling_strategy in sampling:
        print()
        print("#" * 80)
        print(f"SAMPLING STRATEGY: {sampling_strategy}")
        print("#" * 80)

        log = {
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "params": dataset["params"],
            "sampling": sampling_strategy,
            "num_runs": num_runs,
            "runs": [],
        }

        already_selected = set()

        all_ratios = []
        gnn_to_sweep_ratios = []
        sweep_to_sa_ratios = []
        sweep_sa_gap = []
        num_sweep_not_sa = 0

        all_envs = (dataset["train"] + dataset.get("validation", []) + dataset["test"])

        for run in range(num_runs):
            sweep_to_sa_ratios = []
            num_sweep_not_sa = 0

            print()
            print("=" * 60)
            print(f"RUN {run + 1}/{num_runs}")
            print("=" * 60)

            train_sample, test_samples = select_training_graph(
                all_envs,
                sampling_strategy,
                already_selected,
            )

            already_selected.add(train_sample["id"])

            env_id = train_sample["id"]

            print(f"\nTraining on environment {env_id}")

            model = train_gnn(
                train_sample["graph"],
                config,
            )

            print(f"Training cut : {model['cut']}")

            run_log = {
                "run": run + 1,
                "trained_on": env_id,
                "training": {
                    "metadata": train_sample["metadata"],
                    "gnn": {
                        "energy": float(model["energy"]),
                        "cut": int(model["cut"]),
                    },
                },
                "testing": [],
            }

            for sample in test_samples:

                env_id = sample["id"]
                best_known_cut = sample["metadata"]["best_known_cut"]

                gnn = predict_gnn(
                    sample["graph"],
                    model,
                    config,
                    trials=5
                )

                sa = solve_with_sa(
                    sample["graph"],
                    num_reads=sa_reads,
                )

                ratio = (
                    gnn["cut"] / sa["cut"]
                    if sa["cut"] > 0
                    else 0.0
                )

                all_ratios.append(ratio)

                sweep_to_sa_ratio = (
                    best_known_cut / sa["cut"]
                )

                gnn_to_sweep_ratio = (
                    gnn["cut"] / best_known_cut
                )



                gnn_to_sweep_ratios.append(
                    gnn_to_sweep_ratio
                )

                sweep_to_sa_ratios.append(
                    sweep_to_sa_ratio
                )

                if abs(sweep_to_sa_ratio - 1.0) > 1e-9:
                    num_sweep_not_sa += 1
                    sweep_sa_gap.append(
                        abs(1.0 - sweep_to_sa_ratio)
                    )

                print(
                    f"Test {env_id} | "
                    f"GNN {int(gnn['cut']):6d} | "
                    f"SA {int(sa['cut']):6d} | "
                    f"Best {int(best_known_cut):6d} | "
                    f"Ratio {ratio:.4f}"
                    f" | Sweep/SA {sweep_to_sa_ratio:.4f}"
                    f" | GNN/Sweep {gnn_to_sweep_ratio:.4f}"
                )

                run_log["testing"].append(
                    {
                        "test_id": env_id,
                        "metadata": sample["metadata"],
                        "gnn": {
                            "cut": int(gnn["cut"]),
                        },
                        "sa": {
                            "cut": int(sa["cut"]),
                        },
                        "approximation_ratio": float(ratio),
                    }
                )

            log["runs"].append(run_log)



        average_ratio = (
            float(np.mean(all_ratios))
            if all_ratios
            else 0.0
        )

        log["summary"] = {

            "gnn_vs_sa": summarize(
                all_ratios
            ),

            "gnn_vs_sweep": summarize(
                gnn_to_sweep_ratios
            ),

            "sweep_vs_sa": summarize(
                sweep_to_sa_ratios
            ),

            "sweep_not_equal_sa": {

                "count": num_sweep_not_sa,

                "percentage":
                    100.0 * num_sweep_not_sa /
                    len(sweep_to_sa_ratios),

                "gap": summarize(
                    sweep_sa_gap
                ),
            },

            "num_runs": num_runs,
            "num_graphs": len(all_envs),
        }

        strategy_dir = LOG_DIR / sampling_strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)

        log_file = strategy_dir / f"{dataset_name}.yaml"
        with open(log_file, "w") as f:
            yaml.safe_dump(
                log,
                f,
                sort_keys=False,
            )

        print()
        print(f"Sampling strategy: {sampling_strategy}")
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print_summary(
            "GNN / SA",
            log["summary"]["gnn_vs_sa"],
        )

        print_summary(
            "GNN / Sweep",
            log["summary"]["gnn_vs_sweep"],
        )

        print_summary(
            "Sweep / SA",
            log["summary"]["sweep_vs_sa"],
        )

        print(
            f"Sweep != SA : "
            f"{log['summary']['sweep_not_equal_sa']['count']}"
            f"/"
            f"{len(sweep_to_sa_ratios)} "
            f"({log['summary']['sweep_not_equal_sa']['percentage']:.2f}%)"
        )

        print()

        print_summary(
            "Gap when Sweep != SA",
            log["summary"]["sweep_not_equal_sa"]["gap"],
        )

        print(f"Saved benchmark to : {log_file}")
        return average_ratio