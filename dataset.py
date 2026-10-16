# dataset.py
# Script to load Max-Cut datasets from text files and YAML split files. It provides functions to load weighted graphs, read dataset splits, and return structured data for training and testing.

import networkx as nx
from pathlib import Path
import yaml

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PROJECT_ROOT / "envs"

def load_weighted_graph(graph_file):
    """
    Load a weighted graph from a text file.

    Expected format
    ---------------
    # Weighted MaxCut graph
    # Nodes 55
    # Edges 318
    0 3 43
    0 4 36
    ...

    Parameters
    ----------
    graph_file : str or Path

    Returns
    -------
    nx.Graph
    """

    graph_file = Path(graph_file)

    G = nx.Graph()

    with graph_file.open("r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            u, v, w = line.split()

            G.add_edge(
                int(u),
                int(v),
                weight=float(w),
            )

    return G


def load_split(dataset_name):
    """
    Read split.yaml.

    Parameters
    ----------
    dataset_name : str

    Returns
    -------
    dict
        Parsed YAML dictionary.
    """

    split_file = DATASETS_ROOT / dataset_name / "split.yaml"

    if not split_file.exists():
        raise FileNotFoundError(split_file)

    with split_file.open("r") as f:
        split = yaml.safe_load(f)

    return split

def _load_samples(dataset_dir, samples):
    result = []

    for sample in samples:
        env_id = sample["id"]

        graph_path = (
            dataset_dir
            / str(env_id)
            / "maxcut_graph.txt"
        )

        G = load_weighted_graph(graph_path)

        result.append(
            {
                "id": env_id,
                "graph": G,
                "metadata": sample,
            }
        )

    return result

def load_dataset(dataset_name):
    dataset_dir = DATASETS_ROOT / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    split = load_split(dataset_name)

    return {
        "params": split.get("params", {}),
        "train": _load_samples(dataset_dir, split.get("training_set", [])),
        "test": _load_samples(dataset_dir, split.get("test_set", [])),
    }



def load_dataset_all_splits(dataset_name):
    dataset_dir = DATASETS_ROOT / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    split = load_split(dataset_name)

    # Collect all test samples from every split*.yaml
    all_test = {}
    for split_file in sorted(dataset_dir.glob("split*.yaml")):
        with open(split_file) as f:
            split_data = yaml.safe_load(f)

        for sample in split_data.get("test_set", []):
            all_test[sample["id"]] = sample

    return {
        "params": split.get("params", {}),
        "train": _load_samples(dataset_dir, split.get("training_set", [])),
        "test": _load_samples(dataset_dir, all_test.values()),
    }