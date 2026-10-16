# Physics-Inspired GNNs for Weighted Max-Cut

This project extends [GNNs for Solving Combinatorial Optimization Problems](https://github.com/IvanIsCoding/GNN-for-Combinatorial-Optimization), an educational implementation of physics-inspired graph neural networks (PI-GNNs) based on the paper [*Combinatorial Optimization with Physics-Inspired Graph Neural Networks*](https://arxiv.org/abs/2107.01188) by Schuetz et al.

The original project implements GNN-based approaches to Max-Cut and Maximum Independent Set (MIS), and compares their performance against Simulated Annealing (SA).

Building on this work, this project focuses on **weighted Max-Cut** and extends the original implementation with:

- Generation of weighted graphs with configurable sizes and degrees.
- Domain-randomized training, sharing GNN parameters across multiple graphs.
- Evaluation on held-out graphs to assess generalization across several training strategies.
- Hyperparameter optimization and aggregated performance metrics.

This project is an independent extension and is not officially affiliated with or endorsed by the original authors.

## Installation

``` bash
conda env create -f environment.yml
conda activate pignn
```

Alternatively:

``` bash
pip install -r requirements.txt
```

## Quick start

Put options **after** the subcommand.

### Train on one generated graph

``` bash
python main.py train --source generated --nodes 30 --degree 3 --epochs 500 --seed 7
```

A regular graph requires `degree < nodes` and an even `nodes * degree`.

### Train across generated graphs

Train each graph independently:

``` bash
python main.py train --source generated --num-graphs 10 --min-nodes 50 --max-nodes 150 --min-degree 2 --max-degree 6 --training-strategy single --epochs 500
```

Train shared GNN parameters across graphs (domain randomization):

``` bash
python main.py train --source generated --num-graphs 10 --min-nodes 50 --max-nodes 150 --min-degree 2 --max-degree 6 --training-strategy domain_randomization --epochs 500
```

Domain randomization shares model and optimizer state across graphs,
while each graph has its own node embeddings. 

### Train with a dataset

``` bash
python main.py train --source dataset --dataset maxcut_20260411_134115 --sampling smallest random --num-runs 1 --epochs 1000
```

Dataset selection strategies are `smallest`, `largest`, `mean`, and `random`:

- **`smallest`**: Selects the graph with the smallest value of the dataset's `score` field.
- **`largest`**: Selects the graph with the largest value of the `score` field.
- **`mean`**: Selects the graph whose `score` is closest to the mean score across the dataset.
- **`random`**: Randomly selects a graph from the available dataset.

These strategies determine which graph is selected for training. The remaining graphs can then be used for evaluation, depending on the selected workflow.

### Run Simulated Annealing Only

``` bash
python main.py sa --source generated --nodes 30 --degree 3 --reads 2000 --seed 7
```

Similarly, you can train on a dataset and with similar options as above.

### Hyperparameter optimization

Generated graphs (at least two are needed for train/validation  separation):

``` bash
python main.py hyperopt --source generated --num-graphs 10 --min-nodes 50 --max-nodes 150 --min-degree 2 --max-degree 6 --training-strategy domain_randomization --search random --trials 20 --epochs 500
```

Dataset:

``` bash
python main.py hyperopt --source dataset --dataset maxcut_20260411_134115 --search random --trials 20 --epochs 500 --reads 2000
```

Use `--search grid` to evaluate every combination in `SEARCH_SPACE`;  this can be expensive. Generated-graph optimization scores the mean
`cut / total_edge_weight` on held-out graphs. Dataset optimization uses  the existing `run_benchmark()` objective.

## Main CLI Options

| Option | Applies to | Default | Description |
|---|---|---|---|
| `--source` | `train`, `sa`, `hyperopt` | `generated` | Graph source: `generated` or `dataset`. |
| `--dataset` | Dataset source | None | Dataset directory under `envs/`. |
| `--epochs` | `train`, `hyperopt` | `10000` | Training epochs per shot. |
| `--seed` | Graph generation, training | `2023` | Base random seed. |
| `--nodes`, `--degree` | Generated graphs | `100`, `3` | Node count and degree for a single graph. |
| `--num-graphs` | Generated graphs | `1` | Number of graphs to generate. |
| `--min-nodes`, `--max-nodes` | Generated graphs | `50`, `150` | Minimum and maximum node counts. |
| `--min-degree`, `--max-degree` | Generated graphs | `2`, `6` | Minimum and maximum graph degrees. |
| `--weight-low`, `--weight-high` | Generated graphs | `10`, `100` | Minimum and maximum edge weights. |
| `--training-strategy` | `train`, `hyperopt` | `single` | Training strategy: `single` or `domain_randomization`. |
| `--sampling` | Dataset source | `smallest` | One or more graph-selection strategies: `smallest`, `largest`, `mean`, `random`. |
| `--num-runs` | Dataset training | `1` | Number of graph selections per sampling strategy. |
| `--reads` | `sa`, dataset hyperparameter optimization | `2000` | Number of Simulated Annealing reads. |
| `--search` | `hyperopt` | `random` | Search strategy: `random` or `grid`. |
| `--trials` | `hyperopt` | `20` | Number of random-search trials; ignored for grid search. |

For multiple generated graphs, node count and degree are sampled from valid combinations in the requested ranges.

## Dataset format

Datasets are stored under `envs/<dataset-name>/`:

``` text
envs/
└── my_dataset/
    ├── split.yaml
    ├── env_001/maxcut_graph.txt
    └── env_002/maxcut_graph.txt
```

Each `maxcut_graph.txt` contains one weighted edge per line:

``` text
node_u node_v edge_weight
0 1 10
0 2 20
1 2 15
```

Blank lines and lines starting with `#` are ignored. `split.yaml` must contain `training_set` and `test_set` lists. Each entry needs an `id`;
graph selection expects a numeric `score`, and benchmark comparisons may use `best_known_cut`.

``` yaml
params: {}
training_set:
  - id: env_001
    score: 100
    best_known_cut: 1250
test_set:
  - id: env_002
    score: 110
    best_known_cut: 1310
```

These values are examples only. Graph files must exist at
`envs/<dataset-name>/<id>/maxcut_graph.txt`.

The current repository contains a sample set of synthetic graphs under `envs/maxcut_20260411_134115/` for demonstration purposes.

## Related Tutorials and Original Work

This project builds on existing educational implementations of PI-GNNs for CO.

- **[GNNs for Solving Combinatorial Optimization Problems](https://github.com/IvanIsCoding/GNN-for-Combinatorial-Optimization)** — An educational Jupyter notebook series implementing PI-GNNs using JAX and Flax. It explores Max-Cut and MIS, and compares GNN performance with Simulated Annealing. This project builds on that implementation and extends the Max-Cut workflow to weighted graphs, domain-randomized training, held-out evaluation, and hyperparameter optimization.

- **[Amazon Science: Combinatorial Optimization with Graph Neural Networks](https://github.com/amazon-science/co-with-gnns-example)** — The reference implementation accompanying the work by Schuetz et al. It demonstrates PI-GNNs for combinatorial optimization using PyTorch Geometric and includes an example focused on MIS. The broader approach also applies to Max-Cut and other optimization problems.

- **[Original research paper](https://arxiv.org/abs/2107.01188)** — M. J. A. Schuetz, J. K. Brubaker, and H. G. Katzgraber, *Combinatorial Optimization with Physics-Inspired Graph Neural Networks* (2021). This paper introduces the PI-GNN approach underlying these implementations.

This repository is an independent extension. Credit for the original methods and implementations belongs to their respective authors; this project is not affiliated with or endorsed by them.
