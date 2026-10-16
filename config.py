# config.py
# This module defines configuration classes for training and graph generation

from dataclasses import dataclass

@dataclass
class TrainConfig:
    embedding_size: int = 16 #369
    hidden_size: int = 64 #5
    learning_rate: float = 0.0001 #0.00467
    epochs: int = 10000
    dropout: float = 0.05#0.00
    shots: int = 5
    seed: int = 42

@dataclass
class GraphConfig:
    graph_type: str = "erdos"      # erdos | regular
    n_nodes: int = 100
    probability: float = 0.05
    degree: int = 3

    weighted: bool = True
    weight_low: int = 10
    weight_high: int = 100


SEARCH_SPACE = {
    "embedding_size": [16, 32, 64, 128],
    "hidden_size": [8, 16, 32, 64],
    "learning_rate": [
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
    ],
    "dropout": [
        0.0,
        0.05,
        0.10,
        0.20,
        0.30,
    ],
}
