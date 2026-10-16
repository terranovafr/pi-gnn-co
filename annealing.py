# annealing.py
# This module provides a function to solve the Max-Cut problem using Simulated Annealing (SA) via the OpenJij library. It constructs a QUBO model for the Max-Cut problem, samples solutions using SA, and evaluates the resulting cut.

import openjij as oj
from qubo import create_max_cut_model
from evaluate import compute_weighted_cut, sample_to_array

def solve_with_sa(
    G,
    num_reads=2000,
):
    sampler = oj.SASampler()

    model = create_max_cut_model(G)

    qubo, _ = model.to_qubo()

    response = sampler.sample_qubo(
        qubo,
        num_reads=num_reads,
    )

    assignment = sample_to_array(
        response.first.sample,
        G.number_of_nodes(),
    )

    cut = compute_weighted_cut(
        G,
        assignment,
    )

    return {
        "assignment": assignment,
        "energy": response.first.energy,
        "cut": cut,
    }