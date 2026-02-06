"""Tests for genetic.py."""

import numpy as np
import pytest

from chessbot.genetic import (
    crossover,
    initialize_population,
    mutate,
    next_generation,
    select_elite,
)
from chessbot.genome import NUM_GENES, Genome


def test_initialize_population_size():
    pop = initialize_population(10, rng=np.random.default_rng(42))
    assert len(pop) == 10


def test_initialize_population_diversity():
    pop = initialize_population(5, rng=np.random.default_rng(42))
    # All genomes should be different
    genes = [g.genes.tolist() for g in pop]
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            assert genes[i] != genes[j]


def test_initialize_population_positive_material():
    pop = initialize_population(20, rng=np.random.default_rng(42))
    for g in pop:
        assert all(g.genes[:5] > 0), "Material values must be positive"


def test_select_elite():
    pop = [Genome() for _ in range(8)]
    for i, g in enumerate(pop):
        g.fitness = float(i)
    elites = select_elite(pop, elite_fraction=0.25)
    assert len(elites) == 2
    assert elites[0].fitness == 7.0
    assert elites[1].fitness == 6.0


def test_crossover_produces_valid_genome():
    rng = np.random.default_rng(42)
    a = Genome(genes=np.ones(NUM_GENES, dtype=np.float64))
    b = Genome(genes=np.full(NUM_GENES, 2.0, dtype=np.float64))
    child = crossover(a, b, rng)
    assert len(child.genes) == NUM_GENES
    # Each gene should be from one parent
    for i in range(NUM_GENES):
        assert child.genes[i] in (1.0, 2.0)


def test_crossover_independence():
    rng = np.random.default_rng(42)
    a = Genome()
    b = Genome()
    child = crossover(a, b, rng)
    child.genes[0] = 999.0
    assert a.genes[0] != 999.0
    assert b.genes[0] != 999.0


def test_mutate_preserves_length():
    g = Genome()
    m = mutate(g, mutation_rate=1.0, mutation_magnitude=0.5, rng=np.random.default_rng(42))
    assert len(m.genes) == NUM_GENES


def test_mutate_positive_material():
    g = Genome()
    # High mutation to test clamping
    m = mutate(g, mutation_rate=1.0, mutation_magnitude=10.0, rng=np.random.default_rng(42))
    assert all(m.genes[:5] >= 0.1)


def test_next_generation_size():
    pop = initialize_population(8, rng=np.random.default_rng(42))
    for i, g in enumerate(pop):
        g.fitness = float(i)
    new_pop = next_generation(pop, rng=np.random.default_rng(42))
    assert len(new_pop) == 8


def test_next_generation_elites_preserved():
    pop = initialize_population(8, rng=np.random.default_rng(42))
    for i, g in enumerate(pop):
        g.fitness = float(i)
    best_genes = pop[-1].genes.copy()
    new_pop = next_generation(pop, elite_fraction=0.25, rng=np.random.default_rng(42))
    # The best genome should be in the new population
    assert any(np.array_equal(g.genes, best_genes) for g in new_pop)
