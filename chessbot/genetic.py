"""Genetic algorithm operators: selection, crossover, mutation."""

from __future__ import annotations

import numpy as np

from chessbot.genome import DEFAULT_GENES, NUM_GENES, Genome


def initialize_population(
    size: int,
    rng: np.random.Generator | None = None,
    noise_scale: float = 0.3,
) -> list[Genome]:
    """Create a population of genomes with Gaussian noise around defaults."""
    if rng is None:
        rng = np.random.default_rng()

    population = []
    default = np.array(DEFAULT_GENES, dtype=np.float64)
    for _ in range(size):
        genes = default + rng.normal(0, noise_scale, size=NUM_GENES) * default
        # Clamp material values to be positive
        genes[:5] = np.maximum(genes[:5], 0.1)
        population.append(Genome(genes=genes))
    return population


def select_elite(
    population: list[Genome],
    elite_fraction: float = 0.25,
) -> list[Genome]:
    """Select the top fraction of the population by fitness."""
    n_elite = max(1, int(len(population) * elite_fraction))
    sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
    return [g.copy() for g in sorted_pop[:n_elite]]


def crossover(
    parent_a: Genome,
    parent_b: Genome,
    rng: np.random.Generator | None = None,
) -> Genome:
    """Uniform crossover: each gene is taken from one parent at random."""
    if rng is None:
        rng = np.random.default_rng()

    mask = rng.random(NUM_GENES) < 0.5
    child_genes = np.where(mask, parent_a.genes, parent_b.genes)
    return Genome(genes=child_genes.copy())


def mutate(
    genome: Genome,
    mutation_rate: float = 0.2,
    mutation_magnitude: float = 0.1,
    rng: np.random.Generator | None = None,
) -> Genome:
    """Gaussian mutation: each gene has mutation_rate chance of being perturbed."""
    if rng is None:
        rng = np.random.default_rng()

    genes = genome.genes.copy()
    for i in range(NUM_GENES):
        if rng.random() < mutation_rate:
            genes[i] += rng.normal(0, mutation_magnitude) * abs(genes[i])
    # Clamp material values to be positive
    genes[:5] = np.maximum(genes[:5], 0.1)
    return Genome(genes=genes)


def next_generation(
    population: list[Genome],
    elite_fraction: float = 0.25,
    mutation_rate: float = 0.2,
    mutation_magnitude: float = 0.1,
    rng: np.random.Generator | None = None,
) -> list[Genome]:
    """Produce next generation: elitism + crossover + mutation."""
    if rng is None:
        rng = np.random.default_rng()

    target_size = len(population)
    elites = select_elite(population, elite_fraction)

    new_pop = [e.copy() for e in elites]

    # Fill the rest with offspring
    while len(new_pop) < target_size:
        # Tournament selection for parents (pick 2 random, take the fitter)
        idxs = rng.choice(len(population), size=4, replace=True)
        parent_a = max(population[idxs[0]], population[idxs[1]], key=lambda g: g.fitness)
        parent_b = max(population[idxs[2]], population[idxs[3]], key=lambda g: g.fitness)

        child = crossover(parent_a, parent_b, rng)
        child = mutate(child, mutation_rate, mutation_magnitude, rng)
        new_pop.append(child)

    return new_pop[:target_size]
