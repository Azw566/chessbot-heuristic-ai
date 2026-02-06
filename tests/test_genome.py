"""Tests for genome.py."""

import numpy as np
import pytest

from chessbot.genome import DEFAULT_GENES, NUM_GENES, Genome


def test_default_genome():
    g = Genome()
    assert len(g.genes) == NUM_GENES
    np.testing.assert_array_almost_equal(g.genes, DEFAULT_GENES)
    assert g.fitness == 0.0


def test_round_trip_vector():
    g = Genome()
    g.fitness = 42.5
    vec = g.to_vector()
    g2 = Genome.from_vector(vec, fitness=g.fitness)
    np.testing.assert_array_equal(g.genes, g2.genes)
    assert g2.fitness == 42.5


def test_round_trip_dict():
    g = Genome(genes=np.array([1, 2, 3, 4, 5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64), fitness=10.0)
    d = g.to_dict()
    g2 = Genome.from_dict(d)
    np.testing.assert_array_equal(g.genes, g2.genes)
    assert g2.fitness == g.fitness


def test_from_vector_wrong_length():
    with pytest.raises(ValueError):
        Genome.from_vector([1.0, 2.0])


def test_piece_values():
    g = Genome()
    pv = g.piece_values
    assert pv[1] == g.pawn_value  # chess.PAWN == 1


def test_copy_independence():
    g = Genome()
    g2 = g.copy()
    g2.genes[0] = 999.0
    assert g.genes[0] != 999.0


def test_properties():
    g = Genome()
    assert g.pawn_value == 1.0
    assert g.knight_value == 3.0
    assert g.bishop_value == 3.25
    assert g.rook_value == 5.0
    assert g.queen_value == 9.0
    assert g.w_material == 1.0
    assert g.w_mobility == 0.1
    assert g.w_center == 0.3
    assert g.w_king_safety == 0.2
    assert g.w_pawn_structure == 0.2
