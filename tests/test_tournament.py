"""Tests for tournament.py and utils.py."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from chessbot.genetic import initialize_population
from chessbot.genome import Genome
from chessbot.tournament import play_game, run_tournament
from chessbot.utils import (
    Timer,
    load_population,
    save_population,
)


def test_play_game_returns_result():
    g1 = Genome()
    g2 = Genome()
    result = play_game(g1, g2, depth=1, max_moves=20)
    assert result["result"] in ("white", "black", "draw")
    assert result["moves"] > 0
    assert isinstance(result["snapshots"], list)
    assert isinstance(result["move_list"], list)
    assert len(result["move_list"]) == result["moves"]


def test_play_game_max_moves():
    g1 = Genome()
    g2 = Genome()
    result = play_game(g1, g2, depth=1, max_moves=5)
    assert result["moves"] <= 5


def test_run_tournament_assigns_fitness():
    pop = initialize_population(4, rng=np.random.default_rng(42))
    pop, games = run_tournament(pop, depth=1, max_moves=10)
    # At least some genomes should have non-zero fitness
    total_fitness = sum(g.fitness for g in pop)
    assert total_fitness > 0
    # Should have game records
    assert len(games) > 0
    assert "white_idx" in games[0]
    assert "move_list" in games[0]


def test_run_tournament_progress_callback():
    pop = initialize_population(3, rng=np.random.default_rng(42))
    calls = []
    def callback(current, total):
        calls.append((current, total))
    run_tournament(pop, depth=1, max_moves=10, progress_callback=callback)
    assert len(calls) > 0
    # Last call should have current == total
    assert calls[-1][0] == calls[-1][1]


def test_save_load_population():
    pop = initialize_population(4, rng=np.random.default_rng(42))
    pop[0].fitness = 5.0
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    save_population(pop, path)
    loaded = load_population(path)
    assert len(loaded) == 4
    np.testing.assert_array_almost_equal(loaded[0].genes, pop[0].genes)
    assert loaded[0].fitness == 5.0
    Path(path).unlink()


def test_timer():
    import time
    with Timer("test") as t:
        time.sleep(0.05)
    assert t.elapsed >= 0.04
    assert t.label == "test"
