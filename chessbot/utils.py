"""Utility functions: save/load populations, timer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from chessbot.genome import Genome


def save_population(population: list[Genome], path: str | Path) -> None:
    """Save a population to a JSON file."""
    data = {
        "population": [g.to_dict() for g in population],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_population(path: str | Path) -> list[Genome]:
    """Load a population from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [Genome.from_dict(d) for d in data["population"]]


def save_evolution_history(history: list[dict], path: str | Path) -> None:
    """Save evolution history (per-generation stats) to JSON."""
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def load_evolution_history(path: str | Path) -> list[dict]:
    """Load evolution history from JSON."""
    with open(path) as f:
        return json.load(f)


class Timer:
    """Simple context-manager timer."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
