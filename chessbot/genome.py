"""Genome representation for the evolutionary chess engine.

A genome encodes 10 floating-point genes:
  - 5 material piece values: pawn, knight, bishop, rook, queen
  - 5 category weights: material, mobility, center_control, king_safety, pawn_structure
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

# Labels for the 10 genes (used in display / serialization)
GENE_LABELS: list[str] = [
    "pawn_value",
    "knight_value",
    "bishop_value",
    "rook_value",
    "queen_value",
    "w_material",
    "w_mobility",
    "w_center",
    "w_king_safety",
    "w_pawn_structure",
]

# Sensible starting values (classic piece values + equal category weights)
DEFAULT_GENES: list[float] = [
    1.0,   # pawn
    3.0,   # knight
    3.25,  # bishop
    5.0,   # rook
    9.0,   # queen
    1.0,   # material weight
    0.1,   # mobility weight
    0.3,   # center control weight
    0.2,   # king safety weight
    0.2,   # pawn structure weight
]

NUM_GENES = len(DEFAULT_GENES)


@dataclass
class Genome:
    """A single individual in the population."""

    genes: np.ndarray = field(default_factory=lambda: np.array(DEFAULT_GENES, dtype=np.float64))
    fitness: float = 0.0

    # ---- material piece values (indices 0-4) ----
    @property
    def pawn_value(self) -> float:
        return float(self.genes[0])

    @property
    def knight_value(self) -> float:
        return float(self.genes[1])

    @property
    def bishop_value(self) -> float:
        return float(self.genes[2])

    @property
    def rook_value(self) -> float:
        return float(self.genes[3])

    @property
    def queen_value(self) -> float:
        return float(self.genes[4])

    @property
    def piece_values(self) -> dict[int, float]:
        """Map chess.PAWN..chess.QUEEN → material value."""
        import chess
        return {
            chess.PAWN: self.pawn_value,
            chess.KNIGHT: self.knight_value,
            chess.BISHOP: self.bishop_value,
            chess.ROOK: self.rook_value,
            chess.QUEEN: self.queen_value,
        }

    # ---- category weights (indices 5-9) ----
    @property
    def w_material(self) -> float:
        return float(self.genes[5])

    @property
    def w_mobility(self) -> float:
        return float(self.genes[6])

    @property
    def w_center(self) -> float:
        return float(self.genes[7])

    @property
    def w_king_safety(self) -> float:
        return float(self.genes[8])

    @property
    def w_pawn_structure(self) -> float:
        return float(self.genes[9])

    # ---- serialization ----
    def to_vector(self) -> list[float]:
        """Return genes as a plain Python list (JSON-safe)."""
        return self.genes.tolist()

    @classmethod
    def from_vector(cls, vec: list[float], fitness: float = 0.0) -> Genome:
        """Create a Genome from a plain list of floats."""
        if len(vec) != NUM_GENES:
            raise ValueError(f"Expected {NUM_GENES} genes, got {len(vec)}")
        return cls(genes=np.array(vec, dtype=np.float64), fitness=fitness)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "genes": self.to_vector(),
            "fitness": self.fitness,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Genome:
        """Deserialize from a dict."""
        return cls.from_vector(d["genes"], fitness=d.get("fitness", 0.0))

    def copy(self) -> Genome:
        """Return a deep copy."""
        return Genome(genes=self.genes.copy(), fitness=self.fitness)

    def __repr__(self) -> str:
        gene_str = ", ".join(f"{l}={v:.3f}" for l, v in zip(GENE_LABELS, self.genes))
        return f"Genome({gene_str}, fitness={self.fitness:.2f})"
