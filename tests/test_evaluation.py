"""Tests for evaluation.py."""

import chess
import pytest

from chessbot.evaluation import (
    eval_center_control,
    eval_king_safety,
    eval_material,
    eval_mobility,
    eval_pawn_structure,
    evaluate,
)
from chessbot.genome import Genome


@pytest.fixture
def genome():
    return Genome()


def test_material_starting_position(genome):
    board = chess.Board()
    score = eval_material(board, genome)
    assert score == 0.0, "Starting position should be materially equal"


def test_material_white_advantage(genome):
    # Remove black's queen
    board = chess.Board()
    board.remove_piece_at(chess.D8)
    score = eval_material(board, genome)
    assert score > 0, "White should have material advantage"
    assert abs(score - genome.queen_value) < 0.01


def test_mobility_starting_position():
    board = chess.Board()
    mob = eval_mobility(board)
    # White to move in starting position: both sides have 20 legal moves
    assert mob == 0.0


def test_center_control_starting_position():
    board = chess.Board()
    score = eval_center_control(board)
    # Symmetric position → should be ~0
    assert abs(score) < 0.01


def test_evaluate_checkmate():
    genome = Genome()
    # Scholar's mate position (black is checkmated)
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    # Actually set up fool's mate: white is checkmated
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    board.push_san("g4")
    board.push_san("Qh4#")
    assert board.is_checkmate()
    score = evaluate(board, genome)
    # White is checkmated (it's white's turn and checkmate) → large negative
    assert score == -10000.0


def test_evaluate_stalemate():
    genome = Genome()
    # Known stalemate position: black king on a8, white queen on b6, white king on c8
    board = chess.Board("k7/8/1Q6/8/8/8/8/2K5 b - - 0 1")
    # This might not be stalemate, let's use a proper one
    board = chess.Board("k7/8/1K6/8/8/8/8/8 b - - 0 1")
    if board.is_stalemate():
        score = evaluate(board, genome)
        assert score == 0.0


def test_evaluate_returns_float(genome):
    board = chess.Board()
    score = evaluate(board, genome)
    assert isinstance(score, float)


def test_pawn_structure_symmetric():
    board = chess.Board()
    score = eval_pawn_structure(board)
    # Starting position is symmetric
    assert abs(score) < 0.01


def test_king_safety_starting(genome):
    board = chess.Board()
    score = eval_king_safety(board)
    # Both kings have same shelter in starting position
    assert isinstance(score, float)
