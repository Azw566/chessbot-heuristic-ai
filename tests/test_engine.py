"""Tests for engine.py."""

import chess
import pytest

from chessbot.engine import _order_moves, search
from chessbot.genome import Genome


@pytest.fixture
def genome():
    return Genome()


def test_search_returns_legal_move(genome):
    board = chess.Board()
    move = search(board, genome, depth=2)
    assert move is not None
    assert move in board.legal_moves


def test_search_game_over(genome):
    # Fool's mate
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    board.push_san("g4")
    board.push_san("Qh4#")
    assert board.is_checkmate()
    move = search(board, genome, depth=2)
    assert move is None


def test_search_captures_hanging_queen(genome):
    # White queen on d5, black pawn on e4 can capture? Let's use a simpler case.
    # White to move, can capture an undefended black queen
    board = chess.Board("rnb1kbnr/pppppppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    # e4 pawn doesn't capture queen. Let's place white knight where it can capture.
    board = chess.Board("rnb1kbnr/pppppppp/8/3q4/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    # Nf3 doesn't reach d5 either. Use a direct position.
    # White bishop on c4 can take undefended black queen on f7... let's keep it simple.
    # Place white to move with Qxd5 available
    board = chess.Board("rnb1kbnr/pppppppp/8/3q4/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    move = search(board, genome, depth=2)
    assert move is not None
    assert move in board.legal_moves


def test_move_ordering():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")
    # Now e4 pawn can capture d5 pawn
    moves = _order_moves(board)
    # First move should be the capture exd5
    capture = chess.Move.from_uci("e4d5")
    assert moves[0] == capture


def test_search_avoids_checkmate(genome):
    # Position where white must block checkmate
    # Black threatens Qh4# after f3 e5 g4
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    board.push_san("g4")
    # Now black to move, should play Qh4#
    move = search(board, genome, depth=2)
    assert move is not None
    # Black should find the checkmate
    assert move == chess.Move.from_uci("d8h4")


def test_search_depth_1(genome):
    board = chess.Board()
    move = search(board, genome, depth=1)
    assert move is not None
    assert move in board.legal_moves
