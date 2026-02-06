"""Minimax search with alpha-beta pruning and move ordering."""

from __future__ import annotations

import math

import chess

from chessbot.evaluation import evaluate
from chessbot.genome import Genome


def _order_moves(board: chess.Board) -> list[chess.Move]:
    """Order moves: captures first, then checks, then quiet moves."""
    captures = []
    checks = []
    quiet = []

    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append(move)
        else:
            board.push(move)
            is_check = board.is_check()
            board.pop()
            if is_check:
                checks.append(move)
            else:
                quiet.append(move)

    return captures + checks + quiet


def _minimax(
    board: chess.Board,
    genome: Genome,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
) -> float:
    """Alpha-beta minimax. Returns evaluation score."""
    if depth == 0 or board.is_game_over():
        return evaluate(board, genome)

    moves = _order_moves(board)

    if maximizing:
        value = -math.inf
        for move in moves:
            board.push(move)
            value = max(value, _minimax(board, genome, depth - 1, alpha, beta, False))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = math.inf
        for move in moves:
            board.push(move)
            value = min(value, _minimax(board, genome, depth - 1, alpha, beta, True))
            board.pop()
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def search(board: chess.Board, genome: Genome, depth: int = 3) -> chess.Move | None:
    """Find the best move for the current side using alpha-beta search.

    Returns None if no legal moves exist.
    """
    if board.is_game_over():
        return None

    moves = _order_moves(board)
    if not moves:
        return None

    maximizing = board.turn == chess.WHITE
    best_move = moves[0]

    if maximizing:
        best_value = -math.inf
        for move in moves:
            board.push(move)
            value = _minimax(board, genome, depth - 1, -math.inf, math.inf, False)
            board.pop()
            if value > best_value:
                best_value = value
                best_move = move
    else:
        best_value = math.inf
        for move in moves:
            board.push(move)
            value = _minimax(board, genome, depth - 1, -math.inf, math.inf, True)
            board.pop()
            if value < best_value:
                best_value = value
                best_move = move

    return best_move
