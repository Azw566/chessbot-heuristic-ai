"""Minimax search with alpha-beta pruning, transposition table, and iterative deepening."""

from __future__ import annotations

import math
import time

import chess

from chessbot.evaluation import evaluate
from chessbot.genome import Genome

# Transposition table entry flags
_EXACT = 0
_LOWER = 1  # alpha cutoff (failed high)
_UPPER = 2  # beta cutoff (failed low)


def _order_moves(board: chess.Board) -> list[chess.Move]:
    """Order moves: captures first, then quiet moves.

    Skips expensive check-detection (push/pop per move) to keep ordering fast.
    """
    captures = []
    quiet = []

    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append(move)
        else:
            quiet.append(move)

    return captures + quiet


def _minimax(
    board: chess.Board,
    genome: Genome,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    tt: dict,
) -> float:
    """Alpha-beta minimax with transposition table. Returns evaluation score."""
    if depth == 0 or board.is_game_over():
        return evaluate(board, genome)

    key = board.fen()
    tt_entry = tt.get(key)
    if tt_entry is not None:
        tt_depth, tt_flag, tt_value = tt_entry
        if tt_depth >= depth:
            if tt_flag == _EXACT:
                return tt_value
            elif tt_flag == _LOWER and tt_value >= beta:
                return tt_value
            elif tt_flag == _UPPER and tt_value <= alpha:
                return tt_value

    moves = _order_moves(board)
    orig_alpha = alpha

    if maximizing:
        value = -math.inf
        for move in moves:
            board.push(move)
            value = max(value, _minimax(board, genome, depth - 1, alpha, beta, False, tt))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = math.inf
        for move in moves:
            board.push(move)
            value = min(value, _minimax(board, genome, depth - 1, alpha, beta, True, tt))
            board.pop()
            beta = min(beta, value)
            if alpha >= beta:
                break

    # Store in transposition table
    if value <= orig_alpha:
        flag = _UPPER
    elif value >= beta:
        flag = _LOWER
    else:
        flag = _EXACT
    tt[key] = (depth, flag, value)

    return value


def search(
    board: chess.Board,
    genome: Genome,
    depth: int = 3,
    time_limit: float | None = None,
    tt: dict | None = None,
) -> chess.Move | None:
    """Find the best move using iterative deepening with alpha-beta search.

    Uses a transposition table for caching and iterative deepening up to
    *depth* plies.  If *time_limit* (seconds) is given, the search stops
    early when the budget is exhausted and returns the best move found so far.

    An external *tt* dict can be passed to share the table across moves
    within a single game, giving further speedup.

    Returns None if no legal moves exist.
    """
    if board.is_game_over():
        return None

    moves = _order_moves(board)
    if not moves:
        return None

    if tt is None:
        tt = {}

    maximizing = board.turn == chess.WHITE
    best_move = moves[0]
    deadline = (time.perf_counter() + time_limit) if time_limit else None

    # Iterative deepening: search depth 1, 2, … up to *depth*
    for current_depth in range(1, depth + 1):
        if deadline and time.perf_counter() >= deadline:
            break

        current_best = moves[0]
        if maximizing:
            best_value = -math.inf
            for move in moves:
                board.push(move)
                value = _minimax(board, genome, current_depth - 1, -math.inf, math.inf, False, tt)
                board.pop()
                if value > best_value:
                    best_value = value
                    current_best = move
                if deadline and time.perf_counter() >= deadline:
                    break
        else:
            best_value = math.inf
            for move in moves:
                board.push(move)
                value = _minimax(board, genome, current_depth - 1, -math.inf, math.inf, True, tt)
                board.pop()
                if value < best_value:
                    best_value = value
                    current_best = move
                if deadline and time.perf_counter() >= deadline:
                    break

        best_move = current_best

    return best_move
