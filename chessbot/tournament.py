"""Tournament system for evaluating genome fitness."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import chess

from chessbot.engine import search
from chessbot.evaluation import evaluate
from chessbot.genome import Genome


def play_game(
    white_genome: Genome,
    black_genome: Genome,
    depth: int = 2,
    max_moves: int = 100,
) -> dict:
    """Play a game between two genomes.

    A per-game transposition table is shared across all moves so that
    positions evaluated earlier in the game can be reused.

    Returns a dict with:
        - result: "white", "black", or "draw"
        - moves: number of moves played
        - move_list: list of SAN move strings
        - snapshots: list of evaluation snapshots every 10 moves
    """
    board = chess.Board()
    snapshots: list[dict] = []
    move_list: list[str] = []
    move_count = 0
    tt: dict = {}  # shared transposition table for the whole game

    while not board.is_game_over() and move_count < max_moves:
        genome = white_genome if board.turn == chess.WHITE else black_genome
        move = search(board, genome, depth=depth, tt=tt)
        if move is None:
            break
        # Record SAN before pushing (for readable notation)
        move_list.append(board.san(move))
        board.push(move)
        move_count += 1

        # Snapshot every 10 moves
        if move_count % 10 == 0:
            snapshots.append({
                "move": move_count,
                "white_eval": evaluate(board, white_genome),
                "black_eval": evaluate(board, black_genome),
                "fen": board.fen(),
            })

    # Determine result
    if board.is_checkmate():
        # The side whose turn it is got checkmated
        result = "black" if board.turn == chess.WHITE else "white"
    elif board.is_game_over() or move_count >= max_moves:
        result = "draw"
    else:
        result = "draw"

    return {
        "result": result,
        "moves": move_count,
        "move_list": move_list,
        "snapshots": snapshots,
    }


def _play_game_worker(args: tuple) -> tuple[int, int, dict]:
    """Worker function for parallel game execution."""
    i, j, white_genome, black_genome, depth, max_moves = args
    result = play_game(white_genome, black_genome, depth=depth, max_moves=max_moves)
    return i, j, result


def run_tournament(
    population: list[Genome],
    depth: int = 2,
    max_moves: int = 100,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[Genome], list[dict]]:
    """Run a tournament to assign fitness scores.

    For small populations (<=6): full round-robin.
    For larger populations: Swiss-style pairing (sqrt(N) opponents each).

    Games are played in parallel across CPU cores.

    Scoring: Win=3, Draw=1, Loss=0, plus a speed bonus for faster wins.

    Returns (population, game_records) where each game_record contains:
        - white_idx, black_idx: indices in the population
        - result, moves, move_list, snapshots: from play_game()
    """
    n = len(population)
    # Reset fitness
    for g in population:
        g.fitness = 0.0

    if n <= 6:
        # Full round-robin
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        # Swiss-style: each player faces sqrt(N) random opponents
        import numpy as np
        rng = np.random.default_rng()
        num_opponents = max(2, int(math.sqrt(n)))
        pairs_set: set[tuple[int, int]] = set()
        for i in range(n):
            opponents = rng.choice(
                [j for j in range(n) if j != i],
                size=min(num_opponents, n - 1),
                replace=False,
            )
            for j in opponents:
                pair = (min(i, j), max(i, j))
                pairs_set.add(pair)
        pairs = list(pairs_set)

    total_games = len(pairs)
    game_records: list[dict] = []

    # Build work items for parallel execution
    work_items = [
        (i, j, population[i], population[j], depth, max_moves)
        for i, j in pairs
    ]

    # Use parallel execution for larger workloads, sequential for small ones
    if total_games >= 4:
        with ProcessPoolExecutor() as executor:
            for game_idx, (i, j, result) in enumerate(executor.map(_play_game_worker, work_items)):
                _score_game(population, i, j, result, max_moves)
                game_records.append({"white_idx": i, "black_idx": j, **result})
                if progress_callback is not None:
                    progress_callback(game_idx + 1, total_games)
    else:
        for game_idx, (i, j, wg, bg, d, mm) in enumerate(work_items):
            result = play_game(wg, bg, depth=d, max_moves=mm)
            _score_game(population, i, j, result, max_moves)
            game_records.append({"white_idx": i, "black_idx": j, **result})
            if progress_callback is not None:
                progress_callback(game_idx + 1, total_games)

    return population, game_records


def _score_game(
    population: list[Genome],
    i: int,
    j: int,
    result: dict,
    max_moves: int,
) -> None:
    """Apply scoring to population based on game result."""
    speed_bonus = max(0, (max_moves - result["moves"]) / max_moves * 0.5)

    if result["result"] == "white":
        population[i].fitness += 3.0 + speed_bonus
    elif result["result"] == "black":
        population[j].fitness += 3.0 + speed_bonus
    else:
        population[i].fitness += 1.0
        population[j].fitness += 1.0
