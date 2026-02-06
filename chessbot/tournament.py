"""Tournament system for evaluating genome fitness."""

from __future__ import annotations

import math
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

    Returns a dict with:
        - result: "white", "black", or "draw"
        - moves: number of moves played
        - move_list: list of UCI move strings
        - snapshots: list of evaluation snapshots every 10 moves
    """
    board = chess.Board()
    snapshots: list[dict] = []
    move_list: list[str] = []
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        genome = white_genome if board.turn == chess.WHITE else black_genome
        move = search(board, genome, depth=depth)
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


def run_tournament(
    population: list[Genome],
    depth: int = 2,
    max_moves: int = 100,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[Genome], list[dict]]:
    """Run a tournament to assign fitness scores.

    For small populations (<=10): full round-robin.
    For larger populations: Swiss-style pairing (sqrt(N) opponents each).

    Scoring: Win=3, Draw=1, Loss=0, plus a speed bonus for faster wins.

    Returns (population, game_records) where each game_record contains:
        - white_idx, black_idx: indices in the population
        - result, moves, move_list, snapshots: from play_game()
    """
    n = len(population)
    # Reset fitness
    for g in population:
        g.fitness = 0.0

    if n <= 10:
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

    for game_idx, (i, j) in enumerate(pairs):
        result = play_game(population[i], population[j], depth=depth, max_moves=max_moves)

        # Speed bonus: faster wins get a small bonus
        speed_bonus = max(0, (max_moves - result["moves"]) / max_moves * 0.5)

        if result["result"] == "white":
            population[i].fitness += 3.0 + speed_bonus
        elif result["result"] == "black":
            population[j].fitness += 3.0 + speed_bonus
        else:
            population[i].fitness += 1.0
            population[j].fitness += 1.0

        game_records.append({
            "white_idx": i,
            "black_idx": j,
            **result,
        })

        if progress_callback is not None:
            progress_callback(game_idx + 1, total_games)

    return population, game_records
