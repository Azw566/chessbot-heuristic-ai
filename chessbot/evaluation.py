"""Board evaluation with 5 sub-components, parameterized by a Genome."""

from __future__ import annotations

import chess

from chessbot.genome import Genome

# Center squares
CENTER_SQUARES = [chess.E4, chess.D4, chess.E5, chess.D5]


def eval_material(board: chess.Board, genome: Genome) -> float:
    """Sum of piece values (white - black)."""
    score = 0.0
    pv = genome.piece_values
    for piece_type in pv:
        score += len(board.pieces(piece_type, chess.WHITE)) * pv[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * pv[piece_type]
    return score


def eval_mobility(board: chess.Board) -> float:
    """Difference in legal move counts (white - black)."""
    # Count moves for the side to move
    current_moves = board.legal_moves.count()

    # Switch perspective
    board.push(chess.Move.null())
    try:
        opponent_moves = board.legal_moves.count()
    finally:
        board.pop()

    if board.turn == chess.WHITE:
        return float(current_moves - opponent_moves)
    else:
        return float(opponent_moves - current_moves)


def eval_center_control(board: chess.Board) -> float:
    """Occupation + attacks on the 4 center squares."""
    score = 0.0
    for sq in CENTER_SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            score += 0.5 if piece.color == chess.WHITE else -0.5
        score += len(board.attackers(chess.WHITE, sq)) * 0.25
        score -= len(board.attackers(chess.BLACK, sq)) * 0.25
    return score


def eval_king_safety(board: chess.Board) -> float:
    """Pawn shelter bonus minus enemy piece proximity penalty."""
    score = 0.0
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1.0 if color == chess.WHITE else -1.0
        king_sq = board.king(color)
        if king_sq is None:
            continue
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)

        # Pawn shelter: count friendly pawns within 1 file and 1-2 ranks ahead
        shelter = 0
        direction = 1 if color == chess.WHITE else -1
        for df in [-1, 0, 1]:
            f = king_file + df
            if f < 0 or f > 7:
                continue
            for dr in [1, 2]:
                r = king_rank + direction * dr
                if 0 <= r <= 7:
                    sq = chess.square(f, r)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        shelter += 1
        score += sign * shelter * 0.3

        # Enemy proximity penalty: count enemy pieces within Chebyshev distance 2
        enemy = not color
        penalty = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == enemy and piece.piece_type != chess.PAWN:
                dist = max(abs(chess.square_file(sq) - king_file),
                           abs(chess.square_rank(sq) - king_rank))
                if dist <= 2:
                    penalty += 1
        score -= sign * penalty * 0.2

    return score


def eval_pawn_structure(board: chess.Board) -> float:
    """Passed/connected bonuses, isolated/doubled penalties."""
    score = 0.0
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1.0 if color == chess.WHITE else -1.0
        pawns = board.pieces(chess.PAWN, color)
        pawn_files = [chess.square_file(sq) for sq in pawns]

        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)

            # Doubled: another friendly pawn on same file
            if pawn_files.count(f) > 1:
                score -= sign * 0.2

            # Isolated: no friendly pawn on adjacent files
            has_neighbor = any(
                chess.square_file(p) in (f - 1, f + 1) for p in pawns if p != sq
            )
            if not has_neighbor:
                score -= sign * 0.15

            # Passed: no enemy pawn ahead on same or adjacent files
            enemy_pawns = board.pieces(chess.PAWN, not color)
            direction = 1 if color == chess.WHITE else -1
            is_passed = True
            for ep in enemy_pawns:
                ef = chess.square_file(ep)
                er = chess.square_rank(ep)
                if abs(ef - f) <= 1:
                    if direction == 1 and er > r:
                        is_passed = False
                        break
                    if direction == -1 and er < r:
                        is_passed = False
                        break
            if is_passed:
                # Bonus scales with advancement
                advancement = r if color == chess.WHITE else (7 - r)
                score += sign * 0.1 * advancement

            # Connected: friendly pawn on adjacent file and same rank
            connected = any(
                chess.square_file(p) in (f - 1, f + 1) and chess.square_rank(p) == r
                for p in pawns if p != sq
            )
            if connected:
                score += sign * 0.1

    return score


def evaluate(board: chess.Board, genome: Genome) -> float:
    """Full board evaluation combining all 5 sub-evaluators.

    Returns a score from white's perspective: positive = white advantage.
    """
    # Terminal states
    if board.is_checkmate():
        return -10000.0 if board.turn == chess.WHITE else 10000.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0

    material = eval_material(board, genome)
    mobility = eval_mobility(board)
    center = eval_center_control(board)
    king_safety = eval_king_safety(board)
    pawn_struct = eval_pawn_structure(board)

    return (
        genome.w_material * material
        + genome.w_mobility * mobility
        + genome.w_center * center
        + genome.w_king_safety * king_safety
        + genome.w_pawn_structure * pawn_struct
    )
