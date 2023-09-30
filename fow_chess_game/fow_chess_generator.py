from typing import Optional

import numpy as np
from fow_chess.chesscolor import ChessColor
from fow_chess.move import Move
from fow_chess.piece import PieceType
from fow_chess.position import Position
from fow_chess.board import Board


def move_to_action_index(move: Move) -> int:
    # Calculate base index based on the starting position of the piece
    base_idx = (move.piece.file - 1) * (8 * 73) + (move.piece.rank - 1) * 73

    # Calculate the offset based on move direction and distance
    dx = move.to_position.file - move.piece.file
    dy = move.to_position.rank - move.piece.rank

    if (abs(dx), abs(dy)) in [(1, 2), (2, 1)]:
        # It's a knight move
        knight_offsets = {
            (1, 2): 0,
            (2, 1): 1,
            (2, -1): 2,
            (1, -2): 3,
            (-1, -2): 4,
            (-2, -1): 5,
            (-2, 1): 6,
            (-1, 2): 7,
        }
        offset = 56 + knight_offsets[(dx, dy)]
    elif abs(dx) <= 7 and abs(dy) <= 7:
        # It's a queen move (or a simple pawn move, rook move, etc. that can be represented in the same way)
        direction_offsets = {
            (0, 1): 0,  # N
            (1, 1): 7,  # NE
            (1, 0): 14,  # E
            (1, -1): 21,  # SE
            (0, -1): 28,  # S
            (-1, -1): 35,  # SW
            (-1, 0): 42,  # W
            (-1, 1): 49,  # NW
        }
        offset = (
            direction_offsets[(np.sign(dx), np.sign(dy))] + max(abs(dx), abs(dy)) - 1
        )
    else:
        # Pawn underpromotion
        promotion_offsets = {
            PieceType.KNIGHT: 0,
            PieceType.BISHOP: 1,
            PieceType.ROOK: 2,
        }
        underpromotion_directions = {
            (-1, 1): 0,  # left diagonal (from white perspective)
            (0, 1): 3,  # front (from white perspective)
            (1, 1): 6,  # right diagonal (from white perspective)
            (-1, -1): 0,  # left diagonal (from black perspective)
            (0, -1): 3,  # right diagonal (from black perspective)
            (1, -1): 6,  # right diagonal (from black perspective)
        }
        offset = (
            63
            + underpromotion_directions[(dx, dy)]
            + promotion_offsets[move.promotion_piece]
        )

    return base_idx + offset


def get_action_mask(board):
    """Generate a binary mask indicating which actions are legal."""
    legal_moves = board.get_legal_moves(board.side_to_move)
    mask = np.zeros(8 * 8 * 73, dtype=int)  # 73 possible actions for each square

    for move_list in legal_moves.values():
        for move in move_list:
            action_idx = move_to_action_index(move)
            try:
                mask[action_idx] = 1
            except Exception:
                print(f"Invalid move: {move} to {action_idx}")
                raise

    return mask


def action_index_to_move(board, action_idx: int) -> Optional[Move]:
    from_file = action_idx // (8 * 73) + 1  # file
    from_rank = (action_idx // 73) % 8 + 1  # rank
    c = action_idx % (8 * 73) % 73
    promotion_type = PieceType.QUEEN
    # print(f"{act=} {from_file=} {from_rank=} {c=}")
    if c < 56:  # normal move
        direction = c // 7  # N, NE, E, SE, S, SW, W, NW
        amount = c % 7 + 1
        if direction == 0:  # N: increase rank, E: increase file
            to_position = Position(file=from_file, rank=from_rank + amount)
        elif direction == 1:  # NE
            to_position = Position(file=from_file + amount, rank=from_rank + amount)
        elif direction == 2:  # E
            to_position = Position(file=from_file + amount, rank=from_rank)
        elif direction == 3:  # SE
            to_position = Position(file=from_file + amount, rank=from_rank - amount)
        elif direction == 4:  # S
            to_position = Position(file=from_file, rank=from_rank - amount)
        elif direction == 5:  # SW
            to_position = Position(file=from_file - amount, rank=from_rank - amount)
        elif direction == 6:  # W
            to_position = Position(file=from_file - amount, rank=from_rank)
        elif direction == 7:  # NW
            to_position = Position(file=from_file - amount, rank=from_rank + amount)
        else:
            raise ValueError("Should not happen")
    elif c < 63:  # knight moves
        knight_direction = c % 7
        dx = [1, 2, 2, 1, -1, -2, -2, -1]
        dy = [2, 1, -1, -2, -2, -1, 1, 2]
        to_position = Position(
            file=from_file + dx[knight_direction], rank=from_rank + dy[knight_direction]
        )
    else:  # pawn moves, 9 kinds, 63..72
        kind = c - 63  # Underpromotion: left NBR front NBR right NBR
        direction = kind // 3  # 0 1 2
        under_pormotion = kind % 3  # KBR
        promotion_type = [PieceType.KNIGHT, PieceType.BISHOP, PieceType.ROOK][
            under_pormotion
        ]
        if from_rank == 7:
            to_rank = 8
        elif from_rank == 2:
            to_rank = 1
        else:
            # print(
            #     f"From rank is invalid in under promotion: {from_rank=} {c=} {kind=} {under_pormotion=} {direction=} {promotion_type=}"
            # )
            return None
        to_position = Position(file=from_file + direction - 1, rank=to_rank)

    from_position = Position(from_file, from_rank)

    if to_position.rank != 1 and to_position.rank != 8:
        promotion_type = None

    # Find the move from the legal moves
    legal_moves = board.get_legal_moves(board.side_to_move)
    the_move = None
    for move_list in legal_moves.values():
        for legal_move in move_list:
            if (
                legal_move.piece.rank == from_position.rank
                and legal_move.piece.file == from_position.file
                and legal_move.to_position.rank == to_position.rank
                and legal_move.to_position.file == to_position.file
                and legal_move.promotion_piece == promotion_type
            ):
                the_move = legal_move
                break

    if not the_move:
        # raise ValueError("Invalid move")
        return None
    return the_move


def FogChessGenerator(s, act: int):
    """
    s: Current board state (FEN format)
    act: Player's action (a move)
    """
    # Extract the necessary details from the move (act)
    board = Board(s)
    the_move = action_index_to_move(board, act)

    if not the_move:
        # raise ValueError("Invalid move")
        reward = -2
        return s, board.to_fow_fen(board.side_to_move), reward
        # return s, board.to_fow_array(board.side_to_move), reward

    # Apply the move
    winner_color = board.apply_move(the_move)

    # Generate observation
    observation = board.to_fow_fen(
        board.side_to_move
    )  # board.to_fow_array(board.side_to_move)

    # Determine the reward
    if winner_color == ChessColor.BLACK:
        reward = -1  # Assume you're playing as white for this example
    elif winner_color == ChessColor.WHITE:
        reward = 1
    else:
        reward = 0  # No win or loss

    # next_state = (
    #     board.to_array()
    # )  # Convert board state to FEN format for the next state
    next_state = board.to_fen()  # Convert board state to FEN format for the next state

    return next_state, observation, reward
