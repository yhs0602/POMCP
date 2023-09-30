from fow_chess.board import Board
from fow_chess.chesscolor import ChessColor
from fow_chess.piece import PieceType
from fow_chess.position import Position


def FogChessGenerator(s, act: int):
    """
    s: Current board state (FEN format)
    act: Player's action (a move)
    """
    # Extract the necessary details from the move (act)

    from_file = act // (8 * 73) + 1  # file
    from_rank = (act // 73) % 8 + 1  # rank
    c = act % (8 * 73) % 73
    promotion_type = PieceType.QUEEN
    if c < 56:  # normal move
        direction = c // 7  # N, NE, E, SE, S, SW, W, NW
        amount = c % 7
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
            raise ValueError("From rank is invalid in under promotion")
        to_position = Position(file=from_file + direction - 1, rank=to_rank)

    from_position = Position(from_file, from_rank)

    if to_position.rank != 1 and to_position.rank != 8:
        promotion_type = None

    # Create a board from the current state
    board = Board.from_array(s, 0)  # TODO: Manage fullmove number

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
        reward = -2
        return s, board.to_fow_array(board.side_to_move), reward

    # Apply the move
    winner_color = board.apply_move(the_move)

    # Generate observation
    observation = board.to_fow_array(board.side_to_move)

    # Determine the reward
    if winner_color == ChessColor.BLACK:
        reward = -1  # Assume you're playing as white for this example
    elif winner_color == ChessColor.WHITE:
        reward = 1
    else:
        reward = 0  # No win or loss

    next_state = (
        board.to_array()
    )  # Convert board state to FEN format for the next state

    return next_state, observation, reward
