from fow_chess.board import Board
from fow_chess.chesscolor import ChessColor
from fow_chess.piece import PieceType


def FogChessGenerator(s, act: int):
    """
    s: Current board state (FEN format)
    act: Player's action (a move)
    """
    # Extract the necessary details from the move (act)

    x = act // (8 * 73)
    y = (act // 73) % 8
    c = act % (8 * 73) % 73
    if c < 56:  # normal move
        direction = c // 7  # N, NE, E, SE, S, SW, W, NW
        amount = c % 7
    elif c < 63:  # knight moves
        knight_direction = c % 7
    else:  # pawn moves, 9 kinds, 63..72
        kind = c - 63  # left KBR front KBR right KBR
        direction = kind // 3  # 0 1 2
        underpromote = kind % 3  # KBR

    from_position = (x, y)
    to_position = act.to_position
    promotion_piece = act.promotionPiece

    # Create a board from the current state
    board = Board(s)

    # Translate positions to the required format
    from_position_rank = int(from_position[1])
    from_position_file = ord(from_position[0]) - ord("a") + 1
    to_position_rank = int(to_position[1])
    to_position_file = ord(to_position[0]) - ord("a") + 1
    promotion_type = None
    if promotion_piece:
        promotion_type = PieceType(promotion_piece.lower())

    # Find the move from the legal moves
    legal_moves = board.get_legal_moves(board.side_to_move)
    the_move = None
    for move_list in legal_moves.values():
        for legal_move in move_list:
            if (
                legal_move.piece.rank == from_position_rank
                and legal_move.piece.file == from_position_file
                and legal_move.to_position.rank == to_position_rank
                and legal_move.to_position.file == to_position_file
                and legal_move.promotion_piece == promotion_type
            ):
                the_move = legal_move
                break

    if not the_move:
        raise ValueError("Invalid move")

    # Apply the move
    winner_color = board.apply_move(the_move)

    # Generate observation
    observation = board.get_visible_board()

    # Determine the reward
    if winner_color == ChessColor.BLACK:
        reward = -1  # Assume you're playing as white for this example
    elif winner_color == ChessColor.WHITE:
        reward = 1
    else:
        reward = 0  # No win or loss

    next_state = board.fen()  # Convert board state to FEN format for the next state

    return next_state, observation, reward
