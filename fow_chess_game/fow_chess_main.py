import numpy as np
from fow_chess.board import Board
from fow_chess.chesscolor import ChessColor
from fow_chess_generator import (
    FogChessGenerator as Generator,
    get_action_mask,
    action_index_to_move,
)
from pomcp import POMCP

if __name__ == "__main__":
    A = np.arange(8 * 8 * 73)
    """
    [In AlphaChessZero, the] action space is a 8x8x73 dimensional array. Each of the 8×8 positions identifies the square
     from which to “pick up” a piece. The first 56 planes encode possible ‘queen moves’ for any piece: a number of
      squares [1..7] in which the piece will be moved, along one of eight relative compass directions 
      {N, NE, E, SE, S, SW, W, NW}. The next 8 planes encode possible knight moves for that piece. The final 9 planes
       encode possible underpromotions for pawn moves or captures in two possible diagonals, to knight, bishop or rook
        respectively. Other pawn moves or captures from the seventh rank are promoted to a queen.
    """
    board = Board()
    S = board.to_fen()  # board.to_array()
    O = board.to_fow_fen(ChessColor.WHITE)  # board.to_fow_array(ChessColor.WHITE)
    action_mask = get_action_mask(board)

    # setup start
    ab = POMCP(Generator, gamma=0.5, timeout=100)
    ab.initialize(S, A, O, lambda s: get_action_mask(Board(s)))

    # Calculate policy in a loop
    time = 0
    while time <= 100:
        time += 1
        action = ab.Search()
        # print(ab.tree.nodes[-1][:4])
        print(action_index_to_move(board, action))
        board.apply_move(action_index_to_move(board, action))
        observation = board.to_fow_fen(board.side_to_move)  # choice(O)
        print(observation)
        print(board)
        ab.tree.prune_after_action(action, observation)
        ab.UpdateBelief(action, observation)
