import numpy as np
from fow_chess.board import Board
from fow_chess.chesscolor import ChessColor
from fow_chess_generator import FogChessGenerator as Generator
from numpy.random import choice
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
    S = board.to_array()
    O = board.to_fow_array(ChessColor.WHITE)

    # setup start
    ab = POMCP(Generator, gamma=0.5)
    ab.initialize(S, A, O)

    # Calculate policy in a loop
    time = 0
    while time <= 10:
        time += 1
        action = ab.Search()
        print(ab.tree.nodes[-1][:4])
        print(action)
        observation = choice(O)
        ab.tree.prune_after_action(action, observation)
        ab.UpdateBelief(action, observation)
