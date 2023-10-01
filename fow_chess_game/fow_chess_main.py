from functools import partial

import numpy as np
from fow_chess.board import Board
from fow_chess.chesscolor import ChessColor

from fow_chess_game.fow_state import FowState
from fow_chess_game.get_device import get_device
from fow_chess_game.value_network import ValueNetwork
from fow_chess_generator import (
    FogChessGenerator,
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
    S = FowState(board)  # .to_fen()  # board.to_array()
    O1 = board.to_fow_fen(ChessColor.WHITE)  # board.to_fow_array(ChessColor.WHITE)
    action_mask = get_action_mask(board)
    device = get_device()
    value_network = ValueNetwork().to(device)
    # setup start
    player1 = POMCP(
        partial(FogChessGenerator, ChessColor.WHITE),
        gamma=0.9,
        timeout=1000,
        no_particles=300,
        device=device,
        value_network=value_network,
    )
    player1.initialize(S, A, O1, lambda s: get_action_mask(s.board))

    O2 = board.to_fow_fen(ChessColor.BLACK)  # board.to_fow_array(ChessColor.BLACK)
    player2 = POMCP(
        partial(FogChessGenerator, ChessColor.BLACK),
        gamma=0.9,
        timeout=1000,
        no_particles=300,
        device=device,
        value_network=value_network,
    )
    player2.initialize(S, A, O2, lambda s: get_action_mask(s.board))

    # Calculate policy in a loop
    time = 0
    history = []
    while time <= 100:
        time += 1
        action = player1.Search()
        # print(ab.tree.nodes[-1][:4])
        white_move = action_index_to_move(board, action)
        move_str = str(white_move)
        print(move_str)
        history.append(move_str)
        winner = board.apply_move(white_move)
        if winner is not None:
            print("Winner is", winner)
            break
        observation1 = board.to_fow_fen(ChessColor.WHITE)  # choice(O)
        print(observation1)
        print(board)
        player1.tree.prune_after_action(action, observation1)
        player1.UpdateBelief(action, observation1)
        player1.train_value_network()

        action = player2.Search()
        # print(ab.tree.nodes[-1][:4])
        black_move = action_index_to_move(board, action)
        move_str = str(black_move)
        print(move_str)
        history.append(move_str)
        winner = board.apply_move(black_move)
        if winner is not None:
            print("Winner is", winner)
            break
        observation2 = board.to_fow_fen(ChessColor.BLACK)  # choice(O)
        print(observation2)
        print(board)
        player2.tree.prune_after_action(action, observation2)
        player2.UpdateBelief(action, observation2)
        player2.train_value_network()
    print(history)
