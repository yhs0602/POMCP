from functools import partial

import numpy as np
from fow_chess.board import Board
from fow_chess.chesscolor import ChessColor

from fow_chess_game.fow_state import FowObservation, FowState
from fow_chess_game.get_device import get_device
from fow_chess_game.value_network import ValueNetwork
from fow_chess_generator import (
    FogChessGenerator,
    get_action_mask,
    action_index_to_move,
)
from pomcp import POMCP

if __name__ == "__main__":
    action_space = np.arange(8 * 8 * 73)
    """
    [In AlphaChessZero, the] action space is a 8x8x73 dimensional array. Each of the 8×8 positions identifies the square
     from which to “pick up” a piece. The first 56 planes encode possible ‘queen moves’ for any piece: a number of
      squares [1..7] in which the piece will be moved, along one of eight relative compass directions 
      {N, NE, E, SE, S, SW, W, NW}. The next 8 planes encode possible knight moves for that piece. The final 9 planes
       encode possible underpromotions for pawn moves or captures in two possible diagonals, to knight, bishop or rook
        respectively. Other pawn moves or captures from the seventh rank are promoted to a queen.
    """
    board = Board()
    device = get_device()
    value_network = ValueNetwork().to(device)
    # setup start
    gamma = 0.9
    timeout = 1000
    num_particles = 300
    player1 = POMCP(
        partial(FogChessGenerator, ChessColor.WHITE),
        gamma=gamma,
        timeout=timeout,
        num_particles=num_particles,
        device=device,
        value_network=value_network,
    )

    player2 = POMCP(
        partial(FogChessGenerator, ChessColor.BLACK),
        gamma=gamma,
        timeout=timeout,
        num_particles=num_particles,
        device=device,
        value_network=value_network,
    )

    # Calculate policy in a loop
    time = 0
    history = []

    white_initialized, black_initialized = False, False
    while time <= 100:
        time += 1
        print("White turn ======================")
        state = FowState(board)
        observation1 = FowObservation(board.to_fow_fen(ChessColor.WHITE))
        if not white_initialized:
            player1.initialize(
                state, action_space, observation1, lambda s: get_action_mask(s.board)
            )
            white_initialized = True
        else:
            # player1.tree.prune_after_action(action_white, observation1)
            player1.update_belief(action_white, observation1)
            player1.train_value_network()
        action_white = player1.search()
        # print(ab.tree.nodes[-1][:4])
        white_move = action_index_to_move(board, action_white)
        print(f"{action_white=} {white_move=}")
        history.append(str(white_move))
        winner = board.apply_move(white_move)
        if winner is not None:
            print("Winner is", winner)
            break
        observation1 = FowObservation(board.to_fow_fen(ChessColor.WHITE))  # choice(O)
        print(observation1)
        print(board)

        print("Black turn ======================")
        state = FowState(board)
        observation2 = FowObservation(board.to_fow_fen(ChessColor.BLACK))
        if not black_initialized:
            player2.initialize(
                state, action_space, observation2, lambda s: get_action_mask(s.board)
            )
            black_initialized = True
        else:
            # player2.tree.prune_after_action(action_black, observation2)
            player2.update_belief(action_black, observation2)
            player2.train_value_network()
        action_black = player2.search()
        # print(ab.tree.nodes[-1][:4])
        black_move = action_index_to_move(board, action_black)
        print(f"{action_black=} {black_move=}")
        history.append(str(black_move))
        winner = board.apply_move(black_move)
        if winner is not None:
            print("Winner is", winner)
            break
        observation2 = FowObservation(board.to_fow_fen(ChessColor.BLACK))  # choice(O)
        print(observation2)
        print(board)
    print(history)
