from functools import partial

import numpy as np
from fow_chess.board import Board
from fow_chess.chesscolor import ChessColor

from fow_chess_game.fow_chess_generator import FogChessGenerator, get_action_mask
from fow_chess_game.fow_state import FowObservation
from fow_chess_game.get_device import get_device
from fow_chess_game.pomcp2 import POMCP2


def train_policy_network(all_game_data):
    pass


def fog_of_war_self_play_loop(mcts1, mcts2, num_games):
    all_game_data = []
    action_space = np.arange(8 * 8 * 73)
    for game in range(num_games):
        game_data = []
        board = Board()
        observation1 = FowObservation(board.to_fow_fen(ChessColor.WHITE))
        observation2 = FowObservation(board.to_fow_fen(ChessColor.BLACK))

        while True:
            if board.side_to_move == ChessColor.WHITE:
                action_mask = get_action_mask(board)
                action, policy = mcts1.search(observation1, action_mask)
            else:
                action_mask = get_action_mask(board)
                action, policy = mcts2.search(observation2, action_mask)

            game_data.append((board, policy, action))

            winner = board.apply_move(action)
            if winner is not None:
                break

            observation1 = FowObservation(board.to_fow_fen(ChessColor.WHITE))
            observation2 = FowObservation(board.to_fow_fen(ChessColor.BLACK))

        for data in game_data:
            all_game_data.append(data + (winner,))

        # You may choose to train policy networks after each game or after a set of games
        train_policy_network(all_game_data)

    return all_game_data


if __name__ == "__main__":
    gamma = 0.9
    timeout = 1000
    num_particles = 300
    device = get_device()
    value_network = ValueNetwork().to(device)
    player1 = POMCP2(
        partial(FogChessGenerator, ChessColor.WHITE),
        gamma=gamma,
        timeout=timeout,
        num_particles=num_particles,
        device=device,
        value_network=value_network,
    )

    player2 = POMCP2(
        partial(FogChessGenerator, ChessColor.BLACK),
        gamma=gamma,
        timeout=timeout,
        num_particles=num_particles,
        device=device,
        value_network=value_network,
    )
    fog_of_war_self_play_loop(
        mcts1=player1,
        mcts2=player2,
        num_games=100,
    )
