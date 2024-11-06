import numpy as np

from fow_chess_game.fow_chess_generator import action_index_to_move, get_action_mask
from fow_chess_game.fow_state import FowObservation
from fow_chess_game.pomcp2 import Model, State, Observation


class FowModel(Model):
    def __init__(self):
        super().__init__()

    def is_consistent(self, state: State, observation: Observation) -> bool:
        state_array = state.array
        observation_array = observation.array

        # board state

    def __call__(self, state: State, action: int) -> (State, Observation, float):
        board = state.board
        the_move = action_index_to_move(board, action)
        if not the_move:
            # raise ValueError("Invalid move")
            reward = -2
            return state, FowObservation(board.to_fow_fen(board.side_to_move)), reward, True
        board.apply_move(action)
        return state, board.to_fow_fen(board.side_to_move), 0


class FowState(State):
    action_space = np.arange(8 * 8 * 73)

    def __init__(self, board):
        super().__init__()
        self.board = board
        self.array = board.to_array()
        self.fen = board.to_fen()

    def is_terminal(self) -> bool:
        pass

    def sample_random_action(self) -> int:
        action_mask = get_action_mask(self.board)
        valid_actions = np.where(action_mask == 1)[0]
        action_idx = np.random.choice(valid_actions)
        action = self.action_space[action_idx]
        return action.item()
