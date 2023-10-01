from fow_chess.board import Board


class FowState:
    def __init__(self, board: Board):
        self.board = board
        self.array = board.to_array()
        self.fen = board.to_fen()

    @classmethod
    def from_fen(cls, fen):
        return cls(Board(fen))

    def __hash__(self):
        return hash(self.board.to_fen())
