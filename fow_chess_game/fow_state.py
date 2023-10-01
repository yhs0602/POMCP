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
        return hash(self.fen)

    def __eq__(self, other):
        return self.fen == other.fen

    def __repr__(self):
        return self.fen

    def __str__(self):
        return self.fen


class FowObservation:
    def __init__(self, fow_fen: str):
        self.fow_fen = fow_fen

    def __eq__(self, other):
        if not isinstance(other, FowObservation):
            return False
        pieces, turn, castle, en_passant, halfmove, full_move = self.fow_fen.split(" ")
        (
            other_pieces,
            other_turn,
            other_castle,
            other_en_passant,
            other_halfmove,
            other_full_move,
        ) = other.fow_fen.split(" ")
        # if not pieces == other_pieces:
        #     print("Pieces not equal")

        return (
            pieces == other_pieces
            and castle == other_castle
            and en_passant == other_en_passant
        )

    def __str__(self):
        return self.fow_fen

    def __hash__(self):
        return hash(self.fow_fen)

    def __repr__(self):
        return str(self)
