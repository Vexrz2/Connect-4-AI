import numpy as np
from Graphics import *
import torch

class State:
    def __init__(self, board= None, player = 1) -> None:
        self.board = board
        self.player = player
        self.action : int = None
        self.last_action : tuple(int, int) = None

    def get_opponent(self):
        return -self.player

    def switch_player(self):
        self.player *= -1

    def __eq__(self, other) ->bool:
        return np.equal(self.board, other.board).all()

    def __hash__(self) -> int:
        return hash(repr(self.board))
    
    def copy(self):
        newBoard = np.copy(self.board)
        return State(board=newBoard, player=self.player)

    def toTensor (self, device = torch.device('cpu')) -> tuple:
        board_np = self.board.reshape(-1)
        board_tensor = torch.tensor(board_np, dtype=torch.int32, device=device)
        return board_tensor
    
    [staticmethod]
    def tensorToState (state_tensor, player):
        board = state_tensor.reshape([6,7]).cpu().numpy()
        return State(board, player=player)
    
    
    def isInside(self, row_col):
        row, col = row_col
        return 0 <= row < ROWS and 0 <= col < COLS
    