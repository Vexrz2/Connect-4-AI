import numpy as np
from Graphics import *
import torch

class State:
    def __init__(self, board : np.ndarray = None, player = 1) -> None:
        self.board = board 
        self.player = player
        self.last_action : tuple[int, int] = None

    def get_opponent(self):
        return -self.player

    def switch_player(self):
        self.player *= -1

    def __eq__(self, other) ->bool:
        if other == None: 
            return False
        return np.equal(self.board, other.board).all()

    def __hash__(self) -> int:
        return hash(repr(self.board))
    
    def copy(self):
        newBoard = np.copy(self.board)
        return State(board=newBoard, player=self.player)

    def to_tensor (self, device = torch.device('cpu')) -> torch.Tensor:
        board_np = self.board.reshape(-1)
        board_tensor = torch.tensor(board_np, dtype=torch.int32, device=device)
        return board_tensor
    
    [staticmethod]
    def tensor_to_state (state_tensor : torch.Tensor, player):
        board = state_tensor.reshape([6,7]).cpu().numpy()
        return State(board, player=player)
    
    def reverse (self):
        reversed = self.copy()
        reversed.board = reversed.board * -1
        reversed.player = reversed.player * -1
        return reversed
    