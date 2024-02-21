from Connect4 import Connect4
from State import State
import numpy as np



board = np.array([[0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0],
                 [1,0,0,0,0,0,1],
                 [1,0,0,0,0,-1,0],
                 [1,0,0,0,-1,0,0],
                 [1,0,0,-1,1,1,1]])

newState = State(board=board, player=-1)

environment = Connect4()

print(environment.checkNInARow(newState, 4), environment.checkGameWin(newState))