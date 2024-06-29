import numpy as np
from State import State
from Constant import *
import torch

class Connect4:
    def __init__(self, state:State = None) -> None:
        if state == None:
            self.state = self.get_init_state()
        else:
            self.state = state

    def get_init_state(self):
        board = np.zeros([ROWS, COLS],int)
        return State (board, 1) # Board of zeroes, player 1 starts.

    def check_legal_col(self, col, state: State):
        return state.board[0][col] == 0 # Checks if top cell in column is empty

    def get_actions(self, state: State = None):
        actions = []
        for col in range(COLS):
            if self.check_legal_col(col,state):
                actions.append(col)
        return actions

    def move(self, col, state: State):
        if self.check_legal_col(col, state): # Col open
            row = ROWS-1
            while (state.board[row][col] != 0):
                row = row-1
            state.board[row][col] = state.player
            state.last_action = (row,col)
            state.switch_player()
            return True
        return False

    def next_state(self, col, state: State) -> State:
        stateCopy = state.copy()
        self.move(col, stateCopy)
        return stateCopy  
    
    def check_game_win(self, state : State): # Check for win
        if state == self.get_init_state() or not state.last_action: # Still start of game
            return False
        row_col = state.last_action
        state.switch_player() # switch to player that played last action
        win = (self.check_vertical(row_col, state, 4) or self.check_horizontal(row_col, state, 4) 
               or self.check_main_diagonal(row_col, state, 4) or self.check_minor_diagonal(row_col, state, 4))
        state.switch_player()
        return win

    def check_vertical(self, row_col, state:State, length): 
        row, col = row_col
        for startRow in range(max(0,row-length+1), min(row+1,ROWS-length+1)):
            colCheck = state.board[startRow:startRow+length,col]
            if sum(colCheck) == length*state.player: # Horizontal four in a row
                #print(f"vertical check: {row}, {startRow}, {colCheck}")
                return True
        return False

    def check_horizontal(self, row_col, state:State, length): 
        row, col = row_col
        for startCol in range(max(0,col-length+1), min(col+1,COLS-length+1)):
            rowCheck = state.board[row][startCol:startCol+length]
            if sum(rowCheck) == length*state.player: # Horizontal four in a row
                #print(f"horizontal check: {row}, {startCol}, {rowCheck}")
                return True
        return False

    def check_main_diagonal(self, row_col, state:State, length):
        row, col = row_col
        offset = col - row # find offset from main diagonal

        startRow = row - length + 1
        startCol = col - length + 1
        if offset < -2 or offset > 3:  # starting point of diagonal check is outside boundaries
                return False
        
        startRow = max(0, startRow)
        startCol = max(0, startCol) # fix start point if valid

        if offset >= 0:
            startPoint = startRow # start point above main diagonal
            currentDiagonal = np.diag(state.board, offset)
            endPoint = min(len(currentDiagonal) - length, row)
            for i in range(startPoint, endPoint+1):
                diagCheck = currentDiagonal[i:i+length]
                if sum(diagCheck) == length*state.player:
                    return True
        else:
            startPoint = startCol # below main diagonal
            currentDiagonal = np.diag(state.board, offset)
            endPoint = min(len(currentDiagonal) - length, col)
            for i in range(startPoint, endPoint+1):
                diagCheck = currentDiagonal[i:i+length]
                if sum(diagCheck) == length*state.player:
                    return True
        return False

    def check_minor_diagonal(self, row_col, state:State, length):
        flipBoard = np.flipud(state.board)
        row, col = row_col
        if row < ROWS // 2:
            flipRow = ROWS - row - 1
        else:
            flipRow = ROWS - row + 1
        flipRowCol = (flipRow, col)
        flipState = State(flipBoard, state.player)

        return self.check_main_diagonal(flipRowCol, flipState, length)

    def is_end_of_game(self, state: State):
        return self.check_game_draw(state) or self.check_game_win(state)
    
    def check_game_draw(self, state: State): # Checks if the game has come to a draw
        for col in range(0, COLS):
            if (self.check_legal_col(col, state)):
                return False
        return True

    def get_n_sequences(self, state: State, n): # Returns amount of sequences in board of length N (N in a row)
        total = 0
        stateCopy = state.copy()
        for i in range(ROWS):
            for j in range(COLS):
                row_col = (i,j)
                if (self.check_vertical(row_col, stateCopy, n) or self.check_horizontal(row_col, stateCopy, n) 
                or self.check_main_diagonal(row_col, stateCopy, n) or self.check_minor_diagonal(row_col, stateCopy, n)):
                    total += 1

        return total
    
    def get_all_next_states(self, state: State) -> tuple:
        legal_actions = self.get_actions(state)
        next_states = []
        for action in legal_actions:
            next_states.append(self.next_state(action, state))
        return next_states, legal_actions

    def to_tensor (self, list_states : list[State], device = torch.device('cpu')) -> tuple:
        list_board_tensors = []
        list_legal_actions = []
        for state in list_states:
            board_tensor, legal_actions = state.to_tensor(device) 
            list_board_tensors.append(board_tensor)
            list_legal_actions.append(torch.tensor(legal_actions))
        return torch.vstack(list_board_tensors), torch.vstack(list_legal_actions)
    
    def reward (self, state : State, action = None) -> tuple:
        if action:
            next_state = self.next_state(action, state)
        else:
            next_state = state
        if (self.is_end_of_game(next_state)):
            if self.check_game_draw(next_state):
                return 0, True # Draw
            return -state.player, True # Win
        return 0, False # Not end of game

        
        

    

