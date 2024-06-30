import numpy as np
import pygame
import time
from Constant import *

pygame.init()

class Graphics:
    def __init__(self, win, board : np.ndarray):
        self.board = board
        rows, cols = board.shape
        self.win = win
        self.rows = rows
        self.cols = cols

    def draw_Lines_Circles(self):
        for i in range(ROWS):
            pygame.draw.line(self.win, BLACK, (0, i * SQUARE_SIZE), (WIDTH, i * SQUARE_SIZE ), width=LINE_WIDTH)
        for i in range(COLS):
            pygame.draw.line(self.win, BLACK, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE , HEIGHT), width=LINE_WIDTH)

        for i in range(ROWS):
            for j in range(COLS):
                pygame.draw.circle(self.win, BLACK, ((j+0.5)*SQUARE_SIZE, (i+0.5)*SQUARE_SIZE), CIRCLE_RADIUS+LINE_WIDTH)
                pygame.draw.circle(self.win, WHITE, ((j+0.5)*SQUARE_SIZE, (i+0.5)*SQUARE_SIZE), CIRCLE_RADIUS)

    def draw_all_pieces(self):
        for row in range(ROWS):
            for col in range(COLS):
                if self.board[row][col] !=0 :
                    self.draw_piece((row, col), self.board[row][col])
            
    def draw_piece(self, row_col, player):
        center = self.calc_pos(row_col)
        color = self.calc_color(player)
        pygame.draw.circle(self.win,color , center, CIRCLE_RADIUS)
    
    def draw_piece_highlight(self, row_col, player):
        row, col = row_col
        pygame.draw.circle(self.win, BLACK, ((col+0.5)*SQUARE_SIZE, (row+0.5)*SQUARE_SIZE), CIRCLE_RADIUS+HIGHLIGHT_LINE_WIDTH)
        self.draw_piece(row_col, player)

    def calc_pos(self, row_col):
        row, col = row_col
        y = row * SQUARE_SIZE + SQUARE_SIZE//2
        x = col * SQUARE_SIZE + SQUARE_SIZE//2
        return x, y

    def calc_base_pos(self, row_col):
        row, col = row_col
        y = row * SQUARE_SIZE
        x = col * SQUARE_SIZE
        return x, y

    def calc_row_col(self, pos):
        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        return row, col

    def calc_color(self, player):
        if player == 1:
            return RED
        elif player == -1:
            return YELLOW
        else:
            return WHITE

    def draw(self):
        self.win.fill(DARKBLUE)
        self.draw_Lines_Circles()
        self.draw_all_pieces()
        
    def draw_sequence(self, rows_cols, player):
        for row, col in rows_cols:
            self.draw_piece_highlight((row,col), player)

    def draw_square(self, row_col, color):
        pos = self.calc_base_pos(row_col)
        pygame.draw.rect(self.win, color, (*pos, SQUARE_SIZE, SQUARE_SIZE))



    






