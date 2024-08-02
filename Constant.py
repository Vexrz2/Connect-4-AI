WIDTH, HEIGHT = 700, 600
ROWS, COLS = 6, 7
SQUARE_SIZE = WIDTH//COLS
LINE_WIDTH = 3
HIGHLIGHT_LINE_WIDTH = 7
PADDING = SQUARE_SIZE //5
CIRCLE_RADIUS = SQUARE_SIZE/2 - 10
FPS = 60

#RGB
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
DARKBLUE = (0, 0, 200)
LIGHTGRAY = (211,211,211)
GREEN = (0, 128, 0)
YELLOW = (255,255,0)

# epsilon Greedy
epsilon_start = 1
epsilon_final = 0.01
epsilon_decay = 10000

# For minmax
positional_weights = [
        [3, 4, 5, 7, 5, 4, 3],
        [4, 6, 8, 10, 8, 6, 4],
        [5, 8, 11, 13, 11, 8, 5],
        [5, 8, 11, 13, 11, 8, 5],
        [4, 6, 8, 10, 8, 6, 4],
        [3, 4, 5, 7, 5, 4, 3],
    ]
