import pygame
from Graphics import *
from Connect4 import Connect4
from agents import HumanAgent, RandomAgent, minMaxAgent, AlphaBetaAgent, DQNAgent
from Constant import *
import time

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Connect 4')
environment = Connect4()
graphics = Graphics(win, board = environment.state.board)

# player1 = HumanAgent(player=1)
#player1 = RandomAgent(player=1)
# player1 = AlphaBetaAgent(player=1, depth=4, environment=environment)
player1 = DQNAgent(env=environment, player=1, train=False, parameters_path="data/params_1.pth")

player2 = HumanAgent(player=-1)
#player2 = RandomAgent(player=-1)
# player2 = AlphaBetaAgent(player=-1, depth=4, environment=environment)
# player2 = DQNAgent(env=environment, player=-1, train=False, parameters_path="Data/params_4.pth")

def main ():
    run = True
    clock = pygame.time.Clock()
    graphics.draw()
    player = player1

    while(run):
        clock.tick(FPS)
        
        if not isinstance(player, HumanAgent): # Computer playing
            action = player.get_action(state=environment.state)
            if (environment.move(action, environment.state)):
                run = check_end_game(player)
                player = switch_players(player)
        else:
            for event in pygame.event.get(): # Human playing
                if event.type == pygame.QUIT:
                    run = False
                    break
                if event.type == pygame.KEYUP:
                    action = player.get_action(event)
                    if 0 <= action <= 6: 
                        if environment.move(action, environment.state):
                            run = check_end_game(player)
                            player = switch_players(player)
                        else:
                            print("Column full!")
                    elif action == 7:
                        print("Invalid move!")
                    break
        graphics.draw() # Update graphics
        pygame.display.update()
        
    
    pygame.quit() # End game
  
def switch_players(player):
    if player == player1:
       return player2
    else:
        return player1

def check_end_game(player):
    isWin, winning_sequence = environment.check_game_win(environment.state)
    isDraw = environment.check_game_draw(environment.state)
    if isWin:
        graphics.draw_sequence(winning_sequence, -environment.state.player)
        if player == player1:
            playerName = "red"
        else: 
            playerName = "yellow"
        pygame.display.set_caption(f"Player {playerName} wins!")
        pygame.display.update()
        time.sleep(2)
        print(f"Player {playerName} wins!")
        return False
    if isDraw:
        pygame.display.set_caption("Game draw.")
        pygame.display.update()
        print("Game draw.")
        return False
    return True

if __name__ == '__main__':
    main()
    
