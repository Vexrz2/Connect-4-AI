import pygame
from Graphics import *
from Connect4 import Connect4
from Human_Agent import Human_Agent
from Random_Agent import Random_Agent
from minMax_Agent import minMax_Agent
from AlphaBeta_Agent import AlphaBetaAgent
from Constant import *
from DQN_Agent import DQN_Agent
import time
   


win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Connect 4')
environment = Connect4()
graphics = Graphics(win, board = environment.state.board)

player1 = DQN_Agent(env=environment, player=1, train=False, parameters_path="Data/params_1.pth")
player2 = Human_Agent(player=-1)


def main ():
    run = True
    clock = pygame.time.Clock()
    graphics.draw()
    player = player1

    while(run):
        clock.tick(FPS)
        
        if not isinstance(player, Human_Agent): # Computer playing
            action = player.get_Action(state=environment.state, train=False)
            if (environment.move(action, environment.state)):
                player = switchPlayers(player)
                run = checkEndGame(player)
        else:
            for event in pygame.event.get(): # Human playing
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYUP:
                    action = player.get_Action(event, environment.state)
                    if 0 <= action < 7:
                        if environment.move(action, environment.state):
                            player = switchPlayers(player)
                            run = checkEndGame(player)
                    elif action == 7:
                        print("Invalid move!")
        #print(environment.checkNInARow(environment.state, 4), -environment.state.player)
        graphics.draw() # Update graphics
        pygame.display.update()
        #time.sleep(1)
        
    time.sleep(.2) 
    pygame.quit() # End game

  
def switchPlayers(player):
    if player == player1:
       return player2
    else:
        return player1

def checkEndGame(player):
    if environment.checkGameWin(environment.state):
        if player != player1:
            playerName = "red"
        else: 
            playerName = "blue"
        print(f"Player {playerName} wins!")
        return False
    if environment.checkGameDraw(environment.state):
        print("Game draw.")
        return False
    return True

if __name__ == '__main__':
    main()
    
