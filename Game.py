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
player1 = DQN_Agent(player=1, env=environment, train=False, parameters_path="Data/params_1.pth")
#player2 = Human_Agent(player=2)
#player1 = AlphaBetaAgent(player=1, environment=environment)
player2 = Random_Agent(player=-1)


def main ():
    run = True
    clock = pygame.time.Clock()
    graphics.draw()
    player = player1
    wins = [0,0]
    games = 100 # testing
    while(run):
        clock.tick(FPS)

        if not isinstance(player, Human_Agent): # Computer playing
            action = player.get_Action(state=environment.state)
            if (environment.move(action, environment.state)):
                run = checkEndGame(player, wins)
                player = switchPlayers(player)
                if not run and games:
                    games -= 1
                    environment.state.board[:] = 0 
                    run = True
                    print(games)
        else:
            for event in pygame.event.get(): # Human playing
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYUP:
                    action = player.get_Action(event, environment.state)
                    if 0 <= action < 7:
                        if environment.move(action, environment.state):
                            run = checkEndGame(player)
                            player = switchPlayers(player)
                    elif action == 7:
                        print("Invalid move!")

        graphics.draw() # Update graphics
        pygame.display.update()
        
    print(f"dqn vs random win rate: {wins[0]}%")
    time.sleep(.2) 
    pygame.quit() # End game

  
def switchPlayers(player):
    if player == player1:
       return player2
    else:
        return player1

def checkEndGame(player, wins =[0,0]):
    if environment.checkGameWin(environment.state):
        if player == player1:
            playerName = "red"
            wins[0] += 1
        else: 
            playerName = "blue"
            wins[1] += 1
        print(f"Player {playerName} wins!")
        return False
    if environment.checkGameDraw(environment.state):
        print("Game draw.")
        return False
    return True

if __name__ == '__main__':
    main()
    
