from Random_Agent import Random_Agent
from Connect4 import Connect4
from DQN_Agent import DQN_Agent
from AlphaBeta_Agent import AlphaBetaAgent
from minMax_Agent import minMax_Agent

class Tester:
    def __init__(self, env, player1, player2) -> None:
        self.env = env
        self.player1 = player1
        self.player2 = player2
        

    def test (self, games_num):
        env = self.env
        player = self.player1
        player1_win = 0
        player2_win = 0
        games = 0
        while games < games_num:
            action = player.get_Action(state=env.state, train = False)
            env.move(action, env.state)
            player = self.switchPlayers(player)
            if env.is_end_of_game(env.state):
                games += 1
                if env.checkGameDraw(env.state):
                    env.state = env.get_init_state()
                    player = self.player1
                    continue
                if player!=self.player2:
                    player2_win += 1
                else: player1_win += 1
                env.state = env.get_init_state()
                player = self.player1
            
        return player1_win, player2_win        

    def switchPlayers(self, player):
        if player == self.player1:
            return self.player2
        else:
            return self.player1

    def __call__(self, games_num):
        return self.test(games_num)
    

tests = 1000

if __name__ == '__main__':
    env = Connect4()
    player1 = DQN_Agent(env=env, player=1, train=False, parameters_path="Data/params_4.pth")
    player2 = Random_Agent(player=-1)
    test = Tester(env,player1, player2)
    print(test.test(tests))
    player1 = Random_Agent(player=1)
    player2 = DQN_Agent(env=env, player=-1, train=False, parameters_path="Data/params_4.pth")
    test = Tester(env,player1, player2)
    print(test.test(tests))