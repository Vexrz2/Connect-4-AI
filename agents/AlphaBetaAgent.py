from Connect4 import Connect4
from State import State
MAXSCORE = 100000

class AlphaBetaAgent:
    def __init__(self, player, depth = 4, environment: Connect4 = None):
        self.player = player
        if self.player == 1:
            self.opponent = -1
        else:
            self.opponent = 1
        self.depth = depth
        self.environment : Connect4 = environment

    def evaluate (self, gameState : State): 
        score = 6*self.environment.get_n_sequences(gameState, 2) + 40*self.environment.get_n_sequences(gameState, 3) + 3000*self.environment.get_n_sequences(gameState, 4)
        
        opponentState = State(gameState.board, self.opponent)
        score -= 6*self.environment.get_n_sequences(opponentState, 2) + 40*self.environment.get_n_sequences(opponentState, 3) + 3000*self.environment.get_n_sequences(opponentState, 4)
        
        return score
    
    def get_action(self, state: State) -> int:
        value, bestAction = self.min_max(state)
        return bestAction

    def min_max(self, state:State):
        depth = 0
        alpha = -MAXSCORE
        beta = MAXSCORE
        return self.max_value(state, depth, alpha, beta)
        
    def max_value(self, state:State, depth, alpha, beta):
        
        value = -MAXSCORE

        # stop state
        if depth == self.depth or self.environment.is_end_of_game(state):
            value = self.evaluate(state)
            state.last_action[1]
            return value, state.last_action[1]
        
        # start recursion
        bestAction = None
        legal_actions = self.environment.get_actions(state)
        for action in legal_actions:
            newState = self.environment.next_state(action, state)
            newValue = self.min_value(newState, depth + 1, alpha, beta)[0]
            if newValue > value:
                value = newValue
                bestAction = action
                alpha = max(alpha, value)
            if value >= beta:
                return value, bestAction
                    
        
        return value, bestAction 

    def min_value (self, state:State, depth, alpha, beta):
        
        value = MAXSCORE
          
        # stop state
        if depth == self.depth or self.environment.is_end_of_game(state):
            value = self.evaluate(state)
            return value, state.last_action[1]
        
        # start recursion
        bestAction = None
        legal_actions = self.environment.get_actions(state)
        for action in legal_actions:
            newState = self.environment.next_state(action, state)
            newValue = self.max_value(newState,  depth + 1, alpha, beta)[0]
            if newValue < value:
                value = newValue
                bestAction = action
                beta = min(beta, value)
            if value <= alpha:
                return value, bestAction

        return value, bestAction 

