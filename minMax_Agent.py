from Connect4 import Connect4
from State import State
MAXSCORE = 100000

class minMax_Agent:

    def __init__(self, player, depth = 2, environment: Connect4 = None):
        self.player = player
        if self.player == 1:
            self.opponent = -1
        else:
            self.opponent = 1
        self.depth = depth
        self.environment : Connect4 = environment

    def evaluate(self, gameState : State): # TODO 
        score = self.environment.checkNInARow(gameState, 3)
        score += 10*self.environment.checkNInARow(gameState, 4)
        opponentState = State(gameState.board, self.opponent)
        score -= self.environment.checkNInARow(opponentState, 3)
        score -= 10*self.environment.checkNInARow(opponentState, 4)
        
        return score

    def get_Action(self, state: State):
        bestAction = self.minMax(state)[1]
        return bestAction

    def minMax(self, state:State):
        depth = 0
        return self.max_value(state, depth)
        
    def max_value(self, state:State, depth):
        
        value = -MAXSCORE
        # stop state
        if depth == self.depth or self.environment.checkGameWin(state) or self.environment.checkGameDraw(state):
            value = self.evaluate(state)
            return value, state.action
        
        # start recursion
        bestAction = None
        legal_actions = self.environment.get_actions(state)
        for action in legal_actions:
            newState = self.environment.next_state(action, state)
            newValue = self.min_value(newState,  depth + 1)[0]
            if newValue > value:
                value = newValue
                bestAction = action

        return value, bestAction 

    def min_value(self, state:State, depth):
        
        value = MAXSCORE

        # stop state
        if depth == self.depth or self.environment.checkGameWin(state) or self.environment.checkGameDraw(state):
            value = self.evaluate(state)
            return value, state.action
        
        # start recursion
        bestAction = None
        legal_actions = self.environment.get_actions(state)
        for action in legal_actions:
            newState = self.environment.next_state(action, state)
            newValue = self.max_value(newState,  depth + 1)[0]
            if newValue < value:
                value = newValue
                bestAction = action

        return value, bestAction 

