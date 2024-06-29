import random
from State import State

class RandomAgent:

    def __init__(self, player: int) -> None:
        self.player = player

    def get_action(self, state: State = None):
        return random.randint(0,6)