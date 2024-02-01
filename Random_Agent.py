import pygame
import random
from State import State

class Random_Agent:

    def __init__(self, player: int) -> None:
        self.player = player

    def get_Action (self, events = None, graphics=None, state: State = None, epoch = 0, train = None):
        return random.randint(0,6)