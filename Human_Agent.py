import pygame

class Human_Agent:

    def __init__(self, player: int) -> None:
        self.player = player

    def get_Action(self, event= None, state = None):
        if event.type == pygame.KEYUP:
            match event.key:
                case pygame.K_1: return 0
                case pygame.K_2: return 1
                case pygame.K_3: return 2
                case pygame.K_4: return 3
                case pygame.K_5: return 4
                case pygame.K_6: return 5
                case pygame.K_7: return 6
                case _: return 7 # different key pressed other than 1-7
        else:
            return None