import random


class Dummy:

    def __init__(self, env):
        """Initialization"""
        self.env = env

    def select_action(self, valid_moves):
        """Selects random move from the list"""
        return random.choice(valid_moves)
