import random


class Dummy:

    def __init__(self, env):
        self.env = env

    def select_action(self, valid_moves):
        """
        Select an action using epsilon-greedy policy.
        """
        return random.choice(valid_moves)
