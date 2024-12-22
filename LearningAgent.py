import numpy as np
import random
import pickle
from copy import copy

class LearningAgent:

    def __init__(self, step_size, epsilon, env):
        """
        Initialize the learning agent.
        :param step_size: Learning rate for Q-learning updates
        :param epsilon: Exploration rate for epsilon-greedy policy
        :param env: Environment instance (checkers_env)
        """
        self.step_size = step_size
        self.epsilon = epsilon
        self.env = env
        self.q_table = {}  # Q-table to store state-action values

    def save(self):
        with open('saved_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

    def state_to_key(self, board):
        """
        Convert the board state into a tuple key for the Q-table.
        """
        board_tuple = []
        for i in board:
            board_tuple.append(tuple(i))
        return tuple(board_tuple)

    # def evaluation(self, board):
    #     """
    #     Evaluate the board state.
    #     Reward based on piece count, Kings, and positional advantage.
    #     """
    #     player_1_score = np.sum(board == 1) + 1.5 * np.sum(board == 2)
    #     player_minus_1_score = np.sum(board == -1) + 1.5 * np.sum(board == -2)
    #     return player_1_score - player_minus_1_score

    def select_action(self, valid_moves):
        """
        Select an action using epsilon-greedy policy.
        """
        if random.random() < self.epsilon:
            return random.choice(valid_moves)  # Explore: random action

        # Exploit: Choose the action with the highest Q-value
        state_key = self.state_to_key(self.env.board)
        q_values = [self.q_table.get((state_key, tuple(move)), 0) for move in valid_moves]
        max_q = max(q_values)
        best_moves = [valid_moves[i] for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_moves)

    def update_q_table(self, state, lastMove, reward, next_state, next_valid_moves, debug):
        """
        Update the Q-table using the Q-learning update rule.
        """
        state_key = self.state_to_key(state)
        action_key = tuple(lastMove)

        # Compute max Q-value for the next state
        max_next_q = 0
        if next_valid_moves:
            if debug:
                print("A")
            tempEnv = copy(self.env)
            temp = []
            for m in next_valid_moves:
                next_state, next_reward = tempEnv.step(m, -self.env.player)
                next_state_key = self.state_to_key(next_state)
                temp.append(self.q_table.get((next_state_key, tuple(m)), 0))
            max_next_q = max(temp)
        # if max_next_q != 0:
        #     print(max_next_q)

        # Q-learning update rule
        current_q = self.q_table.get((state_key, action_key), 0)
        self.q_table[(state_key, action_key)] = current_q + self.step_size * (reward + max_next_q - current_q)
