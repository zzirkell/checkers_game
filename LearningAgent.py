import random
import pickle


class LearningAgent:

    def __init__(self, step_size, epsilon, env):
        self.step_size = step_size
        self.epsilon = epsilon
        self.env = env
        self.q_table = {}

    def save(self):
        with open('saved_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

    def state_to_key(self, board):
        return tuple(tuple(row) for row in board)

    def select_action(self, valid_moves):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_moves)
        else:
            state_key = self.state_to_key(self.env.board)
            q_values = [self.q_table.get((state_key, tuple(move)), 0) for move in valid_moves]
            max_q = max(q_values)
            best_moves = [valid_moves[i] for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_moves)

    def update_q_table(self, state, last_move, reward, next_state, next_valid_moves, debug=False):
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)

        if debug:
            print("Debug iteration")

        current_q = self.q_table.get((state_key, tuple(last_move)), 0)

        if next_valid_moves:
            future_q = max([self.q_table.get((next_state_key, tuple(move)), 0) for move in next_valid_moves])
        else:
            future_q = 0

        new_q = current_q + self.step_size * (reward + 0.9 * future_q - current_q)
        self.q_table[(state_key, tuple(last_move))] = new_q
