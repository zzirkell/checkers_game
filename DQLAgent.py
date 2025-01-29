import random
import numpy as np
from collections import deque
from keras import Input
from tensorflow import keras


class DQLAgent:
    def __init__(self, env, state_shape, action_size, gamma=0.9, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 learning_rate=0.001):
        """Initialization of the class"""
        self.env = env
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=10000)
        self.model = self._build_model()
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        """Initialization of the model"""
        model = keras.Sequential()
        model.add(Input(shape=(36,)))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Adds info about the last step to the memory"""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, valid_moves):
        """Selects an action for the next move"""
        if np.random.rand() < self.epsilon:
            return random.choice(valid_moves)
        q_values = self.model.predict(state, verbose=0)[0]
        move_q_values = [q_values[move] for move in valid_moves]
        best_move = valid_moves[np.argmax(move_q_values) // 4]
        return best_move

    def replay(self, batch_size=64, episode=0):
        """Reuses experience from memory"""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if episode % 10 == 0:
            self.target_model.set_weights(self.model.get_weights())

    def load(self, path):
        """Loads the agent"""
        self.model.load_weights(path)

    def save(self, path):
        """Saves the agent"""
        self.model.save_weights(path)
