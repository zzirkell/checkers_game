import matplotlib.pyplot as plt
import numpy as np

from CheckersEnv import CheckersEnv
from DQLAgent import DQLAgent
from Dummy import Dummy
from QAgent import QAgent
from CheckersGUI import CheckersGUI
from copy import deepcopy


def play_game(num_episodes=1000):
    """Playing the game either agent vs agent or agent vs player"""
    if not playWithGui:
        player1 = q_agent
        playerNeg1 = dummy

        wins = np.zeros(num_episodes)
        losses = np.zeros(num_episodes)

        for episode in range(num_episodes):
            env.reset()
            print("Playing episode: " + str(episode))

            while True:
                valid_moves = env.get_valid_moves(env.player)
                if not valid_moves:
                    break

                state = env.board.flatten()
                if env.player == 1:
                    if isinstance(player1, DQLAgent):
                        action = player1.select_action(state, valid_moves)
                    else:
                        action = player1.select_action(valid_moves)
                else:
                    if isinstance(playerNeg1, DQLAgent):
                        action = playerNeg1.select_action(state, valid_moves)
                    else:
                        action = playerNeg1.select_action(valid_moves)

                env.step(action, env.player)
                env.player *= -1

            winner = env.check_game_winner()
            if winner == 1:
                wins[episode] = 1
                losses[episode] = 0
            elif winner == -1:
                wins[episode] = 0
                losses[episode] = 1

        cumulative_wins = np.cumsum(wins)
        cumulative_losses = np.cumsum(losses)

        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_wins, label='Wins', color='red')
        plt.plot(cumulative_losses, label='Losses', color='blue')
        plt.xlabel('Episodes')
        plt.ylabel('Results')
        plt.title('Learning Agent Performance')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        player1 = dql_agent
        CheckersGUI(env, player1)


def train(num_episodes=1000):
    """Training of the QL and DQL Agents"""
    print("Training the Q agent...")
    q_agent.epsilon = 1.0

    for episode in range(num_episodes):
        env.reset()
        print("Training episode: " + str(episode))

        while True:
            valid_moves = env.get_valid_moves(env.player)
            if not valid_moves:
                break

            action = q_agent.select_action(valid_moves)
            old_state = deepcopy(env.board)
            next_state, reward = env.step(action, env.player)

            q_agent.update_q_table(old_state, action, reward, next_state, env.get_valid_moves(env.player))

            env.player *= -1

        q_agent.epsilon = max(0.1, q_agent.epsilon - 0.01)

    print("Q training is done")
    q_agent.save()
    print("Q agent is saved")

    print("Training the DQL agent...")
    for episode in range(num_episodes):
        print("Episode " + str(episode))

        env.reset()
        state = env.board.flatten()
        state = np.reshape(state, [1, state_shape[1]])
        while True:
            # dql turn
            valid_moves = env.get_valid_moves(env.player)
            if not valid_moves:
                break

            action = dql_agent.select_action(state, valid_moves)
            _, reward, done = env.step(action, env.player)

            # dummy turn
            valid_moves = env.get_valid_moves(-env.player)
            if not valid_moves:
                break

            action = dummy.select_action(valid_moves)
            next_state, _, _ = env.step(action, -env.player)
            next_state = np.reshape(next_state.flatten(), [1, state_shape[1]])

            dql_agent.remember(state, action, reward, next_state, done)

        dql_agent.replay(batch_size=64, episode=episode)
        if episode % 100 == 0:
            dql_agent.save(f"checkers_dql_weights_{episode}.weights.h5")

    print("DQL training is done")


def load_agents():
    """Loads agents from files if they were trained before"""
    print("Loading the Q agent...")
    q_agent.load('saved_table.pkl')
    print("Q agent is ready")
    print("Loading the DQL agent...")
    dql_agent.load('checkers_dql_weights_100.weights.h5')
    print("DQL agent is ready")


env = CheckersEnv()
training = False
playWithGui = False

# Q agent
q_agent = QAgent(step_size=0.2, epsilon=0, env=env)

# Deep Q agent
state_shape = (1, 36)
action_size = 6 * 6 * 4
dql_agent = DQLAgent(env, state_shape, action_size)

# Dummy
dummy = Dummy(env)

if training:
    train(1000)
else:
    load_agents()
    play_game(1000)
