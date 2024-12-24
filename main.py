import matplotlib.pyplot as plt
import numpy as np

from CheckersEnv import CheckersEnv
from DQLAgent import DQLAgent
from Dummy import Dummy
from LearningAgent import LearningAgent
from CheckersGUI import CheckersGUI
from copy import deepcopy

env = CheckersEnv()
training = False
playWithGui = False

# Q agent
agent1 = LearningAgent(step_size=0.1, epsilon=0.2, env=env)

# Deep Q agent
state_shape = (1, 6 * 6)
action_size = 6 * 6 * 4
agent2 = DQLAgent(env, state_shape, action_size)

# Dummy
dummy = Dummy(env)


def play_game():
    if not playWithGui:
        player1 = agent1
        playerNeg1 = dummy

        num_episodes = 1000
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

        plot_filename = "learning_results.png"
        plt.savefig(plot_filename)
        print(f"Learning results saved as '{plot_filename}'")
    else:
        player1 = agent2
        CheckersGUI(env, player1)


def train():
    print("Training the Q agent...")
    num_episodes = 1000

    for episode in range(num_episodes):
        env.reset()
        print("Training episode: " + str(episode))

        while True:
            valid_moves = env.get_valid_moves(env.player)
            if not valid_moves:
                break

            action = agent1.select_action(valid_moves)
            old_state = deepcopy(env.board)
            next_state, reward = env.step(action, env.player)

            agent1.update_q_table(old_state, action, reward, next_state, env.get_valid_moves(env.player))

            env.player *= -1

        agent1.epsilon = max(0.1, agent1.epsilon - 0.001)

    print("Q training is done")
    agent1.save()
    print("Q agent is saved")

    # print("Training the DQL agent...")
    # episodes = 201
    # for episode in range(episodes):
    #     print("Episode " + str(episode))
    #
    #     env.reset()
    #     state = env.board.flatten()
    #     state = np.reshape(state, [1, state_shape[1]])
    #     while True:
    #         valid_moves = env.get_valid_moves(env.player)
    #         if not valid_moves:
    #             break
    #
    #         action = agent2.select_action(state, valid_moves)
    #         next_state, reward = env.step(action, env.player)
    #         next_state = np.reshape(next_state.flatten(), [1, state_shape[1]])
    #
    #         done = env.check_game_winner() != 0
    #         agent2.remember(state, action, reward, next_state, done)
    #         state = next_state
    #
    #         valid_moves = env.get_valid_moves(-env.player)
    #         if not valid_moves:
    #             break
    #
    #         action = dummy.select_action(valid_moves)
    #         env.step(action, -env.player)
    #
    #     agent2.replay(batch_size=64, episode=episode)
    #     if episode % 100 == 0:
    #         agent2.save(f"checkers_dql_weights_{episode}.weights.h5")
    #
    # print("DQL training is done")


def load_agents():
    print("Loading the Q agent...")
    agent1.load('saved_table.pkl')
    print("Q agent is ready")
    print("Loading the DQL agent...")
    agent2.load('checkers_dql_weights_100.weights.h5')
    print("DQL agent is ready")


if training:
    train()
else:
    load_agents()
    play_game()
