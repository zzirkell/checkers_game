import math

import matplotlib.pyplot as plt
import numpy as np

from CheckersEnv import CheckersEnv
from LearningAgent import LearningAgent

env = CheckersEnv()
agent = LearningAgent(step_size=0.1, epsilon=0.2, env=env)
training = True


def play_game():
    env.reset()
    env.render()

    while True:
        valid_moves = env.get_valid_moves(env.player)
        if not valid_moves:
            print("No valid moves. Game over.")
            winner = env.check_game_winner()
            if winner == 1:
                print("Player 1 wins!")
            elif winner == -1:
                print("Player -1 wins!")
            else:
                print("It's a draw!")
            break

        print(f"Player {env.player}'s turn")
        action = agent.select_action(valid_moves)
        _, _ = env.step(action, env.player)
        env.render()
        env.player *= -1


if training:
    print("Training the agent...")
    num_episodes = 10
    wins = np.zeros(num_episodes)
    losses = np.zeros(num_episodes)
    draws = np.zeros(num_episodes)
    rewards = []

    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        print("Episode: " + str(episode))

        while True:
            valid_moves = env.get_valid_moves(env.player)
            if not valid_moves:
                break

            action = agent.select_action(valid_moves)
            next_state, reward = env.step(action, env.player)

            total_reward += reward

            agent.update_q_table(env.board, action, reward)

            env.player *= -1

        rewards.append(total_reward)

        winner = env.check_game_winner()
        if winner == -1:
            wins[episode] = 1
            losses[episode] = 0
            draws[episode] = 0
        elif winner == 1:
            wins[episode] = 0
            losses[episode] = 1
            draws[episode] = 0
        else:
            wins[episode] = 0
            losses[episode] = 0
            draws[episode] = 1

    print("Training is done")

    cumulative_wins = np.cumsum(wins)
    cumulative_losses = np.cumsum(losses)
    cumulative_draws = np.cumsum(draws)

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_wins, label='Wins', color='red')
    plt.plot(cumulative_losses, label='Losses', color='blue')
    plt.plot(cumulative_draws, label='Draws', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Results')
    plt.title('Learning Agent Performance')
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_filename = "learning_results.png"
    plt.savefig(plot_filename)
    print(f"Learning results saved as '{plot_filename}'")

    agent.save()
    print("Agent saved")
else:
    agent.load('saved_table.pkl')
    play_game()
