import matplotlib.pyplot as plt
import numpy as np

from CheckersEnv import CheckersEnv
from Dummy import Dummy
from LearningAgent import LearningAgent
from copy import deepcopy

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
    num_episodes = 100
    wins = np.zeros(num_episodes)
    losses = np.zeros(num_episodes)
    draws = np.zeros(num_episodes)

    for episode in range(num_episodes):
        env.reset()
        print("Training episode: " + str(episode))

        while True:
            valid_moves = env.get_valid_moves(env.player)
            if not valid_moves:
                break

            action = agent.select_action(valid_moves)
            old_state = deepcopy(env.board)
            next_state, reward = env.step(action, env.player)

            agent.update_q_table(old_state, action, reward, next_state, env.get_valid_moves(env.player))

            env.player *= -1

        agent.epsilon = max(0.1, agent.epsilon * 0.99)

    print("Training is done")
    agent.save()
    print("Agent saved")

    dummy = Dummy(env)

    for episode in range(num_episodes):
        env.reset()
        print("Playing episode: " + str(episode))

        while True:
            valid_moves = env.get_valid_moves(env.player)
            if not valid_moves:
                break

            if env.player == -1:
                action = agent.select_action(valid_moves)
            else:
                action = dummy.select_action(valid_moves)
            next_state, _ = env.step(action, env.player)

            env.player *= -1

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
else:
    agent.load('saved_table.pkl')
    play_game()
