import matplotlib.pyplot as plt

from Dummy import Dummy
from CheckersEnv import CheckersEnv
from LearningAgent import LearningAgent

# Initialize environment and learning agent
env = CheckersEnv()
agent = LearningAgent(step_size=0.1, epsilon=0.2, env=env)
dummy = Dummy(env=env)
training = True
agentVsAgent = True


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
    num_episodes = 1000
    rewards = []

    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        done = False
        print("Episode: " + str(episode))

        while not done:
            if agentVsAgent:
                valid_moves = env.get_valid_moves(env.player)
                if not valid_moves:
                    break

                action = agent.select_action(valid_moves)
                next_state, reward = env.step(action, env.player)

                total_reward += reward

                next_valid_moves = env.get_valid_moves(-env.player)
                if episode == 10:
                    agent.update_q_table(env.board, action, reward, next_state, next_valid_moves, True)
                else:
                    agent.update_q_table(env.board, action, reward, next_state, next_valid_moves, False)

                env.player *= -1
            else:
                if env.player == 1:
                    valid_moves = env.get_valid_moves(env.player)
                    if not valid_moves:
                        print("Player 1 lost")
                        break

                    action = agent.select_action(valid_moves)
                    next_state, reward = env.step(action, env.player)

                    total_reward += reward

                    next_valid_moves = env.get_valid_moves(-env.player)
                    if episode == 100:
                        agent.update_q_table(env.board, action, reward, next_state, next_valid_moves, True)
                    else:
                        agent.update_q_table(env.board, action, reward, next_state, next_valid_moves, False)
                else:
                    valid_moves = env.get_valid_moves(env.player)
                    if not valid_moves:
                        print("Player -1 lost")
                        break

                    action = dummy.select_action(valid_moves)
                    next_state, reward = env.step(action, env.player)

                env.player *= -1

        rewards.append(total_reward)

    print("Training is done")

    plt.plot(rewards, color="red")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plot_filename = "learning_curve.png"
    plt.savefig(plot_filename)
    print(f"Learning curve saved as '{plot_filename}'")

    agent.save()
    print("Agent saved")
else:
    agent.load('saved_table.pkl')
    play_game()
