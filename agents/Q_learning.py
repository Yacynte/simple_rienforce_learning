# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union
from training.train_q_learning import train_q_learning
from utils.utility import _state_index


learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
nr_episodes = 1000  # Number of episodes
q_table_path = "rl-gridworld/agents/q_table.npy" # Save q_table as numpy


def train_and_vizualise(env, hell_state_coordinates: list | np.ndarray, goal_coordinates: tuple, 
                        train: bool = True, visualize_results: bool = True):
    # Execute:
    # --------
    if train:
        # Train a Q-learning agent:
        # -------------------------
        train_q_learning(env=env,
                        nr_episodes=nr_episodes,
                        epsilon=epsilon,
                        epsilon_min=epsilon_min,
                        epsilon_decay=epsilon_decay,
                        alpha=learning_rate,
                        gamma=gamma,
                        q_table_save_path=q_table_path)

    if visualize_results:
        # Visualize the Q-table:
        # ----------------------
        visualize_q_table(env=env,
                        hell_state_coordinates=hell_state_coordinates,
                        goal_coordinates=goal_coordinates,
                        q_values_path=q_table_path)


def navigate_with_q_table(env, q_table):
    grid_size = env.grid_size
    current_pos = env.agent_position
    max_steps = 100  # To avoid infinite loops

    for _ in range(max_steps):
        state = _state_index(current_pos, grid_size)
        best_action = np.argmax(q_table[state])  # Choose best action
        current_pos, done, reward, info = env.step(best_action)
        env.render()

        print(f"Next-state: {current_pos}, Done: {done}, Reward: {reward}, Distance to goal: {info['Distance to goal']}")

        if done:
            env.close()
            break

    return


# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(env,
                      hell_state_coordinates=[(2, 1), (0, 4)],
                      goal_coordinates=(4, 4),
                      actions=["Up", "Down", "Left", "Right", "Up_Left", "Up_Right", "Down_Left", "Down_Right"],
                      q_values_path="rl-gridworld/agents/q_table.npy"):
    # Load the Q-table:
    # -----------------
    try:
        q_table = np.load(q_values_path)
        # Create subplots for each action:
        # --------------------------------
        _, axes = plt.subplots(1, len(actions), figsize=(30, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, i].copy()

            # Mask the goal state's Q-value for visualization:
            # ------------------------------------------------
            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[_state_index(goal_coordinates, env.grid_size)] = True

            if len(hell_state_coordinates) != 0:
                mask[_state_index(hell_state_coordinates, env.grid_size)] = True
            # mask[hell_state_coordinates[0]] = True
            # mask[hell_state_coordinates[1]] = True
            # print(mask.shape, heatmap_data.shape)
            heatmap_data = heatmap_data.reshape((env.grid_size**2,1))
            mask = mask.reshape((env.grid_size**2,1))

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

            # Denote Goal and Hell states:
            # ----------------------------
            ax.text( 0.5, _state_index(goal_coordinates, env.grid_size) + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=max(8, int(70 / env.grid_size)))
            # ax.text(hell_state_coordinates[0][1] + 0.5, hell_state_coordinates[0][0] + 0.5, 'H', color='red',
            #         ha='center', va='center', weight='bold', fontsize=14)
            # ax.text(hell_state_coordinates[1][1] + 0.5, hell_state_coordinates[1][0] + 0.5, 'H', color='red',
            #         ha='center', va='center', weight='bold', fontsize=14)

            if len(hell_state_coordinates) != 0:
                for danger in _state_index(hell_state_coordinates, env.grid_size):
                    ax.text(0.5, danger + 0.5, 'D', color='red', ha='center', va='center',
                            weight='bold', fontsize=max(8, int(70 / env.grid_size)))

            ax.set_title(f'Action: {action}')

        plt.tight_layout()
        plt.savefig("rl-gridworld/agents/q_table_heatmap.png", dpi=100, bbox_inches='tight')
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")
