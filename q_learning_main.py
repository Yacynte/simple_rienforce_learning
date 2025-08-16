# Imports:
# --------
# from my_env import create_env
# from Q_learning import train_q_learning, visualize_q_table, _state_index

from training.train_q_learning import train_q_learning
from agents.Q_learning import visualize_q_table
from utils.utility import _state_index
from env.grid_env import create_env
import numpy as np
import random
# User definitions:
# -----------------
train = True
visualize_results = True

"""
NOTE: Sometimes a fixed initializtion might push the agent to a local minimum.
In this case, it is better to use a random initialization.  
"""

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
nr_episodes = 1000  # Number of episodes
q_table_path = "rl_gridworld/assignment2/q_table.npy" # Save q_table as numpy


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

# Run as a script:
# ----------------
if __name__=="__main__":
    # Set the grid size for the environment
    grid_size = 5
    train = True
    q_table_path = f"rl_gridworld/agents/q_table_{str(grid_size)}.npy"
    # Initialize an empty list to store the coordinates of "hell" (danger) states
    danger_state_coordinates = []

    # Define the goal state to be the bottom-right cell of the grid
    goal_coordinate = (grid_size - 1, grid_size - 1)
    start_coordinate = (0, 0)

    if train:
        # Randomly generate danger states while avoiding the starting cell and goal coordinate
        for _ in range(grid_size):
            y, x = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)  

            # Avoid blocking the starting point and the goal
            if not ((y == start_coordinate[0] and x == start_coordinate[1]) or (y == goal_coordinate[0] and x == goal_coordinate[1])):  

                danger_state_coordinates.append((y, x))


        # Save the generated danger state coordinates to a file for reuse later
        np.save(f"rl_gridworld/agents/danger_coordinates_{str(grid_size)}.npy", np.array(danger_state_coordinates))
        
    
        # Part 1: Train and Visualize the Learning Model
        # ----------------------------------------------

        # Step 1: Create the environment with the specified "hell" (danger) states
        env = create_env(danger_state_coordinates, grid_size, start_coordinate, goal_coordinate)

        # Step 2: Train the Q-learning agent and visualize its learning progress
        train_and_vizualise(env, danger_state_coordinates, goal_coordinate)


    # Part 2: Load the Trained Q-Table and Use it to Navigate the Environment
    # -----------------------------------------------------------------------

    # Step 3: Load the saved Q-table from disk
    q_table = np.load(q_table_path)

    # Step 4: Load the saved "hell" (danger) state coordinates from disk
    danger_state_coordinates = np.load(f"rl_gridworld/agents/danger_coordinates_{str(grid_size)}.npy")

    # Step 5: Re-create the environment with the same configuration
    env = create_env(danger_state_coordinates, grid_size, start_coordinate, goal_coordinate, wait=200)

    # Step 6: Let the agent navigate the environment using the learned Q-table
    navigate_with_q_table(env=env, q_table=q_table)
