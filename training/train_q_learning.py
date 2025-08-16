# Imports:
# --------
import numpy as np
import seaborn as sns
from utils.utility import _state_index

# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,
                     nr_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="rl-gridworld/agents/q_table.npy"):

    # Initialize the Q-table:
    # -----------------------
    q_table = np.zeros((env.grid_size * env.grid_size, env.action_space.n))

    # print("q_table in train: ", q_table.shape)
    # Q-learning algorithm:
    # ---------------------
    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------
    for episode in range(nr_episodes):
        state, _ = env.reset()

        # state = tuple(state)
        state_index = _state_index(state, env.grid_size)
        total_reward = 0

        #! Step 2: Take actions in the environment until "Done" flag is triggered
        #! -------
        while True:
            #! Step 3: Define Exploration vs. Exploitation
            #! -------
            # print("state index: ", state_index)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state_index])  # Exploit
                # print("action: ", action)
            # print("state action:", action)
            next_state, done, reward, _ = env.step(action)
            env.render()
            next_state_index = _state_index(next_state, env.grid_size)
            # next_state = tuple(next_state)
            total_reward += reward

            #! Step 4: Update the Q-values using the Q-value update rule (Bellman's Equation)
            #! -------
            q_table[state_index][action] = q_table[state_index][action] + alpha * \
                (reward + gamma *
                 np.max(q_table[next_state_index]) - q_table[state_index][action])
            # print(q_table[state_index][action])
            state_index = next_state_index

            #! Step 5: Stop the episode if the agent reaches Goal or Hell-states
            #! -------
            if done:
                break
        
        #! Step 6: Perform epsilon decay
        #! -------
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        # if done and reward > 0:
        #     break

    #! Step 7: Close the environment window
    #! -------
    env.close()
    print("Training finished.\n")

    #! Step 8: Save the trained Q-table
    #! -------
    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.", q_table.shape)

