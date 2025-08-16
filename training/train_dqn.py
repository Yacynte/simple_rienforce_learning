import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
# from agents.dqn_model import DQN, ReplayBuffer


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # from input to hidden
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)  # output Q-values for all actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def compute_shaped_reward(state, next_state, goal_pos):
    # Euclidean or Manhattan distance
    old_dist = np.linalg.norm(np.array(state) - np.array(goal_pos))
    new_dist = np.linalg.norm(np.array(next_state) - np.array(goal_pos))
    
    # Positive reward for getting closer, negative for going away
    # print("Reward to goal: ", 10*(old_dist - new_dist))
    return 10*(old_dist - new_dist)



def train_dqn(env, episodes=1000, batch_size=128, gamma=0.99,
              epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, lr=1e-3):

    device = torch.device("cpu")

    input_dim = 2  # x and y position
    output_dim = env.action_space.n
    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        total_reward = 0

        done = False
        while not done:
            # env.render()
            # Epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = torch.argmax(q_values).item()
                    # q_values = policy_net(state.unsqueeze(0))  # Add batch dim: shape [1, n_actions]
                    # probabilities = F.softmax(q_values.squeeze(0), dim=0)  # Remove batch dim before softmax
                    # action = torch.multinomial(probabilities, 1).item()

            next_state, reward, done, _, _ = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)

            # if next_state == state:
            #     reward -= 0.2  # discourage doing nothing
            # Intermediary shaping reward
            # shaping = compute_shaped_reward(state.cpu().numpy(), next_state, env.goal_pos)
            # reward += shaping

            replay_buffer.push(state.cpu().numpy(), action, reward, next_state_tensor.cpu().numpy(), done)
            state = next_state_tensor
            total_reward += reward

            # Training step
            if len(replay_buffer) >= 1000:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                actions = torch.tensor(np.array(actions)).unsqueeze(1).to(device)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
                dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)

                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    target_q = rewards + gamma * target_net(next_states).max(1, keepdim=True)[0] * (1 - dones)

                loss = F.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # print(f"Episode {episode + 1}, Reward: {reward}, Epsilon: {epsilon:.3f}")

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())  # hard update
        print(f"Episode {episode + 1}, Total reward: {total_reward}, Epsilon: {epsilon:.3f}")

        torch.save(policy_net.state_dict(), "rl-gridworld/agents/dqn_agent.pth")