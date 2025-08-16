import torch
import torch.nn.functional as F
import numpy as np

from training.train_dqn import DQN


def test_agent(env, model_path):
    device = torch.device("cpu")
    policy_net = DQN(input_dim=2, output_dim=env.action_space.n).to(device)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    for ep in range(5):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        done = False
        while not done:
            env.render()
            with torch.no_grad():
                # action = torch.argmax(policy_net(state)).item()
                q_values = policy_net(state.unsqueeze(0))  # Add batch dim: shape [1, n_actions]
                probabilities = F.softmax(q_values.squeeze(0), dim=0)  # Remove batch dim before softmax
                action = torch.multinomial(probabilities, 1).item()
            next_state, reward, done, _, _ = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32).to(device)

