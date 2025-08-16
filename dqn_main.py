from training.train_dqn import train_dqn
from agents.dqn_model import test_agent
from env import continous_env as env

if __name__ == "__main__":
    new_env = env.ContinuousMazeEnv()
    
    # train_dqn(new_env)
    test_agent(new_env, "rl_gridworld/agents/dqn_agent.pth")