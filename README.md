# RL Gridworld Environment

Custom **NxN Gridworld Environment** built with **Gymnasium** and **Pygame**, where an agent must find a goal while avoiding danger.  
The project evolves from a baseline **random agent**, to **Q-learning**, and finally a **Deep Q-Network (DQN)** implemented in PyTorch.

---

## Features
- ✅ NxN customizable environment
- ✅ 8 possible discrete actions
- ✅ Reward system:
  - -0.1 per vertical or horizontal step
  - -0.15 per diagonal step
  - -10 on danger (episode restart)
  - +10 on reaching the goal
- ✅ Baseline random agent
- ✅ Q-learning implementation
- ✅ Deep Q-Network (DQN) training
- ✅ Live visualization with Pygame
- ✅ Training curve plots

---

## Installation
```bash
git clone https://github.com/<your-username>/rl-gridworld.git
cd simple_RL
pip install -r requirements.txt
