# Enhanced Autonomous Driving RL System
This repository contains a deep reinforcement learning project based on the `highway-env` library and uses the `stable-baselines3` PPO agent. The project involves training an agent to run on the three predefined environments in the highway-env package: Highway, Merge, and Intersection, with training being done using Lidar and Grayscale observations. The agent is also trained on a custom environment that combines the Highway, Merge, and Intersection environments and adds in aggressive drivers as an obstacle.


## Project Structure

```
root/
├── multi_scenario_env.py             # Custom multi-scenario env with aggressive traffic
├── visualize_multi_scenario.py       # Render trained multi-scenario agent
├── train_multi_scenario.py           # PPO training on the custom multi-scenario environment
├── visualize_merge_highway.py        # Render a trained merge-v0 agent
├── train_merge_highway.py            # PPO training on merge-v0 with Lidar observations
├── multi_wandb_helpers.py
├── requirements
├── README.md
├── tb_merge_v0/PPO_1                 # Merge log files
│   └── PPO_1
│       └── events.out.tfevents.1764480636.Purrfection.29308.0
└── tb_multi_scenario                 # Multi-scenario log files
    ├──  PPO_1
    │    └── events.out.tfevents.1764489725.Purrfection.36408.0
    ├──  PPO_2
    │    └── events.out.tfevents.1764489759.Purrfection.36752.0
    └──  ...
```
## Setup
```
# Install dependencies
pip install -r requirements.txt

# Ensure you have pygame for visualization (optional)
pip install pygame

# Create output directories
mkdir -p outputs/models
```
## Training Example

Run:
```
python train_merge_highway.py
```

By default a model is saved to:
```
outputs/models/merge_v0_ppo_100k.zip
```

## Visualization Example
Run:
```
python visualize_multi_scenario.py
```
This should open a small window that renders the environment. For the multi-scenario environment, 5 environments from (highway, merge, and intersection) are chosen at random.
