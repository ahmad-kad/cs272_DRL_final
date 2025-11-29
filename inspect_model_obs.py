from stable_baselines3 import PPO

MODEL_PATH = "outputs/models/adaptive_lidar_200000.zip"

model = PPO.load(MODEL_PATH)
print("Model path:", MODEL_PATH)
print("Model observation_space:", model.observation_space)
print("Model action_space:", model.action_space)

from environments.urban_junction_env import UrbanJunctionEnv
from utils.config import get_curriculum_config

scenario = "merge"
modality = "lidar"

config = get_curriculum_config(scenario, "hard", modality)
env = UrbanJunctionEnv(config=config, scenario=scenario, modality=modality)

print("Env observation_space:", env.observation_space)
