import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gymnasium as gym

def check():
    path = r'c:\Users\tung\Downloads\RL\Project\models\best_ppo_PPO_Run_20260410_102429\best_model.zip'
    stats_path = r'c:\Users\tung\Downloads\RL\Project\models\best_ppo_PPO_Run_20260410_102429\vec_normalize.pkl'
    
    print(f"Checking {path}")
    model = PPO.load(path)
    print(f"Model Observation Space: {model.observation_space}")
    
    vn = VecNormalize.load(stats_path, DummyVecEnv([lambda: gym.make('CartPole-v1')])) # Dummy env for loading
    print(f"VecNormalize Observation Space: {vn.observation_space}")

if __name__ == "__main__":
    check()
