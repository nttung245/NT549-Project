import os
import sys
from stable_baselines3 import PPO
import gymnasium as gym

def check_all_models():
    models_dir = r'c:\Users\tung\Downloads\RL\Project\models'
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.zip'):
                path = os.path.join(root, file)
                try:
                    model = PPO.load(path)
                    print(f"Model: {path}")
                    print(f"  Obs Space: {model.observation_space}")
                except Exception as e:
                    print(f"Error loading {path}: {e}")

if __name__ == "__main__":
    check_all_models()
