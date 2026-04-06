# -*- coding: utf-8 -*-
"""
Aircraft Digital Twin - Predictive Maintenance with PPO.
Entry point: loads data, creates environment, and runs a demo episode.
"""

import os
import sys
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from scripts.data_processor import prepare_data, FEATURES, KEY_SENSORS
from scripts.lstm_model import create_sequences, train_model
from scripts.aircraft_env import AircraftEnv


def main():
    # ── 1. Load Data ────────────────────────────────────────────
    data_dir = os.path.join(PROJECT_ROOT, 'CMAPSSData')
    print("=" * 60)
    print("  ✈️  Aircraft Digital Twin - Predictive Maintenance")
    print("=" * 60)
    train_rolling, test_rolling, true_rul, scaler = prepare_data(data_dir)

    # ── 2. Train or Load LSTM ───────────────────────────────────
    model_path = os.path.join(PROJECT_ROOT, 'models', 'lstm_rul_model.keras')
    if os.path.exists(model_path):
        import tensorflow as tf
        lstm_model = tf.keras.models.load_model(model_path)
        print(f"📦 Loaded LSTM model from {model_path}")
    else:
        print("🧠 Training LSTM model (first run, this may take a few minutes)...")
        X_train, y_train = create_sequences(train_rolling, scaler)
        lstm_model, _ = train_model(X_train, y_train)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        lstm_model.save(model_path)
        print(f"💾 LSTM model saved to {model_path}")

    # ── 3. Random Forest Comparison & Charting ─────────────────
    from scripts.train_ppo import train_rf_model, evaluate_and_compare
    rf_model = train_rf_model(train_rolling, scaler, FEATURES)
    evaluate_and_compare(lstm_model, rf_model, test_rolling, scaler, true_rul)

    # ── 4. Create Environment ───────────────────────────────────
    env = AircraftEnv(
        fleet_data=train_rolling,
        model=lstm_model,
        scaler=scaler,
        sensor_list=KEY_SENSORS,
        features_list=FEATURES
    )

    # ── 5. Run Demo Episode ─────────────────────────────────────
    print("\n🎮 Running demo episode with random actions...")
    print("-" * 75)
    obs, info = env.reset()
    print(f"  Engine ID: {env.twin.engine_id}")
    print(f"  Route: {env.TOTAL_DISTANCE:.0f} units with {env.NUM_SUB_AIRPORTS} sub-airports")
    print(f"  Initial fuel: {env.twin.fuel:.1f}")
    print("-" * 75)

    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 500:
        action = env.action_space.sample() # Randomly chooses 0 (Cruise), 1 (Descend), or 2 (Climb)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if steps % 20 == 0 or done or 'event' in info:
            action_map = {0: "CRUISE", 1: "DESCEND", 2: "CLIMB"}
            action_name = action_map.get(action, "UNKNOWN")
            print(f"  Step {steps:>4} | Action: {action_name:>7} | "
                  f"Alt: {obs[0]:>6.0f} | RUL: {obs[3]:>6.1f} | "
                  f"Fuel: {obs[2]:>5.1f} | Dist: {obs[5]:>7.0f} | Reward: {reward:>+8.0f}")

    print("-" * 60)
    event = info.get('event', 'TIMEOUT')
    print(f"  Episode finished: {event}")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
