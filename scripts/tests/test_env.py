# -*- coding: utf-8 -*-
"""
Environment Smoke Test.
Runs AircraftEnv with random actions to verify stability.
"""

import os
import sys
import numpy as np

# Ensure project root is in path when executed directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.data.data_processor import prepare_data, FEATURES, KEY_SENSORS
from scripts.models.lstm_model import create_sequences, train_model
from scripts.core.aircraft_env import AircraftEnv


def main():
    data_dir = os.path.join(PROJECT_ROOT, 'CMAPSSData')
    print("[DATA] Loading data...")
    train_rolling, test_rolling, true_rul, scaler = prepare_data(data_dir)

    # Train or load model
    model_path = os.path.join(PROJECT_ROOT, 'models', 'lstm_rul_model.keras')
    if os.path.exists(model_path):
        import tensorflow as tf
        lstm_model = tf.keras.models.load_model(model_path)
        print(f"[MODEL] Loaded LSTM model from {model_path}")
    else:
        print("[MODEL] Training LSTM model (first run)...")
        X_train, y_train = create_sequences(train_rolling, scaler)
        lstm_model, _ = train_model(X_train, y_train)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        lstm_model.save(model_path)
        print(f"[MODEL] Saved to {model_path}")

    # Create environment
    env = AircraftEnv(
        fleet_data=train_rolling,
        model_path=model_path,
        scaler=scaler,
        sensor_list=KEY_SENSORS,
        features_list=FEATURES
    )

    print("\n[TEST] Running Smoke Test: 5 episodes with random actions")
    print("=" * 70)
    expected_obs_shape = env.observation_space.shape

    for ep in range(5):
        obs, info = env.reset()
        weather_zones = info.get("weather_zones", [])
        print(f"\n--- Episode {ep + 1} ---")
        print(f"  Initial obs shape: {obs.shape}, dtype: {obs.dtype}")
        print(f"  Initial obs: {obs}")
        print(f"  Weather zones: {len(weather_zones)}")

        assert obs.shape == expected_obs_shape == (16,), f"Unexpected obs shape: {obs.shape}"
        assert 2 <= len(weather_zones) <= 5, f"Unexpected weather zone count: {len(weather_zones)}"
        assert env.observation_space.contains(obs), "Initial observation is outside observation_space bounds"

        ep_reward = 0
        done = False
        step_count = 0
        events = []

        while not done and step_count < 500:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            step_count += 1

            if 'event' in info:
                events.append(info['event'])

            # Log key moments
            if step_count % 50 == 0 or done:
                assert obs.shape == expected_obs_shape == (16,), f"Unexpected obs shape: {obs.shape}"
                assert env.observation_space.contains(obs), "Step observation is outside observation_space bounds"
                rul = obs[2]
                fuel = obs[1]
                dist = obs[9]
                weather = info.get("weather", "clear")
                print(f"  Step {step_count:>4}: Action={action}, RUL={rul:.1f}, "
                      f"Fuel={fuel:.1f}, Dist={dist:.0f}, Weather={weather}, Reward={reward:.0f}")

        final_event = info.get('event', 'TIMEOUT')
        print(f"  [OK] Episode {ep + 1} done: Steps={step_count}, "
              f"Total Reward={ep_reward:.0f}, Final Event={final_event}")
        if events:
            print(f"  Events: {events}")

    print("\n" + "=" * 70)
    print("[OK] Smoke test complete! Environment is stable.")


if __name__ == "__main__":
    main()
