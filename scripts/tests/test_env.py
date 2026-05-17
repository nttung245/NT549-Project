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
from scripts.core.weather import WeatherEffect


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

    high_rul_env = AircraftEnv(
        fleet_data=train_rolling,
        model_path=model_path,
        scaler=scaler,
        sensor_list=KEY_SENSORS,
        features_list=FEATURES,
        min_initial_rul=250.0,
        initial_rul_max=335.0,
    )
    assert high_rul_env.engine_units, "Expected at least one high-RUL engine"
    assert all(
        250.0 <= high_rul_env.unit_initial_rul_map[unit_id] <= 335.0
        for unit_id in high_rul_env.engine_units
    ), "High-RUL filtering selected an engine outside the requested band"

    high_rul_env.set_initial_rul_range(140.0, 170.0)
    assert high_rul_env.engine_units, "Expected at least one medium-RUL engine after curriculum update"
    assert all(
        140.0 <= high_rul_env.unit_initial_rul_map[unit_id] <= 170.0
        for unit_id in high_rul_env.engine_units
    ), "Curriculum RUL update selected an engine outside the requested band"

    obs, _ = env.reset(seed=123)
    no_need_pressure = env._maintenance_need_pressure(
        effective_rul=300.0,
        fuel=env.FUEL_CAPACITY,
        current_pos=0.0,
        next_hazard_zone=None,
    )
    low_rul_pressure = env._maintenance_need_pressure(
        effective_rul=30.0,
        fuel=env.FUEL_CAPACITY,
        current_pos=0.0,
        next_hazard_zone=None,
    )
    assert no_need_pressure < 0.15, f"Healthy aircraft should not need early maintenance: {no_need_pressure}"
    assert low_rul_pressure > no_need_pressure, "Low RUL should increase maintenance pressure"
    assert env._weather_reward(WeatherEffect(zone_type="tailwind"), 0) > env._weather_reward(
        WeatherEffect(zone_type="tailwind"), 1
    ), "Cruising in tailwind should be more valuable than descending in tailwind"

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

        assert obs.shape == expected_obs_shape == (19,), f"Unexpected obs shape: {obs.shape}"
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
                assert obs.shape == expected_obs_shape == (19,), f"Unexpected obs shape: {obs.shape}"
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
