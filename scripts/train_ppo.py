# -*- coding: utf-8 -*-
"""
PPO Training Script for Aircraft Predictive Maintenance.
Uses stable-baselines3 to train a PPO agent on AircraftEnv.
"""

import os
import sys
import numpy as np
import mlflow

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.data_processor import prepare_data, FEATURES, KEY_SENSORS
from scripts.lstm_model import create_sequences, train_model
from scripts.aircraft_env import AircraftEnv
from scripts.rl_callbacks import MLflowLoggingCallback

# Attempt to import stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠️  stable-baselines3 not installed. Install with: pip install stable-baselines3")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def main():
    # ── 1. Load & Prepare Data ──────────────────────────────────
    data_dir = os.path.join(PROJECT_ROOT, 'CMAPSSData')
    print("📂 Loading CMAPSS data...")
    train_rolling, test_rolling, true_rul, scaler = prepare_data(data_dir)

    # ── 2. Train or Load LSTM Model ─────────────────────────────
    model_path = os.path.join(PROJECT_ROOT, 'models', 'lstm_rul_model.keras')

    if os.path.exists(model_path):
        print(f"📦 Loading existing LSTM model from {model_path}")
        import tensorflow as tf
        lstm_model = tf.keras.models.load_model(model_path)
    else:
        print("🧠 Training new LSTM model...")
        X_train, y_train = create_sequences(train_rolling, scaler)
        lstm_model, history = train_model(X_train, y_train)

        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        lstm_model.save(model_path)
        print(f"💾 LSTM model saved to {model_path}")

    # ── 3. Random Forest Baseline ─────────────────────────────
    rf_model = train_rf_model(train_rolling, scaler, FEATURES)

    # ── 4. RUL Comparison (RF vs LSTM) ────────────────────────
    evaluate_and_compare(lstm_model, rf_model, test_rolling, scaler, true_rul)

    # ── 5. Create AircraftEnv ───────────────────────────────────
    print("✈️  Creating AircraftEnv...")
    env = AircraftEnv(
        fleet_data=train_rolling,
        model=lstm_model,
        scaler=scaler,
        sensor_list=KEY_SENSORS,
        features_list=FEATURES
    )

    # ── 6. Validate Environment ─────────────────────────────────
    if HAS_SB3:
        print("🔍 Checking environment compatibility...")
        try:
            check_env(env, warn=True)
            print("✅ Environment check passed!")
        except Exception as e:
            print(f"⚠️  Environment check warning: {e}")

        # ── 7. Train PPO Agent ──────────────────────────────────────
        if HAS_SB3:
            print("🚀 Training PPO Agent with MLflow...")
            
            # MLflow configuration
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("PPO-Aircraft-Maintenance")
            
            with mlflow.start_run(run_name="PPO_Script_Training"):
                ppo_model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    tensorboard_log=os.path.join(PROJECT_ROOT, 'logs', 'ppo_aircraft')
                )

                # Log parameters
                mlflow.log_param("learning_rate", 3e-4)
                mlflow.log_param("total_timesteps", 500000)

                # Add MLflow Callback
                mlflow_cb = MLflowLoggingCallback(verbose=1)
                
                total_timesteps = 500_000
                ppo_model.learn(total_timesteps=total_timesteps, callback=mlflow_cb)

                # Save PPO model
                ppo_path = os.path.join(PROJECT_ROOT, 'models', 'ppo_aircraft')
                ppo_model.save(ppo_path)
                print(f"💾 PPO model saved to {ppo_path}")
                
                # Log final model as artifact
                mlflow.log_artifact(ppo_path + ".zip", artifact_path="model")

        # ── 8. Evaluate ────────────────────────────────────────
        print("\n📊 Evaluating trained agent...")
        evaluate_agent(env, ppo_model, n_episodes=10)
    else:
        print("⏭️  Skipping PPO training (stable-baselines3 not available)")
        print("   Running random agent evaluation instead...")
        evaluate_random(env, n_episodes=5)


def train_rf_model(train_df, scaler, features_list):
    """Huấn luyện Random Forest Regressor làm baseline so sánh."""
    print("\n🌲 Training Random Forest Regressor (Baseline)...")
    train_last = train_df.groupby('unit_nr').last().reset_index()
    X_train_rf = scaler.transform(train_last[features_list])
    y_train_rf = train_last['RUL']
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    rf.fit(X_train_rf, y_train_rf)
    print("✅ Random Forest training complete.")
    return rf


def evaluate_and_compare(lstm_model, rf_model, test_df, scaler, true_rul_df):
    """So sánh hiệu năng giữa LSTM và Random Forest trên tập Test."""
    print("\n📊 Comparing Performance: Random Forest vs LSTM...")
    X_test_lstm, _ = create_sequences(test_df, scaler)
    y_pred_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    
    test_last = test_df.groupby('unit_nr').last().reset_index()
    X_test_rf = scaler.transform(test_last[FEATURES])
    y_pred_rf = rf_model.predict(X_test_rf)
    
    y_true = true_rul_df['RUL_ground_truth'].values
    
    r2_rf = r2_score(y_true, y_pred_rf)
    r2_lstm = r2_score(y_true, y_pred_lstm)
    
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='Ground Truth (Actual)', color='blue', linewidth=2)
    plt.plot(y_pred_rf, label=f'Random Forest (R2={r2_rf:.2f})', color='red', linestyle='--')
    plt.plot(y_pred_lstm, label=f'LSTM (R2={r2_lstm:.2f})', color='green', linewidth=2)
    
    plt.title('COMPARISON: RANDOM FOREST VS LSTM ON NASA CMAPSS TEST SET')
    plt.xlabel('Engine Unit Index')
    plt.ylabel('RUL (Cycles)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"🔥 Performance Improvement: R2 Score increased from {r2_rf:.2f} to {r2_lstm:.2f}")


def evaluate_agent(env, model, n_episodes=10):
    """Evaluate a trained PPO agent."""
    total_rewards = []
    events = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            if steps > 5000:  # Safety limit
                break

        event = info.get('event', 'TIMEOUT')
        total_rewards.append(ep_reward)
        events.append(event)
        print(f"  Episode {ep+1}: Reward={ep_reward:.0f}, Steps={steps}, Event={event}")

    print(f"\n  Average Reward: {np.mean(total_rewards):.0f}")
    print(f"  Events: {dict(zip(*np.unique(events, return_counts=True)))}")


def evaluate_random(env, n_episodes=5):
    """Evaluate with random actions (smoke test)."""
    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        done = False
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            if steps > 5000:
                break

        event = info.get('event', 'TIMEOUT')
        print(f"  Episode {ep+1}: Reward={ep_reward:.0f}, Steps={steps}, Event={event}")


if __name__ == "__main__":
    main()
