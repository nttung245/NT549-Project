import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.aircraft_env import AircraftEnv
from scripts.data_processor import prepare_data, FEATURES, KEY_SENSORS

def parse_args():
    parser = argparse.ArgumentParser(description="Run one PPO aircraft demo episode.")
    parser.add_argument(
        "--run-name",
        type=str,
        default="PPO_Run_20260510_124810",
        help="PPO run name under models/best_ppo_<run-name>.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1200,
        help="Maximum demo episode steps before stopping.",
    )
    parser.add_argument(
        "--eligible-units",
        type=int,
        nargs="*",
        default=[14, 62, 3],
        help="Engine unit IDs to sample from. Pass no values after the flag to use all eligible units.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions. Default keeps stochastic demo behavior.",
    )
    return parser.parse_args()


args = parse_args()

# ============================================================
# 1. DATA PREPARATION (Load and prepare training data)
# ============================================================
print("Loading and preparing data...")
data_dir = os.path.join(PROJECT_ROOT, 'CMAPSSData')
train_rolling, _, _, scaler = prepare_data(data_dir)
print(f"✅ Data loaded: {len(train_rolling)} rows, scaler fitted on {len(FEATURES)} features")

# ============================================================
# 2. LSTM MODEL PATH
# ============================================================
lstm_model_path = os.path.join(PROJECT_ROOT, 'models', 'lstm_rul_model.keras')
if not os.path.exists(lstm_model_path):
    raise FileNotFoundError(f"LSTM model not found at {lstm_model_path}. Please train the LSTM model first.")

# ============================================================
# 3. LOAD PPO MODEL AND STATS
# ============================================================
best_model_dir = os.path.join(PROJECT_ROOT, "models", f"best_ppo_{args.run_name}")
model_path = os.path.join(best_model_dir, "best_model.zip")
stats_path = os.path.join(best_model_dir, "vec_normalize.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"PPO model not found at {model_path}")
if not os.path.exists(stats_path):
    raise FileNotFoundError(f"VecNormalize stats not found at {stats_path}")

# 4. Create and wrap environment
def make_env():
    eligible_units = args.eligible_units if args.eligible_units else None
    return AircraftEnv(
        fleet_data=train_rolling,
        model_path=lstm_model_path,
        scaler=scaler,
        sensor_list=KEY_SENSORS,
        features_list=FEATURES,
        eligible_units=eligible_units,
    )

env = DummyVecEnv([make_env])

# Load normalization statistics
env = VecNormalize.load(stats_path, env)
env.training = False
env.norm_reward = False

# Load PPO model
ppo_model = PPO.load(model_path, env=env)
print(f"✅ Successfully loaded model and stats from {best_model_dir}")
print(
    f"⚙️  Demo config: run_name={args.run_name}, max_steps={args.max_steps}, "
    f"deterministic={args.deterministic}, eligible_units={args.eligible_units or 'all'}"
)

# ============================================================
# 5. RUN DEMO EPISODE
# ============================================================
obs = env.reset()  # VecEnv.reset() returns observation (handled automatically)
done = False
steps = 0
total_reward = 0
action_names = {0: "CRUISE", 1: "DESCEND", 2: "CLIMB"}

# Get raw env for accessing actual values (not normalized)
raw_env = env.envs[0]

print("=" * 80)
print(f"🔹 STARTING EPISODE | Engine: {raw_env.twin.engine_id}")
print("=" * 80)

# Match PPO's stochastic training policy during demos/evaluation by default.
# Deterministic=True takes only the most likely action and can collapse to CRUISE/DESCEND loops.
deterministic_eval = args.deterministic

while not done and steps < args.max_steps:
    # Predict action (obs is automatically normalized by VecNormalize wrapper)
    action, _ = ppo_model.predict(obs, deterministic=deterministic_eval)  # type: ignore[arg-type, call-arg]
    
    # Step the environment
    obs, reward, done_array, info_array = env.step(action)
    done = bool(done_array[0])
    
    total_reward += reward[0]
    steps += 1
    
    # Print status every 2 steps or on special events
    if steps % 2 == 0 or done or 'event' in info_array[0]:
        act_name = action_names.get(int(action[0]), "UNKNOWN")
        twin = raw_env.twin
        
        # Handle RUL value to avoid None errors
        rul_display = twin.current_rul if twin.current_rul is not None else 0.0
        
        # Find next airport ahead
        current_pos = raw_env.TOTAL_DISTANCE - raw_env.distance_to_destination
        ahead = [ap - current_pos for ap in raw_env.sub_airports if (ap - current_pos) > 0]
        dist_ap = min(ahead) if ahead else 0
        
        print(f"Step {steps:>4} | Action: {act_name:>7} | Alt: {twin.altitude:>5.0f}m | "
              f"RUL: {rul_display:>5.1f} | Fuel: {twin.fuel:>5.1f} | "
              f"Next AP: {dist_ap:>6.0f} | Dest: {raw_env.distance_to_destination:>6.0f}")

print("=" * 80)
final_event = info_array[0].get('event', 'TIMEOUT')
print(f"🛬 END: {final_event}")
print(f"   Total Steps: {steps} | Total Reward: {total_reward:.2f}")
