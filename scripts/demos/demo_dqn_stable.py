# -*- coding: utf-8 -*-
"""Run one Stable-Baselines3 DQN aircraft demo episode."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Configure TensorFlow before AircraftEnv loads the LSTM model.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path for imports when executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.aircraft_env import AircraftEnv
from scripts.data.data_processor import FEATURES, KEY_SENSORS, prepare_data


def ensure_ndarray_obs(obs: Any) -> np.ndarray:
    """Return VecEnv observations as an ndarray for SB3 predict type checkers."""
    if not isinstance(obs, np.ndarray):
        raise TypeError(f"Expected ndarray observation from AircraftEnv, got {type(obs).__name__}")
    return obs



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DQN aircraft demo episode.")
    parser.add_argument(
        "--run-name",
        type=str,
        default="DQN_Run_20260524_043809",
        help="DQN run name under models/best_dqn_<run-name>.",
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
        default=None,
        help="Engine unit IDs to sample from. Pass no values after the flag to use all eligible units.",
    )
    parser.add_argument(
        "--min-initial-rul",
        type=float,
        default=120.0,
        help="Minimum initial RUL filter used by AircraftEnv.",
    )
    parser.add_argument(
        "--initial-rul-max",
        type=float,
        default=335.0,
        help="Maximum initial RUL filter used by AircraftEnv. Use a negative value to disable.",
    )
    parser.add_argument(
        "--airport-noise",
        type=float,
        default=200.0,
        help="Airport randomization amplitude. Defaults to the full DQN training preset.",
    )
    parser.add_argument(
        "--no-weather",
        dest="enable_weather",
        action="store_false",
        help="Disable weather zones in the demo environment.",
    )
    parser.set_defaults(enable_weather=True)
    parser.add_argument(
        "--no-maintenance-reset-health",
        dest="maintenance_resets_health",
        action="store_false",
        help="Disable engine health reset after successful intermediate maintenance.",
    )
    parser.set_defaults(maintenance_resets_health=True)
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Use DQN exploratory prediction instead of deterministic greedy actions.",
    )
    parser.set_defaults(deterministic=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Torch device for loading the DQN policy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eligible_units = args.eligible_units if args.eligible_units else None
    initial_rul_max = args.initial_rul_max if args.initial_rul_max >= 0 else None

    print("Loading and preparing data...")
    data_dir = PROJECT_ROOT / "CMAPSSData"
    train_rolling, _, _, scaler = prepare_data(str(data_dir))
    print(f"✅ Data loaded: {len(train_rolling)} rows, scaler fitted on {len(FEATURES)} features")

    lstm_model_path = PROJECT_ROOT / "models" / "lstm_rul_model.keras"
    if not lstm_model_path.exists():
        raise FileNotFoundError(f"LSTM model not found at {lstm_model_path}. Please train the LSTM model first.")

    best_model_dir = PROJECT_ROOT / "models" / f"best_dqn_{args.run_name}"
    model_path = best_model_dir / "best_model.zip"
    stats_path = best_model_dir / "vec_normalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"DQN model not found at {model_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found at {stats_path}")

    def make_env() -> AircraftEnv:
        return AircraftEnv(
            fleet_data=train_rolling,
            model_path=str(lstm_model_path),
            scaler=scaler,
            sensor_list=KEY_SENSORS,
            features_list=FEATURES,
            eligible_units=eligible_units,
            min_initial_rul=args.min_initial_rul,
            initial_rul_max=initial_rul_max,
            maintenance_resets_health=args.maintenance_resets_health,
            airport_noise=args.airport_noise,
            enable_weather=args.enable_weather,
        )

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(str(stats_path), env)
    env.training = False
    env.norm_reward = False

    dqn_model = DQN.load(str(model_path), env=env, device=args.device)
    print(f"✅ Successfully loaded DQN model and stats from {best_model_dir}")
    print(
        f"⚙️  Demo config: run_name={args.run_name}, max_steps={args.max_steps}, "
        f"deterministic={args.deterministic}, eligible_units={eligible_units or 'all'}, "
        f"min_initial_rul={args.min_initial_rul}, initial_rul_max={initial_rul_max}, "
        f"airport_noise={args.airport_noise}, weather={args.enable_weather}"
    )

    obs = ensure_ndarray_obs(env.reset())
    done = False
    steps = 0
    total_reward = 0.0
    last_info: dict[str, Any] = {}
    action_names = {0: "CRUISE", 1: "DESCEND", 2: "CLIMB"}
    raw_env = env.envs[0]

    print("=" * 80)
    print(f"🔹 STARTING DQN EPISODE | Engine: {raw_env.twin.engine_id}")
    print("=" * 80)

    while not done and steps < args.max_steps:
        action, _ = dqn_model.predict(obs, deterministic=args.deterministic)
        next_obs, reward, done_array, info_array = env.step(action)
        obs = ensure_ndarray_obs(next_obs)
        done = bool(done_array[0])
        total_reward += float(reward[0])
        steps += 1

        info = info_array[0]
        last_info = info

        if steps % 2 == 0 or done or "event" in info:
            act_name = action_names.get(int(np.asarray(action)[0]), "UNKNOWN")
            twin = raw_env.twin

            altitude_display = float(info.get("altitude", twin.altitude))
            rul_display = float(info.get("rul", twin.current_rul if twin.current_rul is not None else 0.0))
            fuel_display = float(info.get("fuel", twin.fuel))
            next_target_display = float(info.get("dist_to_next_target", 0.0))
            dest_display = float(info.get("distance_to_destination", raw_env.distance_to_destination))
            event_display = f" | Event: {info['event']}" if "event" in info else ""

            print(
                f"Step {steps:>4} | Action: {act_name:>7} | Alt: {altitude_display:>5.0f}m | "
                f"RUL: {rul_display:>5.1f} | Fuel: {fuel_display:>5.1f} | "
                f"Next Target: {next_target_display:>6.0f} | Dest: {dest_display:>6.0f}"
                f"{event_display}"
            )

    print("=" * 80)
    final_event = last_info.get("event", "TIMEOUT")
    print(f"🛬 END: {final_event}")
    print(f"   Total Steps: {steps} | Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
