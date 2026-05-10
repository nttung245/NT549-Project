# -*- coding: utf-8 -*-
"""
PPO training script for Aircraft predictive maintenance.

This script mirrors the notebook PPO flow but keeps it reproducible from CLI:
- subprocess/vectorized training environments
- VecNormalize for observations/rewards
- deterministic EvalCallback for model selection
- deterministic and stochastic diagnostic evaluations
- MLflow logging for SB3 metrics, parameters, models, and VecNormalize stats
- AircraftEnv feasibility filtering and maintenance-as-checkpoint semantics
"""

from __future__ import annotations

import argparse
import os

# TensorFlow is imported indirectly by scripts.lstm_model and inside each
# AircraftEnv worker. Configure runtime before any TensorFlow import so CPU-only
# machines do not spam CUDA/oneDNN initialization warnings in every subprocess.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import mlflow

# Ensure project root is importable when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_processor import FEATURES, KEY_SENSORS, prepare_data
from scripts.lstm_model import create_sequences, train_model
from scripts.aircraft_env import AircraftEnv
from scripts.rl_callbacks import (
    EvalDiagnosticsCallback,
    MLflowLoggingCallback,
    SaveVecNormalizeCallback,
    SyncVecNormalizeCallback,
)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠️  stable-baselines3 not installed. Install with: pip install stable-baselines3")


# Medium-RUL curriculum subset: high enough to avoid impossible starts, but low
# enough that the agent should learn maintenance instead of always flying direct.
DEFAULT_TRAIN_ELIGIBLE_UNITS = [14, 62, 3]


def ensure_lstm_model(model_path: Path, train_rolling, scaler) -> None:
    """Train the LSTM only if the saved model is missing."""
    if model_path.exists():
        print(f"📦 Using existing LSTM model: {model_path}")
        return

    print("🧠 LSTM model not found. Training a new LSTM model first...")
    x_train, y_train = create_sequences(train_rolling, scaler)
    lstm_model, _ = train_model(x_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    lstm_model.save(model_path)
    print(f"💾 LSTM model saved to: {model_path}")


def make_aircraft_env_factory(
    *,
    model_path: str,
    min_initial_rul: float,
    maintenance_resets_health: bool,
    eligible_units: list[int] | None,
) -> Callable[[], AircraftEnv]:
    """
    Create a pickle-safe environment factory for SubprocVecEnv.

    Each worker reloads CMAPSS data and scaler locally instead of capturing large
    pandas objects from a notebook/main process.
    """

    def _make_env() -> AircraftEnv:
        worker_data_dir = PROJECT_ROOT / "CMAPSSData"
        worker_train_rolling, _, _, worker_scaler = prepare_data(str(worker_data_dir))

        return AircraftEnv(
            fleet_data=worker_train_rolling,
            model_path=model_path,
            scaler=worker_scaler,
            sensor_list=KEY_SENSORS,
            features_list=FEATURES,
            eligible_units=eligible_units,
            min_initial_rul=min_initial_rul,
            maintenance_resets_health=maintenance_resets_health,
        )

    return _make_env


def load_or_train_ppo(args: argparse.Namespace):
    if not HAS_SB3:
        raise RuntimeError("stable-baselines3 is required for PPO training.")

    data_dir = PROJECT_ROOT / "CMAPSSData"
    models_dir = PROJECT_ROOT / "models"
    logs_dir = PROJECT_ROOT / "logs"
    model_path = models_dir / "lstm_rul_model.keras"

    print("📂 Loading CMAPSS data for preprocessing/LSTM readiness...")
    train_rolling, _, _, scaler = prepare_data(str(data_dir))
    ensure_lstm_model(model_path, train_rolling, scaler)

    run_name = args.run_name or f"PPO_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"🏷️ Training run name: {run_name}")
    print(f"🛩️ Eligible training engine units: {args.eligible_units}")

    repo_ppo_path = models_dir / f"ppo_aircraft_{run_name}"
    stats_path = models_dir / f"ppo_aircraft_{run_name}_vec_normalize.pkl"
    best_model_dir = models_dir / f"best_ppo_{run_name}"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    make_env = make_aircraft_env_factory(
        model_path=str(model_path),
        min_initial_rul=args.min_initial_rul,
        maintenance_resets_health=args.maintenance_resets_health,
        eligible_units=args.eligible_units,
    )

    if args.check_env:
        print("🔍 Checking AircraftEnv compatibility...")
        env = make_env()
        check_env(env, warn=True)
        print("✅ Environment check passed.")

    if (repo_ppo_path.with_suffix(".zip")).exists() and stats_path.exists() and not args.force_train:
        print(f"📦 Loading existing PPO agent and VecNormalize stats from: {repo_ppo_path}")
        train_env = make_vec_env(make_env, n_envs=1)
        train_env = VecNormalize.load(str(stats_path), train_env)
        train_env.training = False
        train_env.norm_reward = False
        ppo_model = PPO.load(str(repo_ppo_path), env=train_env, device=args.device)
        return ppo_model, train_env, run_name, repo_ppo_path, stats_path, best_model_dir

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or args.mlflow_tracking_uri
    print(f"📊 Connecting to MLflow at: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    print(f"🚀 Training PPO Agent ({args.total_timesteps:,} steps) with EvalCallback and MLflow...")

    with mlflow.start_run(run_name=run_name):
        # 1. Setup vectorized training environment.
        train_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=args.clip_obs,
        )

        # 2. Setup eval environment. Its obs stats are synced from train_env.
        eval_env = make_vec_env(make_env, n_envs=1)
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=args.clip_obs,
            training=False,
        )

        # 3. Initialize PPO.
        # Default to CPU because this project also loads TensorFlow in each env
        # worker; forcing CPU avoids noisy CUDA probing on machines without a
        # properly configured GPU stack.
        ppo_model = PPO(
            "MlpPolicy",
            train_env,
            verbose=args.verbose,
            device=args.device,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            tensorboard_log=str(logs_dir / "ppo_aircraft"),
        )

        # 4. Log all training/environment parameters to MLflow.
        params = {
            "run_name": run_name,
            "total_timesteps": args.total_timesteps,
            "device": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tf_cpp_min_log_level": os.environ.get("TF_CPP_MIN_LOG_LEVEL"),
            "tf_enable_onednn_opts": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
            "learning_rate": args.learning_rate,
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "clip_obs": args.clip_obs,
            "norm_obs": True,
            "norm_reward_train": True,
            "norm_reward_eval": False,
            "eval_freq": args.eval_freq,
            "eval_n_episodes": args.eval_n_episodes,
            "eval_deterministic": True,
            "diag_det_freq": args.diag_det_freq,
            "diag_stoch_freq": args.diag_stoch_freq,
            "diag_n_episodes": args.diag_n_episodes,
            "min_initial_rul": args.min_initial_rul,
            "maintenance_resets_health": args.maintenance_resets_health,
            "eligible_units": args.eligible_units,
            "model_path": str(model_path),
            "repo_ppo_path": str(repo_ppo_path),
            "stats_path": str(stats_path),
            "best_model_dir": str(best_model_dir),
        }
        mlflow.log_params(params)

        # 5. Set up callback chain.
        sync_cb = SyncVecNormalizeCallback(eval_env=eval_env, verbose=1)
        save_vec_stats_cb = SaveVecNormalizeCallback(save_path=str(best_model_dir), verbose=1)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(logs_dir / "eval_results"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_n_episodes,
            deterministic=True,
            callback_on_new_best=save_vec_stats_cb,
        )
        eval_diag_det_cb = EvalDiagnosticsCallback(
            eval_env=eval_env,
            eval_freq=args.diag_det_freq,
            n_eval_episodes=args.diag_n_episodes,
            deterministic=True,
            log_prefix="eval_diag_det",
            verbose=1,
        )
        eval_diag_stoch_cb = EvalDiagnosticsCallback(
            eval_env=eval_env,
            eval_freq=args.diag_stoch_freq,
            n_eval_episodes=args.diag_n_episodes,
            deterministic=False,
            log_prefix="eval_diag_stoch",
            verbose=1,
        )
        mlflow_cb = MLflowLoggingCallback(verbose=1)

        callback_list = CallbackList([
            sync_cb,
            eval_callback,
            eval_diag_det_cb,
            eval_diag_stoch_cb,
            mlflow_cb,
        ])

        # 6. Start training.
        ppo_model.learn(total_timesteps=args.total_timesteps, callback=callback_list)

        # 7. Final saving and artifact logging.
        ppo_model.save(str(repo_ppo_path))
        train_env.save(str(stats_path))
        print(f"💾 PPO model saved to: {repo_ppo_path}.zip")
        print(f"💾 VecNormalize stats saved to: {stats_path}")

        mlflow.log_artifact(str(repo_ppo_path.with_suffix(".zip")), artifact_path="model_final")
        mlflow.log_artifact(str(stats_path), artifact_path="vecnormalize_final")

        best_model_path = best_model_dir / "best_model.zip"
        best_stats_path = best_model_dir / "vec_normalize.pkl"
        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path), artifact_path="model_best")
        if best_stats_path.exists():
            mlflow.log_artifact(str(best_stats_path), artifact_path="vecnormalize_best")

        print("✅ PPO training complete and artifacts logged to MLflow.")
        return ppo_model, train_env, run_name, repo_ppo_path, stats_path, best_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO Agent for Aircraft Predictive Maintenance")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name. Default: timestamped PPO_Run_*.")
    parser.add_argument("--force-train", action="store_true", help="Train even if a model with the same run name already exists.")
    parser.add_argument("--check-env", action="store_true", help="Run stable-baselines3 check_env before training.")

    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Torch device for PPO. Default CPU avoids CUDA warnings on machines without GPU drivers.",
    )

    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-n-episodes", type=int, default=10)
    parser.add_argument("--diag-det-freq", type=int, default=10000)
    parser.add_argument("--diag-stoch-freq", type=int, default=20000)
    parser.add_argument("--diag-n-episodes", type=int, default=10)

    parser.add_argument("--min-initial-rul", type=float, default=120.0)
    parser.add_argument(
        "--no-maintenance-reset-health",
        dest="maintenance_resets_health",
        action="store_false",
        help="Disable health/RUL reset after successful intermediate maintenance.",
    )
    parser.set_defaults(maintenance_resets_health=True)
    parser.add_argument(
        "--eligible-units",
        type=int,
        nargs="*",
        default=DEFAULT_TRAIN_ELIGIBLE_UNITS,
        help=(
            "Engine unit IDs to sample from before min_initial_rul filtering. "
            f"Default uses a fast curriculum subset: {DEFAULT_TRAIN_ELIGIBLE_UNITS}. "
            "Pass no values after the flag to use all units."
        ),
    )

    parser.add_argument("--mlflow-tracking-uri", type=str, default="http://localhost:5000")
    parser.add_argument("--mlflow-experiment", type=str, default="Aircraft_Predictive_Maintenance_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eligible_units == []:
        args.eligible_units = None
    load_or_train_ppo(args)
    print("✅ PPO System Ready!")


if __name__ == "__main__":
    main()
