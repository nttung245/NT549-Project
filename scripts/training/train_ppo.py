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

# TensorFlow is imported indirectly by scripts.models.lstm_model and inside each
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.data_processor import FEATURES, KEY_SENSORS, prepare_data
from scripts.models.lstm_model import create_sequences, train_model
from scripts.core.aircraft_env import AircraftEnv
from scripts.training.rl_callbacks import (
    EntCoefScheduleCallback,
    EvalDiagnosticsCallback,
    FixedSeedEvalCallback,
    MLflowLoggingCallback,
    RulCurriculumCallback,
    SaveVecNormalizeCallback,
    SyncVecNormalizeCallback,
)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠️  stable-baselines3 not installed. Install with: pip install stable-baselines3")


# By default, train across all engines and let the RUL curriculum narrow the
# sampled band. Passing --eligible-units still supports targeted experiments.
DEFAULT_TRAIN_ELIGIBLE_UNITS = None
DEFAULT_RUL_CURRICULUM = "0:140:170,0.3:170:260,0.65:120:335"


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


def parse_linear_schedule(spec: str | None) -> Callable[[float], float] | None:
    """
    Parse a compact linear schedule spec.

    Format: "linear:start:end".
    The returned callable uses elapsed progress in [0, 1].
    """
    if spec is None or spec.strip().lower() in {"", "none", "off", "fixed"}:
        return None

    parts = spec.split(":")
    if len(parts) != 3 or parts[0].lower() != "linear":
        raise ValueError(f"Unsupported schedule spec: {spec!r}. Expected 'linear:start:end'.")

    start = float(parts[1])
    end = float(parts[2])

    def _schedule(progress_elapsed: float) -> float:
        progress = min(1.0, max(0.0, float(progress_elapsed)))
        return start + progress * (end - start)

    return _schedule


def make_sb3_learning_rate(
    *,
    fixed_value: float,
    schedule_spec: str | None,
) -> float | Callable[[float], float]:
    """
    Return either a fixed learning rate or an SB3-compatible schedule.

    Stable-Baselines3 passes progress_remaining in [1, 0], so convert it to
    elapsed progress before applying the user-facing schedule.
    """
    elapsed_schedule = parse_linear_schedule(schedule_spec)
    if elapsed_schedule is None:
        return fixed_value

    def _sb3_schedule(progress_remaining: float) -> float:
        progress_elapsed = 1.0 - float(progress_remaining)
        return elapsed_schedule(progress_elapsed)

    return _sb3_schedule


def schedule_start_value(schedule_spec: str | None, fallback: float) -> float:
    schedule = parse_linear_schedule(schedule_spec)
    return float(schedule(0.0)) if schedule is not None else float(fallback)


def parse_rul_curriculum(
    spec: str | None,
) -> list[tuple[float, float, float | None]] | None:
    """
    Parse "progress:min:max" curriculum stages.

    Example: "0:140:170,0.3:170:260,0.65:120:335".
    Use max as "none" for an open-ended upper bound.
    """
    if spec is None or spec.strip().lower() in {"", "none", "off", "fixed"}:
        return None

    stages: list[tuple[float, float, float | None]] = []
    for raw_stage in spec.split(","):
        parts = raw_stage.strip().split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Unsupported RUL curriculum stage {raw_stage!r}. Expected 'progress:min:max'."
            )
        progress = float(parts[0])
        min_rul = float(parts[1])
        max_rul = None if parts[2].lower() in {"none", "inf", "open"} else float(parts[2])
        if not 0.0 <= progress <= 1.0:
            raise ValueError(f"Curriculum progress must be in [0, 1], got {progress}.")
        if max_rul is not None and max_rul < min_rul:
            raise ValueError(f"Curriculum max RUL {max_rul} is lower than min RUL {min_rul}.")
        stages.append((progress, min_rul, max_rul))

    stages.sort(key=lambda item: item[0])
    if not stages or stages[0][0] != 0.0:
        raise ValueError("RUL curriculum must start at progress 0.")
    return stages


def make_aircraft_env_factory(
    *,
    model_path: str,
    min_initial_rul: float,
    initial_rul_max: float | None,
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
            initial_rul_max=initial_rul_max,
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

    rul_curriculum = parse_rul_curriculum(args.rul_curriculum_schedule)
    print(f"RUL curriculum: {rul_curriculum}")

    repo_ppo_path = models_dir / f"ppo_aircraft_{run_name}"
    stats_path = models_dir / f"ppo_aircraft_{run_name}_vec_normalize.pkl"
    best_model_dir = models_dir / f"best_ppo_{run_name}"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    make_env = make_aircraft_env_factory(
        model_path=str(model_path),
        min_initial_rul=args.min_initial_rul,
        initial_rul_max=args.initial_rul_max,
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
        raw_train_env = make_vec_env(make_env, n_envs=1)
        train_env = None
        try:
            train_env = VecNormalize.load(str(stats_path), raw_train_env)
            train_env.training = False
            train_env.norm_reward = False
            ppo_model = PPO.load(str(repo_ppo_path), env=train_env, device=args.device)
        except (AssertionError, ValueError) as exc:
            if train_env is not None:
                train_env.close()
            else:
                raw_train_env.close()
            current_shape = getattr(raw_train_env.observation_space, "shape", None)
            raise RuntimeError(
                "Existing PPO/VecNormalize artifacts are incompatible with the current AircraftEnv "
                f"observation space {current_shape}. This usually means the environment features changed "
                "after the model was trained. Re-run with --force-train or choose a newer --run-name. "
                f"Model path: {repo_ppo_path.with_suffix('.zip')}; VecNormalize path: {stats_path}."
            ) from exc
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
        learning_rate = make_sb3_learning_rate(
            fixed_value=args.learning_rate,
            schedule_spec=args.learning_rate_schedule,
        )
        initial_ent_coef = schedule_start_value(args.ent_coef_schedule, args.ent_coef)

        ppo_model = PPO(
            "MlpPolicy",
            train_env,
            verbose=args.verbose,
            device=args.device,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=initial_ent_coef,
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
            "learning_rate_schedule": args.learning_rate_schedule,
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "ent_coef_schedule": args.ent_coef_schedule,
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
            "eval_seed": args.eval_seed,
            "min_initial_rul": args.min_initial_rul,
            "initial_rul_max": args.initial_rul_max,
            "rul_curriculum_schedule": args.rul_curriculum_schedule,
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
        eval_callback = FixedSeedEvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(logs_dir / "eval_results"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_n_episodes,
            deterministic=True,
            callback_on_new_best=save_vec_stats_cb,
            eval_seed=args.eval_seed,
        )
        eval_diag_det_cb = EvalDiagnosticsCallback(
            eval_env=eval_env,
            eval_freq=args.diag_det_freq,
            n_eval_episodes=args.diag_n_episodes,
            deterministic=True,
            log_prefix="eval_diag_det",
            eval_seed=args.eval_seed,
            verbose=1,
        )
        eval_diag_stoch_cb = EvalDiagnosticsCallback(
            eval_env=eval_env,
            eval_freq=args.diag_stoch_freq,
            n_eval_episodes=args.diag_n_episodes,
            deterministic=False,
            log_prefix="eval_diag_stoch",
            eval_seed=args.eval_seed,
            verbose=1,
        )
        ent_coef_schedule = parse_linear_schedule(args.ent_coef_schedule)
        ent_coef_schedule_cb = (
            EntCoefScheduleCallback(
                schedule=ent_coef_schedule,
                total_timesteps=args.total_timesteps,
                verbose=1,
            )
            if ent_coef_schedule is not None
            else None
        )
        rul_curriculum_cb = (
            RulCurriculumCallback(
                schedule=rul_curriculum,
                total_timesteps=args.total_timesteps,
                eval_env=eval_env,
                verbose=1,
            )
            if rul_curriculum is not None
            else None
        )
        mlflow_cb = MLflowLoggingCallback(verbose=1)

        callbacks = [
            sync_cb,
            eval_callback,
            eval_diag_det_cb,
            eval_diag_stoch_cb,
        ]
        if rul_curriculum_cb is not None:
            callbacks.append(rul_curriculum_cb)
        if ent_coef_schedule_cb is not None:
            callbacks.append(ent_coef_schedule_cb)
        callbacks.append(mlflow_cb)
        callback_list = CallbackList(callbacks)

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

    parser.add_argument("--total-timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--learning-rate-schedule",
        type=str,
        default="linear:3e-4:3e-5",
        help="Learning-rate schedule spec, e.g. 'linear:3e-4:3e-5'. Use 'none' for fixed --learning-rate.",
    )
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument(
        "--ent-coef-schedule",
        type=str,
        default="linear:0.05:0.005",
        help="Entropy coefficient schedule spec, e.g. 'linear:0.05:0.005'. Use 'none' for fixed --ent-coef.",
    )
    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Torch device for PPO. Default CPU avoids CUDA warnings on machines without GPU drivers.",
    )

    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--eval-n-episodes", type=int, default=50)
    parser.add_argument("--diag-det-freq", type=int, default=20000)
    parser.add_argument("--diag-stoch-freq", type=int, default=40000)
    parser.add_argument("--diag-n-episodes", type=int, default=30)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Fixed seed for checkpoint-selection and diagnostic evals. Use a negative value to disable fixed-seed eval.",
    )

    parser.add_argument("--min-initial-rul", type=float, default=120.0)
    parser.add_argument(
        "--initial-rul-max",
        type=float,
        default=None,
        help="Optional maximum initial RUL. Use with --rul-curriculum-schedule for banded sampling.",
    )
    parser.add_argument(
        "--rul-curriculum-schedule",
        type=str,
        default=DEFAULT_RUL_CURRICULUM,
        help=(
            "Initial-RUL curriculum as 'progress:min:max' stages. "
            "Use 'none' to keep a fixed --min-initial-rul/--initial-rul-max range."
        ),
    )
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
            "Default uses all units so the RUL curriculum can span low/high-RUL bands. "
            "Pass no values after the flag to use all units."
        ),
    )

    parser.add_argument("--mlflow-tracking-uri", type=str, default="http://localhost:5000")
    parser.add_argument("--mlflow-experiment", type=str, default="Aircraft_Predictive_Maintenance_v3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eligible_units == []:
        args.eligible_units = None
    if args.eval_seed is not None and args.eval_seed < 0:
        args.eval_seed = None
    load_or_train_ppo(args)
    print("✅ PPO System Ready!")


if __name__ == "__main__":
    main()
