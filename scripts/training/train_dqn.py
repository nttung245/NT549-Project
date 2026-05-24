# -*- coding: utf-8 -*-
"""
DQN training script for Aircraft predictive maintenance.

This script is intended as a value-based RL baseline for comparison with PPO:
- same AircraftEnv dynamics and reward design
- same LSTM/RUL readiness flow
- optional VecNormalize observation normalization
- fixed-seed checkpoint selection and diagnostic eval callbacks
- MLflow logging for SB3 metrics, parameters, models, and VecNormalize stats
"""

from __future__ import annotations

import argparse
import os

# TensorFlow is imported indirectly by the LSTM model and by AircraftEnv. Keep
# the runtime CPU-first and quiet by default, matching the PPO training script.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
from datetime import datetime
from pathlib import Path

import mlflow

# Ensure project root is importable when this script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.data_processor import prepare_data
from scripts.training.rl_callbacks import (
    EvalDiagnosticsCallback,
    FixedSeedEvalCallback,
    MLflowLoggingCallback,
    SaveVecNormalizeCallback,
    SyncVecNormalizeCallback,
)
from scripts.training.train_ppo import (
    ABLATION_PRESETS,
    ABLATION_RUN_ORDER,
    DEFAULT_AIRPORT_NOISE,
    DEFAULT_TRAIN_ELIGIBLE_UNITS,
    apply_ablation_preset,
    collect_explicit_cli_options,
    ensure_lstm_model,
    make_aircraft_env_factory,
    make_sb3_learning_rate,
)

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠️  stable-baselines3 not installed. Install with: pip install stable-baselines3")


def load_or_train_dqn(args: argparse.Namespace):
    """Load an existing DQN run or train a new DQN baseline."""
    if not HAS_SB3:
        raise RuntimeError("stable-baselines3 is required for DQN training.")

    data_dir = PROJECT_ROOT / "CMAPSSData"
    models_dir = PROJECT_ROOT / "models"
    logs_dir = PROJECT_ROOT / "logs"
    model_path = models_dir / "lstm_rul_model.keras"

    print("📂 Loading CMAPSS data for preprocessing/LSTM readiness...")
    train_rolling, _, _, scaler = prepare_data(str(data_dir))
    ensure_lstm_model(model_path, train_rolling, scaler)

    run_name = args.run_name or f"DQN_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"🏷️ Training run name: {run_name}")
    print(f"🧪 Ablation preset: {args.ablation_preset}")
    print(f"🛩️ Eligible training engine units: {args.eligible_units}")
    print(f"🌦️ Weather enabled: {args.enable_weather}; airport noise: ±{args.airport_noise:g}")
    print(
        "DQN baseline distribution: "
        f"RUL=[{args.min_initial_rul:g}, {args.initial_rul_max:g}], "
        f"airport_noise=±{args.airport_noise:g}, weather={args.enable_weather}."
    )

    repo_dqn_path = models_dir / f"dqn_aircraft_{run_name}"
    stats_path = models_dir / f"dqn_aircraft_{run_name}_vec_normalize.pkl"
    best_model_dir = models_dir / f"best_dqn_{run_name}"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    make_env = make_aircraft_env_factory(
        model_path=str(model_path),
        min_initial_rul=args.min_initial_rul,
        initial_rul_max=args.initial_rul_max,
        maintenance_resets_health=args.maintenance_resets_health,
        eligible_units=args.eligible_units,
        airport_noise=args.airport_noise,
        enable_weather=args.enable_weather,
    )

    if args.check_env:
        print("🔍 Checking AircraftEnv compatibility...")
        env = make_env()
        check_env(env, warn=True)
        print("✅ Environment check passed.")

    if (repo_dqn_path.with_suffix(".zip")).exists() and stats_path.exists() and not args.force_train:
        print(f"📦 Loading existing DQN agent and VecNormalize stats from: {repo_dqn_path}")
        raw_train_env = make_vec_env(make_env, n_envs=1)
        train_env = None
        try:
            train_env = VecNormalize.load(str(stats_path), raw_train_env)
            train_env.training = False
            train_env.norm_reward = False
            dqn_model = DQN.load(str(repo_dqn_path), env=train_env, device=args.device)
        except (AssertionError, ValueError) as exc:
            if train_env is not None:
                train_env.close()
            else:
                raw_train_env.close()
            current_shape = getattr(raw_train_env.observation_space, "shape", None)
            raise RuntimeError(
                "Existing DQN/VecNormalize artifacts are incompatible with the current AircraftEnv "
                f"observation space {current_shape}. Re-run with --force-train or choose a newer --run-name. "
                f"Model path: {repo_dqn_path.with_suffix('.zip')}; VecNormalize path: {stats_path}."
            ) from exc
        return dqn_model, train_env, run_name, repo_dqn_path, stats_path, best_model_dir

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or args.mlflow_tracking_uri
    print(f"📊 Connecting to MLflow at: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    print(f"🚀 Training DQN Agent ({args.total_timesteps:,} steps) with EvalCallback and MLflow...")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("algorithm", "DQN")

        # Keep the same VecEnv shape as PPO by default for fairer wall-clock and
        # data-collection conditions. DQN-specific replay-buffer behavior remains
        # controlled by train_freq/gradient_steps/learning_starts.
        vec_env_cls = SubprocVecEnv if args.n_envs > 1 else None
        train_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=vec_env_cls)
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=args.clip_obs,
        )

        eval_env = make_vec_env(make_env, n_envs=1)
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=args.clip_obs,
            training=False,
        )

        learning_rate = make_sb3_learning_rate(
            fixed_value=args.learning_rate,
            schedule_spec=args.learning_rate_schedule,
        )
        policy_kwargs = {"net_arch": args.net_arch} if args.net_arch else None

        dqn_model = DQN(
            "MlpPolicy",
            train_env,
            verbose=args.verbose,
            device=args.device,
            learning_rate=learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=(args.train_freq, "step"),
            gradient_steps=args.gradient_steps,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(logs_dir / "dqn_aircraft"),
        )

        params = {
            "algorithm": "DQN",
            "run_name": run_name,
            "total_timesteps": args.total_timesteps,
            "device": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tf_cpp_min_log_level": os.environ.get("TF_CPP_MIN_LOG_LEVEL"),
            "tf_enable_onednn_opts": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
            "learning_rate": args.learning_rate,
            "learning_rate_schedule": args.learning_rate_schedule,
            "n_envs": args.n_envs,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "tau": args.tau,
            "gamma": args.gamma,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "target_update_interval": args.target_update_interval,
            "exploration_fraction": args.exploration_fraction,
            "exploration_initial_eps": args.exploration_initial_eps,
            "exploration_final_eps": args.exploration_final_eps,
            "max_grad_norm": args.max_grad_norm,
            "net_arch": args.net_arch,
            "clip_obs": args.clip_obs,
            "norm_obs": True,
            "norm_reward_train": True,
            "norm_reward_eval": False,
            "eval_freq": args.eval_freq,
            "eval_n_episodes": args.eval_n_episodes,
            "eval_deterministic": True,
            "eval_policy": "deterministic_greedy",
            "diag_det_freq": args.diag_det_freq,
            "diag_stoch_freq": args.diag_stoch_freq,
            "diag_n_episodes": args.diag_n_episodes,
            "eval_seed": args.eval_seed,
            "ablation_preset": args.ablation_preset,
            "ablation_run_order": ",".join(ABLATION_RUN_ORDER),
            "enable_weather": args.enable_weather,
            "airport_noise": args.airport_noise,
            "min_initial_rul": args.min_initial_rul,
            "initial_rul_max": args.initial_rul_max,
            "maintenance_resets_health": args.maintenance_resets_health,
            "eligible_units": args.eligible_units,
            "model_path": str(model_path),
            "repo_dqn_path": str(repo_dqn_path),
            "stats_path": str(stats_path),
            "best_model_dir": str(best_model_dir),
        }
        mlflow.log_params(params)

        sync_cb = SyncVecNormalizeCallback(eval_env=eval_env, verbose=1)
        save_vec_stats_cb = SaveVecNormalizeCallback(save_path=str(best_model_dir), verbose=1)
        eval_callback = FixedSeedEvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(logs_dir / "eval_results_dqn"),
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
        mlflow_cb = MLflowLoggingCallback(verbose=1)
        callback_list = CallbackList([sync_cb, eval_callback, eval_diag_det_cb, eval_diag_stoch_cb, mlflow_cb])

        dqn_model.learn(total_timesteps=args.total_timesteps, callback=callback_list)

        dqn_model.save(str(repo_dqn_path))
        train_env.save(str(stats_path))
        print(f"💾 DQN model saved to: {repo_dqn_path}.zip")
        print(f"💾 VecNormalize stats saved to: {stats_path}")

        mlflow.log_artifact(str(repo_dqn_path.with_suffix(".zip")), artifact_path="model_final")
        mlflow.log_artifact(str(stats_path), artifact_path="vecnormalize_final")

        best_model_path = best_model_dir / "best_model.zip"
        best_stats_path = best_model_dir / "vec_normalize.pkl"
        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path), artifact_path="model_best")
        if best_stats_path.exists():
            mlflow.log_artifact(str(best_stats_path), artifact_path="vecnormalize_best")

        print("✅ DQN training complete and artifacts logged to MLflow.")
        return dqn_model, train_env, run_name, repo_dqn_path, stats_path, best_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN Agent for Aircraft Predictive Maintenance")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name. Default: timestamped DQN_Run_*.")
    parser.add_argument(
        "--ablation-preset",
        type=str,
        default="full",
        choices=list(ABLATION_PRESETS.keys()),
        help=(
            "Preset for ablation progression. Recommended order: "
            f"{', '.join(ABLATION_RUN_ORDER)}. Use 'custom' to rely only on explicit CLI flags."
        ),
    )
    parser.add_argument("--force-train", action="store_true", help="Train even if a model with the same run name already exists.")
    parser.add_argument("--check-env", action="store_true", help="Run stable-baselines3 check_env before training.")

    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--learning-rate-schedule",
        type=str,
        default="linear:3e-4:3e-5",
        help="Learning-rate schedule spec, e.g. 'linear:3e-4:3e-5'. Use 'none' for fixed --learning-rate.",
    )
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=10_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.35)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--net-arch", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Torch device for DQN. Default CPU avoids CUDA warnings on machines without GPU drivers.",
    )

    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--eval-n-episodes", type=int, default=50)
    parser.add_argument("--diag-det-freq", type=int, default=40000)
    parser.add_argument("--diag-stoch-freq", type=int, default=40000)
    parser.add_argument("--diag-n-episodes", type=int, default=30)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Fixed seed for checkpoint-selection and diagnostic evals. Use a negative value to disable fixed-seed eval.",
    )
    parser.add_argument("--min-initial-rul", type=float, default=170.0)
    parser.add_argument(
        "--initial-rul-max",
        type=float,
        default=260.0,
        help="Optional maximum initial RUL for the fixed training distribution.",
    )
    parser.add_argument(
        "--no-maintenance-reset-health",
        dest="maintenance_resets_health",
        action="store_false",
        help="Disable health/RUL reset after successful intermediate maintenance.",
    )
    parser.set_defaults(maintenance_resets_health=True)
    parser.add_argument(
        "--airport-noise",
        type=float,
        default=DEFAULT_AIRPORT_NOISE,
        help="Fixed airport-position randomization amplitude.",
    )
    parser.add_argument(
        "--weather",
        dest="enable_weather",
        action="store_true",
        help="Enable weather zones in AircraftEnv.",
    )
    parser.add_argument(
        "--no-weather",
        dest="enable_weather",
        action="store_false",
        help="Disable weather zones for simpler ablation runs.",
    )
    parser.set_defaults(enable_weather=True)
    parser.add_argument(
        "--eligible-units",
        type=int,
        nargs="*",
        default=DEFAULT_TRAIN_ELIGIBLE_UNITS,
        help=(
            "Engine unit IDs to sample from before min_initial_rul filtering. "
            "Default uses all units across the fixed RUL band. "
            "Pass no values after the flag to use all units."
        ),
    )

    parser.add_argument("--mlflow-tracking-uri", type=str, default="http://localhost:5000")
    parser.add_argument("--mlflow-experiment", type=str, default="Aircraft_Predictive_Maintenance_v4")

    args = parser.parse_args()
    apply_ablation_preset(args, collect_explicit_cli_options(sys.argv[1:]))
    return args


def main() -> None:
    args = parse_args()
    if args.eligible_units == []:
        args.eligible_units = None
    if args.eval_seed is not None and args.eval_seed < 0:
        args.eval_seed = None
    load_or_train_dqn(args)
    print("✅ DQN System Ready!")


if __name__ == "__main__":
    main()
