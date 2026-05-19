import os
import re
from collections import Counter
from typing import Any, Callable, cast

import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import KVWriter


class SaveVecNormalizeCallback(BaseCallback):
    """
    Log original VecNormalize statistics whenever a new best model is saved.
    This should be passed as callback_on_new_best to EvalCallback.
    """

    def __init__(self, save_path: str, verbose: int = 0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        # Save the vec_normalize stats from the training env (which is usually where normalization is learned)
        # We need to access the training env from the model
        vec_normalize_env = self.model.get_vec_normalize_env()
        if vec_normalize_env is not None:
            stats_path = os.path.join(self.save_path, "vec_normalize.pkl")
            vec_normalize_env.save(stats_path)
            if self.verbose > 0:
                print(f"Saved VecNormalize stats to {stats_path}")
        return True


class FixedSeedEvalCallback(EvalCallback):
    """
    EvalCallback variant that re-seeds the eval VecEnv before each evaluation.

    This makes checkpoint-to-checkpoint model selection much less noisy because
    every evaluation starts from the same deterministic seed stream. Random evals
    should still be run separately for robustness/generalization reporting.
    """

    def __init__(self, *args, eval_seed: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_seed = eval_seed

    def _on_step(self) -> bool:
        if self.eval_seed is not None and self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.seed(self.eval_seed)
        return super()._on_step()


class EntCoefScheduleCallback(BaseCallback):
    """Linearly schedule PPO entropy coefficient during training."""

    def __init__(
        self,
        schedule: Callable[[float], float],
        total_timesteps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.schedule = schedule
        self.total_timesteps = max(1, int(total_timesteps))

    def _on_step(self) -> bool:
        progress = min(1.0, float(self.num_timesteps) / float(self.total_timesteps))
        ent_coef = float(self.schedule(progress))
        ppo_model = cast(Any, self.model)
        ppo_model.ent_coef = ent_coef
        self.logger.record("train/ent_coef_schedule", ent_coef)
        if mlflow.active_run():
            mlflow.log_metric("train/ent_coef_schedule", ent_coef, step=self.num_timesteps)
        return True


class MLflowOutputFormat(KVWriter):
    """
    Custom writer to push Stable Baselines 3 logger metrics (rollout/, train/, time/) to MLflow.
    This is more robust than a callback because it hooks into the logger's dump() cycle.
    """

    def write(self, key_values, key_excluded, step=0):
        if not mlflow.active_run():
            return

        for key, value in key_values.items():
            if isinstance(value, (int, float, np.number)):
                # Clean key for MLflow compatibility (no spaces, parentheses, etc.)
                clean_key = re.sub(r"[^\w\-\./]", "_", key)
                mlflow.log_metric(clean_key, float(value), step=step)


class MLflowLoggingCallback(BaseCallback):
    """
    Callback for registering the MLflowOutputFormat into the Stable Baselines 3 logger.
    """

    def __init__(self, verbose: int = 0):
        super(MLflowLoggingCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        """
        Register the MLflow writer into the logger when training starts.
        """
        if any(isinstance(output_format, MLflowOutputFormat) for output_format in self.logger.output_formats):
            if self.verbose > 0:
                print("MLflowOutputFormat already registered; skipping duplicate registration.")
            return

        self.logger.output_formats.append(MLflowOutputFormat())
        if self.verbose > 0:
            print("🚀 MLflowOutputFormat registered to SB3 Logger.")

    def _on_step(self) -> bool:
        return True


class EvalDiagnosticsCallback(BaseCallback):
    """
    Run short evaluation probes and log
    event/action/altitude summaries to SB3 and MLflow. This is intentionally
    separate from EvalCallback: EvalCallback keeps model selection stable, while
    this callback explains why evaluation succeeds or collapses.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        deterministic: bool = False,
        log_prefix: str = "eval_diag",
        eval_seed: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.log_prefix = log_prefix
        self.eval_seed = eval_seed

    def _sync_vec_normalize(self) -> None:
        train_venv = self.model.get_vec_normalize_env()
        if train_venv is not None and hasattr(self.eval_env, "obs_rms"):
            self.eval_env.obs_rms = train_venv.obs_rms

    @staticmethod
    def _unwrap_infos(infos: Any) -> list[dict[str, Any]]:
        if isinstance(infos, dict):
            return [infos]
        if isinstance(infos, (list, tuple)):
            return [info for info in infos if isinstance(info, dict)]
        return []

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        self._sync_vec_normalize()
        if self.eval_seed is not None:
            self.eval_env.seed(self.eval_seed)

        event_counts: Counter[str] = Counter()
        step_event_counts: Counter[str] = Counter()
        action_counts: Counter[int] = Counter()
        rewards: list[float] = []
        lengths: list[int] = []
        final_altitudes: list[float] = []
        first_descend_distances: list[float] = []
        nearest_landing_distances: list[float] = []
        landing_profile_errors: list[float] = []
        soft_field_crash_distances: list[float] = []
        soft_landing_distances: list[float] = []
        airport_noises: list[float] = []
        weather_enabled_flags: list[float] = []
        landing_feasible_count = 0
        landing_feasible_checks = 0

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = np.array([False])
            episode_reward = 0.0
            episode_length = 0
            first_descend_seen = False
            last_info: dict[str, Any] = {}

            while not bool(done[0]):
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                action_array = np.asarray(action).reshape(-1)
                action_int = int(action_array[0])
                action_counts[action_int] += 1

                obs, reward, done, infos = self.eval_env.step(action)
                info_list = self._unwrap_infos(infos)
                if info_list:
                    last_info = info_list[0]

                reward_array = np.asarray(reward).reshape(-1)
                episode_reward += float(reward_array[0])
                episode_length += 1

                if action_int == 1 and not first_descend_seen:
                    first_descend_seen = True
                    first_descend_distances.append(float(last_info.get("dist_to_next_target", np.nan)))

                step_event = last_info.get("event")
                if step_event:
                    step_event_counts[str(step_event)] += 1

                if "nearest_landing_distance" in last_info:
                    nearest_landing_distances.append(float(last_info.get("nearest_landing_distance", np.nan)))
                if "landing_profile_error" in last_info:
                    landing_profile_errors.append(float(last_info.get("landing_profile_error", np.nan)))
                if "soft_field_crash_distance" in last_info:
                    soft_field_crash_distances.append(float(last_info.get("soft_field_crash_distance", np.nan)))
                if "soft_landing_distance" in last_info:
                    soft_landing_distances.append(float(last_info.get("soft_landing_distance", np.nan)))
                if "airport_noise" in last_info:
                    airport_noises.append(float(last_info.get("airport_noise", np.nan)))
                if "weather_enabled" in last_info:
                    weather_enabled_flags.append(1.0 if bool(last_info.get("weather_enabled")) else 0.0)

                if "landing_feasible_now" in last_info:
                    landing_feasible_checks += 1
                    if bool(last_info["landing_feasible_now"]):
                        landing_feasible_count += 1

            event_counts[str(last_info.get("event", "NO_EVENT"))] += 1
            rewards.append(episode_reward)
            lengths.append(episode_length)
            final_altitudes.append(float(last_info.get("altitude", np.nan)))

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        mean_length = float(np.mean(lengths)) if lengths else 0.0
        mean_final_altitude = float(np.nanmean(final_altitudes)) if final_altitudes else 0.0
        mean_first_descend_distance = (
            float(np.nanmean(first_descend_distances)) if first_descend_distances else -1.0
        )
        feasible_ratio = (
            float(landing_feasible_count / landing_feasible_checks)
            if landing_feasible_checks else 0.0
        )
        mean_nearest_landing_distance = (
            float(np.nanmean(nearest_landing_distances)) if nearest_landing_distances else -1.0
        )
        mean_landing_profile_error = (
            float(np.nanmean(landing_profile_errors)) if landing_profile_errors else -1.0
        )
        mean_soft_field_crash_distance = (
            float(np.nanmean(soft_field_crash_distances)) if soft_field_crash_distances else -1.0
        )
        mean_soft_landing_distance = (
            float(np.nanmean(soft_landing_distances)) if soft_landing_distances else -1.0
        )
        mean_airport_noise = float(np.nanmean(airport_noises)) if airport_noises else -1.0
        weather_enabled_ratio = float(np.mean(weather_enabled_flags)) if weather_enabled_flags else -1.0
        total_actions = sum(action_counts.values()) or 1

        metrics = {
            f"{self.log_prefix}/mean_reward": mean_reward,
            f"{self.log_prefix}/mean_length": mean_length,
            f"{self.log_prefix}/mean_final_altitude": mean_final_altitude,
            f"{self.log_prefix}/mean_first_descend_distance": mean_first_descend_distance,
            f"{self.log_prefix}/landing_feasible_ratio": feasible_ratio,
            f"{self.log_prefix}/mean_nearest_landing_distance": mean_nearest_landing_distance,
            f"{self.log_prefix}/mean_landing_profile_error": mean_landing_profile_error,
            f"{self.log_prefix}/mean_soft_field_crash_distance": mean_soft_field_crash_distance,
            f"{self.log_prefix}/mean_soft_landing_distance": mean_soft_landing_distance,
            f"{self.log_prefix}/mean_airport_noise": mean_airport_noise,
            f"{self.log_prefix}/weather_enabled_ratio": weather_enabled_ratio,
            f"{self.log_prefix}/action_cruise_ratio": action_counts[0] / total_actions,
            f"{self.log_prefix}/action_descend_ratio": action_counts[1] / total_actions,
            f"{self.log_prefix}/action_climb_ratio": action_counts[2] / total_actions,
            f"{self.log_prefix}/reward_std": float(np.std(rewards)) if rewards else 0.0,
            f"{self.log_prefix}/length_std": float(np.std(lengths)) if lengths else 0.0,
        }
        for event_name, count in event_counts.items():
            metrics[f"{self.log_prefix}/event_{event_name}"] = float(count)
        for event_name, count in step_event_counts.items():
            metrics[f"{self.log_prefix}/step_event_{event_name}"] = float(count)

        for key, value in metrics.items():
            self.logger.record(key, value)
            if mlflow.active_run():
                mlflow.log_metric(key, value, step=self.num_timesteps)

        self.logger.dump(step=self.num_timesteps)

        if self.verbose > 0:
            print(
                f"🔎 {self.log_prefix}: reward={mean_reward:.2f}, "
                f"len={mean_length:.1f}, events={dict(event_counts)}, "
                f"actions={dict(action_counts)}"
            )

        return True


class SyncVecNormalizeCallback(BaseCallback):
    """
    Synchronize the statistics of the evaluation environment with the
    statistics of the training environment.
    """

    def __init__(self, eval_env, verbose: int = 0):
        super(SyncVecNormalizeCallback, self).__init__(verbose)
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        # Sync stats from train_env to eval_env
        train_venv = self.model.get_vec_normalize_env()
        if train_venv is not None:
            # We assume eval_env is also a VecNormalize or contains one
            from stable_baselines3.common.vec_env import VecNormalize

            eval_venv = self.eval_env

            # If eval_env is a list or something else, we need to be careful
            # But usually it's passed as the VecNormalize object directly in the training script
            if isinstance(eval_venv, VecNormalize):
                eval_venv.obs_rms = train_venv.obs_rms
                # We usually don't sync reward_rms for eval
                if self.verbose > 1:
                    print("🔄 Synced VecNormalize stats to eval_env")
        return True
