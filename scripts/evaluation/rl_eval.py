import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.core.aircraft_env import AircraftEnv
from scripts.data.data_processor import FEATURES, KEY_SENSORS

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from typing import Any

def _get_base_env(vec_env) -> AircraftEnv:
    """Unwrap DummyVecEnv/VecNormalize to access the underlying AircraftEnv."""
    env = vec_env
    while hasattr(env, "venv"):
        env = env.venv
    return env.envs[0]  # type: ignore[no-any-return]


def evaluate_rl_agent(
    ppo_model,
    test_rolling,
    model_path: str,
    scaler,
    num_episodes=20,
    stats_path=None,
    deterministic: bool = True,
    eligible_units: list[int] | None = None,
    min_initial_rul: float = 120.0,
    maintenance_resets_health: bool = True,
    max_steps: int = 2500,
    plot: bool = True,
):
    """
    Run the PPO agent on separate engines from the test set and report route-arrival success.

    Args:
        ppo_model: Loaded Stable-Baselines3 policy, or None for heuristic fallback.
        test_rolling: Test dataframe with RUL column.
        model_path: Path to trained Keras LSTM/GRU model file used by AircraftEnv.
        scaler: Fitted StandardScaler.
        num_episodes: Number of complete routes to evaluate.
        stats_path: Optional VecNormalize stats. Must match PPO training stats.
        deterministic: Use deterministic policy actions for stable benchmark eval.
        eligible_units: Optional unit IDs to sample from.
        min_initial_rul: Filter out units that start below this RUL threshold.
        maintenance_resets_health: Match training env maintenance semantics.
        max_steps: Safety cap for pathological eval loops.
        plot: Whether to render the RUL-at-maintenance histogram.
    """
    def make_eval_env():
        return AircraftEnv(
            fleet_data=test_rolling,
            model_path=model_path,
            scaler=scaler,
            sensor_list=KEY_SENSORS,
            features_list=FEATURES,
            eligible_units=eligible_units,
            min_initial_rul=min_initial_rul,
            maintenance_resets_health=maintenance_resets_health,
        )
        
    eval_env = DummyVecEnv([make_eval_env])
    
    if stats_path is not None:
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"VecNormalize stats not found: {stats_path}")
        eval_env = VecNormalize.load(stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    results = []
    print(f"🚀 Evaluating PPO on {num_episodes} test routes (deterministic={deterministic})...")
    
    for i in range(num_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0
        maintenance_count = 0
        maintenance_ruls: list[float] = []
        events: list[str] = []
        info: dict[str, Any] = {}
        
        while not done and step_count < max_steps:
            if ppo_model is not None:
                action, _ = ppo_model.predict(obs, deterministic=deterministic)
            else:
                current_obs: np.ndarray = obs[0]  # type: ignore[index]
                current_rul = float(current_obs[2])
                airport_dists = current_obs[3:9]
                dists_ahead = [d for d in airport_dists if d > 0]
                dist_next = float(min(dists_ahead) if dists_ahead else current_obs[9])
                
                if current_rul > 60:
                    action = np.array([0])
                elif dist_next < 1000 and current_obs[0] > 0:
                    action = np.array([1])
                elif current_obs[0] <= 0:
                    action = np.array([2])
                else:
                    action = np.array([0])
            
            obs, reward, done_array, info_array = eval_env.step(action)
            done = bool(done_array[0])
            ep_reward += float(reward[0])
            step_count += 1
            info = info_array[0]

            event = str(info.get('event', ''))
            if event:
                events.append(event)
            if event == 'MAINTAINED':
                maintenance_count += 1
                maintenance_ruls.append(float(info.get('rul_at_landing', np.nan)))

        if not done and step_count >= max_steps:
            info = {**info, 'event': 'EVAL_MAX_STEPS'}
            events.append('EVAL_MAX_STEPS')
        
        final_event = str(info.get('event', 'TIMEOUT'))
        original_env = _get_base_env(eval_env)
        arrived = final_event == 'ARRIVED'
        safe_route = arrived and all(event not in {'CRASHED', 'FIELD_CRASH', 'FUEL_EMPTY'} for event in events)
        
        results.append({
            'engine_id': original_env.twin.engine_id if original_env.twin is not None else None,
            'event': final_event,
            'arrived': arrived,
            'safe_route': safe_route,
            'reward': ep_reward,
            'maintenance_count': maintenance_count,
            'mean_maintenance_rul': float(np.nanmean(maintenance_ruls)) if maintenance_ruls else np.nan,
            'steps': step_count,
            'events': events,
        })
        
    df_results = pd.DataFrame(results)
    arrival_rate = float(df_results['arrived'].mean() * 100) if not df_results.empty else 0.0
    safe_route_rate = float(df_results['safe_route'].mean() * 100) if not df_results.empty else 0.0
    avg_reward = float(df_results['reward'].mean()) if not df_results.empty else 0.0
    avg_maintenance_count = float(df_results['maintenance_count'].mean()) if not df_results.empty else 0.0
    avg_maintenance_rul = float(df_results['mean_maintenance_rul'].dropna().mean()) if df_results['mean_maintenance_rul'].notna().any() else float('nan')
    
    print("\n" + "=" * 48)
    print("📊 RL EVALUATION RESULTS (Test Set)")
    print(f"   - Arrival Rate:       {arrival_rate:.1f}%")
    print(f"   - Safe Route Rate:    {safe_route_rate:.1f}%")
    print(f"   - Average Reward:     {avg_reward:.1f}")
    print(f"   - Avg Maintenances:   {avg_maintenance_count:.2f}")
    print(f"   - Avg RUL at Maint.:  {avg_maintenance_rul:.1f} cycles")
    print("=" * 48)
    
    if plot:
        maint_ruls = df_results['mean_maintenance_rul'].dropna()
        if not maint_ruls.empty:
            plt.figure(figsize=(10, 4))
            plt.hist(maint_ruls, bins=10, color='green', alpha=0.6, label='Maintenance Checkpoints')
            plt.axvline(0, color='red', linestyle='--', label='Critical Failure')
            plt.title('Mean Safety Margin at Maintenance Checkpoints')
            plt.xlabel('Remaining Useful Life (RUL)')
            plt.ylabel('Count')
            plt.legend()
            plt.show()
    
    return df_results
