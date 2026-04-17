import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.aircraft_env import AircraftEnv
from scripts.data_processor import FEATURES, KEY_SENSORS

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def evaluate_rl_agent(ppo_model, test_rolling, lstm_model, scaler, num_episodes=20, stats_path=None):
    """
    Run the PPO agent on separate engines from the test set and report success rate.
    If stats_path is provided, it will wrap the env in a VecNormalize to ensure observations match the training scales.
    """
    def make_eval_env():
        return AircraftEnv(
            fleet_data=test_rolling,
            model=lstm_model,
            scaler=scaler,
            sensor_list=KEY_SENSORS,
            features_list=FEATURES
        )
        
    # We must wrap it in DummyVecEnv first for VecNormalize to work
    eval_env = DummyVecEnv([make_eval_env])
    
    if stats_path is not None and os.path.exists(stats_path):
        eval_env = VecNormalize.load(stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    results = []
    print(f"🚀 Evaluating PPO on {num_episodes} test engines...")
    
    for i in range(num_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0
        step_count = 0
        
        while not done:
            if ppo_model is not None:
                action, _ = ppo_model.predict(obs, deterministic=True)
            else:
                # Heuristic fallback: Cruise if RUL is high, Descend if RUL is low
                # New 10-dim indexing:
                # obs[0][3] is Current_RUL
                # obs[0][4:9] are Sub-airports
                # obs[0][9] is Dist_to_Destination
                
                current_rul = obs[0][3]
                airport_dists = obs[0][4:9]
                dists_ahead = [d for d in airport_dists if d > 0]
                dist_next = min(dists_ahead) if dists_ahead else obs[0][9]
                
                if current_rul > 60:
                    action = [0]  # CRUISE
                elif dist_next < 1000 and obs[0][0] > 0:
                    action = [1]  # DESCEND
                else:
                    action = [0]  # CRUISE
            
            # VecEnv returns 4 values: obs, rewards, dones, infos
            obs, reward, done_array, info_array = eval_env.step(action)
            done = done_array[0]
            ep_reward += reward[0]
            step_count += 1
        
        info = info_array[0]
        
        # When an episode is done, VecEnv automatically resets the environment and stores the pre-reset obs in info.
        # But we can just use the final event tracking.
        final_event = info.get('event', 'TIMEOUT')
        
        # We need to get engine_id from the original unwrapped environment
        # DummyVecEnv has a method 'envs' which contains the original envs
        original_env = eval_env.envs[0] if hasattr(eval_env, 'envs') \
                       else eval_env.venv.envs[0]
        
        results.append({
            'engine_id': original_env.twin.engine_id,
            'event': final_event,
            'reward': ep_reward,
            'landed_rul': info.get('rul_at_landing', 0) if final_event in ('ARRIVED', 'MAINTAINED') else 0,
            'steps': step_count
        })
        
    df_results = pd.DataFrame(results)
    success_rate = (df_results['event'].isin(['ARRIVED', 'MAINTAINED'])).mean() * 100
    avg_reward = df_results['reward'].mean()
    avg_landed_rul = df_results[df_results['event'].isin(['ARRIVED', 'MAINTAINED'])]['landed_rul'].mean()
    
    print("\n" + "=" * 40)
    print(f"📊 RL EVALUATION RESULTS (Test Set)")
    print(f"   - Success Rate:      {success_rate:.1f}%")
    print(f"   - Average Reward:    {avg_reward:.1f}")
    print(f"   - Avg RUL at Landing: {avg_landed_rul:.1f} cycles")
    print("=" * 40)
    
    # Visualization
    plt.figure(figsize=(10, 4))
    plt.hist(df_results[df_results['event'] == 'LANDED']['landed_rul'], bins=10, color='green', alpha=0.6, label='Safe Landings')
    plt.axvline(0, color='red', linestyle='--', label='Critical Failure')
    plt.title('Safety Margin (RUL at Landing)')
    plt.xlabel('Remaining Useful Life (RUL)')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    
    return df_results
