import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.data_processor import prepare_data, FEATURES, KEY_SENSORS
from scripts.aircraft_env import AircraftEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

def test_load():
    print("--- Initializing Test Load ---")
    data_dir = os.path.join(PROJECT_ROOT, 'CMAPSSData')
    train_rolling, _, _, scaler = prepare_data(data_dir)
    
    env = AircraftEnv(
        fleet_data=train_rolling,
        model=None,
        scaler=scaler,
        sensor_list=KEY_SENSORS,
        features_list=FEATURES
    )
    
    ppo_path = os.path.join(PROJECT_ROOT, 'models', 'ppo_aircraft')
    stats_path = os.path.join(PROJECT_ROOT, 'models', 'ppo_aircraft_vec_normalize.pkl')
    
    if os.path.exists(ppo_path + ".zip") and os.path.exists(stats_path):
        try:
            def make_dummy_env(): return env
            dummy_vec_env = DummyVecEnv([make_dummy_env])
            vec_normalize = VecNormalize.load(stats_path, dummy_vec_env)
            ppo_model = PPO.load(ppo_path, env=vec_normalize)
            print("SUCCESS: PPO model and VecNormalize loaded!")
            
            # Test a single prediction
            obs, _ = env.reset()
            norm_obs = vec_normalize.normalize_obs(obs)
            action, _ = ppo_model.predict(norm_obs, deterministic=True)
            print(f"Prediction Test: Action={action}")
            
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("FAILED: Files missing")

if __name__ == "__main__":
    test_load()
