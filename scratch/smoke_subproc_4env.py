import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

PROJECT_ROOT = os.getcwd()
model_path = os.path.join(PROJECT_ROOT, "models", "lstm_rul_model.keras")


def make_env():
    from scripts.aircraft_env import AircraftEnv
    from scripts.data_processor import FEATURES, KEY_SENSORS, prepare_data

    data_dir = os.path.join(PROJECT_ROOT, "CMAPSSData")
    train_rolling, _, _, scaler = prepare_data(data_dir)
    return AircraftEnv(
        fleet_data=train_rolling,
        model_path=model_path,
        scaler=scaler,
        sensor_list=KEY_SENSORS,
        features_list=FEATURES,
    )


if __name__ == "__main__":
    train_env = make_vec_env(make_env, n_envs=4, vec_env_cls=SubprocVecEnv)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", train_env, verbose=0, n_steps=16, batch_size=8, n_epochs=1)
    model.learn(total_timesteps=32)
    train_env.close()
    print("SUBPROC_4ENV_SMOKE_OK")
