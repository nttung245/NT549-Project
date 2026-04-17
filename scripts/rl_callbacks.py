import os
import re
import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
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
        if self.model.get_vec_normalize_env() is not None:
            stats_path = os.path.join(self.save_path, "vec_normalize.pkl")
            self.model.get_vec_normalize_env().save(stats_path)
            if self.verbose > 0:
                print(f"Saved VecNormalize stats to {stats_path}")
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
                clean_key = re.sub(r'[^\w\-\./]', '_', key)
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
        self.logger.output_formats.append(MLflowOutputFormat())
        if self.verbose > 0:
            print("🚀 MLflowOutputFormat registered to SB3 Logger.")

    def _on_step(self) -> bool:
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
        if self.model.get_vec_normalize_env() is not None:
            # We assume eval_env is also a VecNormalize or contains one
            from stable_baselines3.common.vec_env import VecNormalize
            
            train_venv = self.model.get_vec_normalize_env()
            eval_venv = self.eval_env
            
            # If eval_env is a list or something else, we need to be careful
            # But usually it's passed as the VecNormalize object directly in the training script
            if isinstance(eval_venv, VecNormalize):
                eval_venv.obs_rms = train_venv.obs_rms
                # We usually don't sync reward_rms for eval
                if self.verbose > 1:
                    print("🔄 Synced VecNormalize stats to eval_env")
        return True
