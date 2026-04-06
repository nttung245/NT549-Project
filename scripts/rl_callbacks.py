import os
import mlflow
from stable_baselines3.common.callbacks import BaseCallback

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

class MLflowLoggingCallback(BaseCallback):
    """
    Callback for logging metrics to MLflow.
    """
    def __init__(self, verbose: int = 0):
        super(MLflowLoggingCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        # Log metrics occasionally or on every step
        # SB3 logs mostly on rollout end, but we can catch them here
        return True

    def _on_rollout_end(self) -> None:
        """
        Log metrics from the logger to MLflow.
        """
        # Get metrics from SB3 logger
        # Stable Baselines 3 Logger uses `name_to_value` to store log data
        if self.logger is not None:
            if hasattr(self.logger, 'name_to_value'):
                metrics = self.logger.name_to_value
            elif hasattr(self.logger, 'get_log_dict'):
                metrics = self.logger.get_log_dict()
            else:
                metrics = {}
                
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    # Clean up key name for MLflow if needed
                    mlflow.log_metric(key.replace("/", "_"), val, step=self.num_timesteps)
