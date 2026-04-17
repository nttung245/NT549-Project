import sys
import numpy
try:
    from stable_baselines3 import PPO
    import cloudpickle
    print(f"Python: {sys.version}")
    print(f"Numpy: {numpy.__version__}")
    print(f"Loading models/ppo_aircraft...")
    model = PPO.load("models/ppo_aircraft")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:")
    import traceback
    traceback.print_exc()
