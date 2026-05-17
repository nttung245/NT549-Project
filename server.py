import asyncio
import json
import os
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.data.data_processor import prepare_data, FEATURES, KEY_SENSORS
from scripts.core.aircraft_env import AircraftEnv
from scripts.core.digital_twin import AircraftDigitalTwin

# RL Components
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

app = FastAPI(title="Aircraft Digital Twin - WebSocket Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
env: AircraftEnv | None = None
simulation_task = None
is_running = False
current_state = {}
clients = set()
lstm_model = None
ppo_model = None
vec_normalize = None

# Match the latest trained PPO run used by the stable demo.
DEFAULT_PPO_RUN_NAME = os.environ.get("PPO_RUN_NAME", "PPO_Run_20260516_143227")
DEFAULT_ELIGIBLE_UNITS = [14, 62, 3]

@app.on_event("startup")
async def startup_event():
    global env, lstm_model
    print("=" * 60)
    print("🚀 Initializing FastAPI WebSocket Backend...")
    print("=" * 60)
    
    data_dir = os.path.join(PROJECT_ROOT, 'CMAPSSData')
    train_rolling, _, _, scaler = prepare_data(data_dir)

    model_path = os.path.join(PROJECT_ROOT, 'models', 'lstm_rul_model.keras')
    if os.path.exists(model_path):
        import tensorflow as tf
        lstm_model = tf.keras.models.load_model(model_path)
        print("📦 Loaded LSTM Model.")
    else:
        print("⚠️ Warning: lstm_rul_model.keras not found, RUL prediction will rely on defaults.")
        lstm_model = None

    env = AircraftEnv(
        fleet_data=train_rolling,
        model_path=model_path,
        scaler=scaler,
        sensor_list=KEY_SENSORS,
        features_list=FEATURES,
        eligible_units=DEFAULT_ELIGIBLE_UNITS,
    )
    
    obs, info = env.reset()
    
    # --- Load RL Agent ---
    best_ppo_dir = os.path.join(PROJECT_ROOT, "models", f"best_ppo_{DEFAULT_PPO_RUN_NAME}")
    ppo_path = os.path.join(best_ppo_dir, "best_model.zip")
    stats_path = os.path.join(best_ppo_dir, "vec_normalize.pkl")
    print(f"🔎 PPO run for UI: {DEFAULT_PPO_RUN_NAME}")
    print(f"🔎 PPO model path: {ppo_path}")
    print(f"🔎 VecNormalize path: {stats_path}")
    
    global ppo_model, vec_normalize
    if os.path.exists(ppo_path) and os.path.exists(stats_path):
        try:
            # We need a DummyVecEnv to wrap for VecNormalize
            def make_dummy_env():
                if env is None:
                    raise RuntimeError("Environment has not been initialized.")
                return env
            dummy_vec_env = DummyVecEnv([make_dummy_env])
            
            # Load normalization stats
            vec_normalize = VecNormalize.load(stats_path, dummy_vec_env)
            vec_normalize.training = False
            vec_normalize.norm_reward = False
            
            # Load PPO Model
            ppo_model = PPO.load(ppo_path, env=vec_normalize)
            print("🤖 Loaded PPO RL Agent and Normalization stats.")
        except Exception as e:
            print(f"❌ Error loading PPO model: {e}")
            ppo_model = None
            vec_normalize = None
    else:
        print("⚠️ Warning: PPO model or stats not found. Falling back to heuristic policy.")

    update_current_state(obs, info, 0.0, 0, "INIT")


def update_current_state(obs, info, reward, step, action_str):
    global current_state
    if env is None:
        return
    # obs = [Altitude, Fuel, Effective_RUL, Dist_AP1..AP6, Dist_Dest,
    #        In_Approach_Zone, Wind, Fuel_Mult, RUL_Mult, Next_Hazard_Dist, Next_Hazard_Type]
    # "dist_next" in UI should show the distance to the closest airport AHEAD.
    airport_dists = obs[3:9]
    dists_ahead = [d for d in airport_dists if d > 0]
    dist_next = min(dists_ahead) if dists_ahead else obs[9] # Fallback to destination if all passed
    twin = env.twin

    current_state = {
        "step": step,
        "altitude": float(obs[0]),
        "velocity": float(twin.velocity if twin is not None else 0.0),
        "fuel": float(obs[1]),
        "rul": float(obs[2]),
        "dist_next": float(dist_next),
        "dist_dest": float(obs[9]),
        "in_approach_zone": bool(obs[10] > 0.5),
        "wind_strength": float(obs[11]),
        "weather_fuel_multiplier": float(obs[12]),
        "weather_rul_multiplier": float(obs[13]),
        "next_hazard_distance": float(obs[14]),
        "next_hazard_type": int(obs[15]),
        "weather": str(info.get("weather", "clear")),
        "weather_zones": env.weather_map.to_dicts(),
        "fuel_capacity": float(env.FUEL_CAPACITY),
        "total_distance": float(env.TOTAL_DISTANCE),
        "sub_airports": [float(x) for x in env.sub_airports],
        "flight_phase": env.flight_phase,
        "action": action_str,
        "reward": float(reward),
        "info": info
    }

async def broadcast_state():
    if clients:
        msg = json.dumps({"type": "state", "data": current_state})
        await asyncio.gather(*[client.send_text(msg) for client in clients])

async def simulation_loop():
    global is_running, env
    step_count = 0

    while True:
        if is_running and env:
            # Use RL Agent if available, else fallback to heuristic
            active_env = env
            obs = active_env._get_obs() # Get raw observation
            
            if ppo_model and vec_normalize:
                # 1. Normalize observation
                norm_obs = vec_normalize.normalize_obs(obs)
                # 2. Predict action with the stochastic policy used during PPO training.
                # Deterministic argmax can collapse to the single highest-probability action.
                action, _ = ppo_model.predict(norm_obs, deterministic=False)
                action = int(action)
            else:
                # Simple heuristic policy fallback
                dist_closest = active_env._dist_to_nearest_airport()
                twin = active_env.twin
                if twin is None:
                    raise RuntimeError("Environment must be reset before simulation loop runs.")
                raw_obs = active_env._get_obs()
                current_rul = float(raw_obs[2])
                action = 0  # Default fly
                if (current_rul < 30 or twin.fuel < 20) and active_env.flight_phase == "CRUISING":
                    if dist_closest < active_env.LANDING_THRESHOLD:
                        action = 1 # Land

            obs, r, done, t, info = active_env.step(action)
            step_count += 1
            
            # Map actions to readable strings
            action_map = {0: "CRUISE", 1: "DESCEND", 2: "CLIMB"}
            act_str = action_map.get(action, "UNKNOWN")
            
            update_current_state(obs, info, r, step_count, act_str)
            await broadcast_state()

            if done:
                is_running = False
                await broadcast_state()

        # Update frame every 0.3 seconds for visual pacing
        await asyncio.sleep(0.3)  

@app.on_event("startup")
async def start_sim():
    global simulation_task
    simulation_task = asyncio.create_task(simulation_loop())

@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    global is_running, env

    await websocket.accept()
    clients.add(websocket)
    try:
        # Give UI the initial state immediately when they connect
        if env:
            await websocket.send_text(json.dumps({"type": "state", "data": current_state}))
        
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            cmd = msg.get("command")
            
            if cmd == "TOGGLE_PLAY":
                is_running = not is_running
                
            elif cmd == "RESET":
                is_running = False
                if env is None:
                    continue
                obs, info = env.reset()
                update_current_state(obs, info, 0.0, 0, "RESET")
                await broadcast_state()
                
            elif cmd in ["FORCE_FLY", "FORCE_CRUISE", "FORCE_LAND", "FORCE_DESCEND", "FORCE_CLIMB"]:
                if not is_running and env: # Manual step controls
                    active_env = env
                    # Map websocket commands to environment actions
                    cmd_to_action = {
                        "FORCE_FLY": 0,
                        "FORCE_CRUISE": 0,
                        "FORCE_LAND": 1,
                        "FORCE_DESCEND": 1,
                        "FORCE_CLIMB": 2
                    }
                    action = cmd_to_action[cmd]
                    obs, r, done, t, info = active_env.step(action)
                    
                    action_map = {0: "CRUISE", 1: "DESCEND", 2: "CLIMB"}
                    act_str = action_map[action]
                    
                    current_state["step"] += 1
                    update_current_state(obs, info, r, current_state["step"], act_str)
                    
                    if done:
                        is_running = False
                    await broadcast_state()
                    
    except WebSocketDisconnect:
        clients.remove(websocket)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
