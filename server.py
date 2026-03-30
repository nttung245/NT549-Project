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

from scripts.data_processor import prepare_data, FEATURES, KEY_SENSORS
from scripts.aircraft_env import AircraftEnv

app = FastAPI(title="Aircraft Digital Twin - WebSocket Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
env = None
simulation_task = None
is_running = False
current_state = {}
clients = set()
lstm_model = None

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
        model=lstm_model,
        scaler=scaler,
        sensor_list=KEY_SENSORS,
        features_list=FEATURES
    )
    
    obs, info = env.reset()
    update_current_state(obs, info, 0.0, 0, "INIT")


def update_current_state(obs, info, reward, step, action_str):
    global current_state
    # obs = [Altitude, Velocity, Fuel, Current_RUL, Dist_to_Next_Airport, Dist_to_Destination]
    current_state = {
        "step": step,
        "altitude": float(obs[0]),
        "velocity": float(obs[1]),
        "fuel": float(obs[2]),
        "rul": float(obs[3]),
        "dist_next": float(obs[4]),
        "dist_dest": float(obs[5]),
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
            # Simple policy for auto-run demonstration
            dist_closest = env._dist_to_nearest_airport()
            current_rul = env.twin.current_rul if env.twin.current_rul else 150
            action = 0  # Default fly
            
            if (current_rul < 30 or env.twin.fuel < 20) and env.flight_phase == "CRUISING":
                if dist_closest < env.LANDING_THRESHOLD:
                    action = 1 # Land

            obs, r, done, t, info = env.step(action)
            step_count += 1
            act_str = "LAND" if action == 1 else "FLY"
            
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
            
            global is_running, env
            if cmd == "TOGGLE_PLAY":
                is_running = not is_running
                
            elif cmd == "RESET":
                is_running = False
                obs, info = env.reset()
                update_current_state(obs, info, 0.0, 0, "RESET")
                await broadcast_state()
                
            elif cmd in ["FORCE_FLY", "FORCE_LAND"]:
                if not is_running and env: # Manual step controls
                    action = 0 if cmd == "FORCE_FLY" else 1
                    obs, r, done, t, info = env.step(action)
                    act_str = "LAND" if action == 1 else "FLY"
                    current_state["step"] += 1
                    update_current_state(obs, info, r, current_state["step"], act_str)
                    
                    if done:
                        is_running = False
                    await broadcast_state()
                    
    except WebSocketDisconnect:
        clients.remove(websocket)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
