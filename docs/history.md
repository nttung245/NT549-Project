# Project History

## 2026-05-10 — PPO Curriculum + Maintenance Checkpoint Training

### Problem
- PPO training on many engines was slow and mixed easy/impossible starts.
- Very high-RUL engines let the policy fly directly to destination without learning maintenance.
- High altitude incentives made the agent climb too much and miss landing timing.
- Notebook training behavior needed a reproducible script with MLflow, VecNormalize, and deterministic best-model eval.

### Changes
- `scripts/core/aircraft_env.py`:
  - Added `eligible_units`, `min_initial_rul`, and `maintenance_resets_health` controls.
  - Maintenance is now an intermediate checkpoint: successful subairport landing refuels and resets/swaps engine health/RUL, then route continues.
  - Reduced `MAX_ALTITUDE` to `5000`, reduced `CLIMB_RATE` to `500`, removed altitude cruise bonus, and increased feasible approach `DESCEND` reward.
- `scripts/training/train_ppo.py`:
  - Rebuilt CLI training flow to match notebook PPO: `SubprocVecEnv`, `VecNormalize`, `EvalCallback`, diagnostics callbacks, and MLflow params/artifacts.
  - Uses deterministic eval for best-model selection; stochastic eval remains diagnostics-only.
  - Defaults to a 3-engine medium-RUL curriculum subset `[14, 62, 3]` with initial RUL near `150`.
  - Defaults PPO device to CPU and sets TensorFlow runtime env vars to reduce CUDA/oneDNN warning noise.
- `scripts/evaluation/rl_eval.py`:
  - Evaluates with saved `VecNormalize` stats, deterministic by default, and route-level success metrics.
- `scripts/training/rl_callbacks.py`:
  - Keeps normalization sync, VecNormalize best-stat saving, MLflow logging, and deterministic/stochastic diagnostics.

### Validation
- `.venv/bin/python -m py_compile scripts/training/train_ppo.py scripts/core/aircraft_env.py scripts/evaluation/rl_eval.py scripts/training/rl_callbacks.py` passed.
- Confirmed default training engine subset: `[14, 62, 3]`.

---

## 2026-05-09 — Stochastic Eval Alignment + Grounded Landing Fix

### Problem
- PPO demo/eval often collapsed into repeated `CRUISE` or repeated `DESCEND` because evaluation used deterministic argmax while PPO training uses a stochastic policy distribution.
- After `MAINTAINED`, aircraft was grounded at altitude `0`, but `DESCEND` still moved it horizontally and consumed fuel, creating an infinite ground-slide pattern.
- `RUL <= 0` could still terminate the episode while aircraft was already grounded at an airport after maintenance.

### Changes
- Switched PPO inference/eval paths to stochastic sampling (`deterministic=False`) so evaluation/demo/server behavior matches PPO rollout training policy:
  - `scripts/evaluation/rl_eval.py`
  - `scripts/demos/demo_ppo_stable.py`
  - `scripts/training/rl_callbacks.py`
  - `scripts/training/train_ppo.py`
  - `server.py`
- Updated `scripts/core/aircraft_env.py` grounded handling:
  - If grounded and action is `CRUISE` or `DESCEND`, aircraft does not move, does not burn fuel, stays `GROUNDED`, and receives a small invalid-ground-action penalty.
  - Only `CLIMB` can resume the route after maintenance.
  - `RUL <= 0` crash is ignored while `flight_phase == "GROUNDED"`.
- Fixed safe-landing histogram in `scripts/evaluation/rl_eval.py` to count `ARRIVED` and `MAINTAINED` instead of stale `LANDED`.

### Validation
- `python3 -m py_compile scripts/core/aircraft_env.py scripts/evaluation/rl_eval.py scripts/training/rl_callbacks.py scripts/demos/demo_ppo_stable.py scripts/training/train_ppo.py server.py` passed.

---

## 2026-05-09 — Landing Continuity + MLflow + SubprocVecEnv 4 Workers

### Problem
- Intermediate maintenance landing was treated as terminal (`done=True`), so SB3 VecEnv auto-reset the route and invalidated multi-leg journey learning.
- Deferred refuel logic conflicted with VecEnv auto-reset.
- `mlflow` dependency was missing from the project environment.
- Notebook `SubprocVecEnv` factory captured pandas/scaler objects, causing pickle errors with multiprocessing.

### Changes
- `scripts/core/aircraft_env.py`:
  - `MAINTAINED` is now non-terminal (`done=False`).
  - Maintenance landing sets `altitude=0`, `velocity=0`, `fuel=FUEL_CAPACITY`, `flight_phase="GROUNDED"`.
  - Reset `dist_since_last_maintenance=0` after maintenance.
  - `_reset_to_new_engine()` documented as not used for intermediate maintenance.
- Dependency/environment:
  - Added `mlflow` through project-managed environment (`uv`), updated lock/config files.
  - Added VSCode interpreter settings for project `.venv`.
- Notebook/vector env:
  - Kept `SubprocVecEnv`, `n_envs=4`, `VecNormalize`.
  - Made `make_env()` pickle-safe by loading data/scaler inside each worker instead of capturing notebook objects.

### Validation
- Forced maintenance landing verified: `event=MAINTAINED`, `done=False`, `GROUNDED`, fuel refilled, next `CLIMB` continues same route.
- Dependency imports verified in `.venv`.
- 4-worker smoke test passed with `SUBPROC_4ENV_SMOKE_OK`.

---

## 2026-05-08 — PPO Reward Stabilization + Eval Diagnostics

### Problem
- Rollout reward looked good, but eval reward collapsed: stochastic rollout sometimes found good behavior, deterministic eval often failed quickly.
- Dense progress reward dominated terminal success objectives.
- Approach reward encouraged generic `DESCEND` near airports, even when landing was physically infeasible.
- Hyperparameters in notebook drifted from earlier recommendations.

### Changes
- `scripts/core/aircraft_env.py` reward/physics tuning:
  - Reduced progress reward from `distance_covered / 1000` to `/ 1500`.
  - Added small altitude-efficiency reward only for `CRUISE` at higher altitude.
  - Added landing feasibility checks using remaining descent steps and descent distance.
  - Reward `DESCEND` in approach zone only when landing is feasible.
  - Penalize too-early infeasible `DESCEND` and `CLIMB` in approach zone lightly.
  - Increased ARRIVED reward to `+150`.
  - Increased crash/fuel/timeout penalties.
  - Added landing accuracy bonus for maintenance.
  - Added detailed `info` diagnostics: reward components, altitude, fuel, RUL, position, next target distance, nearest airport distance, approach zone, feasibility, descent requirements, phase.
- `scripts/training/rl_callbacks.py`:
  - Added `EvalDiagnosticsCallback` with action ratios, event counts, mean reward/length/final altitude, first descend distance, landing feasible ratio.
  - Syncs `VecNormalize.obs_rms` from train env before probes.
- `demo_flow.ipynb`:
  - Fixed learning rate to `3e-4`.
  - Increased entropy coefficient to `0.05`.
  - Increased eval episodes.
  - Added deterministic and stochastic diagnostic probes.

### Validation
- `python3 -m compileall -q scripts server.py main.py` passed.
- Notebook JSON validated.

---

## 2026-05-02 — Altitude-Dependent Physics + Landing Window Tuning

### Problem
- Agent had little incentive to climb to high altitude.
- Deterministic eval often crashed because the approach window and landing threshold were inconsistent with descent distance from high altitude.

### Changes
- `scripts/core/aircraft_env.py`:
  - `CRUISE` speed now scales with altitude: about `25` at ground to `40` at `12000m`.
  - Fuel rate decreases with altitude: about `0.5` at ground to `0.3` at `12000m`.
  - `LANDING_THRESHOLD` adjusted to `400`.
  - `APPROACH_DISTANCE` adjusted to `800`.
  - Removed penalty for skipping airports inside approach zone.

### Impact
- High-altitude cruising became beneficial but not a guaranteed objective by itself.
- Landing timing became more physically consistent with descent profile.

---

## 2026-05-02 — Dense Approach Reward + Simplified Startup

### Problem
- After long training, PPO still rarely landed successfully in deterministic eval.
- Landing reward was too sparse, startup from ground made the task harder, and entropy was too low.

### Changes
- `scripts/core/aircraft_env.py`:
  - Added dense approach reward to guide descent near airports.
  - Reduced `DESCEND_RATE` from `1000` to `500` for smoother descent.
  - Increased `LANDING_THRESHOLD` from `300` to `500` at that stage.
  - Increased `MAX_STEPS` to `2000`.
  - Start episodes at altitude `2000` instead of ground.
  - Removed old takeoff/idle branch.
  - Added timeout penalty.
  - Redesigned observation to 11 dimensions: altitude, fuel, RUL, six signed airport distances, destination distance, approach-zone flag.
- `scripts/training/train_ppo.py`:
  - Increased `ent_coef` to `0.05`.
  - Increased total training timesteps to `1,000,000`.

### Note
- Observation space changed, so older PPO models became incompatible and required retraining.

---

## 2026-05-01 — Anti-Farm Reward + Exploration Fixes

### Problem
- PPO plateaued around the fuel limit.
- Maintenance reward could be negative for healthy landings, discouraging refuel/maintenance.
- Naively making maintenance reward positive introduced a CLIMB/DESCEND farming exploit near the same airport.

### Changes
- `scripts/core/aircraft_env.py`:
  - Set `FUEL_CAPACITY=250` so at least one maintenance/refuel is needed for the full route.
  - Reduced airport noise to `±200` for lower variance.
  - Added `MAX_STEPS` and `current_step`.
  - Standardized major failure penalties.
  - Added progress reward proportional to distance covered.
  - Reworked maintenance reward to depend on distance since last maintenance, low-RUL urgency, fuel usage, and later accuracy.
  - Added/reset `dist_since_last_maintenance` to prevent short-loop reward farming.
- Repo hygiene:
  - Added training output folders such as `logs/` and `mlruns/` to `.gitignore`.

### Hyperparameter Guidance
- Prefer fixed learning rate `3e-4`.
- Keep entropy high enough (`ent_coef` around `0.05`) during landing-behavior discovery.

---

## Earlier Baseline — Single-Agent AircraftEnv Setup

### Key State
- Single-agent PPO environment for aircraft predictive maintenance.
- Route length increased to `20000`.
- Six sub-airports distributed along the route with controlled random noise.
- Initially fixed training to engine unit `1` for proof-of-concept feasibility.
- Core actions: `CRUISE`, `DESCEND`, `CLIMB`.
- Reward and physics were iteratively shaped toward: fly efficiently, land for maintenance/refuel, avoid crashes/fuel-empty, and eventually reach destination.
