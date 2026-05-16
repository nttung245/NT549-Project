# Weather + Wind System — Implementation Plan

## Overview

Add a dynamic weather system to the aircraft maintenance RL environment. Each episode generates random weather zones along the 20,000-unit route that affect fuel consumption, speed, and RUL decay. Weather zones drift slowly during the episode, forcing the agent to react and adapt rather than memorize a fixed strategy.

## Goals

- **No memorization**: Every episode has different weather patterns
- **Strategic depth**: Altitude choice becomes meaningful beyond just fuel efficiency
- **Backward compatible**: Existing training pipeline, evaluation scripts, and visualizations continue to work
- **Emergent behavior**: Keep 3 actions (CRUISE, DESCEND, CLIMB) — weather avoidance emerges naturally from reward shaping

## Design

### Weather Zones

Each zone spans a distance range and altitude band along the route:

| Zone type  | Fuel multiplier | Speed modifier | RUL decay multiplier | Spawn weight   |
| ---------- | --------------- | -------------- | -------------------- | -------------- |
| Clear      | 1.0x            | 1.0x           | 1.0x                 | (default fill) |
| Headwind   | 1.5x            | -20%           | 1.0x                 | 35%            |
| Tailwind   | 0.7x            | +15%           | 1.0x                 | 35%            |
| Storm      | 2.0x            | -10%           | 1.5x                 | 20%            |
| Turbulence | 1.0x            | -5%            | 1.3x                 | 10%            |

- **2-5 zones per episode** (random count, weighted spawn)
- **Random positions** along the 20,000-unit route
- **Random altitude bands** (e.g., storm might be 2000-4000m, headwind at 3000-5000m)
- **Zones do NOT overlap** — zones are placed sequentially with gaps

### Weather Drift

- Each zone drifts **30-80 units/step** along the route direction
- Drift direction is random per zone (can move toward or away from aircraft)
- Zones that drift completely off-route are removed
- Drift rate is deterministic per zone (fixed at spawn from seed)

### Observation Space (11 → 16 dims)

```
obs[0]  = altitude (0-5000)
obs[1]  = fuel (0-250)
obs[2]  = RUL (0-300)
obs[3:9] = signed distances to 6 sub-airports
obs[9]  = distance to destination
obs[10] = in_approach_zone

-- NEW: Weather --
obs[11] = wind_strength (-1.0 headwind → +1.0 tailwind, normalized)
obs[12] = fuel_multiplier (1.0-2.0)
obs[13] = rul_multiplier (1.0-1.5)
obs[14] = next_zone_distance (distance to nearest weather zone ahead, normalized 0-1)
obs[15] = next_zone_type (0=clear, 1=headwind, 2=tailwind, 3=storm, 4=turbulence)
```

### Actions (unchanged — 3 actions)

Keep CRUISE / DESCEND / CLIMB. Weather avoidance emerges:

- DESCEND to escape high-altitude storms/headwinds
- CLIMB to escape low-altitude turbulence
- CRUISE to ride tailwind zones efficiently

### Reward Adjustments

| Component             | Value                  | Condition                                                |
| --------------------- | ---------------------- | -------------------------------------------------------- |
| Tailwind step bonus   | +0.02                  | Each step in tailwind zone (encourage utilization)       |
| Storm step penalty    | -0.05                  | Each step in storm zone (discourage prolonged exposure)  |
| Weather-aware landing | +3.0                   | Land at sub-airport when storm is within 500 units ahead |
| Rough landing         | -50% maintenance bonus | Land while in turbulence zone                            |

### Curriculum Landing — Precision Progression

**Problem**: Agent needs to learn landing gradually, but shrinking thresholds risks destabilizing training if done on a fixed schedule.

**Solution**: Reduce `APPROACH_DISTANCE` (not `LANDING_THRESHOLD`) based on agent performance.

#### Schedule

| Stage      | APPROACH_DISTANCE | Trigger to advance                                 |
| ---------- | ----------------- | -------------------------------------------------- |
| 1 (Easy)   | 600 units         | Land success rate ≥ 80% over last 50 eval episodes |
| 2 (Medium) | 400 units         | Land success rate ≥ 80% over last 50 eval episodes |
| 3 (Hard)   | 250 units         | Land success rate ≥ 80% over last 50 eval episodes |
| 4 (Final)  | 150 units         | —                                                  |

- `LANDING_THRESHOLD` stays constant at all stages (only approach zone shrinks)
- Agent can always land — just needs to be more precise to earn approach bonus

#### Anti-Reward-Hacking: Approach Reward Restructure

**Problem**: Per-step approach bonus incentivizes agent to loop inside approach zone without committing to landing, or to enter approach zone for bonus then crash outside threshold.

**Fix**: Move reward from per-step → one-time landing commitment.

| Component                            | Old (per-step) | New (commitment-based)             |
| ------------------------------------ | -------------- | ---------------------------------- |
| DESCEND in approach zone             | +0.75/step     | +0.1/step (small guide only)       |
| Successful land WITHIN approach zone | (none)         | **+5.0 one-time**                  |
| Land outside approach zone           | 0              | 0 (neutral)                        |
| Left approach zone without landing   | 0              | **-1.0** (wasted approach penalty) |

**Why this works**: The largest reward comes only from **completing** the landing, not from lingering in the zone. Agent has no incentive to farm approach steps.

#### Implementation

**File: `scripts/core/aircraft_env.py`**

- Add `self.curriculum_stage` tracking (starts at 0)
- Add `self.approach_distances = [600, 400, 250, 150]`
- In `reset()`: set `APPROACH_DISTANCE = self.approach_distances[self.curriculum_stage]`
- In `FixedSeedEvalCallback`: after each eval window, check land success rate over last 50 episodes, advance stage if ≥ 80%
- In `step()`: restructure approach reward per table above
- Track `self._entered_approach` and `self._landed_in_approach` flags per episode to detect wasted approaches

### Fuel & RUL Calculation Changes

In `step()`, before applying fuel/RUL changes:

```python
weather = self._get_current_weather()
fuel_spent *= weather.fuel_multiplier
rul_decay *= weather.rul_multiplier
velocity *= weather.speed_modifier
```

### Visual Changes (sim_2d.py)

- Draw weather bands as colored vertical strips on the route map
- Color code: gray=headwind, cyan=tailwind, red=storm, yellow=turbulence
- Weather bands drift visually in real-time
- Aircraft shake effect in turbulence zone

### Server Changes (server.py)

- Include weather zone list in initial state broadcast
- Include current weather effects in per-step state

## Implementation Steps

### Phase 1: Core Weather Engine

**File: `scripts/core/weather.py`** (new)

- `WeatherZone` dataclass: type, start, end, alt_min, alt_max, drift_rate
- `WeatherMap` class:
  - `generate(route_length, rng)` — create 2-5 random zones
  - `get_weather_at(distance, altitude)` — return (fuel_mult, rul_mult, speed_mod, wind_strength)
  - `update(step)` — drift all zones, remove off-route zones
  - `get_next_zone(distance)` — find nearest zone ahead

### Phase 2: Integrate into AircraftEnv

**File: `scripts/core/aircraft_env.py`**

1. Import `WeatherMap`, create in `reset()` alongside sub-airport placement
2. In `step()`, query weather before fuel/RUL/velocity calculations
3. Apply weather multipliers to fuel burn, RUL decay, velocity
4. Apply weather-aware rewards (tailwind bonus, storm penalty, weather-aware landing)
5. Extend observation space from 11 → 16 dims
6. Update `observation_space` bounds to reflect new ranges

### Phase 3: Reward Tuning

**File: `scripts/core/aircraft_env.py`** (reward section)

- Add `tailwind_step_bonus` (+0.02/step in tailwind)
- Add `storm_step_penalty` (-0.05/step in storm)
- Add `weather_aware_landing_bonus` (+3.0 when landing before storm)
- Add `rough_landing_penalty` (halve maintenance bonus in turbulence)
- Verify existing reward components still sum correctly

### Phase 4: Visual & Server Updates

**File: `scripts/demos/sim_2d.py`**

- Draw weather bands as semi-transparent colored rectangles on route
- Add aircraft shake effect in turbulence

**File: `server.py`**

- Add weather zone list to WebSocket initial state
- Add current weather effects to per-step broadcast

### Phase 5: Testing

**File: `scripts/tests/test_env.py`** (update)

- Verify weather zones generate correctly (count, position, no overlap)
- Verify weather drift moves zones as expected
- Verify observation space returns correct weather values
- Verify fuel/RUL multipliers apply correctly
- Smoke test: run 10 episodes, ensure no crashes, all zone types appear
- Verify backward compat: `make_env` factory still works with VecNormalize

### Phase 6: Training Validation

- Run short training (~50k steps) to verify convergence
- Compare metrics vs baseline (no weather): expect slightly lower rewards initially (harder environment), but stable learning curve
- Verify PPO training pipeline works unchanged with new obs dim

### Phase 7: Future Enhancements Deferred from Initial Static Version

- Weather drift: intentionally deferred. Initial version keeps zones static during each episode while randomizing zones between episodes.
- Landing curriculum: intentionally deferred. Initial version keeps the existing landing thresholds and reward structure.
- Dynamic anti-memorization variants: optional future work after the static-weather PPO baseline is stable.

## Files Changed

| File                      | Action     | Scope                                    |
| ------------------------- | ---------- | ---------------------------------------- |
| `scripts/core/weather.py`      | **Create** | Static weather zone generation and query |
| `scripts/core/aircraft_env.py` | Modify     | Integrate weather into step/obs/reward   |
| `scripts/demos/sim_2d.py`       | Modify     | Draw weather bands, turbulence shake     |
| `server.py`               | Modify     | Broadcast weather in WebSocket           |
| `scripts/tests/test_env.py`     | Modify     | Add weather-specific tests               |

## Risks & Mitigations

| Risk                                         | Mitigation                                                                                                   |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Weather makes environment too hard           | Start with lighter multipliers, tune after training                                                          |
| Observation space change breaks saved models | Existing saved PPO models are incompatible with the new 16-dim obs space; retrain PPO and VecNormalize stats |
| Weather drift too chaotic                    | Drift is deferred to Phase 7; static zones are randomized per episode first                                  |
| VecNormalize needs reset                     | New obs dim requires re-training VecNormalize stats — expected                                               |

## Acceptance Criteria

1. `python -m scripts.tests.test_env` passes with weather zones active
2. `python main.py` runs without errors (weather zones visible in demo)
3. PPO training converges on short run (50k steps, reward curve trending up)
4. Weather bands visible in `sim_2d.py` visualization
5. Each episode has visibly different weather patterns
6. No changes to action space, training script structure, or VecNormalize logic required
