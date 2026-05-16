# RL Evaluation Stabilization and Improvement Plan

## Context

The current PPO evaluation curves fluctuate heavily even when the rollout curves look smoother. This is expected because the evaluation protocol and the environment both introduce high variance:

- Evaluation currently uses too few episodes.
- The environment reset still samples random engines and random airport layouts.
- Terminal rewards and penalties are large compared with per-step rewards.
- Landing is a threshold-sensitive, hit-or-miss event.
- The current PPO policy still appears uncertain, especially when comparing deterministic and stochastic evaluation.

This plan prioritizes stabilizing evaluation first, then improving PPO training stability, then adjusting environment/reward design if needed.

## Current Observations

### Evaluation settings are too noisy

Current defaults in `scripts/training/train_ppo.py`:

- `eval_n_episodes = 10`
- `diag_n_episodes = 10`
- `eval_freq = 5000`

With only 10 evaluation episodes, one additional success or failure shifts event rates by 10 percentage points. Because `ARRIVED` gives a large positive terminal reward and `FIELD_CRASH` gives a large negative penalty, the mean reward can jump sharply between evaluations.

### Evaluation environment is still stochastic

Even with deterministic policy evaluation, the environment itself remains random:

- `reset()` randomly samples an engine unit.
- `reset()` randomly perturbs sub-airport positions.
- successful maintenance can randomly choose a replacement engine.

Therefore deterministic evaluation currently means deterministic action selection, not deterministic scenarios.

### Landing is threshold-sensitive

The environment uses a narrow landing success/failure boundary. A few action steps of difference can change the outcome from `MAINTAINED` or `ARRIVED` to `FIELD_CRASH`.

### Policy is not yet robust

Recent evaluation showed a large gap between deterministic and stochastic policy behavior. That suggests the policy distribution is still not confident enough and the model can easily flip between actions around important states.

## Priority 1: Stabilize Evaluation Protocol

### 1. Increase evaluation episode count

Use more episodes per evaluation so each point is less affected by scenario luck.

Recommended full run:

```bash
uv run python -m scripts.training.train_ppo \
  --force-train \
  --eval-n-episodes 50 \
  --diag-n-episodes 30 \
  --eval-freq 20000 \
  --diag-det-freq 20000 \
  --diag-stoch-freq 40000
```

Recommended lighter run:

```bash
uv run python -m scripts.training.train_ppo \
  --force-train \
  --eval-n-episodes 30 \
  --diag-n-episodes 20 \
  --eval-freq 25000
```

Rationale: fewer evaluation points, but each point is much more reliable.

### 2. Add fixed-scenario evaluation

Use two separate evaluation modes:

| Evaluation mode | Purpose |
|---|---|
| Fixed-seed/fixed-scenario eval | Select the best checkpoint fairly |
| Random eval | Measure generalization and robustness |

The current model selection should not depend only on random scenario luck.

### 3. Track event rates as first-class metrics

Mean reward alone is not enough. Log and compare:

- arrival rate,
- safe route rate,
- field crash rate,
- fuel empty rate,
- crashed rate,
- mean maintenance count,
- mean/median/std reward,
- mean/median/std episode length.

The existing diagnostics callback already logs event counts, but these should be treated as core evaluation metrics, not secondary debugging metrics.

## Priority 2: Stabilize PPO Training

### 4. Use scheduled entropy coefficient

Current fixed value:

```bash
--ent-coef 0.05
```

Better approach: decay entropy over training instead of using one fixed value.

Recommended schedule:

| Training progress | Suggested `ent_coef` |
|---|---:|
| Early training | `0.05` |
| Middle training | `0.02` |
| Late training | `0.005` - `0.01` |

Rationale: early training needs exploration, but late training needs precise timing. Keeping entropy high until the end makes action probabilities too spread out and hurts deterministic/stochastic consistency.

Implementation note: Stable-Baselines3 directly supports schedules for `learning_rate`, but `ent_coef` is usually a fixed value in PPO. To schedule entropy cleanly, add a small callback that updates `model.ent_coef` based on `num_timesteps / total_timesteps`, or run staged fine-tuning with decreasing `--ent-coef` values.

Practical staged option without code changes:

```bash
# Stage 1: exploration
--total-timesteps 500000 --ent-coef 0.05

# Stage 2: transition
--total-timesteps 500000 --ent-coef 0.02

# Stage 3: stabilization
--total-timesteps 500000 --ent-coef 0.005
```

### 5. Use scheduled learning rate

Current fixed value:

```bash
--learning-rate 3e-4
```

Better approach: decay learning rate over training.

Recommended schedule:

| Training progress | Suggested learning rate |
|---|---:|
| Early training | `3e-4` |
| Middle training | `1e-4` |
| Late training | `3e-5` - `5e-5` |

Rationale: early training benefits from faster updates, but late training should use smaller updates to reduce action-boundary flipping between `CRUISE`, `DESCEND`, and `CLIMB`.

Implementation note: Stable-Baselines3 PPO accepts callable schedules for `learning_rate`, so this can be implemented directly in `scripts/training/train_ppo.py` by parsing a schedule option and passing a function instead of a fixed float.

### 6. Increase batch size

Current value:

```bash
--batch-size 64
```

Recommended:

```bash
--batch-size 256
```

Alternative:

```bash
--batch-size 512
```

Rationale: larger batches reduce gradient noise, especially with a rollout size of `n_envs * n_steps = 4 * 2048`.

### Recommended immediate PPO run

Preferred direction: use schedules for both learning rate and entropy coefficient.

Target schedule:

```text
learning_rate: 3e-4 -> 1e-4 -> 3e-5
ent_coef:      0.05 -> 0.02 -> 0.005
```

If schedule support is implemented in `scripts/training/train_ppo.py`, use a run like:

```bash
uv run python -m scripts.training.train_ppo \
  --force-train \
  --total-timesteps 1500000 \
  --learning-rate-schedule linear:3e-4:3e-5 \
  --ent-coef-schedule linear:0.05:0.005 \
  --batch-size 256 \
  --eval-freq 20000 \
  --eval-n-episodes 50 \
  --diag-det-freq 20000 \
  --diag-stoch-freq 40000 \
  --diag-n-episodes 30
```

If schedule support is not implemented yet, approximate it with staged fine-tuning:

```bash
# Stage 1: exploration
uv run python -m scripts.training.train_ppo \
  --force-train \
  --total-timesteps 500000 \
  --learning-rate 3e-4 \
  --ent-coef 0.05 \
  --batch-size 256

# Stage 2: transition
uv run python -m scripts.training.train_ppo \
  --force-train \
  --total-timesteps 500000 \
  --learning-rate 1e-4 \
  --ent-coef 0.02 \
  --batch-size 256

# Stage 3: stabilization
uv run python -m scripts.training.train_ppo \
  --force-train \
  --total-timesteps 500000 \
  --learning-rate 3e-5 \
  --ent-coef 0.005 \
  --batch-size 256
```

## Priority 3: Improve Environment and Reward Design

### 7. Add smoother landing alignment reward

Current landing reward is highly threshold-based. Add small dense shaping so the agent receives useful feedback when it is close to a correct landing, not only when it succeeds exactly.

Possible additions:

- reward descending near airport when landing is feasible,
- reward low altitude near target,
- penalty for being low far from target,
- distance-proportional penalty for field crash instead of only a hard fixed penalty.

### 8. Add airport-noise curriculum

Current sub-airport positions are randomly perturbed by up to ±200. Use a curriculum:

1. phase 1: fixed airports, noise 0,
2. phase 2: moderate noise, e.g. ±50,
3. phase 3: full noise, e.g. ±200.

This makes the agent learn the landing mechanism first, then learn robustness.

### 9. Consider adding explicit `LAND` action

Currently `DESCEND` both reduces altitude and implicitly causes landing when altitude reaches zero. This makes timing very brittle.

A cleaner action design would be:

- `CRUISE`,
- `DESCEND`,
- `CLIMB`,
- `LAND`.

Then landing is an explicit decision and can be validated based on airport proximity.

## Priority 4: Reduce Train/Test Distribution Mismatch

### 10. Expand eligible training units gradually

Current default training subset is narrow:

```python
DEFAULT_TRAIN_ELIGIBLE_UNITS = [14, 62, 3]
```

Recommended curriculum:

1. train on the small subset until the agent learns basic landing,
2. fine-tune on 10-20 selected engines,
3. fine-tune on all eligible training engines,
4. evaluate on held-out test engines.

Do not jump immediately to all engines if the agent has not learned the basic route-management behavior.

## Recommended Order of Work

1. Increase evaluation episode count and reduce evaluation frequency.
2. Add fixed-scenario evaluation for checkpoint selection.
3. Replace fixed entropy coefficient with a decaying schedule, e.g. `0.05 -> 0.02 -> 0.005`.
4. Replace fixed learning rate with a decaying schedule, e.g. `3e-4 -> 1e-4 -> 3e-5`.
5. Increase batch size from `64` to `256`.
6. Re-run training and compare event rates, not only mean reward.
7. If fluctuation remains high, add smoother landing reward shaping.
8. Add airport-noise curriculum.
9. Expand eligible training units gradually.
10. Consider explicit `LAND` action only after the above simpler changes are tested.

## Success Criteria

A better run should show:

- smoother fixed-scenario evaluation curve,
- increasing arrival rate,
- decreasing field crash rate,
- smaller gap between deterministic and stochastic evaluation,
- less violent action-ratio flipping,
- stable or improving rollout reward without severe eval collapse.
