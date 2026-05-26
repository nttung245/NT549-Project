# Aircraft Digital Twin Frontend

This is a Next.js application for the aircraft digital twin dashboard and the static DQN/PPO MLflow training comparison page.

## Development

Run the development server from this directory:

```bash
bun dev
```

Open `http://localhost:3000` for the live digital-twin dashboard, or `http://localhost:3000/training-comparison` for the DQN/PPO training comparison page.

## Static DQN vs PPO MLflow comparison

The comparison page at `/training-comparison` is a fully static Next.js route. It does not call MLflow from the browser and does not require a Next.js API route. Instead, a Python exporter reads MLflow once and saves reusable offline files under `frontend/public/training-metrics/`. After those files are exported, you can stop MLflow and keep using or building the web page from the saved DQN/PPO metrics.

### Refresh MLflow data

From the repository root:

```bash
source .venv/bin/activate
python scripts/export_mlflow_training_history.py \
  --mlflow-tracking-uri http://localhost:5000 \
  --experiment-name Aircraft_Predictive_Maintenance_v5 \
  --target-date 2026-05-26 \
  --algorithms DQN,PPO
```

The exporter writes both a backwards-compatible single JSON file and the offline snapshot folder used by the page:

```text
frontend/public/mlflow/dqn_ppo_training_history_20260526.json
frontend/public/training-metrics/dqn-ppo-20260526/comparison.json
frontend/public/training-metrics/dqn-ppo-20260526/manifest.json
frontend/public/training-metrics/dqn-ppo-20260526/dqn.json
frontend/public/training-metrics/dqn-ppo-20260526/ppo.json
frontend/public/training-metrics/dqn-ppo-20260526/metrics.csv
```

The frontend loads `frontend/public/training-metrics/dqn-ppo-20260526/comparison.json`, so MLflow is not required at runtime.

Useful exporter options:

```bash
# Export only the latest DQN and latest PPO run for the date.
python scripts/export_mlflow_training_history.py --run-selection latest-per-algorithm

# Include all run statuses instead of only FINISHED/RUNNING.
python scripts/export_mlflow_training_history.py --statuses ALL

# Also export a flat CSV with one row per metric-history point outside the snapshot folder.
python scripts/export_mlflow_training_history.py --csv-output mlflow_training_history_20260526.csv

# Write the reusable offline folder somewhere else.
python scripts/export_mlflow_training_history.py --snapshot-dir frontend/public/training-metrics/my-snapshot

# Skip writing the offline folder if you only need the legacy single JSON output.
python scripts/export_mlflow_training_history.py --skip-snapshot
```

The exporter discovers every saved metric key dynamically from the selected MLflow runs, then preserves every history point. It does not trim unequal training lengths, so a PPO run trained for 500k timesteps and a DQN run trained for 1.5M timesteps can still be compared on absolute-step charts and normalized-progress charts.

### Build the static site

From `frontend/`:

```bash
bun run build
```

The project uses `output: "export"`, so the production static files are generated in `frontend/out/`. Host that directory with any static file server.

## Routes

- `/` — live aircraft digital twin dashboard.
- `/training-comparison` — static DQN vs PPO MLflow metric-history comparison.
