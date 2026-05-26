#!/usr/bin/env python3
"""Export DQN/PPO MLflow metric histories for a static Next.js comparison page.

The script discovers every metric key stored on selected MLflow runs and writes a
self-contained JSON artifact that the frontend can serve from its public folder.
It does not resample or truncate histories, so algorithms trained for different
numbers of timesteps remain directly comparable on absolute-step and normalized
progress charts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any
from zoneinfo import ZoneInfo

try:
    import mlflow
    from mlflow.entities import Run
    from mlflow.tracking import MlflowClient
except ImportError as exc:  # pragma: no cover - user environment guard
    raise SystemExit(
        "MLflow is not installed in the active Python environment. "
        "Activate the project virtual environment first, for example: "
        "source .venv/bin/activate && python scripts/export_mlflow_training_history.py"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = "Aircraft_Predictive_Maintenance_v5"
DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_TARGET_DATE = "2026-05-26"
DEFAULT_ALGORITHMS = ("DQN", "PPO")
DEFAULT_STATUSES = ("FINISHED", "RUNNING")
DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "frontend" / "public" / "training-metrics"

GROUP_ORDER = {
    "rollout": 0,
    "train": 1,
    "eval": 2,
    "eval_diag_det": 3,
    "eval_diag_stoch": 4,
    "time": 5,
    "diagnostics": 6,
    "other": 7,
}

GROUP_LABELS = {
    "rollout": "Rollout training signals",
    "train": "Optimizer and policy training",
    "eval": "Evaluation checkpoint metrics",
    "eval_diag_det": "Deterministic evaluation diagnostics",
    "eval_diag_stoch": "Stochastic evaluation diagnostics",
    "time": "Runtime and throughput",
    "diagnostics": "Environment diagnostics",
    "other": "Other MLflow metrics",
}

MINIMIZE_KEYWORDS = (
    "loss",
    "error",
    "distance",
    "crash",
    "collision",
    "violation",
    "kl",
    "entropy_loss",
)

MAXIMIZE_KEYWORDS = (
    "reward",
    "rew",
    "success",
    "ratio",
    "return",
    "ep_rew",
    "mean_reward",
    "feasible",
    "fps",
)


JsonDict = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export complete MLflow metric histories for DQN/PPO runs into a "
            "static JSON payload consumed by the Next.js comparison dashboard."
        )
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI or http://localhost:5000.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name. Default: {DEFAULT_EXPERIMENT}.",
    )
    parser.add_argument(
        "--target-date",
        default=DEFAULT_TARGET_DATE,
        help="Run start date to include. Accepts YYYY-MM-DD, DD/MM/YYYY, or DD/MM. Default: 2026-05-26.",
    )
    parser.add_argument(
        "--timezone",
        default="UTC",
        help="Timezone used to interpret --target-date boundaries. Default: UTC.",
    )
    parser.add_argument(
        "--algorithms",
        default=",".join(DEFAULT_ALGORITHMS),
        help="Comma-separated algorithm labels to include. Default: DQN,PPO.",
    )
    parser.add_argument(
        "--statuses",
        default=",".join(DEFAULT_STATUSES),
        help="Comma-separated MLflow run statuses to include. Use 'ALL' to include every status.",
    )
    parser.add_argument(
        "--run-selection",
        choices=("all", "latest-per-algorithm"),
        default="all",
        help="Whether to export every matching run or only the latest run for each algorithm. Default: all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to frontend/public/mlflow/dqn_ppo_training_history_<date>.json.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help=(
            "Offline static snapshot directory. Defaults to "
            "frontend/public/training-metrics/dqn-ppo-<date>. The directory receives "
            "comparison.json, manifest.json, DQN.json, PPO.json, and metrics.csv so the "
            "frontend can be used without starting MLflow after export."
        ),
    )
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="Only write --output and optional --csv-output; do not write the offline snapshot folder.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV output path with one row per metric point.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=500,
        help="Maximum number of MLflow runs to inspect before filtering. Default: 500.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write minified JSON instead of pretty-printed JSON.",
    )
    return parser.parse_args()


def parse_csv_arg(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_target_date(raw: str, now_year: int) -> date:
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass

    for fmt in ("%d/%m", "%d-%m"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return date(now_year, parsed.month, parsed.day)
        except ValueError:
            pass

    raise ValueError(
        f"Unsupported --target-date value {raw!r}. Use YYYY-MM-DD, DD/MM/YYYY, DD/MM, or DD-MM."
    )


def date_bounds_ms(target: date, tz_name: str) -> tuple[int, int, str]:
    tz = ZoneInfo(tz_name)
    start = datetime.combine(target, time.min, tzinfo=tz)
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000), tz.key


def utc_iso_from_ms(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_algorithm(run: Run, algorithms: list[str]) -> str | None:
    tags = run.data.tags
    params = run.data.params
    run_name = tags.get("mlflow.runName") or run.info.run_name or run.info.run_id
    searchable_fields = [
        tags.get("algorithm"),
        tags.get("Algorithm"),
        params.get("algorithm"),
        params.get("Algorithm"),
        run_name,
    ]
    searchable_text = " ".join(str(field) for field in searchable_fields if field).lower()

    for algorithm in algorithms:
        if re.search(rf"(^|[^a-z0-9]){re.escape(algorithm.lower())}([^a-z0-9]|$)", searchable_text):
            return algorithm
    return None


def metric_group(metric_key: str) -> str:
    if metric_key.startswith("eval_diag_det/"):
        return "eval_diag_det"
    if metric_key.startswith("eval_diag_stoch/"):
        return "eval_diag_stoch"
    if metric_key.startswith("eval/") or metric_key.startswith("eval_"):
        return "eval"
    if metric_key.startswith("rollout/"):
        return "rollout"
    if metric_key.startswith("train/"):
        return "train"
    if metric_key.startswith("time/"):
        return "time"
    if "diag" in metric_key or "event_" in metric_key or "action_" in metric_key:
        return "diagnostics"
    return "other"


def display_metric_name(metric_key: str) -> str:
    without_prefix = metric_key.split("/", maxsplit=1)[-1]
    return without_prefix.replace("_", " ").replace("-", " ").title()


def metric_direction(metric_key: str) -> str:
    lowered = metric_key.lower()
    if any(keyword in lowered for keyword in MINIMIZE_KEYWORDS):
        return "minimize"
    if any(keyword in lowered for keyword in MAXIMIZE_KEYWORDS):
        return "maximize"
    return "contextual"


def summarize_values(points: list[JsonDict]) -> JsonDict:
    valid_values = [point["value"] for point in points if isinstance(point.get("value"), (int, float))]
    valid_steps = [point["step"] for point in points if isinstance(point.get("step"), int)]

    if not valid_values:
        return {
            "count": len(points),
            "valid_count": 0,
            "first": None,
            "last": None,
            "min": None,
            "max": None,
            "mean": None,
            "min_step": min(valid_steps) if valid_steps else None,
            "max_step": max(valid_steps) if valid_steps else None,
        }

    first_point = next(point for point in points if isinstance(point.get("value"), (int, float)))
    last_point = next(point for point in reversed(points) if isinstance(point.get("value"), (int, float)))
    return {
        "count": len(points),
        "valid_count": len(valid_values),
        "first": first_point["value"],
        "last": last_point["value"],
        "min": min(valid_values),
        "max": max(valid_values),
        "mean": fmean(valid_values),
        "min_step": min(valid_steps) if valid_steps else None,
        "max_step": max(valid_steps) if valid_steps else None,
    }


def metric_history_to_points(history: Any) -> list[JsonDict]:
    points: list[JsonDict] = []
    for metric in history:
        timestamp = safe_int(getattr(metric, "timestamp", None))
        points.append(
            {
                "step": safe_int(getattr(metric, "step", None)),
                "value": safe_float(getattr(metric, "value", None)),
                "timestamp": timestamp,
                "time": utc_iso_from_ms(timestamp),
            }
        )

    points.sort(
        key=lambda point: (
            point["step"] if point["step"] is not None else -1,
            point["timestamp"] if point["timestamp"] is not None else -1,
        )
    )
    return points


def run_to_payload(client: MlflowClient, run: Run, algorithm: str) -> JsonDict:
    run_name = run.data.tags.get("mlflow.runName") or run.info.run_name or run.info.run_id
    metric_keys = sorted(run.data.metrics.keys())
    metrics: dict[str, JsonDict] = {}

    for key in metric_keys:
        history = client.get_metric_history(run.info.run_id, key)
        points = metric_history_to_points(history)
        metrics[key] = {
            "key": key,
            "display_name": display_metric_name(key),
            "group": metric_group(key),
            "group_label": GROUP_LABELS.get(metric_group(key), GROUP_LABELS["other"]),
            "direction": metric_direction(key),
            "chart_type": "line" if len(points) > 1 else "bar",
            "summary": summarize_values(points),
            "history": points,
        }

    return {
        "run_id": run.info.run_id,
        "run_name": run_name,
        "algorithm": algorithm,
        "status": run.info.status,
        "artifact_uri": run.info.artifact_uri,
        "start_time": run.info.start_time,
        "start_time_iso": utc_iso_from_ms(run.info.start_time),
        "end_time": run.info.end_time,
        "end_time_iso": utc_iso_from_ms(run.info.end_time),
        "duration_ms": (
            run.info.end_time - run.info.start_time
            if run.info.start_time is not None and run.info.end_time is not None
            else None
        ),
        "params": dict(sorted(run.data.params.items())),
        "tags": dict(sorted(run.data.tags.items())),
        "metric_keys": metric_keys,
        "metric_count": len(metric_keys),
        "total_metric_points": sum(metric["summary"]["count"] for metric in metrics.values()),
        "max_step": max(
            (
                metric["summary"]["max_step"]
                for metric in metrics.values()
                if metric["summary"]["max_step"] is not None
            ),
            default=None,
        ),
        "metrics": metrics,
    }


def build_metric_catalog(runs: list[JsonDict]) -> list[JsonDict]:
    metric_index: dict[str, JsonDict] = {}

    for run in runs:
        for key, metric in run["metrics"].items():
            entry = metric_index.setdefault(
                key,
                {
                    "key": key,
                    "display_name": metric["display_name"],
                    "group": metric["group"],
                    "group_label": metric["group_label"],
                    "direction": metric["direction"],
                    "chart_type": "line",
                    "algorithms": set(),
                    "run_ids": [],
                    "run_names": [],
                    "points_by_algorithm": defaultdict(int),
                    "points_by_run": {},
                    "min_step": None,
                    "max_step": None,
                    "min_value": None,
                    "max_value": None,
                    "total_points": 0,
                },
            )
            entry["algorithms"].add(run["algorithm"])
            entry["run_ids"].append(run["run_id"])
            entry["run_names"].append(run["run_name"])

            summary = metric["summary"]
            point_count = int(summary["count"])
            entry["points_by_algorithm"][run["algorithm"]] += point_count
            entry["points_by_run"][run["run_id"]] = point_count
            entry["total_points"] += point_count

            for field in ("min_step", "max_step"):
                value = summary[field]
                if value is None:
                    continue
                if entry[field] is None:
                    entry[field] = value
                elif field == "min_step":
                    entry[field] = min(entry[field], value)
                else:
                    entry[field] = max(entry[field], value)

            for field in ("min_value", "max_value"):
                summary_field = "min" if field == "min_value" else "max"
                value = summary[summary_field]
                if value is None:
                    continue
                if entry[field] is None:
                    entry[field] = value
                elif field == "min_value":
                    entry[field] = min(entry[field], value)
                else:
                    entry[field] = max(entry[field], value)

            if point_count <= 1 and entry["chart_type"] != "line":
                entry["chart_type"] = "bar"

    catalog: list[JsonDict] = []
    for entry in metric_index.values():
        catalog.append(
            {
                **entry,
                "algorithms": sorted(entry["algorithms"]),
                "run_ids": sorted(set(entry["run_ids"])),
                "run_names": sorted(set(entry["run_names"])),
                "points_by_algorithm": dict(sorted(entry["points_by_algorithm"].items())),
                "points_by_run": dict(sorted(entry["points_by_run"].items())),
            }
        )

    catalog.sort(key=lambda item: (GROUP_ORDER.get(item["group"], 999), item["key"]))
    return catalog


def select_runs(runs: list[tuple[Run, str]], selection: str) -> list[tuple[Run, str]]:
    if selection == "all":
        return runs

    latest_by_algorithm: dict[str, tuple[Run, str]] = {}
    for run, algorithm in runs:
        if algorithm not in latest_by_algorithm:
            latest_by_algorithm[algorithm] = (run, algorithm)
    return list(latest_by_algorithm.values())


def flatten_rows(payload: JsonDict) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for run in payload["runs"]:
        for metric_key, metric in run["metrics"].items():
            for point in metric["history"]:
                rows.append(
                    {
                        "run_id": run["run_id"],
                        "run_name": run["run_name"],
                        "algorithm": run["algorithm"],
                        "metric": metric_key,
                        "metric_group": metric["group"],
                        "step": point["step"],
                        "value": point["value"],
                        "timestamp": point["timestamp"],
                        "time": point["time"],
                    }
                )
    return rows


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "snapshot"


def write_json(payload: JsonDict, output_path: Path, *, compact: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=None if compact else 2,
            allow_nan=False,
        )
        handle.write("\n")


def write_csv(rows: list[JsonDict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_name",
        "algorithm",
        "metric",
        "metric_group",
        "step",
        "value",
        "timestamp",
        "time",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_offline_snapshot(payload: JsonDict, rows: list[JsonDict], snapshot_dir: Path, *, compact: bool) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = snapshot_dir / "comparison.json"
    manifest_path = snapshot_dir / "manifest.json"
    csv_path = snapshot_dir / "metrics.csv"

    write_json(payload, comparison_path, compact=compact)
    write_csv(rows, csv_path)

    algorithm_files: dict[str, str] = {}
    for algorithm in payload["algorithms"]:
        algorithm_payload = {
            "schema_version": payload["schema_version"],
            "generated_at": payload["generated_at"],
            "source_comparison": "comparison.json",
            "experiment": payload["experiment"],
            "target_date": payload["target_date"],
            "timezone": payload["timezone"],
            "algorithm": algorithm,
            "summary": {
                "run_count": payload["summary"].get("runs_by_algorithm", {}).get(algorithm, 0),
                "metric_count": len(
                    [
                        metric
                        for metric in payload["metric_catalog"]
                        if algorithm in metric.get("algorithms", [])
                    ]
                ),
                "total_metric_points": sum(
                    run["total_metric_points"] for run in payload["runs"] if run["algorithm"] == algorithm
                ),
                "max_step": payload["summary"].get("max_step_by_algorithm", {}).get(algorithm),
            },
            "metric_catalog": [
                metric for metric in payload["metric_catalog"] if algorithm in metric.get("algorithms", [])
            ],
            "runs": [run for run in payload["runs"] if run["algorithm"] == algorithm],
        }
        filename = f"{slugify(algorithm)}.json"
        algorithm_files[algorithm] = filename
        write_json(algorithm_payload, snapshot_dir / filename, compact=compact)

    manifest = {
        "schema_version": payload["schema_version"],
        "generated_at": payload["generated_at"],
        "description": "Offline DQN/PPO MLflow metrics snapshot. These files can be served statically without starting MLflow.",
        "comparison": "comparison.json",
        "csv": "metrics.csv",
        "algorithms": payload["algorithms"],
        "algorithm_files": algorithm_files,
        "experiment": payload["experiment"],
        "target_date": payload["target_date"],
        "timezone": payload["timezone"],
        "summary": payload["summary"],
        "metric_groups": payload["metric_groups"],
    }
    write_json(manifest, manifest_path, compact=compact)


def build_empty_payload(
    *,
    tracking_uri: str,
    experiment_name: str,
    experiment_id: str | None,
    target: date,
    tz_name: str,
    algorithms: list[str],
    run_selection: str,
    reason: str,
) -> JsonDict:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "tracking_uri": tracking_uri,
        "experiment": {
            "name": experiment_name,
            "experiment_id": experiment_id,
        },
        "target_date": target.isoformat(),
        "timezone": tz_name,
        "algorithms": algorithms,
        "run_selection": run_selection,
        "status": "empty",
        "message": reason,
        "summary": {
            "run_count": 0,
            "metric_count": 0,
            "total_metric_points": 0,
            "runs_by_algorithm": {algorithm: 0 for algorithm in algorithms},
        },
        "metric_groups": [
            {"key": key, "label": GROUP_LABELS[key], "order": GROUP_ORDER[key]}
            for key in sorted(GROUP_ORDER, key=lambda value: GROUP_ORDER[value])
        ],
        "metric_catalog": [],
        "runs": [],
    }


def main() -> None:
    args = parse_args()
    target = parse_target_date(args.target_date, now_year=datetime.now(tz=timezone.utc).year)
    start_ms, end_ms, tz_name = date_bounds_ms(target, args.timezone)
    algorithms = [algorithm.upper() for algorithm in parse_csv_arg(args.algorithms)]
    statuses = [status.upper() for status in parse_csv_arg(args.statuses)]
    date_slug = target.strftime("%Y%m%d")
    output = args.output or (
        PROJECT_ROOT
        / "frontend"
        / "public"
        / "mlflow"
        / f"dqn_ppo_training_history_{date_slug}.json"
    )
    snapshot_dir = args.snapshot_dir or DEFAULT_SNAPSHOT_ROOT / f"dqn-ppo-{date_slug}"

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=args.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(args.experiment_name)

    if experiment is None:
        payload = build_empty_payload(
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.experiment_name,
            experiment_id=None,
            target=target,
            tz_name=tz_name,
            algorithms=algorithms,
            run_selection=args.run_selection,
            reason=f"Experiment {args.experiment_name!r} was not found.",
        )
    else:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=args.max_runs,
        )

        matching_runs: list[tuple[Run, str]] = []
        for run in runs:
            start_time = run.info.start_time
            if start_time is None or not (start_ms <= start_time < end_ms):
                continue
            if statuses != ["ALL"] and run.info.status.upper() not in statuses:
                continue
            algorithm = infer_algorithm(run, algorithms)
            if algorithm is None:
                continue
            matching_runs.append((run, algorithm))

        selected_runs = select_runs(matching_runs, args.run_selection)
        runs_payload = [run_to_payload(client, run, algorithm) for run, algorithm in selected_runs]
        runs_by_algorithm = {algorithm: 0 for algorithm in algorithms}
        for run_payload in runs_payload:
            runs_by_algorithm[run_payload["algorithm"]] = runs_by_algorithm.get(run_payload["algorithm"], 0) + 1

        metric_catalog = build_metric_catalog(runs_payload)
        total_metric_points = sum(run["total_metric_points"] for run in runs_payload)
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "tracking_uri": args.mlflow_tracking_uri,
            "experiment": {
                "name": experiment.name,
                "experiment_id": experiment.experiment_id,
                "artifact_location": experiment.artifact_location,
                "lifecycle_stage": experiment.lifecycle_stage,
            },
            "target_date": target.isoformat(),
            "timezone": tz_name,
            "date_window": {
                "start_ms": start_ms,
                "end_ms_exclusive": end_ms,
                "start_iso": utc_iso_from_ms(start_ms),
                "end_iso_exclusive": utc_iso_from_ms(end_ms),
            },
            "algorithms": algorithms,
            "run_selection": args.run_selection,
            "status": "ok" if runs_payload else "empty",
            "message": (
                "Exported matching MLflow runs."
                if runs_payload
                else "No matching DQN/PPO runs were found for the selected date and statuses."
            ),
            "summary": {
                "run_count": len(runs_payload),
                "metric_count": len(metric_catalog),
                "total_metric_points": total_metric_points,
                "runs_by_algorithm": runs_by_algorithm,
                "max_step_by_algorithm": {
                    algorithm: max(
                        (
                            run["max_step"]
                            for run in runs_payload
                            if run["algorithm"] == algorithm and run["max_step"] is not None
                        ),
                        default=None,
                    )
                    for algorithm in algorithms
                },
            },
            "metric_groups": [
                {"key": key, "label": GROUP_LABELS[key], "order": GROUP_ORDER[key]}
                for key in sorted(GROUP_ORDER, key=lambda value: GROUP_ORDER[value])
            ],
            "metric_catalog": metric_catalog,
            "runs": runs_payload,
        }

    write_json(payload, output, compact=args.compact)

    rows = flatten_rows(payload)
    if not args.skip_snapshot:
        write_offline_snapshot(payload, rows, snapshot_dir, compact=args.compact)
    if args.csv_output is not None:
        write_csv(rows, args.csv_output)

    print(
        "Exported MLflow comparison data: "
        f"runs={payload['summary']['run_count']}, "
        f"metrics={payload['summary']['metric_count']}, "
        f"points={payload['summary']['total_metric_points']} -> {output}"
    )
    if not args.skip_snapshot:
        print(f"Exported offline static snapshot -> {snapshot_dir}")
    if args.csv_output is not None:
        print(f"Exported CSV rows={len(rows)} -> {args.csv_output}")


if __name__ == "__main__":
    main()
