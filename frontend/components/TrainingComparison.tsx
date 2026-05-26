"use client";

import { useEffect, useMemo, useState } from "react";

const DATA_URL = "/training-metrics/dqn-ppo-20260526/comparison.json";
const ALGORITHM_COLORS: Record<string, string> = {
    DQN: "#38bdf8",
    PPO: "#c084fc",
};
const FALLBACK_COLORS = ["#f59e0b", "#34d399", "#fb7185", "#facc15", "#2dd4bf"];

type MetricPoint = {
    step: number | null;
    value: number | null;
    timestamp: number | null;
    time: string | null;
};

type MetricSummary = {
    count: number;
    valid_count: number;
    first: number | null;
    last: number | null;
    min: number | null;
    max: number | null;
    mean: number | null;
    min_step: number | null;
    max_step: number | null;
};

type MetricPayload = {
    key: string;
    display_name: string;
    group: string;
    group_label: string;
    direction: "maximize" | "minimize" | "contextual" | string;
    chart_type: "line" | "bar" | string;
    summary: MetricSummary;
    history: MetricPoint[];
};

type RunPayload = {
    run_id: string;
    run_name: string;
    algorithm: string;
    status: string;
    artifact_uri?: string;
    start_time: number | null;
    start_time_iso: string | null;
    end_time: number | null;
    end_time_iso: string | null;
    duration_ms: number | null;
    params: Record<string, string>;
    tags: Record<string, string>;
    metric_keys: string[];
    metric_count: number;
    total_metric_points: number;
    max_step: number | null;
    metrics: Record<string, MetricPayload>;
};

type MetricCatalogEntry = {
    key: string;
    display_name: string;
    group: string;
    group_label: string;
    direction: string;
    chart_type: string;
    algorithms: string[];
    run_ids: string[];
    run_names: string[];
    points_by_algorithm: Record<string, number>;
    points_by_run: Record<string, number>;
    min_step: number | null;
    max_step: number | null;
    min_value: number | null;
    max_value: number | null;
    total_points: number;
};

type MetricGroup = {
    key: string;
    label: string;
    order: number;
};

type TrainingHistoryPayload = {
    schema_version: number;
    generated_at: string | null;
    tracking_uri: string;
    experiment: {
        name: string;
        experiment_id: string | null;
        artifact_location?: string;
        lifecycle_stage?: string;
    };
    target_date: string;
    timezone: string;
    algorithms: string[];
    run_selection: string;
    status: string;
    message: string;
    summary: {
        run_count: number;
        metric_count: number;
        total_metric_points: number;
        runs_by_algorithm: Record<string, number>;
        max_step_by_algorithm?: Record<string, number | null>;
    };
    metric_groups: MetricGroup[];
    metric_catalog: MetricCatalogEntry[];
    runs: RunPayload[];
};

type ChartSeries = {
    id: string;
    label: string;
    algorithm: string;
    color: string;
    points: Array<{ x: number; y: number; step: number; value: number }>;
};

type AxisMode = "step" | "progress";

type ComparisonDatum = {
    algorithm: string;
    runName: string;
    value: number | null;
    color: string;
};

function formatNumber(value: number | null | undefined, compact = false): string {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    if (!Number.isFinite(value)) return "—";
    const absolute = Math.abs(value);
    if (compact && absolute >= 1000) {
        return new Intl.NumberFormat("en", {
            notation: "compact",
            maximumFractionDigits: 2,
        }).format(value);
    }
    if (absolute >= 1000) {
        return new Intl.NumberFormat("en", { maximumFractionDigits: 0 }).format(value);
    }
    if (absolute >= 10) return value.toFixed(2).replace(/\.00$/, "");
    if (absolute >= 1) return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
    if (absolute === 0) return "0";
    return value.toExponential(2);
}

function formatDuration(durationMs: number | null): string {
    if (durationMs === null || durationMs < 0) return "—";
    const seconds = Math.round(durationMs / 1000);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${remainingSeconds}s`;
    return `${remainingSeconds}s`;
}

function formatDateTime(value: string | null): string {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return new Intl.DateTimeFormat("en", {
        dateStyle: "medium",
        timeStyle: "short",
        timeZone: "UTC",
    }).format(date);
}

function algorithmColor(algorithm: string, index = 0): string {
    return ALGORITHM_COLORS[algorithm] ?? FALLBACK_COLORS[index % FALLBACK_COLORS.length];
}

function shortRunId(runId: string): string {
    return runId.length <= 10 ? runId : `${runId.slice(0, 8)}…`;
}

function uniqueGroups(payload: TrainingHistoryPayload | null): MetricGroup[] {
    if (!payload) return [];
    const groupsFromCatalog = new Set(payload.metric_catalog.map((metric) => metric.group));
    return payload.metric_groups
        .filter((group) => groupsFromCatalog.has(group.key) || payload.metric_catalog.length === 0)
        .sort((a, b) => a.order - b.order);
}

function buildInitialSelections(payload: TrainingHistoryPayload | null): Record<string, string> {
    if (!payload) return {};
    const selections: Record<string, string> = {};
    for (const algorithm of payload.algorithms) {
        const latest = payload.runs.find((run) => run.algorithm === algorithm);
        if (latest) selections[algorithm] = latest.run_id;
    }
    return selections;
}

function toSeries(
    metricKey: string,
    selectedRuns: RunPayload[],
    visibleAlgorithms: Record<string, boolean>,
    axisMode: AxisMode,
): ChartSeries[] {
    return selectedRuns
        .filter((run) => visibleAlgorithms[run.algorithm] !== false)
        .map((run, index) => {
            const metric = run.metrics[metricKey];
            const maxStep = metric?.summary.max_step ?? run.max_step ?? 1;
            const denominator = Math.max(1, maxStep);
            const points = (metric?.history ?? [])
                .filter((point): point is MetricPoint & { step: number; value: number } => {
                    return typeof point.step === "number" && typeof point.value === "number" && Number.isFinite(point.value);
                })
                .map((point) => ({
                    x: axisMode === "progress" ? point.step / denominator : point.step,
                    y: point.value,
                    step: point.step,
                    value: point.value,
                }));

            return {
                id: `${run.run_id}:${metricKey}`,
                label: `${run.algorithm} · ${run.run_name}`,
                algorithm: run.algorithm,
                color: algorithmColor(run.algorithm, index),
                points,
            };
        });
}

function selectedRunList(payload: TrainingHistoryPayload | null, selectedRunIds: Record<string, string>): RunPayload[] {
    if (!payload) return [];
    return payload.algorithms
        .map((algorithm) => payload.runs.find((run) => run.run_id === selectedRunIds[algorithm]))
        .filter((run): run is RunPayload => Boolean(run));
}

function collectFinalValues(metricKey: string, selectedRuns: RunPayload[]): ComparisonDatum[] {
    return selectedRuns.map((run, index) => {
        const metric = run.metrics[metricKey];
        return {
            algorithm: run.algorithm,
            runName: run.run_name,
            value: metric?.summary.last ?? null,
            color: algorithmColor(run.algorithm, index),
        };
    });
}

function latestInsight(metricKey: string, selectedRuns: RunPayload[]): string {
    const values = collectFinalValues(metricKey, selectedRuns).filter((item) => item.value !== null) as Array<
        ComparisonDatum & { value: number }
    >;
    if (values.length < 2) return "Need both algorithms for a direct latest-value delta.";
    const [first, second] = values;
    const delta = first.value - second.value;
    const percent = second.value !== 0 ? (delta / Math.abs(second.value)) * 100 : null;
    const direction = delta >= 0 ? "higher" : "lower";
    return `${first.algorithm} latest value is ${formatNumber(Math.abs(delta))} ${direction} than ${second.algorithm}${percent !== null ? ` (${formatNumber(Math.abs(percent))}%)` : ""
        }.`;
}

function normalizeMetricKey(key: string): string {
    return key.toLowerCase().replace(/[\s_/-]+/g, " ");
}

function EmptyState({ message }: { message: string }) {
    return (
        <section className="rounded-3xl border border-slate-700 bg-slate-950/70 p-8 shadow-2xl shadow-slate-950/50">
            <div className="mx-auto max-w-3xl text-center">
                <p className="text-sm font-semibold uppercase tracking-[0.3em] text-cyan-300">No static data yet</p>
                <h2 className="mt-4 text-3xl font-black tracking-tight text-white">MLflow export is required</h2>
                <p className="mt-4 text-slate-300">{message}</p>
                <div className="mt-6 rounded-2xl border border-slate-700 bg-slate-900/70 p-4 text-left font-mono text-sm text-slate-200">
                    <p>source .venv/bin/activate</p>
                    <p>python scripts/export_mlflow_training_history.py</p>
                    <p>cd frontend && bun run build</p>
                </div>
            </div>
        </section>
    );
}

function SummaryCard({ label, value, detail }: { label: string; value: string; detail: string }) {
    return (
        <div className="rounded-2xl border border-slate-700/80 bg-slate-950/70 p-5 shadow-lg shadow-slate-950/30">
            <p className="text-xs font-bold uppercase tracking-[0.22em] text-slate-500">{label}</p>
            <p className="mt-3 text-3xl font-black tracking-tight text-white">{value}</p>
            <p className="mt-2 text-sm text-slate-400">{detail}</p>
        </div>
    );
}

function TogglePill({
    active,
    color,
    label,
    onClick,
}: {
    active: boolean;
    color: string;
    label: string;
    onClick: () => void;
}) {
    return (
        <button
            type="button"
            onClick={onClick}
            className={`rounded-full border px-4 py-2 text-sm font-bold transition ${active ? "border-white/20 bg-white/10 text-white shadow-lg" : "border-slate-700 bg-slate-950/80 text-slate-500"
                }`}
            style={active ? { boxShadow: `0 0 28px ${color}30` } : undefined}
        >
            <span className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: active ? color : "#475569" }} />
            {label}
        </button>
    );
}

function RunSelector({
    payload,
    selectedRunIds,
    onChange,
}: {
    payload: TrainingHistoryPayload;
    selectedRunIds: Record<string, string>;
    onChange: (algorithm: string, runId: string) => void;
}) {
    return (
        <div className="grid gap-4 lg:grid-cols-2">
            {payload.algorithms.map((algorithm, index) => {
                const runs = payload.runs.filter((run) => run.algorithm === algorithm);
                return (
                    <label key={algorithm} className="rounded-2xl border border-slate-700 bg-slate-950/70 p-4">
                        <span className="flex items-center justify-between gap-3">
                            <span className="text-sm font-black uppercase tracking-[0.22em]" style={{ color: algorithmColor(algorithm, index) }}>
                                {algorithm}
                            </span>
                            <span className="text-xs text-slate-500">{runs.length} run(s)</span>
                        </span>
                        <select
                            value={selectedRunIds[algorithm] ?? ""}
                            onChange={(event) => onChange(algorithm, event.target.value)}
                            className="mt-3 w-full rounded-xl border border-slate-700 bg-slate-900 px-3 py-3 text-sm font-semibold text-slate-100 outline-none transition focus:border-cyan-400"
                            disabled={runs.length === 0}
                        >
                            {runs.length === 0 ? <option value="">No {algorithm} run found</option> : null}
                            {runs.map((run) => (
                                <option key={run.run_id} value={run.run_id}>
                                    {run.run_name} · {formatNumber(run.max_step, true)} steps · {run.status}
                                </option>
                            ))}
                        </select>
                    </label>
                );
            })}
        </div>
    );
}

function RunOverview({ runs }: { runs: RunPayload[] }) {
    if (runs.length === 0) return null;
    return (
        <div className="grid gap-4 lg:grid-cols-2">
            {runs.map((run, index) => (
                <article key={run.run_id} className="overflow-hidden rounded-3xl border border-slate-700 bg-slate-950/70 shadow-xl">
                    <div className="h-1.5" style={{ backgroundColor: algorithmColor(run.algorithm, index) }} />
                    <div className="p-5">
                        <div className="flex flex-wrap items-start justify-between gap-3">
                            <div>
                                <p className="text-xs font-black uppercase tracking-[0.24em] text-slate-500">{run.algorithm}</p>
                                <h3 className="mt-2 text-xl font-black text-white">{run.run_name}</h3>
                                <p className="mt-1 font-mono text-xs text-slate-500">{shortRunId(run.run_id)}</p>
                            </div>
                            <span className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs font-bold text-emerald-200">
                                {run.status}
                            </span>
                        </div>
                        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
                            <div className="rounded-2xl bg-slate-900/80 p-3">
                                <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Metrics</p>
                                <p className="mt-1 text-lg font-black text-white">{run.metric_count}</p>
                            </div>
                            <div className="rounded-2xl bg-slate-900/80 p-3">
                                <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Points</p>
                                <p className="mt-1 text-lg font-black text-white">{formatNumber(run.total_metric_points, true)}</p>
                            </div>
                            <div className="rounded-2xl bg-slate-900/80 p-3">
                                <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Max step</p>
                                <p className="mt-1 text-lg font-black text-white">{formatNumber(run.max_step, true)}</p>
                            </div>
                            <div className="rounded-2xl bg-slate-900/80 p-3">
                                <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Duration</p>
                                <p className="mt-1 text-lg font-black text-white">{formatDuration(run.duration_ms)}</p>
                            </div>
                        </div>
                        <p className="mt-4 text-sm text-slate-400">Started {formatDateTime(run.start_time_iso)} UTC</p>
                    </div>
                </article>
            ))}
        </div>
    );
}

function LineChart({ series, axisMode }: { series: ChartSeries[]; axisMode: AxisMode }) {
    const drawableSeries = series.filter((item) => item.points.length > 0);
    if (drawableSeries.length === 0) {
        return (
            <div className="flex h-72 items-center justify-center rounded-2xl border border-dashed border-slate-700 bg-slate-900/50 text-sm font-semibold text-slate-500">
                No metric history points for the selected visible runs.
            </div>
        );
    }

    const allPoints = drawableSeries.flatMap((item) => item.points);
    const minX = axisMode === "progress" ? 0 : Math.min(...allPoints.map((point) => point.x));
    const maxX = axisMode === "progress" ? 1 : Math.max(...allPoints.map((point) => point.x));
    const minY = Math.min(...allPoints.map((point) => point.y));
    const maxY = Math.max(...allPoints.map((point) => point.y));
    const width = 920;
    const height = 330;
    const padLeft = 68;
    const padRight = 26;
    const padTop = 24;
    const padBottom = 46;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;
    const xSpan = maxX - minX || 1;
    const yPadding = (maxY - minY) * 0.08 || Math.max(1, Math.abs(maxY) * 0.08);
    const yMin = minY - yPadding;
    const yMax = maxY + yPadding;
    const ySpan = yMax - yMin || 1;
    const xScale = (x: number) => padLeft + ((x - minX) / xSpan) * plotWidth;
    const yScale = (y: number) => padTop + (1 - (y - yMin) / ySpan) * plotHeight;
    const yTicks = Array.from({ length: 5 }, (_, index) => yMin + (ySpan * index) / 4);
    const xTicks = Array.from({ length: 5 }, (_, index) => minX + (xSpan * index) / 4);

    return (
        <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/70">
            <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img" aria-label="MLflow metric line chart">
                <defs>
                    <linearGradient id="chartBackground" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#0f172a" stopOpacity="0.95" />
                        <stop offset="100%" stopColor="#020617" stopOpacity="0.98" />
                    </linearGradient>
                </defs>
                <rect width={width} height={height} fill="url(#chartBackground)" />
                {yTicks.map((tick) => {
                    const y = yScale(tick);
                    return (
                        <g key={`y-${tick}`}>
                            <line x1={padLeft} x2={width - padRight} y1={y} y2={y} stroke="#1e293b" strokeDasharray="4 8" />
                            <text x={padLeft - 12} y={y + 4} textAnchor="end" fill="#94a3b8" fontSize="12">
                                {formatNumber(tick, true)}
                            </text>
                        </g>
                    );
                })}
                {xTicks.map((tick) => {
                    const x = xScale(tick);
                    return (
                        <g key={`x-${tick}`}>
                            <line x1={x} x2={x} y1={padTop} y2={height - padBottom} stroke="#0f172a" />
                            <text x={x} y={height - 17} textAnchor="middle" fill="#94a3b8" fontSize="12">
                                {axisMode === "progress" ? `${formatNumber(tick * 100)}%` : formatNumber(tick, true)}
                            </text>
                        </g>
                    );
                })}
                <line x1={padLeft} x2={width - padRight} y1={height - padBottom} y2={height - padBottom} stroke="#334155" />
                <line x1={padLeft} x2={padLeft} y1={padTop} y2={height - padBottom} stroke="#334155" />
                {drawableSeries.map((item) => {
                    const path = item.points.map((point, index) => `${index === 0 ? "M" : "L"}${xScale(point.x).toFixed(2)},${yScale(point.y).toFixed(2)}`).join(" ");
                    return (
                        <g key={item.id}>
                            <path d={path} fill="none" stroke={item.color} strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" />
                            {item.points.map((point, index) => {
                                if (item.points.length > 60 && index % Math.ceil(item.points.length / 24) !== 0 && index !== item.points.length - 1) {
                                    return null;
                                }
                                return (
                                    <circle key={`${item.id}-${point.step}-${index}`} cx={xScale(point.x)} cy={yScale(point.y)} r="3.5" fill={item.color} opacity="0.9">
                                        <title>{`${item.label}\nstep: ${formatNumber(point.step, true)}\nvalue: ${formatNumber(point.value)}`}</title>
                                    </circle>
                                );
                            })}
                        </g>
                    );
                })}
            </svg>
        </div>
    );
}

function FinalValueBars({ values }: { values: ComparisonDatum[] }) {
    const finiteValues = values.filter((item): item is ComparisonDatum & { value: number } => typeof item.value === "number" && Number.isFinite(item.value));
    if (finiteValues.length === 0) {
        return <p className="text-sm text-slate-500">No final values available.</p>;
    }
    const minValue = Math.min(0, ...finiteValues.map((item) => item.value));
    const maxValue = Math.max(0, ...finiteValues.map((item) => item.value));
    const span = maxValue - minValue || 1;
    return (
        <div className="space-y-3">
            {values.map((item) => {
                const numericValue = typeof item.value === "number" && Number.isFinite(item.value) ? item.value : null;
                const width = numericValue !== null ? Math.max(4, ((numericValue - minValue) / span) * 100) : 0;
                return (
                    <div key={`${item.algorithm}-${item.runName}`}>
                        <div className="mb-1 flex items-center justify-between gap-3 text-xs">
                            <span className="font-bold text-slate-300">{item.algorithm}</span>
                            <span className="font-mono text-slate-400">{formatNumber(item.value)}</span>
                        </div>
                        <div className="h-3 overflow-hidden rounded-full bg-slate-800">
                            <div className="h-full rounded-full" style={{ width: `${width}%`, backgroundColor: item.color }} />
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

function MetricCard({
    metric,
    selectedRuns,
    visibleAlgorithms,
    axisMode,
}: {
    metric: MetricCatalogEntry;
    selectedRuns: RunPayload[];
    visibleAlgorithms: Record<string, boolean>;
    axisMode: AxisMode;
}) {
    const series = toSeries(metric.key, selectedRuns, visibleAlgorithms, axisMode);
    const finalValues = collectFinalValues(metric.key, selectedRuns).filter((value) => visibleAlgorithms[value.algorithm] !== false);
    const availableAlgorithms = series.filter((item) => item.points.length > 0).map((item) => item.algorithm);
    const missingAlgorithms = selectedRuns
        .filter((run) => visibleAlgorithms[run.algorithm] !== false && !run.metrics[metric.key])
        .map((run) => run.algorithm);

    return (
        <article className="rounded-3xl border border-slate-700/80 bg-slate-950/80 p-5 shadow-2xl shadow-slate-950/40">
            <div className="mb-5 flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                <div>
                    <div className="flex flex-wrap items-center gap-2">
                        <span className="rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs font-black uppercase tracking-[0.18em] text-cyan-200">
                            {metric.group_label}
                        </span>
                        <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-xs font-bold uppercase tracking-[0.14em] text-slate-400">
                            {metric.direction}
                        </span>
                    </div>
                    <h3 className="mt-3 break-words font-mono text-xl font-black text-white">{metric.key}</h3>
                    <p className="mt-2 text-sm text-slate-400">{latestInsight(metric.key, selectedRuns)}</p>
                </div>
                <div className="grid min-w-[260px] grid-cols-2 gap-3 text-sm">
                    <div className="rounded-2xl bg-slate-900/80 p-3">
                        <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Points</p>
                        <p className="mt-1 text-lg font-black text-white">{formatNumber(metric.total_points, true)}</p>
                    </div>
                    <div className="rounded-2xl bg-slate-900/80 p-3">
                        <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Step range</p>
                        <p className="mt-1 text-lg font-black text-white">
                            {formatNumber(metric.min_step, true)}–{formatNumber(metric.max_step, true)}
                        </p>
                    </div>
                </div>
            </div>

            <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_280px]">
                <LineChart series={series} axisMode={axisMode} />
                <aside className="space-y-5 rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
                    <div>
                        <p className="mb-3 text-xs font-black uppercase tracking-[0.22em] text-slate-500">Latest values</p>
                        <FinalValueBars values={finalValues} />
                    </div>
                    <div className="border-t border-slate-800 pt-4">
                        <p className="text-xs font-black uppercase tracking-[0.22em] text-slate-500">Availability</p>
                        <p className="mt-2 text-sm text-slate-300">Shown: {availableAlgorithms.length ? availableAlgorithms.join(", ") : "none"}</p>
                        {missingAlgorithms.length ? <p className="mt-1 text-sm text-amber-300">Missing: {missingAlgorithms.join(", ")}</p> : null}
                    </div>
                </aside>
            </div>
        </article>
    );
}

export default function TrainingComparison() {
    const [payload, setPayload] = useState<TrainingHistoryPayload | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [visibleAlgorithms, setVisibleAlgorithms] = useState<Record<string, boolean>>({ DQN: true, PPO: true });
    const [selectedRunIds, setSelectedRunIds] = useState<Record<string, string>>({});
    const [axisMode, setAxisMode] = useState<AxisMode>("step");
    const [query, setQuery] = useState("");
    const [selectedGroup, setSelectedGroup] = useState("all");

    useEffect(() => {
        let cancelled = false;
        async function loadData() {
            try {
                const response = await fetch(DATA_URL, { cache: "no-store" });
                if (!response.ok) throw new Error(`Could not load ${DATA_URL}: ${response.status}`);
                const data = (await response.json()) as TrainingHistoryPayload;
                if (!cancelled) {
                    setPayload(data);
                    setSelectedRunIds(buildInitialSelections(data));
                    setVisibleAlgorithms(Object.fromEntries(data.algorithms.map((algorithm) => [algorithm, true])));
                }
            } catch (loadError) {
                if (!cancelled) setError(loadError instanceof Error ? loadError.message : "Unknown data loading error");
            } finally {
                if (!cancelled) setLoading(false);
            }
        }
        loadData();
        return () => {
            cancelled = true;
        };
    }, []);

    const selectedRuns = useMemo(() => selectedRunList(payload, selectedRunIds), [payload, selectedRunIds]);
    const groups = useMemo(() => uniqueGroups(payload), [payload]);
    const filteredMetrics = useMemo(() => {
        if (!payload) return [];
        const normalizedQuery = normalizeMetricKey(query);
        return payload.metric_catalog.filter((metric) => {
            if (selectedGroup !== "all" && metric.group !== selectedGroup) return false;
            if (!normalizedQuery) return true;
            const haystack = normalizeMetricKey(`${metric.key} ${metric.display_name} ${metric.group_label}`);
            return haystack.includes(normalizedQuery);
        });
    }, [payload, query, selectedGroup]);

    if (loading) {
        return (
            <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,#164e63_0%,transparent_34%),linear-gradient(180deg,#020617_0%,#0f172a_48%,#111827_100%)] px-4 py-8 md:px-8">
                <div className="mx-auto max-w-7xl animate-pulse rounded-3xl border border-slate-800 bg-slate-950/70 p-10 text-center text-slate-300">
                    Loading static MLflow comparison data…
                </div>
            </main>
        );
    }

    if (error || !payload) {
        return (
            <main className="min-h-screen bg-[linear-gradient(180deg,#020617_0%,#0f172a_50%,#111827_100%)] px-4 py-8 md:px-8">
                <div className="mx-auto max-w-7xl">
                    <EmptyState message={error ?? "The static MLflow payload could not be loaded."} />
                </div>
            </main>
        );
    }

    const hasData = payload.runs.length > 0 && payload.metric_catalog.length > 0;

    return (
        <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,#155e75_0%,transparent_32%),radial-gradient(circle_at_top_right,#581c87_0%,transparent_30%),linear-gradient(180deg,#020617_0%,#0f172a_50%,#111827_100%)] px-4 py-8 selection:bg-cyan-500/30 md:px-8">
            <div className="mx-auto max-w-7xl">
                <header className="overflow-hidden rounded-[2rem] border border-slate-700 bg-slate-950/70 shadow-2xl shadow-slate-950/60 backdrop-blur">
                    <div className="relative p-6 md:p-9">
                        <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-sky-400 via-fuchsia-400 to-amber-300" />
                        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                            <div>
                                <p className="text-sm font-black uppercase tracking-[0.35em] text-cyan-300">Static MLflow Dashboard</p>
                                <h1 className="mt-4 max-w-4xl text-4xl font-black tracking-tight text-white md:text-6xl">
                                    DQN vs PPO Training History
                                </h1>
                                <p className="mt-4 max-w-3xl text-base leading-7 text-slate-300 md:text-lg">
                                    Full metric-history comparison for all MLflow metrics exported from {payload.experiment.name} on {payload.target_date}. Toggle algorithms, compare unequal timestep ranges, and inspect every saved metric.
                                </p>
                            </div>
                            <div className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4 text-sm text-slate-300">
                                <p className="font-mono text-xs text-slate-500">Generated</p>
                                <p className="mt-1 font-bold text-white">{formatDateTime(payload.generated_at)}</p>
                                <p className="mt-3 font-mono text-xs text-slate-500">Tracking URI</p>
                                <p className="mt-1 break-all font-semibold text-cyan-200">{payload.tracking_uri}</p>
                            </div>
                        </div>
                    </div>
                </header>

                <section className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                    <SummaryCard label="Runs" value={formatNumber(payload.summary.run_count, true)} detail="Selected MLflow runs exported for the date." />
                    <SummaryCard label="Metrics" value={formatNumber(payload.summary.metric_count, true)} detail="Every discovered metric key, not a hard-coded subset." />
                    <SummaryCard label="History points" value={formatNumber(payload.summary.total_metric_points, true)} detail="All metric-history samples preserved." />
                    <SummaryCard label="Date" value={payload.target_date} detail={`Bounded with ${payload.timezone} calendar day.`} />
                </section>

                {!hasData ? (
                    <div className="mt-6">
                        <EmptyState message={payload.message} />
                    </div>
                ) : (
                    <>
                        <section className="mt-6 rounded-3xl border border-slate-700 bg-slate-950/70 p-5 shadow-2xl shadow-slate-950/40">
                            <div className="flex flex-col gap-5 xl:flex-row xl:items-center xl:justify-between">
                                <div className="flex flex-wrap gap-3">
                                    {payload.algorithms.map((algorithm, index) => (
                                        <TogglePill
                                            key={algorithm}
                                            active={visibleAlgorithms[algorithm] !== false}
                                            color={algorithmColor(algorithm, index)}
                                            label={`${algorithm} (${payload.summary.runs_by_algorithm[algorithm] ?? 0})`}
                                            onClick={() => setVisibleAlgorithms((current) => ({ ...current, [algorithm]: current[algorithm] === false }))}
                                        />
                                    ))}
                                </div>
                                <div className="flex flex-wrap gap-2 rounded-2xl border border-slate-800 bg-slate-900/60 p-1">
                                    <button
                                        type="button"
                                        onClick={() => setAxisMode("step")}
                                        className={`rounded-xl px-4 py-2 text-sm font-bold transition ${axisMode === "step" ? "bg-cyan-400 text-slate-950" : "text-slate-400 hover:text-white"}`}
                                    >
                                        Absolute step
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setAxisMode("progress")}
                                        className={`rounded-xl px-4 py-2 text-sm font-bold transition ${axisMode === "progress" ? "bg-cyan-400 text-slate-950" : "text-slate-400 hover:text-white"}`}
                                    >
                                        Normalized progress
                                    </button>
                                </div>
                            </div>

                            <div className="mt-5">
                                <RunSelector payload={payload} selectedRunIds={selectedRunIds} onChange={(algorithm, runId) => setSelectedRunIds((current) => ({ ...current, [algorithm]: runId }))} />
                            </div>
                        </section>

                        <section className="mt-6">
                            <RunOverview runs={selectedRuns} />
                        </section>

                        <section className="mt-6 rounded-3xl border border-slate-700 bg-slate-950/70 p-5 shadow-2xl shadow-slate-950/40">
                            <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_280px]">
                                <label>
                                    <span className="text-xs font-black uppercase tracking-[0.22em] text-slate-500">Search metrics</span>
                                    <input
                                        value={query}
                                        onChange={(event) => setQuery(event.target.value)}
                                        placeholder="rollout/ep_rew_mean, train/loss, eval_diag_det/..."
                                        className="mt-2 w-full rounded-2xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm font-semibold text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-400"
                                    />
                                </label>
                                <label>
                                    <span className="text-xs font-black uppercase tracking-[0.22em] text-slate-500">Metric group</span>
                                    <select
                                        value={selectedGroup}
                                        onChange={(event) => setSelectedGroup(event.target.value)}
                                        className="mt-2 w-full rounded-2xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm font-semibold text-white outline-none transition focus:border-cyan-400"
                                    >
                                        <option value="all">All metric groups</option>
                                        {groups.map((group) => (
                                            <option key={group.key} value={group.key}>
                                                {group.label}
                                            </option>
                                        ))}
                                    </select>
                                </label>
                            </div>
                            <div className="mt-4 flex flex-wrap gap-2">
                                {groups.map((group) => (
                                    <button
                                        type="button"
                                        key={group.key}
                                        onClick={() => setSelectedGroup(group.key)}
                                        className={`rounded-full border px-3 py-1.5 text-xs font-bold transition ${selectedGroup === group.key ? "border-cyan-300 bg-cyan-300/15 text-cyan-100" : "border-slate-700 bg-slate-900/70 text-slate-400 hover:text-white"
                                            }`}
                                    >
                                        {group.label}
                                    </button>
                                ))}
                            </div>
                        </section>

                        <section className="mt-6 space-y-5">
                            <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
                                <div>
                                    <p className="text-sm font-black uppercase tracking-[0.28em] text-cyan-300">Metric histories</p>
                                    <h2 className="mt-2 text-3xl font-black tracking-tight text-white">{filteredMetrics.length} chart(s)</h2>
                                </div>
                                <p className="max-w-2xl text-sm text-slate-400">
                                    Metrics that exist for only one algorithm remain visible and are marked as missing for the other algorithm instead of being dropped.
                                </p>
                            </div>
                            {filteredMetrics.length === 0 ? (
                                <div className="rounded-3xl border border-dashed border-slate-700 bg-slate-950/60 p-8 text-center text-slate-400">
                                    No metrics match the current search and group filters.
                                </div>
                            ) : (
                                filteredMetrics.map((metric) => (
                                    <MetricCard key={metric.key} metric={metric} selectedRuns={selectedRuns} visibleAlgorithms={visibleAlgorithms} axisMode={axisMode} />
                                ))
                            )}
                        </section>
                    </>
                )}
            </div>
        </main>
    );
}
