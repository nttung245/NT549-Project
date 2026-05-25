import React from 'react';
import { SimState } from '../hooks/useSimulation';

type Tone = 'slate' | 'cyan' | 'emerald' | 'amber' | 'red' | 'sky' | 'violet' | 'orange';

const HAZARD_LABELS: Record<number, string> = {
  0: 'Clear',
  1: 'Headwind',
  2: 'Tailwind',
  3: 'Storm',
  4: 'Turbulence',
};

const tonePillClasses: Record<Tone, string> = {
  slate: 'border-slate-600/70 bg-slate-900/80 text-slate-200',
  cyan: 'border-cyan-400/40 bg-cyan-400/12 text-cyan-100',
  emerald: 'border-emerald-400/35 bg-emerald-400/12 text-emerald-100',
  amber: 'border-amber-300/45 bg-amber-300/12 text-amber-100',
  red: 'border-red-400/45 bg-red-400/12 text-red-100',
  sky: 'border-sky-400/40 bg-sky-400/12 text-sky-100',
  violet: 'border-violet-400/40 bg-violet-400/12 text-violet-100',
  orange: 'border-orange-400/45 bg-orange-400/12 text-orange-100',
};

const toneTextClasses: Record<Tone, string> = {
  slate: 'text-slate-100',
  cyan: 'text-cyan-100',
  emerald: 'text-emerald-100',
  amber: 'text-amber-100',
  red: 'text-red-100',
  sky: 'text-sky-100',
  violet: 'text-violet-100',
  orange: 'text-orange-100',
};

const toneBorderClasses: Record<Tone, string> = {
  slate: 'border-slate-600/70',
  cyan: 'border-cyan-400/35',
  emerald: 'border-emerald-400/35',
  amber: 'border-amber-300/45',
  red: 'border-red-400/45',
  sky: 'border-sky-400/35',
  violet: 'border-violet-400/35',
  orange: 'border-orange-400/45',
};

const progressClasses: Record<Tone, string> = {
  slate: 'bg-slate-300',
  cyan: 'bg-cyan-300',
  emerald: 'bg-emerald-300',
  amber: 'bg-amber-300',
  red: 'bg-red-300',
  sky: 'bg-sky-300',
  violet: 'bg-violet-300',
  orange: 'bg-orange-300',
};

const actionToneClasses: Record<'primary' | 'neutral' | 'sky' | 'orange' | 'danger', string> = {
  primary: 'border-cyan-400/45 bg-cyan-400/15 text-cyan-50 hover:border-cyan-300/80 hover:bg-cyan-400/24 hover:shadow-cyan-950/35',
  neutral: 'border-slate-600/70 bg-slate-800/72 text-slate-100 hover:border-slate-500 hover:bg-slate-700/88',
  sky: 'border-sky-400/45 bg-sky-400/15 text-sky-50 hover:border-sky-300/80 hover:bg-sky-400/24 hover:shadow-sky-950/35',
  orange: 'border-orange-400/45 bg-orange-400/15 text-orange-50 hover:border-orange-300/80 hover:bg-orange-400/24 hover:shadow-orange-950/35',
  danger: 'border-red-400/45 bg-red-400/12 text-red-50 hover:border-red-300/80 hover:bg-red-400/22 hover:shadow-red-950/35',
};

function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(' ');
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function formatNumber(value: number, digits = 0) {
  return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '0';
}

function formatMultiplier(value: number, digits = 2) {
  return `x${(Number.isFinite(value) ? value : 1).toFixed(digits)}`;
}

function formatLabel(value: string) {
  const normalized = value.replace(/_/g, ' ').trim();
  if (!normalized) return 'Unknown';
  return normalized.replace(/\w\S*/g, (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase());
}

function StatusPill({
  children,
  tone = 'slate',
  className,
}: {
  children: React.ReactNode;
  tone?: Tone;
  className?: string;
}) {
  return (
    <span className={cx('inline-flex max-w-full min-w-0 items-center rounded-full border px-3 py-1.5 text-[0.68rem] font-black uppercase tracking-[0.1em] shadow-sm', tonePillClasses[tone], className)}>
      <span className="min-w-0 truncate">{children}</span>
    </span>
  );
}

function Panel({
  title,
  eyebrow,
  right,
  className,
  children,
}: {
  title: string;
  eyebrow?: string;
  right?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <section className={cx('aviation-panel min-w-0 overflow-hidden rounded-[1.5rem] p-4 shadow-2xl sm:p-5', className)}>
      <div className="mb-5 flex min-w-0 flex-col gap-3 border-b border-slate-700/55 pb-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          {eyebrow && <p className="truncate text-[0.68rem] font-bold uppercase tracking-[0.18em] text-cyan-300/80">{eyebrow}</p>}
          <h3 className="mt-1 text-lg font-black tracking-tight text-white sm:text-xl">{title}</h3>
        </div>
        {right && <div className="flex min-w-0 flex-wrap items-center gap-2 sm:justify-end">{right}</div>}
      </div>
      {children}
    </section>
  );
}

function ProgressMeter({
  label,
  value,
  detail,
  percent,
  tone,
}: {
  label: string;
  value: string;
  detail?: string;
  percent: number;
  tone: Tone;
}) {
  const boundedPercent = clamp(percent, 0, 100);

  return (
    <div className="min-w-0 rounded-2xl border border-slate-700/55 bg-slate-950/38 p-3.5">
      <div className="mb-2 flex min-w-0 items-center justify-between gap-3 text-sm">
        <span className="min-w-0 font-semibold text-slate-300">{label}</span>
        <span className={cx('shrink-0 font-mono text-xs font-black', toneTextClasses[tone])}>{value}</span>
      </div>
      <div className="h-3 overflow-hidden rounded-full border border-slate-700/70 bg-slate-950/85 shadow-inner">
        <div className={cx('h-full rounded-full transition-all duration-500', progressClasses[tone])} style={{ width: `${boundedPercent}%` }} />
      </div>
      {detail && <p className="mt-2 text-xs leading-5 text-slate-500">{detail}</p>}
    </div>
  );
}

function MetricCard({
  label,
  value,
  unit,
  helper,
  tone = 'slate',
  emphasis = false,
}: {
  label: string;
  value: React.ReactNode;
  unit?: string;
  helper?: string;
  tone?: Tone;
  emphasis?: boolean;
}) {
  return (
    <div className={cx('min-w-0 overflow-hidden rounded-2xl border bg-slate-950/42 p-3.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.045)] transition duration-200 hover:bg-slate-900/68', toneBorderClasses[tone], emphasis && 'bg-slate-900/64')}>
      <p className="text-[0.68rem] font-bold uppercase tracking-[0.14em] text-slate-500">{label}</p>
      <div className="mt-2 flex min-w-0 items-baseline gap-1.5">
        <p className={cx('min-w-0 font-mono text-xl font-black leading-none tracking-tight sm:text-2xl', toneTextClasses[tone])}>{value}</p>
        {unit && <span className="shrink-0 text-xs font-bold text-slate-500">{unit}</span>}
      </div>
      {helper && <p className="mt-2 text-xs leading-5 text-slate-500">{helper}</p>}
    </div>
  );
}

function DetailRow({
  label,
  value,
  tone = 'slate',
}: {
  label: string;
  value: React.ReactNode;
  tone?: Tone;
}) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-3 border-b border-slate-800/80 py-2.5 last:border-b-0">
      <span className="min-w-0 text-sm font-medium text-slate-400">{label}</span>
      <span className={cx('shrink-0 text-right font-mono text-sm font-black', toneTextClasses[tone])}>{value}</span>
    </div>
  );
}

function ActionButton({
  onClick,
  icon,
  label,
  description,
  tone,
  className,
}: {
  onClick: () => void;
  icon: string;
  label: string;
  description: string;
  tone: 'primary' | 'neutral' | 'sky' | 'orange' | 'danger';
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cx(
        'group flex min-w-0 items-start gap-3 overflow-hidden rounded-2xl border p-3.5 text-left shadow-lg shadow-slate-950/18 outline-none transition duration-200 hover:-translate-y-0.5 focus-visible:ring-2 focus-visible:ring-cyan-300/70 active:translate-y-0 active:scale-[0.99] sm:p-4',
        actionToneClasses[tone],
        className
      )}
    >
      <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-white/10 bg-white/8 text-lg leading-none shadow-inner sm:h-11 sm:w-11 sm:text-xl">{icon}</span>
      <span className="min-w-0">
        <span className="block text-sm font-black uppercase tracking-[0.1em] text-current">{label}</span>
        <span className="mt-1 block text-xs font-medium leading-5 tracking-normal text-slate-300/78">
          {description}
        </span>
      </span>
    </button>
  );
}

export default function Dashboard({
  state,
  onPlayPause,
  onStepFly,
  onStepLand,
  onStepClimb,
  onReset,
}: {
  state: SimState | null;
  onPlayPause: () => void;
  onStepFly: () => void;
  onStepLand: () => void;
  onStepClimb: () => void;
  onReset: () => void;
}) {
  if (!state) return null;

  const fuelCapacity = Math.max(1, state.fuel_capacity || 1);
  const rulPct = clamp((state.rul / 150) * 100, 0, 100);
  const fuelPct = clamp((state.fuel / fuelCapacity) * 100, 0, 100);
  const isRiskyWeather = state.weather === 'storm' || state.weather === 'turbulence';
  const algorithm = state.algorithm || 'RL';
  const runName = state.run_name || 'Live simulation stream';
  const policyMode = state.policy_mode || 'deterministic';
  const maintenancePressure = clamp(state.maintenance_pressure ?? 0, 0, 1);
  const healthTone: Tone = state.rul < 30 ? 'red' : state.rul < 60 ? 'amber' : 'emerald';
  const fuelTone: Tone = state.fuel < 20 ? 'orange' : 'sky';
  const weatherTone: Tone = isRiskyWeather ? 'amber' : 'emerald';
  const approachStatus = state.in_approach_zone ? (state.landing_feasible_now ? 'Ready' : 'In zone') : 'En route';
  const approachTone: Tone = state.landing_feasible_now ? 'emerald' : state.in_approach_zone ? 'cyan' : 'slate';
  const nextHazard = HAZARD_LABELS[state.next_hazard_type] ?? 'Unknown';
  const nextHazardTone: Tone = nextHazard === 'Clear' ? 'emerald' : nextHazard === 'Storm' ? 'violet' : 'amber';
  const phase = formatLabel(state.flight_phase || 'CRUISING');
  const action = formatLabel(state.action || 'AUTO');
  const weather = formatLabel(state.weather || 'clear');
  const reward = Number.isFinite(state.reward) ? state.reward.toFixed(2) : '0.00';

  return (
    <div className="grid min-w-0 grid-cols-1 gap-5 xl:grid-cols-[minmax(300px,0.92fr)_minmax(0,1.55fr)_minmax(300px,0.92fr)]">
      <Panel
        title="Engine Health"
        eyebrow={`${algorithm} telemetry`}
        right={<StatusPill tone={healthTone}>RUL {Math.round(state.rul)}</StatusPill>}
      >
        <div className="mb-4 rounded-2xl border border-slate-700/65 bg-slate-950/48 p-3.5">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="text-[0.68rem] font-bold uppercase tracking-[0.16em] text-slate-500">Active run</p>
              <p className="mt-1 break-words font-mono text-xs font-semibold leading-5 text-slate-300">{runName}</p>
            </div>
            <StatusPill tone="cyan" className="shrink-0">Step {state.step}</StatusPill>
          </div>
        </div>

        <div className="space-y-3">
          <ProgressMeter
            label="Remaining useful life"
            value={`${Math.round(state.rul)} cycles`}
            detail={state.rul < 30 ? 'Critical inspection threshold is close.' : 'Health margin remains inside operating range.'}
            percent={rulPct}
            tone={healthTone}
          />
          <ProgressMeter
            label="Fuel reserves"
            value={`${state.fuel.toFixed(1)} / ${fuelCapacity.toFixed(0)}`}
            detail={state.fuel < 20 ? 'Reserve warning: consider alternate or landing.' : 'Reserve level is currently nominal.'}
            percent={fuelPct}
            tone={fuelTone}
          />
        </div>

        <div className="mt-4 grid min-w-0 grid-cols-2 gap-3">
          <MetricCard label="Pressure" value={(maintenancePressure * 100).toFixed(0)} unit="%" tone={maintenancePressure > 0.45 ? 'amber' : 'emerald'} helper="Maintenance load" />
          <MetricCard label="Reward" value={reward} tone={Number(state.reward) < 0 ? 'amber' : 'cyan'} helper="Current step" />
        </div>
      </Panel>

      <Panel
        className="xl:col-span-1"
        title="Flight Operations"
        eyebrow="Route and decision state"
        right={(
          <>
            <StatusPill tone="cyan">{algorithm}</StatusPill>
            <StatusPill tone="sky">{action}</StatusPill>
            <StatusPill tone="slate">{policyMode}</StatusPill>
          </>
        )}
      >
        <div className="grid min-w-0 grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-6">
          <MetricCard label="Altitude" value={formatNumber(Math.round(state.altitude))} unit="m" tone="cyan" helper={phase} emphasis />
          <MetricCard label="Velocity" value={formatNumber(Math.round(state.velocity))} tone="sky" helper="Ground speed" emphasis />
          <MetricCard label="Destination" value={formatNumber(Math.round(state.dist_dest))} unit="m" tone="slate" helper="Remaining" />
          <MetricCard label="Next AP" value={formatNumber(Math.round(state.dist_next))} unit="m" tone="violet" helper="Nearest alternate" />
          <MetricCard label="Service" value={(maintenancePressure * 100).toFixed(0)} unit="%" tone={maintenancePressure > 0.45 ? 'amber' : 'emerald'} helper="Need level" />
          <MetricCard label="Approach" value={approachStatus} tone={approachTone} helper={state.in_approach_zone ? 'Landing window' : 'Route cruise'} />
        </div>

        <div className="mt-4 grid min-w-0 grid-cols-1 gap-3 md:grid-cols-3">
          <div className="rounded-2xl border border-slate-700/60 bg-slate-950/42 p-3.5">
            <p className="text-[0.68rem] font-bold uppercase tracking-[0.16em] text-slate-500">Flight phase</p>
            <p className="mt-1 text-sm font-black text-slate-100">{phase}</p>
          </div>
          <div className="rounded-2xl border border-slate-700/60 bg-slate-950/42 p-3.5">
            <p className="text-[0.68rem] font-bold uppercase tracking-[0.16em] text-slate-500">Action mode</p>
            <p className="mt-1 text-sm font-black text-slate-100">{action}</p>
          </div>
          <div className="rounded-2xl border border-slate-700/60 bg-slate-950/42 p-3.5">
            <p className="text-[0.68rem] font-bold uppercase tracking-[0.16em] text-slate-500">Policy</p>
            <p className="mt-1 text-sm font-black text-slate-100">{policyMode}</p>
          </div>
        </div>
      </Panel>

      <Panel
        title="Weather Cell"
        eyebrow="Hazard load"
        right={<StatusPill tone={weatherTone}>{weather}</StatusPill>}
      >
        <div className="grid min-w-0 grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-2 2xl:grid-cols-3">
          <MetricCard label="Wind" value={state.wind_strength.toFixed(1)} tone={isRiskyWeather ? 'amber' : 'emerald'} helper="Strength" emphasis={isRiskyWeather} />
          <MetricCard label="Fuel burn" value={formatMultiplier(state.weather_fuel_multiplier)} tone="sky" helper="Load factor" />
          <MetricCard label="Speed" value={formatMultiplier(state.weather_speed_multiplier ?? 1)} tone="cyan" helper="Velocity factor" />
          <MetricCard label="Wear" value={formatMultiplier(state.weather_rul_multiplier)} tone={isRiskyWeather ? 'orange' : 'emerald'} helper="RUL factor" />
          <MetricCard label="Next" value={nextHazard} tone={nextHazardTone} helper={`${formatNumber(Math.round(state.next_hazard_distance))} m ahead`} />
          <MetricCard label="Alt window" value={`${formatNumber(Math.round(state.next_hazard_alt_min))}-${formatNumber(Math.round(state.next_hazard_alt_max))}`} unit="m" tone="violet" helper="Hazard band" />
        </div>

        <div className="mt-4 rounded-2xl border border-slate-700/60 bg-slate-950/42 px-3.5">
          <DetailRow label="Fuel load" value={formatMultiplier(state.weather_fuel_multiplier)} tone="sky" />
          <DetailRow label="RUL load" value={formatMultiplier(state.weather_rul_multiplier)} tone={isRiskyWeather ? 'orange' : 'emerald'} />
          <DetailRow label="Speed modifier" value={formatMultiplier(state.weather_speed_multiplier ?? 1)} tone="cyan" />
        </div>
      </Panel>

      <section className="aviation-panel min-w-0 overflow-hidden rounded-[1.5rem] p-4 shadow-2xl sm:p-5 xl:col-span-3">
        <div className="mb-5 flex min-w-0 flex-col gap-3 border-b border-slate-700/55 pb-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="min-w-0">
            <p className="text-[0.68rem] font-bold uppercase tracking-[0.18em] text-cyan-300/80">Manual command deck</p>
            <h3 className="mt-1 text-lg font-black tracking-tight text-white sm:text-xl">Simulation Controls</h3>
          </div>
          <p className="max-w-2xl text-sm leading-6 text-slate-500">
            Commands are sent immediately to the live simulation stream. Use manual overrides for inspection, climb, descent, and episode reset.
          </p>
        </div>

        <div className="grid min-w-0 grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <ActionButton onClick={onPlayPause} icon="⏯" label="Play / Pause" description="Hold or resume the live policy stream." tone="primary" />
          <ActionButton onClick={onStepFly} icon="✈" label="Cruise" description="Maintain route and current operating profile." tone="neutral" />
          <ActionButton onClick={onStepClimb} icon="↗" label="Climb" description="Increase altitude for the next command step." tone="sky" />
          <ActionButton onClick={onStepLand} icon="↘" label="Descend" description="Move toward approach or landing altitude." tone="orange" />
          <ActionButton onClick={onReset} icon="⟲" label="Reset" description="Restart the episode and clear live state." tone="danger" className="sm:col-span-2 lg:col-span-1" />
        </div>
      </section>
    </div>
  );
}
