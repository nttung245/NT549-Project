import React from 'react';
import { SimState } from '../hooks/useSimulation';

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

  const rulPct = Math.max(0, Math.min(100, (state.rul / 150) * 100));
  const fuelPct = Math.max(0, Math.min(100, (state.fuel / state.fuel_capacity) * 100));
  const isRiskyWeather = state.weather === 'storm' || state.weather === 'turbulence';
  const policyMode = state.policy_mode ?? 'deterministic';
  const maintenancePressure = Math.max(0, Math.min(1, state.maintenance_pressure ?? 0));

  return (
    <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-[1.05fr_1.2fr_0.95fr]">
      <section className="rounded-lg border border-slate-700 bg-slate-950/70 p-5 shadow-xl">
        <div className="mb-5 flex items-start justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-white">Engine</h3>
            <p className="mt-1 text-xs uppercase tracking-[0.18em] text-slate-500">PPO telemetry</p>
          </div>
          <span className={`rounded-md border px-2 py-1 text-xs font-bold ${state.rul < 30 ? 'border-red-400/40 bg-red-400/10 text-red-200' : 'border-emerald-400/30 bg-emerald-400/10 text-emerald-200'}`}>
            RUL {Math.round(state.rul)}
          </span>
        </div>

        <div className="space-y-5">
          <div>
            <div className="mb-2 flex justify-between text-sm font-medium">
              <span className="text-slate-400">Remaining useful life</span>
              <span className={state.rul < 30 ? 'text-red-300' : 'text-emerald-300'}>{Math.round(state.rul)} cycles</span>
            </div>
            <div className="h-3 overflow-hidden rounded bg-slate-800">
              <div
                className={`h-full transition-all duration-300 ${state.rul < 30 ? 'bg-red-400' : 'bg-emerald-400'}`}
                style={{ width: `${rulPct}%` }}
              />
            </div>
          </div>

          <div>
            <div className="mb-2 flex justify-between text-sm font-medium">
              <span className="text-slate-400">Fuel reserves</span>
              <span className={state.fuel < 20 ? 'text-orange-300' : 'text-sky-300'}>{state.fuel.toFixed(1)}</span>
            </div>
            <div className="h-3 overflow-hidden rounded bg-slate-800">
              <div className="h-full bg-sky-400 transition-all duration-300" style={{ width: `${fuelPct}%` }} />
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-slate-700 bg-slate-950/70 p-5 shadow-xl">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-base font-semibold text-white">Flight State</h3>
          <span className="rounded-md border border-slate-600 bg-slate-900 px-2 py-1 text-xs font-bold uppercase tracking-[0.14em] text-slate-200">
            {state.action} / {policyMode}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
          <div className="rounded-md border border-slate-800 bg-slate-900/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Altitude</p>
            <p className="mt-2 font-mono text-xl font-bold text-slate-100">{Math.round(state.altitude)}m</p>
          </div>
          <div className="rounded-md border border-slate-800 bg-slate-900/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Velocity</p>
            <p className="mt-2 font-mono text-xl font-bold text-slate-100">{Math.round(state.velocity)}</p>
          </div>
          <div className="rounded-md border border-slate-800 bg-slate-900/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Dest</p>
            <p className="mt-2 font-mono text-xl font-bold text-slate-100">{Math.round(state.dist_dest)}</p>
          </div>
          <div className="rounded-md border border-slate-800 bg-slate-900/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Next AP</p>
            <p className="mt-2 font-mono text-xl font-bold text-slate-100">{Math.round(state.dist_next)}</p>
          </div>
          <div className="rounded-md border border-slate-800 bg-slate-900/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Need</p>
            <p className={`mt-2 font-mono text-xl font-bold ${maintenancePressure > 0.45 ? 'text-amber-200' : 'text-slate-100'}`}>
              {(maintenancePressure * 100).toFixed(0)}%
            </p>
          </div>
          <div className="rounded-md border border-slate-800 bg-slate-900/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Approach</p>
            <p className={`mt-2 font-mono text-xl font-bold ${state.in_approach_zone ? 'text-cyan-200' : 'text-slate-100'}`}>
              {state.in_approach_zone ? (state.landing_feasible_now ? 'READY' : 'ZONE') : 'NO'}
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-slate-700 bg-slate-950/70 p-5 shadow-xl">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-base font-semibold text-white">Weather</h3>
          <span className={`rounded-md border px-2 py-1 text-xs font-bold uppercase ${isRiskyWeather ? 'border-amber-300/50 bg-amber-300/10 text-amber-100' : 'border-emerald-300/30 bg-emerald-300/10 text-emerald-100'}`}>
            {state.weather}
          </span>
        </div>

        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="rounded-md bg-slate-900 p-2">
            <p className="text-[11px] uppercase tracking-[0.12em] text-slate-500">Wind</p>
            <p className="mt-1 font-mono text-lg text-slate-100">{state.wind_strength.toFixed(1)}</p>
          </div>
          <div className="rounded-md bg-slate-900 p-2">
            <p className="text-[11px] uppercase tracking-[0.12em] text-slate-500">Fuel</p>
            <p className="mt-1 font-mono text-lg text-slate-100">x{state.weather_fuel_multiplier.toFixed(1)}</p>
          </div>
          <div className="rounded-md bg-slate-900 p-2">
            <p className="text-[11px] uppercase tracking-[0.12em] text-slate-500">Speed</p>
            <p className="mt-1 font-mono text-lg text-slate-100">x{(state.weather_speed_multiplier ?? 1).toFixed(1)}</p>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-slate-700 bg-slate-950/70 p-4 shadow-xl lg:col-span-3">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          <button onClick={onPlayPause} className="rounded-md border border-cyan-400/40 bg-cyan-400/15 px-4 py-3 text-sm font-bold uppercase tracking-[0.14em] text-cyan-50 transition hover:bg-cyan-400/25 active:scale-[0.98]">
            Play / Pause
          </button>
          <button onClick={onStepFly} className="rounded-md border border-slate-600 bg-slate-800 px-4 py-3 text-sm font-bold uppercase tracking-[0.14em] text-slate-100 transition hover:bg-slate-700 active:scale-[0.98]">
            Cruise
          </button>
          <button onClick={onStepClimb} className="rounded-md border border-sky-400/40 bg-sky-400/15 px-4 py-3 text-sm font-bold uppercase tracking-[0.14em] text-sky-50 transition hover:bg-sky-400/25 active:scale-[0.98]">
            Climb
          </button>
          <button onClick={onStepLand} className="rounded-md border border-orange-400/40 bg-orange-400/15 px-4 py-3 text-sm font-bold uppercase tracking-[0.14em] text-orange-50 transition hover:bg-orange-400/25 active:scale-[0.98]">
            Descend
          </button>
          <button onClick={onReset} className="col-span-2 rounded-md border border-red-400/40 bg-red-400/10 px-4 py-3 text-sm font-bold uppercase tracking-[0.14em] text-red-50 transition hover:bg-red-400/20 active:scale-[0.98] sm:col-span-1">
            Reset
          </button>
        </div>
      </section>
    </div>
  );
}
