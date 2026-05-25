"use client";

import { useSimulation } from '../hooks/useSimulation';
import Dashboard from '../components/Dashboard';
import Environment2D from '../components/Environment2D';

export default function Home() {
  const { state, connected, sendCommand } = useSimulation();
  const algorithm = state?.algorithm || 'PPO';
  const phase = state?.flight_phase?.replace(/_/g, ' ') || 'Awaiting telemetry';
  const step = state?.step ?? 0;

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#030814] px-3 py-4 text-slate-100 selection:bg-cyan-500/30 sm:px-5 md:px-7 lg:px-8 lg:py-7">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute left-[-12%] top-[-18%] h-[34rem] w-[34rem] rounded-full bg-cyan-500/10 blur-3xl" />
        <div className="absolute right-[-10%] top-[8%] h-[30rem] w-[30rem] rounded-full bg-blue-600/10 blur-3xl" />
        <div className="absolute bottom-[-18%] left-[28%] h-[30rem] w-[30rem] rounded-full bg-emerald-500/10 blur-3xl" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,#10213b_0%,rgba(3,8,20,0.35)_38%,#030814_78%)]" />
        <div className="absolute inset-0 opacity-[0.07] [background-image:linear-gradient(rgba(148,163,184,0.6)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.6)_1px,transparent_1px)] [background-size:64px_64px]" />
      </div>

      <div className="relative z-10 mx-auto flex max-w-[1600px] flex-col gap-5">
        <header className="aviation-panel overflow-hidden rounded-[1.5rem] p-4 sm:p-5 lg:p-6">
          <div className="flex min-w-0 flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
            <div className="min-w-0">
              <div className="mb-4 flex min-w-0 flex-wrap items-center gap-2">
                <span className="inline-flex max-w-full items-center rounded-full border border-cyan-400/35 bg-cyan-400/10 px-3 py-1.5 text-[0.68rem] font-bold uppercase tracking-[0.18em] text-cyan-100 shadow-[0_0_28px_rgba(34,211,238,0.12)]">
                  <span className="min-w-0 truncate">Autonomous Flight Ops</span>
                </span>
                <span className="inline-flex max-w-full items-center rounded-full border border-slate-600/70 bg-slate-950/55 px-3 py-1.5 text-[0.68rem] font-semibold uppercase tracking-[0.16em] text-slate-300">
                  <span className="min-w-0 truncate">Digital Twin Console</span>
                </span>
                <span className="inline-flex max-w-full items-center rounded-full border border-emerald-400/25 bg-emerald-400/10 px-3 py-1.5 text-[0.68rem] font-semibold uppercase tracking-[0.16em] text-emerald-100">
                  <span className="min-w-0 truncate">{algorithm} Policy</span>
                </span>
              </div>
              <h1 className="max-w-5xl text-balance bg-[linear-gradient(110deg,#f8fafc_0%,#a5f3fc_42%,#bfdbfe_72%,#f8fafc_100%)] bg-clip-text text-3xl font-black tracking-[-0.045em] text-transparent sm:text-4xl md:text-5xl xl:text-6xl">
                Aircraft Digital Twin
              </h1>
              <p className="mt-3 max-w-3xl text-pretty text-sm font-medium leading-6 text-slate-400 sm:text-base md:text-lg">
                A cleaner operations console for route control, engine health, weather hazards, and live reinforcement-learning telemetry.
              </p>
            </div>

            <div className="grid min-w-0 gap-3 sm:grid-cols-3 xl:w-[35rem]">
              <div className="rounded-2xl border border-slate-700/70 bg-slate-950/60 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)] sm:col-span-2">
                <div className="flex min-w-0 items-center justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-[0.68rem] font-bold uppercase tracking-[0.18em] text-slate-500">
                      Simulation Link
                    </p>
                    <p className="mt-1 text-sm font-semibold text-slate-200">
                      WebSocket telemetry stream
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2 rounded-full border border-slate-700 bg-slate-900/80 px-3 py-1.5">
                    <span className={`h-2.5 w-2.5 rounded-full ${connected ? 'bg-emerald-400 shadow-[0_0_16px_rgba(52,211,153,0.95)]' : 'bg-rose-500 shadow-[0_0_16px_rgba(244,63,94,0.95)]'} animate-pulse`} />
                    <span className={`text-xs font-black uppercase tracking-[0.12em] ${connected ? 'text-emerald-200' : 'text-rose-200'}`}>
                      {connected ? 'Online' : 'Offline'}
                    </span>
                  </div>
                </div>
                <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-800">
                  <div className={`h-full rounded-full transition-all duration-500 ${connected ? 'w-full bg-cyan-300' : 'w-1/3 bg-rose-400'}`} />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 sm:grid-cols-1">
                <div className="rounded-2xl border border-slate-700/70 bg-slate-950/52 p-3.5">
                  <p className="text-[0.66rem] font-bold uppercase tracking-[0.16em] text-slate-500">Step</p>
                  <p className="mt-1 font-mono text-xl font-black text-cyan-100">{step.toLocaleString()}</p>
                </div>
                <div className="rounded-2xl border border-slate-700/70 bg-slate-950/52 p-3.5">
                  <p className="text-[0.66rem] font-bold uppercase tracking-[0.16em] text-slate-500">Phase</p>
                  <p className="mt-1 truncate text-sm font-black capitalize text-slate-100">{phase.toLowerCase()}</p>
                </div>
              </div>
            </div>
          </div>
        </header>

        <Environment2D state={state} />

        <Dashboard
          state={state}
          onPlayPause={() => sendCommand('TOGGLE_PLAY')}
          onStepFly={() => sendCommand('FORCE_FLY')}
          onStepLand={() => sendCommand('FORCE_LAND')}
          onStepClimb={() => sendCommand('FORCE_CLIMB')}
          onReset={() => sendCommand('RESET')}
        />
      </div>
    </main>
  );
}
