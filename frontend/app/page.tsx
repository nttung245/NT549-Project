"use client";

import { useSimulation } from '../hooks/useSimulation';
import Dashboard from '../components/Dashboard';
import Environment2D from '../components/Environment2D';

export default function Home() {
  const { state, connected, sendCommand } = useSimulation();

  return (
    <main className="min-h-screen bg-[linear-gradient(180deg,#06111f_0%,#0f172a_48%,#111827_100%)] px-4 py-8 selection:bg-cyan-500/30 md:px-8">
      <div className="relative z-10 mx-auto max-w-7xl">
        <header className="mb-6 flex flex-col gap-4 border-b border-slate-700 pb-5 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight text-white md:text-5xl">
              Aircraft Digital Twin
            </h1>
            <p className="mt-2 text-base font-medium tracking-wide text-slate-400 md:text-lg">
              PPO route control with live engine risk and weather hazards
            </p>
          </div>
          <div className="flex w-fit items-center gap-3 rounded-md border border-slate-700 bg-slate-950/60 px-4 py-2">
            <span className="text-sm font-semibold text-slate-300">Server</span>
            <div className={`h-3 w-3 rounded-full ${connected ? 'bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.8)]' : 'bg-rose-500 shadow-[0_0_12px_rgba(244,63,94,0.8)]'} animate-pulse`} />
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
