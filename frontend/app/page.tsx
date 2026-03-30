"use client";

import { useSimulation } from '../hooks/useSimulation';
import Dashboard from '../components/Dashboard';
import Environment2D from '../components/Environment2D';

export default function Home() {
  const { state, connected, sendCommand } = useSimulation();

  return (
    <main className="min-h-screen pt-12 pb-24 px-6 md:px-12 selection:bg-indigo-500/30">
      {/* Decorative Lights */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-600/20 blur-[120px] rounded-full pointer-events-none" />
      <div className="absolute bottom-[10%] right-[-10%] w-[30%] h-[30%] bg-cyan-600/10 blur-[120px] rounded-full pointer-events-none" />

      <div className="max-w-6xl mx-auto relative z-10">
        
        {/* Header */}
        <header className="flex justify-between items-end mb-10 pb-6 border-b border-white/10">
          <div>
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400">
              Antigravity Engine
            </h1>
            <p className="text-slate-400 mt-2 text-lg font-medium tracking-wide">
              Predictive Maintenance Simulation
            </p>
          </div>
          <div className="flex items-center space-x-3 mb-2 px-4 py-2 bg-slate-800/50 rounded-full border border-white/5">
            <span className="text-sm font-semibold text-slate-300">Server</span>
            <div className={`w-3 h-3 rounded-full ${connected ? 'bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.8)]' : 'bg-rose-500 shadow-[0_0_12px_rgba(244,63,94,0.8)]'} animate-pulse`} />
          </div>
        </header>

        {/* 2D Environment Render */}
        <Environment2D state={state} />

        {/* Dashboard */}
        <Dashboard
          state={state}
          onPlayPause={() => sendCommand('TOGGLE_PLAY')}
          onStepFly={() => sendCommand('FORCE_FLY')}
          onStepLand={() => sendCommand('FORCE_LAND')}
          onReset={() => sendCommand('RESET')}
        />

      </div>
    </main>
  );
}
