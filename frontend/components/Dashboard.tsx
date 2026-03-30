import React from 'react';
import { SimState } from '../hooks/useSimulation';

export default function Dashboard({
  state,
  onPlayPause,
  onStepFly,
  onStepLand,
  onReset,
}: {
  state: SimState | null;
  onPlayPause: () => void;
  onStepFly: () => void;
  onStepLand: () => void;
  onReset: () => void;
}) {
  if (!state) return null;

  const rulPct = Math.max(0, Math.min(100, (state.rul / 150) * 100));
  const fuelPct = Math.max(0, Math.min(100, (state.fuel / state.fuel_capacity) * 100));

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
      {/* Metrics Card */}
      <div className="bg-slate-800/80 backdrop-blur-lg border border-white/10 rounded-3xl p-6 shadow-2xl flex flex-col space-y-6">
        <h3 className="text-xl font-semibold tracking-tight text-white mb-2">Engine Analytics</h3>
        
        {/* RUL Bar */}
        <div>
          <div className="flex justify-between text-sm mb-1 font-medium">
            <span className="text-slate-400">RUL (Cycles)</span>
            <span className={state.rul < 30 ? "text-red-400" : "text-green-400"}>{Math.round(state.rul)}</span>
          </div>
          <div className="w-full bg-slate-700/50 rounded-full h-3 backdrop-blur-sm overflow-hidden border border-white/5">
            <div
              className={`h-full rounded-full transition-all duration-300 ${state.rul < 30 ? 'bg-gradient-to-r from-red-600 to-red-400 shadow-[0_0_10px_rgba(239,68,68,0.5)]' : 'bg-gradient-to-r from-emerald-600 to-emerald-400'}`}
              style={{ width: `${rulPct}%` }}
            />
          </div>
        </div>

        {/* Fuel Bar */}
        <div>
          <div className="flex justify-between text-sm mb-1 font-medium">
            <span className="text-slate-400">Fuel Reserves</span>
            <span className={state.fuel < 20 ? "text-orange-400" : "text-sky-400"}>{state.fuel.toFixed(1)}</span>
          </div>
          <div className="w-full bg-slate-700/50 rounded-full h-3 backdrop-blur-sm overflow-hidden border border-white/5">
            <div
              className="h-full bg-gradient-to-r from-sky-600 to-sky-400 rounded-full transition-all duration-300"
              style={{ width: `${fuelPct}%` }}
            />
          </div>
        </div>
      </div>

      {/* Flight Stats Card */}
      <div className="bg-slate-800/80 backdrop-blur-lg border border-white/10 rounded-3xl p-6 shadow-2xl flex flex-col justify-center space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-900/50 rounded-2xl p-4 border border-white/5 cursor-default hover:bg-slate-900/70 transition-colors">
            <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Altitude</p>
            <p className="text-2xl font-bold text-slate-100 font-mono mt-1">{Math.round(state.altitude)} m</p>
          </div>
          <div className="bg-slate-900/50 rounded-2xl p-4 border border-white/5 cursor-default hover:bg-slate-900/70 transition-colors">
            <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Dist Dest.</p>
            <p className="text-2xl font-bold text-slate-100 font-mono mt-1">{Math.round(state.dist_dest)}</p>
          </div>
          <div className="bg-slate-900/50 rounded-2xl p-4 border border-white/5 cursor-default hover:bg-slate-900/70 transition-colors">
            <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Dist Next</p>
            <p className="text-2xl font-bold text-slate-100 font-mono mt-1">{Math.round(state.dist_next)}</p>
          </div>
          <div className="bg-slate-900/50 rounded-2xl p-4 border border-white/5 cursor-default hover:bg-slate-900/70 transition-colors">
            <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Last Action</p>
            <p className={`text-xl font-bold font-mono mt-1 ${state.action === 'LAND' ? 'text-orange-400 drop-shadow-[0_0_8px_rgba(251,146,60,0.5)]' : 'text-cyan-400'}`}>{state.action}</p>
          </div>
        </div>
      </div>

      {/* Controls Card */}
      <div className="bg-slate-800/80 backdrop-blur-lg border border-white/10 rounded-3xl p-6 shadow-2xl flex flex-col items-center justify-center space-y-4 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/10 rounded-full blur-3xl" />
        <h3 className="text-lg font-semibold tracking-tight text-white/80 shrink-0 w-full text-center">Nav Controls</h3>
        
        <div className="grid grid-cols-2 gap-3 w-full">
            <button
            onClick={onPlayPause}
            className="col-span-2 py-3 rounded-2xl font-bold tracking-widest uppercase bg-indigo-600 hover:bg-indigo-500 text-white shadow-[0_4px_14px_0_rgba(79,70,229,0.39)] transition-all active:scale-95 border border-indigo-400/50"
            >
            Play / Pause
            </button>
            <button
            onClick={onStepFly}
            className="py-3 rounded-2xl font-bold tracking-widest text-sm uppercase bg-cyan-900/80 hover:bg-cyan-800 text-cyan-100 border border-cyan-500/30 transition-all active:scale-95"
            >
            Step Fly
            </button>
            <button
            onClick={onStepLand}
            className="py-3 rounded-2xl font-bold tracking-widest text-sm uppercase bg-orange-900/80 hover:bg-orange-800 text-orange-100 border border-orange-500/30 transition-all active:scale-95"
            >
            Step Land
            </button>
             <button
            onClick={onReset}
            className="col-span-2 py-3 rounded-2xl mt-2 font-bold tracking-widest text-xs uppercase bg-slate-700/50 hover:bg-slate-600 text-slate-300 border border-slate-500/30 transition-all active:scale-95"
            >
            Reset Episode
            </button>
        </div>
      </div>

    </div>
  );
}
