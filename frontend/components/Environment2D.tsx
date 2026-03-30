import React from 'react';
import { SimState } from '../hooks/useSimulation';

export default function Environment2D({ state }: { state: SimState | null }) {
  if (!state) return <div className="h-[400px] flex items-center justify-center text-white/50 animate-pulse bg-white/5 rounded-3xl border border-white/10 backdrop-blur-md">Connecting to Simulation...</div>;

  const width = 1000;
  const height = 400;

  const mapX = (dist: number) => {
    return 50 + (dist / state.total_distance) * (width - 100);
  };

  const mapY = (alt: number) => {
    return 300 - (alt / 10000) * 200;
  };

  const currentPos = state.total_distance - state.dist_dest;
  const planeX = mapX(currentPos);
  const planeY = mapY(state.altitude);

  return (
    <div className="relative w-full overflow-hidden bg-gradient-to-b from-slate-900 to-slate-800 rounded-3xl border border-white/10 shadow-2xl glassmorphism p-6">
      
      <div className="absolute top-4 left-6 flex items-center space-x-2">
        <div className={`w-3 h-3 rounded-full ${state.flight_phase === 'CRUISING' ? 'bg-cyan-400' : 'bg-orange-400'} animate-pulse shadow-[0_0_10px_currentColor]`} />
        <span className="text-sm font-semibold tracking-wider text-slate-300 uppercase">{state.flight_phase}</span>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto mt-4 drop-shadow-xl overflow-visible">
        {/* Sky Background Element */}
        <defs>
          <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#0ea5e9" stopOpacity="0.1" />
            <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.0" />
          </linearGradient>
        </defs>
        <rect width="100%" height="320" fill="url(#skyGrad)" />

        {/* Ground */}
        <line x1="50" y1="320" x2={width - 50} y2="320" stroke="#10b981" strokeWidth="4" />

        {/* Destination */}
        <rect x={mapX(state.total_distance) - 20} y="300" width="40" height="20" fill="#ef4444" rx="4" />
        <text x={mapX(state.total_distance)} y="285" fill="#f8fafc" fontSize="14" textAnchor="middle" className="font-bold">DEST</text>

        {/* Sub-Airports */}
        {state.sub_airports.map((ap, idx) => (
          <g key={idx}>
            <rect x={mapX(ap) - 15} y="305" width="30" height="15" fill="#3b82f6" rx="2" />
            <text x={mapX(ap)} y="340" fill="#cbd5e1" fontSize="12" textAnchor="middle">Sub {idx + 1}</text>
          </g>
        ))}

        {/* Aircraft Trail */}
        <path d={`M ${50} ${mapY(10000)} L ${planeX} ${planeY}`} stroke="#38bdf8" strokeDasharray="5,5" strokeWidth="2" opacity="0.3" />

        {/* Aircraft */}
        <g style={{ transform: `translate(${planeX}px, ${planeY}px) rotate(${state.flight_phase === 'DESCENDING' ? 15 : 0}deg)`, transition: 'all 0.3s linear' }}>
          <circle cx="0" cy="0" r="10" fill="#f8fafc" className="drop-shadow-lg" />
          <polygon points="10,0 -8,-8 -8,8" fill="#f8fafc" />
          <polygon points="-8,-8 -12,-8 -8,0" fill="#cbd5e1" />
        </g>
      </svg>
    </div>
  );
}
