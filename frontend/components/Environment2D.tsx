import React from 'react';
import { SimState, WeatherZone } from '../hooks/useSimulation';

const WEATHER_STYLES: Record<string, { fill: string; stroke: string; label: string }> = {
  headwind: { fill: '#f59e0b', stroke: '#fbbf24', label: 'HEADWIND' },
  tailwind: { fill: '#10b981', stroke: '#34d399', label: 'TAILWIND' },
  storm: { fill: '#7c3aed', stroke: '#facc15', label: 'STORM' },
  turbulence: { fill: '#f97316', stroke: '#fb923c', label: 'TURBULENCE' },
};

const HAZARD_LABELS: Record<number, string> = {
  0: 'Clear',
  1: 'Headwind',
  2: 'Tailwind',
  3: 'Storm',
  4: 'Turbulence',
};

function getWeatherStyle(type: string) {
  return WEATHER_STYLES[type] ?? { fill: '#94a3b8', stroke: '#cbd5e1', label: type.toUpperCase() };
}

function AircraftSvg({ isHazard }: { isHazard: boolean }) {
  return (
    <g className={isHazard ? 'aircraft-shake' : undefined}>
      <ellipse cx="0" cy="0" rx="34" ry="8" fill="#f8fafc" />
      <path d="M 30 -6 L 45 0 L 30 6 Z" fill="#e2e8f0" />
      <path d="M -6 -7 L -25 -38 L 6 -8 Z" fill="#dbeafe" stroke="#93c5fd" strokeWidth="1" />
      <path d="M -6 7 L -25 38 L 6 8 Z" fill="#dbeafe" stroke="#93c5fd" strokeWidth="1" />
      <path d="M -30 -5 L -43 -22 L -24 -7 Z" fill="#cbd5e1" />
      <path d="M -30 5 L -43 22 L -24 7 Z" fill="#cbd5e1" />
      <circle cx="23" cy="0" r="3" fill="#0f172a" opacity="0.55" />
      <rect x="-18" y="-3" width="23" height="6" rx="3" fill="#38bdf8" opacity="0.85" />
      <circle cx="-8" cy="0" r="2" fill="#0f172a" opacity="0.35" />
      <circle cx="-1" cy="0" r="2" fill="#0f172a" opacity="0.35" />
      <path d="M -44 0 L -58 -5 L -55 0 L -58 5 Z" fill="#38bdf8" opacity="0.55" />
    </g>
  );
}

function WeatherGlyphs({
  zone,
  x,
  y,
  width,
  height,
  clipId,
}: {
  zone: WeatherZone;
  x: number;
  y: number;
  width: number;
  height: number;
  clipId: string;
}) {
  const type = zone.type;
  const rows = Array.from({ length: Math.max(1, Math.floor(height / 34)) });
  const cols = Array.from({ length: Math.max(2, Math.floor(width / 46)) });

  if (type === 'storm') {
    return (
      <g clipPath={`url(#${clipId})`}>
        <rect x={x} y={y} width={width} height={height} fill="#facc15" opacity="0.04" className="storm-flash" />
        {cols.map((_, col) => {
          const lx = x + 18 + col * 48;
          const ly = y + 16 + ((col % 2) * 22);
          return (
            <path
              key={`storm-${col}`}
              d={`M ${lx} ${ly} L ${lx + 11} ${ly + 22} L ${lx + 3} ${ly + 22} L ${lx + 18} ${ly + 52}`}
              fill="none"
              stroke="#fde68a"
              strokeWidth="3"
              strokeLinejoin="round"
              className="lightning-strike"
            />
          );
        })}
        {rows.map((_, row) => (
          <path
            key={`rain-${row}`}
            d={`M ${x + 8} ${y + 18 + row * 34} H ${x + width - 12}`}
            stroke="#a78bfa"
            strokeWidth="2"
            strokeDasharray="10 16"
            opacity="0.42"
            className="weather-flow-reverse"
          />
        ))}
      </g>
    );
  }

  if (type === 'turbulence') {
    return (
      <g clipPath={`url(#${clipId})`}>
        {rows.map((_, row) => (
          <path
            key={`turb-${row}`}
            d={`M ${x + 8} ${y + 18 + row * 32} C ${x + 34} ${y + 3 + row * 32}, ${x + 62} ${y + 33 + row * 32}, ${x + 92} ${y + 18 + row * 32} S ${x + 156} ${y + 18 + row * 32}, ${x + width - 8} ${y + 18 + row * 32}`}
            fill="none"
            stroke="#fed7aa"
            strokeWidth="2"
            strokeDasharray="14 14"
            opacity="0.5"
            className="turbulence-wave"
          />
        ))}
      </g>
    );
  }

  const reverse = type === 'headwind';
  return (
    <g clipPath={`url(#${clipId})`}>
      {rows.map((_, row) =>
        cols.map((_, col) => {
          const ax = x + 18 + col * 48;
          const ay = y + 18 + row * 34;
          const endX = reverse ? ax - 20 : ax + 20;
          return (
            <g key={`wind-${row}-${col}`} className={reverse ? 'weather-flow-reverse' : 'weather-flow'}>
              <line x1={ax} y1={ay} x2={endX} y2={ay} stroke="#ecfeff" strokeWidth="2" opacity="0.5" />
              <path
                d={reverse ? `M ${endX} ${ay} L ${endX + 7} ${ay - 5} L ${endX + 7} ${ay + 5} Z` : `M ${endX} ${ay} L ${endX - 7} ${ay - 5} L ${endX - 7} ${ay + 5} Z`}
                fill="#ecfeff"
                opacity="0.5"
              />
            </g>
          );
        })
      )}
    </g>
  );
}

export default function Environment2D({ state }: { state: SimState | null }) {
  if (!state) {
    return (
      <div className="h-[460px] flex items-center justify-center text-white/50 animate-pulse bg-slate-900 border border-white/10 rounded-lg">
        Connecting to Simulation...
      </div>
    );
  }

  const width = 1000;
  const height = 460;
  const routeTop = 56;
  const routeBottom = 355;
  const maxAltitude = 5000;

  const mapX = (dist: number) => 62 + (dist / state.total_distance) * (width - 124);
  const mapY = (alt: number) => routeBottom - (Math.max(0, Math.min(maxAltitude, alt)) / maxAltitude) * (routeBottom - routeTop);

  const currentPos = state.total_distance - state.dist_dest;
  const currentWeather = state.weather || 'clear';
  const inHazard = currentWeather === 'storm' || currentWeather === 'turbulence';
  const planeX = mapX(currentPos);
  const planeY = mapY(state.altitude);
  const phase = state.flight_phase || 'CRUISING';
  const planeRotation = phase === 'DESCENDING' ? 9 : phase === 'CLIMBING' ? -9 : 0;
  const nextHazard = HAZARD_LABELS[state.next_hazard_type] ?? 'Unknown';
  const weatherSpeedMultiplier = state.weather_speed_multiplier ?? 1;

  return (
    <section className="relative overflow-hidden rounded-lg border border-slate-700 bg-[#07111f] shadow-2xl">
      <div className="absolute left-4 top-4 z-10 flex flex-wrap items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em]">
        <span className="rounded-md border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-cyan-100">{phase}</span>
        <span className={`rounded-md border px-3 py-1 ${inHazard ? 'border-amber-300/60 bg-amber-300/15 text-amber-100' : 'border-emerald-300/30 bg-emerald-300/10 text-emerald-100'}`}>
          Weather: {currentWeather}
        </span>
        <span className="rounded-md border border-slate-500/50 bg-slate-950/45 px-3 py-1 text-slate-200">
          Next: {nextHazard} / {Math.round(state.next_hazard_distance)}m
        </span>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="block h-auto w-full" role="img" aria-label="Aircraft route simulation with weather zones">
        <defs>
          <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#0f2a44" />
            <stop offset="56%" stopColor="#0b1829" />
            <stop offset="100%" stopColor="#111827" />
          </linearGradient>
          <linearGradient id="groundGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#14532d" />
            <stop offset="50%" stopColor="#166534" />
            <stop offset="100%" stopColor="#14532d" />
          </linearGradient>
          <filter id="aircraftGlow" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="5" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <rect width={width} height={height} fill="url(#skyGradient)" />
        <g opacity="0.2">
          {[0, 1250, 2500, 3750, 5000].map((alt) => (
            <g key={alt}>
              <line x1="62" x2={width - 62} y1={mapY(alt)} y2={mapY(alt)} stroke="#94a3b8" strokeWidth="1" strokeDasharray="5 12" />
              <text x="22" y={mapY(alt) + 4} fill="#cbd5e1" fontSize="12">{alt}m</text>
            </g>
          ))}
        </g>

        {state.weather_zones.map((zone, idx) => {
          const style = getWeatherStyle(zone.type);
          const x = mapX(zone.start);
          const zoneWidth = Math.max(12, mapX(zone.end) - x);
          const y = mapY(zone.alt_max);
          const zoneHeight = Math.max(18, mapY(zone.alt_min) - y);
          const clipId = `weather-zone-${idx}`;
          const isCurrentZone = currentPos >= zone.start && currentPos <= zone.end && state.altitude >= zone.alt_min && state.altitude <= zone.alt_max;

          return (
            <g key={`${zone.type}-${idx}`}>
              <clipPath id={clipId}>
                <rect x={x} y={y} width={zoneWidth} height={zoneHeight} rx="8" />
              </clipPath>
              <rect
                x={x}
                y={y}
                width={zoneWidth}
                height={zoneHeight}
                rx="8"
                fill={style.fill}
                opacity={isCurrentZone ? 0.34 : 0.2}
                stroke={style.stroke}
                strokeWidth={isCurrentZone ? 3 : 1.5}
              />
              <WeatherGlyphs zone={zone} x={x} y={y} width={zoneWidth} height={zoneHeight} clipId={clipId} />
              <text x={x + 8} y={y + 17} fill="#f8fafc" fontSize="11" fontWeight="700" letterSpacing="1">
                {style.label}
              </text>
            </g>
          );
        })}

        <path d={`M 62 ${routeBottom} H ${width - 62}`} stroke="url(#groundGradient)" strokeWidth="9" strokeLinecap="round" />
        <path d={`M 62 ${mapY(2000)} C 270 ${mapY(5200)}, 560 ${mapY(6300)}, ${planeX} ${planeY}`} fill="none" stroke="#67e8f9" strokeWidth="2" strokeDasharray="8 12" opacity="0.4" />

        {state.sub_airports.map((ap, idx) => {
          const x = mapX(ap);
          return (
            <g key={ap}>
              <rect x={x - 20} y={routeBottom - 10} width="40" height="10" rx="2" fill="#60a5fa" />
              <line x1={x - 30} y1={routeBottom + 10} x2={x + 30} y2={routeBottom + 10} stroke="#93c5fd" strokeWidth="3" strokeLinecap="round" />
              <text x={x} y={routeBottom + 32} fill="#bfdbfe" fontSize="12" fontWeight="600" textAnchor="middle">AP {idx + 1}</text>
            </g>
          );
        })}

        <g>
          <rect x={mapX(state.total_distance) - 28} y={routeBottom - 14} width="56" height="14" rx="2" fill="#f87171" />
          <line x1={mapX(state.total_distance) - 40} y1={routeBottom + 10} x2={mapX(state.total_distance) + 40} y2={routeBottom + 10} stroke="#fecaca" strokeWidth="4" strokeLinecap="round" />
          <text x={mapX(state.total_distance)} y={routeBottom + 34} fill="#fecaca" fontSize="12" fontWeight="700" textAnchor="middle">DEST</text>
        </g>

        {inHazard && (
          <circle cx={planeX} cy={planeY} r="46" fill="none" stroke={currentWeather === 'storm' ? '#facc15' : '#fb923c'} strokeWidth="2" strokeDasharray="5 8" opacity="0.75" className="hazard-ring" />
        )}

        <g transform={`translate(${planeX} ${planeY}) rotate(${planeRotation})`} filter="url(#aircraftGlow)">
          <AircraftSvg isHazard={inHazard} />
        </g>

        <g transform={`translate(${width - 280} 28)`}>
          <rect width="244" height="84" rx="8" fill="#020617" opacity="0.62" stroke="#334155" />
          <text x="16" y="26" fill="#cbd5e1" fontSize="12" fontWeight="700">Weather load</text>
          <text x="16" y="52" fill="#f8fafc" fontSize="18" fontWeight="700">Fuel x{state.weather_fuel_multiplier.toFixed(2)}</text>
          <text x="136" y="52" fill="#f8fafc" fontSize="18" fontWeight="700">RUL x{state.weather_rul_multiplier.toFixed(2)}</text>
          <text x="16" y="74" fill="#93c5fd" fontSize="12">Wind {state.wind_strength.toFixed(1)} / Speed x{weatherSpeedMultiplier.toFixed(2)}</text>
        </g>
      </svg>
    </section>
  );
}
