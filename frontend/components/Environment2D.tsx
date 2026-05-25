import React from 'react';
import { SimState, WeatherZone } from '../hooks/useSimulation';

const WEATHER_STYLES: Record<string, { fill: string; stroke: string; label: string; accent: string }> = {
  headwind: { fill: '#d97706', stroke: '#fbbf24', label: 'Headwind', accent: '#fde68a' },
  tailwind: { fill: '#059669', stroke: '#34d399', label: 'Tailwind', accent: '#bbf7d0' },
  storm: { fill: '#6d28d9', stroke: '#facc15', label: 'Storm', accent: '#fef08a' },
  turbulence: { fill: '#ea580c', stroke: '#fb923c', label: 'Turbulence', accent: '#fed7aa' },
};

const HAZARD_LABELS: Record<number, string> = {
  0: 'Clear',
  1: 'Headwind',
  2: 'Tailwind',
  3: 'Storm',
  4: 'Turbulence',
};

type BadgeTone = 'slate' | 'cyan' | 'emerald' | 'amber' | 'violet' | 'rose';

const badgeToneClasses: Record<BadgeTone, string> = {
  slate: 'border-slate-700/70 bg-slate-950/62 text-slate-100',
  cyan: 'border-cyan-400/35 bg-cyan-400/10 text-cyan-50',
  emerald: 'border-emerald-400/35 bg-emerald-400/10 text-emerald-50',
  amber: 'border-amber-300/45 bg-amber-300/12 text-amber-50',
  violet: 'border-violet-400/35 bg-violet-400/10 text-violet-50',
  rose: 'border-rose-400/35 bg-rose-400/10 text-rose-50',
};

const weatherToneMap: Record<string, BadgeTone> = {
  clear: 'emerald',
  headwind: 'amber',
  tailwind: 'emerald',
  storm: 'violet',
  turbulence: 'amber',
};

function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(' ');
}

function getWeatherStyle(type: string) {
  return WEATHER_STYLES[type] ?? { fill: '#64748b', stroke: '#cbd5e1', label: titleCase(type), accent: '#e2e8f0' };
}

function titleCase(value: string) {
  const normalized = value.replace(/_/g, ' ').trim();
  if (!normalized) return 'Unknown';
  return normalized.replace(/\w\S*/g, (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase());
}

function formatMeters(value: number) {
  return `${Math.round(value).toLocaleString()} m`;
}

function compactText(value: string, maxLength: number) {
  const normalized = value.replace(/_/g, ' ').trim();
  return normalized.length > maxLength ? `${normalized.slice(0, Math.max(1, maxLength - 1))}…` : normalized;
}

function formatZoneLabel(label: string, width: number) {
  if (width < 54) return compactText(label, 3).toUpperCase();
  if (width < 86) return compactText(label, 6).toUpperCase();
  if (width < 124) return compactText(label, 9).toUpperCase();
  return compactText(label, 14).toUpperCase();
}

function MapBadge({
  label,
  value,
  detail,
  tone = 'slate',
}: {
  label: string;
  value: React.ReactNode;
  detail?: React.ReactNode;
  tone?: BadgeTone;
}) {
  const title = typeof value === 'string' ? value : undefined;

  return (
    <div className={cx('min-w-0 rounded-2xl border px-3.5 py-3 shadow-lg shadow-slate-950/20 ring-1 ring-white/[0.03] backdrop-blur-md', badgeToneClasses[tone])}>
      <span className="block truncate text-[0.66rem] font-bold uppercase tracking-[0.14em] text-slate-400">{label}</span>
      <span className="mt-1 block min-w-0 truncate text-sm font-black tracking-tight sm:text-base" title={title}>
        {value}
      </span>
      {detail && <span className="mt-1 block truncate text-[0.72rem] font-medium text-slate-400">{detail}</span>}
    </div>
  );
}

function AircraftSvg({ isHazard }: { isHazard: boolean }) {
  return (
    <g className={isHazard ? 'aircraft-shake' : undefined}>
      <ellipse cx="0" cy="0" rx="36" ry="8.5" fill="#f8fafc" />
      <path d="M 31 -6.5 L 49 0 L 31 6.5 Z" fill="#e2e8f0" />
      <path d="M -7 -7.5 L -28 -40 L 7 -8.5 Z" fill="#dbeafe" stroke="#93c5fd" strokeWidth="1.1" />
      <path d="M -7 7.5 L -28 40 L 7 8.5 Z" fill="#dbeafe" stroke="#93c5fd" strokeWidth="1.1" />
      <path d="M -31 -5.5 L -45 -24 L -25 -7.5 Z" fill="#cbd5e1" />
      <path d="M -31 5.5 L -45 24 L -25 7.5 Z" fill="#cbd5e1" />
      <circle cx="24" cy="0" r="3.2" fill="#0f172a" opacity="0.56" />
      <rect x="-19" y="-3.3" width="24" height="6.6" rx="3.3" fill="#38bdf8" opacity="0.86" />
      <circle cx="-9" cy="0" r="2" fill="#0f172a" opacity="0.36" />
      <circle cx="-1" cy="0" r="2" fill="#0f172a" opacity="0.36" />
      <path d="M -46 0 L -61 -5.5 L -58 0 L -61 5.5 Z" fill="#38bdf8" opacity="0.58" />
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
  const cols = Array.from({ length: Math.max(2, Math.floor(width / 48)) });

  if (type === 'storm') {
    return (
      <g clipPath={`url(#${clipId})`}>
        <rect x={x} y={y} width={width} height={height} fill="#facc15" opacity="0.045" className="storm-flash" />
        {cols.map((_, col) => {
          const lx = x + 18 + col * 52;
          const ly = y + 18 + ((col % 2) * 24);
          return (
            <path
              key={`storm-${col}`}
              d={`M ${lx} ${ly} L ${lx + 11} ${ly + 22} L ${lx + 3} ${ly + 22} L ${lx + 19} ${ly + 54}`}
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
            d={`M ${x + 10} ${y + 22 + row * 34} H ${x + width - 12}`}
            stroke="#c4b5fd"
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
            d={`M ${x + 10} ${y + 21 + row * 32} C ${x + 36} ${y + 4 + row * 32}, ${x + 64} ${y + 36 + row * 32}, ${x + 96} ${y + 21 + row * 32} S ${x + 164} ${y + 21 + row * 32}, ${x + width - 10} ${y + 21 + row * 32}`}
            fill="none"
            stroke="#fed7aa"
            strokeWidth="2.2"
            strokeDasharray="14 14"
            opacity="0.54"
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
          const ax = x + 20 + col * 50;
          const ay = y + 22 + row * 34;
          const endX = reverse ? ax - 22 : ax + 22;
          return (
            <g key={`wind-${row}-${col}`} className={reverse ? 'weather-flow-reverse' : 'weather-flow'}>
              <line x1={ax} y1={ay} x2={endX} y2={ay} stroke="#ecfeff" strokeWidth="2" opacity="0.48" />
              <path
                d={reverse ? `M ${endX} ${ay} L ${endX + 7} ${ay - 5} L ${endX + 7} ${ay + 5} Z` : `M ${endX} ${ay} L ${endX - 7} ${ay - 5} L ${endX - 7} ${ay + 5} Z`}
                fill="#ecfeff"
                opacity="0.48"
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
      <section className="aviation-panel flex h-[520px] min-w-0 items-center justify-center overflow-hidden rounded-[1.5rem] text-sm font-semibold uppercase tracking-[0.16em] text-white/55">
        <span className="animate-pulse truncate px-6 text-center">Connecting to simulation...</span>
      </section>
    );
  }

  const width = 1120;
  const height = 540;
  const routeLeft = 88;
  const routeRight = 1046;
  const routeTop = 72;
  const routeBottom = 410;
  const routeSpan = routeRight - routeLeft;
  const maxAltitude = 5000;
  const routeDistance = Math.max(1, state.total_distance || 1);
  const weatherZones = state.weather_zones ?? [];
  const subAirports = state.sub_airports ?? [];
  const fuelMultiplier = Number.isFinite(state.weather_fuel_multiplier) ? state.weather_fuel_multiplier : 1;
  const rulMultiplier = Number.isFinite(state.weather_rul_multiplier) ? state.weather_rul_multiplier : 1;
  const weatherSpeedMultiplier = Number.isFinite(state.weather_speed_multiplier) ? state.weather_speed_multiplier : 1;
  const windStrength = Number.isFinite(state.wind_strength) ? state.wind_strength : 0;

  const mapX = (dist: number) => routeLeft + (Math.max(0, Math.min(routeDistance, dist)) / routeDistance) * routeSpan;
  const mapY = (alt: number) => routeBottom - (Math.max(0, Math.min(maxAltitude, alt)) / maxAltitude) * (routeBottom - routeTop);

  const currentPos = Math.max(0, Math.min(routeDistance, routeDistance - state.dist_dest));
  const currentWeather = (state.weather || 'clear').toLowerCase();
  const currentWeatherLabel = titleCase(currentWeather);
  const inHazard = currentWeather === 'storm' || currentWeather === 'turbulence';
  const planeX = mapX(currentPos);
  const planeY = mapY(state.altitude);
  const phase = state.flight_phase || 'CRUISING';
  const planeRotation = phase === 'DESCENDING' ? 9 : phase === 'CLIMBING' ? -9 : 0;
  const nextHazard = HAZARD_LABELS[state.next_hazard_type] ?? 'Unknown';
  const nextHazardTone: BadgeTone = nextHazard === 'Clear' ? 'emerald' : nextHazard === 'Storm' ? 'violet' : 'amber';
  const weatherTone = weatherToneMap[currentWeather] ?? 'slate';
  const altitudeMarks = [0, 1250, 2500, 3750, 5000];
  const distanceMarks = [0, 0.25, 0.5, 0.75, 1];
  const departureY = mapY(2100);
  const actualPath = `M ${routeLeft} ${departureY} C ${routeLeft + routeSpan * 0.23} ${mapY(3300)}, ${routeLeft + routeSpan * 0.5} ${mapY(4550)}, ${planeX} ${planeY}`;
  const projectedPath = `M ${planeX} ${planeY} C ${Math.min(routeRight, planeX + routeSpan * 0.16)} ${Math.max(routeTop, planeY - 20)}, ${routeRight - routeSpan * 0.18} ${mapY(2300)}, ${routeRight} ${routeBottom}`;

  return (
    <section className="aviation-panel min-w-0 overflow-hidden rounded-[1.5rem] shadow-2xl">
      <div className="border-b border-slate-700/65 px-4 py-4 sm:px-5 lg:px-6">
        <div className="flex min-w-0 flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
          <div className="min-w-0">
            <p className="truncate text-[0.7rem] font-bold uppercase tracking-[0.18em] text-cyan-300/80">2D flight environment</p>
            <h2 className="mt-1 text-xl font-black tracking-tight text-white sm:text-2xl">Route, altitude and weather map</h2>
            <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-400">
              Live position is plotted against route distance and altitude. Weather cells are separated from controls to keep the operating picture readable.
            </p>
          </div>
          <div className="grid min-w-0 grid-cols-2 gap-2 sm:grid-cols-4 xl:w-[39rem]">
            <MapBadge label="Phase" value={phase} tone="cyan" detail="Flight mode" />
            <MapBadge label="Weather" value={currentWeatherLabel} tone={weatherTone} detail={`Wind ${windStrength.toFixed(1)}`} />
            <MapBadge label="Next hazard" value={nextHazard} tone={nextHazardTone} detail={formatMeters(state.next_hazard_distance)} />
            <MapBadge label="Load" value={`F ${fuelMultiplier.toFixed(2)} / R ${rulMultiplier.toFixed(2)}`} tone={inHazard ? 'amber' : 'slate'} detail={`Speed x${weatherSpeedMultiplier.toFixed(2)}`} />
          </div>
        </div>
      </div>

      <div className="p-3 sm:p-4 lg:p-5">
        <div className="overflow-hidden rounded-[1.25rem] border border-slate-700/70 bg-slate-950/45 shadow-inner">
          <svg viewBox={`0 0 ${width} ${height}`} className="block h-auto w-full" role="img" aria-label="Aircraft route simulation with weather zones" preserveAspectRatio="xMidYMid meet">
            <defs>
              <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#102a43" />
                <stop offset="46%" stopColor="#081827" />
                <stop offset="100%" stopColor="#0f172a" />
              </linearGradient>
              <linearGradient id="groundGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#14532d" />
                <stop offset="50%" stopColor="#22c55e" />
                <stop offset="100%" stopColor="#14532d" />
              </linearGradient>
              <linearGradient id="routeLineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.25" />
                <stop offset="55%" stopColor="#67e8f9" stopOpacity="0.9" />
                <stop offset="100%" stopColor="#c4b5fd" stopOpacity="0.42" />
              </linearGradient>
              <radialGradient id="radarGlow" cx="50%" cy="42%" r="62%">
                <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.13" />
                <stop offset="48%" stopColor="#22d3ee" stopOpacity="0.035" />
                <stop offset="100%" stopColor="#020617" stopOpacity="0" />
              </radialGradient>
              <filter id="aircraftGlow" x="-90%" y="-90%" width="280%" height="280%">
                <feGaussianBlur stdDeviation="5" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            <rect width={width} height={height} fill="url(#skyGradient)" />
            <rect width={width} height={height} fill="url(#radarGlow)" />
            <rect x={routeLeft} y={routeTop} width={routeSpan} height={routeBottom - routeTop} rx="18" fill="#020617" opacity="0.16" stroke="#334155" strokeWidth="1" />

            <g opacity="0.18">
              {distanceMarks.map((pct) => {
                const x = routeLeft + pct * routeSpan;
                return <line key={`vgrid-${pct}`} x1={x} x2={x} y1={routeTop} y2={routeBottom} stroke="#67e8f9" strokeWidth="1" strokeDasharray="4 16" />;
              })}
            </g>

            <g>
              {altitudeMarks.map((alt) => {
                const y = mapY(alt);
                return (
                  <g key={alt}>
                    <line x1={routeLeft} x2={routeRight} y1={y} y2={y} stroke="#94a3b8" strokeWidth="1" strokeDasharray="6 14" opacity="0.26" />
                    <text x={routeLeft - 58} y={y + 5} fill="#94a3b8" fontSize="13" fontWeight="800" textAnchor="start">{alt} m</text>
                  </g>
                );
              })}
            </g>

            <text x={routeLeft} y={routeTop - 28} fill="#cbd5e1" fontSize="13" fontWeight="900" letterSpacing="1.4">ALTITUDE PROFILE</text>
            <text x={routeRight} y={routeTop - 28} fill="#64748b" fontSize="11" fontWeight="800" textAnchor="end" letterSpacing="1.2">DISTANCE TO DESTINATION</text>

            {weatherZones.map((zone, idx) => {
              const style = getWeatherStyle(zone.type);
              const x = mapX(zone.start);
              const zoneWidth = Math.max(12, mapX(zone.end) - x);
              const y = mapY(zone.alt_max);
              const zoneHeight = Math.max(18, mapY(zone.alt_min) - y);
              const clipId = `weather-zone-${idx}`;
              const isCurrentZone = currentPos >= zone.start && currentPos <= zone.end && state.altitude >= zone.alt_min && state.altitude <= zone.alt_max;
              const label = formatZoneLabel(style.label, zoneWidth);
              const chipWidth = Math.max(30, Math.min(zoneWidth - 12, label.length * 7.2 + 18));

              return (
                <g key={`${zone.type}-${idx}`}>
                  <clipPath id={clipId}>
                    <rect x={x} y={y} width={zoneWidth} height={zoneHeight} rx="14" />
                  </clipPath>
                  <rect
                    x={x}
                    y={y}
                    width={zoneWidth}
                    height={zoneHeight}
                    rx="14"
                    fill={style.fill}
                    opacity={isCurrentZone ? 0.34 : 0.18}
                    stroke={style.stroke}
                    strokeWidth={isCurrentZone ? 3 : 1.4}
                  />
                  {isCurrentZone && <rect x={x + 3} y={y + 3} width={Math.max(6, zoneWidth - 6)} height={Math.max(6, zoneHeight - 6)} rx="12" fill="none" stroke={style.accent} strokeWidth="1.4" strokeDasharray="7 9" opacity="0.7" />}
                  <WeatherGlyphs zone={zone} x={x} y={y} width={zoneWidth} height={zoneHeight} clipId={clipId} />
                  {zoneWidth > 44 && zoneHeight > 27 && (
                    <g>
                      <rect x={x + 7} y={y + 8} width={chipWidth} height="20" rx="8" fill="#020617" opacity="0.66" />
                      <text x={x + 16} y={y + 22} fill="#f8fafc" fontSize={zoneWidth < 78 ? 9 : 10.5} fontWeight="900" letterSpacing="0.9">
                        {label}
                      </text>
                    </g>
                  )}
                </g>
              );
            })}

            <path d={`M ${routeLeft} ${routeBottom} H ${routeRight}`} stroke="url(#groundGradient)" strokeWidth="9" strokeLinecap="round" opacity="0.95" />
            <path d={actualPath} fill="none" stroke="url(#routeLineGradient)" strokeWidth="3" strokeDasharray="9 11" opacity="0.72" />
            <path d={actualPath} fill="none" stroke="#cffafe" strokeWidth="0.9" opacity="0.72" />
            <path d={projectedPath} fill="none" stroke="#94a3b8" strokeWidth="1.7" strokeDasharray="5 11" opacity="0.32" />

            {subAirports.map((ap, idx) => {
              const x = mapX(ap);
              return (
                <g key={ap}>
                  <rect x={x - 24} y={routeBottom - 16} width="48" height="16" rx="4" fill="#60a5fa" />
                  <line x1={x - 36} y1={routeBottom + 16} x2={x + 36} y2={routeBottom + 16} stroke="#93c5fd" strokeWidth="3.5" strokeLinecap="round" />
                  <text x={x} y={routeBottom + 42} fill="#bfdbfe" fontSize="13" fontWeight="900" textAnchor="middle">AP {idx + 1}</text>
                </g>
              );
            })}

            <g>
              <rect x={mapX(routeDistance) - 33} y={routeBottom - 18} width="66" height="18" rx="4" fill="#f87171" />
              <line x1={mapX(routeDistance) - 46} y1={routeBottom + 16} x2={mapX(routeDistance) + 46} y2={routeBottom + 16} stroke="#fecaca" strokeWidth="4.5" strokeLinecap="round" />
              <text x={mapX(routeDistance)} y={routeBottom + 43} fill="#fecaca" fontSize="13" fontWeight="950" textAnchor="middle">DEST</text>
            </g>

            {inHazard && (
              <circle cx={planeX} cy={planeY} r="52" fill="none" stroke={currentWeather === 'storm' ? '#facc15' : '#fb923c'} strokeWidth="2.5" strokeDasharray="6 9" opacity="0.78" className="hazard-ring" />
            )}

            <g transform={`translate(${planeX} ${planeY}) rotate(${planeRotation})`} filter="url(#aircraftGlow)">
              <AircraftSvg isHazard={inHazard} />
            </g>

            <g transform={`translate(${routeLeft} ${height - 56})`}>
              <text x="0" y="0" fill="#64748b" fontSize="11" fontWeight="900" letterSpacing="1.3">LEGEND</text>
              {Object.entries(WEATHER_STYLES).map(([key, style], idx) => (
                <g key={key} transform={`translate(${78 + idx * 150} -11)`}>
                  <rect x="0" y="0" width="16" height="16" rx="5" fill={style.fill} stroke={style.stroke} opacity="0.78" />
                  <text x="24" y="12" fill="#cbd5e1" fontSize="12" fontWeight="800">{style.label}</text>
                </g>
              ))}
            </g>
          </svg>
        </div>
      </div>
    </section>
  );
}
