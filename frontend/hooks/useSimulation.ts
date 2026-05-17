import { useState, useEffect, useCallback, useRef } from 'react';

export type WeatherZone = {
  type: 'headwind' | 'tailwind' | 'storm' | 'turbulence' | string;
  type_id: number;
  start: number;
  end: number;
  alt_min: number;
  alt_max: number;
  fuel_multiplier: number;
  speed_multiplier: number;
  rul_multiplier: number;
  wind_strength: number;
};

export type SimState = {
  step: number;
  altitude: number;
  velocity: number;
  fuel: number;
  rul: number;
  dist_next: number;
  dist_dest: number;
  in_approach_zone: boolean;
  wind_strength: number;
  weather_fuel_multiplier: number;
  weather_rul_multiplier: number;
  weather_speed_multiplier: number;
  next_hazard_distance: number;
  next_hazard_type: number;
  next_hazard_alt_min: number;
  next_hazard_alt_max: number;
  next_hazard_speed_multiplier: number;
  weather: string;
  weather_zones: WeatherZone[];
  policy_mode: string;
  maintenance_pressure: number;
  landing_feasible_now: boolean;
  fuel_capacity: number;
  total_distance: number;
  sub_airports: number[];
  flight_phase: string;
  action: string;
  reward: number;
  info: Record<string, unknown>;
};

const DEFAULT_WS_URLS = ['ws://localhost:8001/ws/simulation', 'ws://localhost:8000/ws/simulation'];

export function useSimulation() {
  const [state, setState] = useState<SimState | null>(null);
  const [connected, setConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    const configuredUrl = process.env.NEXT_PUBLIC_SIM_WS_URL;
    const urls = configuredUrl ? [configuredUrl] : DEFAULT_WS_URLS;
    let closedByReact = false;
    let urlIndex = 0;

    const connect = () => {
      const socket = new WebSocket(urls[urlIndex]);

      socket.onopen = () => {
        setConnected(true);
        console.log(`Connected to simulation server at ${urls[urlIndex]}`);
      };

      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'state') {
          setState(msg.data);
        }
      };

      socket.onclose = () => {
        setConnected(false);
        if (!closedByReact && urlIndex < urls.length - 1) {
          urlIndex += 1;
          window.setTimeout(connect, 250);
        } else {
          console.log('Disconnected from simulation server');
        }
      };

      ws.current = socket;
    };

    connect();

    return () => {
      closedByReact = true;
      ws.current?.close();
    };
  }, []);

  const sendCommand = useCallback((cmd: string) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ command: cmd }));
    }
  }, []);

  return { state, connected, sendCommand };
}
