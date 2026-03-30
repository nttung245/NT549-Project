import { useState, useEffect, useCallback, useRef } from 'react';

export type SimState = {
  step: number;
  altitude: number;
  velocity: number;
  fuel: number;
  rul: number;
  dist_next: number;
  dist_dest: number;
  fuel_capacity: number;
  total_distance: number;
  sub_airports: number[];
  flight_phase: string;
  action: string;
  reward: number;
  info: any;
};

export function useSimulation() {
  const [state, setState] = useState<SimState | null>(null);
  const [connected, setConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect to FastAPI Backend
    const socket = new WebSocket('ws://localhost:8000/ws/simulation');
    
    socket.onopen = () => {
      setConnected(true);
      console.log('Connected to simulation server');
    };

    socket.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'state') {
        setState(msg.data);
      }
    };

    socket.onclose = () => {
      setConnected(false);
      console.log('Disconnected from simulation server');
    };

    ws.current = socket;

    return () => {
      socket.close();
    };
  }, []);

  const sendCommand = useCallback((cmd: string) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ command: cmd }));
    }
  }, []);

  return { state, connected, sendCommand };
}
