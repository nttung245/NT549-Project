# -*- coding: utf-8 -*-
"""
Simple 2D Simulation for Aircraft Digital Twin (Pygame).
Visualizes the aircraft traveling towards the destination,
sub-airports, and static weather zones.
"""

import math
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import pygame
from scripts.data.data_processor import prepare_data, FEATURES, KEY_SENSORS
from scripts.core.aircraft_env import AircraftEnv

# Constants for drawing
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 150, 255)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
GRAY = (150, 150, 150)
YELLOW = (255, 220, 0)

WEATHER_ZONE_COLORS = {
    "headwind": (255, 165, 0, 65),
    "tailwind": (60, 200, 120, 65),
    "storm": (120, 80, 200, 85),
    "turbulence": (240, 120, 50, 65),
}

WEATHER_FLOW_DIRECTIONS = {
    "headwind": -1,
    "tailwind": 1,
    "storm": -1,
    "turbulence": 1,
}

WEATHER_FLOW_SPEEDS = {
    "headwind": 7,
    "tailwind": 9,
    "storm": 4,
    "turbulence": 12,
}


def draw_weather_flow(screen, rect, zone_type, step):
    """Draw animated flow streaks inside a static weather zone.

    This is intentionally visual-only: weather physics remains static in
    AircraftEnv, while the zone overlay communicates wind/turbulence motion.
    """
    if rect.width <= 3 or rect.height <= 3:
        return

    direction = WEATHER_FLOW_DIRECTIONS.get(zone_type, 1)
    speed = WEATHER_FLOW_SPEEDS.get(zone_type, 6)
    spacing = 42
    shift = int((step * speed) % spacing)
    if direction < 0:
        shift = -shift

    line_color = (30, 30, 30)
    y_spacing = max(12, min(24, rect.height // 3 if rect.height >= 36 else 12))

    for y in range(rect.top + 8, rect.bottom - 5, y_spacing):
        for base_x in range(rect.left - spacing, rect.right + spacing, spacing):
            x = base_x + shift
            if not (rect.left <= x <= rect.right):
                continue

            if zone_type == "storm":
                points = [
                    (x, y),
                    (x + 6, min(rect.bottom - 2, y + 7)),
                    (x + 1, min(rect.bottom - 2, y + 7)),
                    (x + 8, min(rect.bottom - 2, y + 17)),
                ]
                pygame.draw.lines(screen, line_color, False, points, 2)
            elif zone_type == "turbulence":
                amplitude = 4
                points = []
                for i in range(0, 20, 4):
                    wave_y = y + int(math.sin((step + i) * 0.8) * amplitude)
                    points.append((x + i, max(rect.top + 2, min(rect.bottom - 2, wave_y))))
                if len(points) >= 2:
                    pygame.draw.lines(screen, line_color, False, points, 2)
            else:
                end_x = x + direction * 20
                end_x = max(rect.left + 2, min(rect.right - 2, end_x))
                pygame.draw.line(screen, line_color, (x, y), (end_x, y), 2)
                arrow_x = end_x
                pygame.draw.polygon(
                    screen,
                    line_color,
                    [
                        (arrow_x, y),
                        (arrow_x - direction * 6, y - 4),
                        (arrow_x - direction * 6, y + 4),
                    ],
                )


def draw_env(screen, env, font, step, action, total_reward):
    screen.fill(WHITE)

    twin = env.twin
    if twin is None:
        return

    # Coordinates mapping
    def map_x(distance_from_start):
        return int(50 + (distance_from_start / env.TOTAL_DISTANCE) * (SCREEN_WIDTH - 100))

    def map_y(altitude):
        # MAX_ALTITUDE = top (100), 0m = bottom (500)
        return int(500 - (altitude / env.MAX_ALTITUDE) * 400)

    # Ground line
    pygame.draw.line(screen, GREEN, (50, 500), (SCREEN_WIDTH - 50, 500), 5)

    # Static weather zones with visual-only animated drift/flow streaks.
    for zone in env.weather_map.zones:
        x1 = map_x(zone.start)
        x2 = map_x(zone.end)
        y_top = map_y(zone.alt_max)
        y_bottom = map_y(zone.alt_min)
        rect = pygame.Rect(x1, y_top, max(2, x2 - x1), max(2, y_bottom - y_top))
        color = WEATHER_ZONE_COLORS.get(zone.zone_type, (180, 180, 180, 50))
        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        overlay.fill(color)
        screen.blit(overlay, rect.topleft)
        pygame.draw.rect(screen, color[:3], rect, 1)
        draw_weather_flow(screen, rect, zone.zone_type, step)
        screen.blit(font.render(zone.zone_type, True, BLACK), (rect.x + 3, rect.y + 3))

    # Destination Airport
    dest_rect = pygame.Rect(map_x(env.TOTAL_DISTANCE) - 20, 480, 40, 20)
    pygame.draw.rect(screen, RED, dest_rect)
    dest_text = font.render("Dest", True, BLACK)
    screen.blit(dest_text, (dest_rect.x, dest_rect.y - 25))

    # Sub-airports
    for ap in env.sub_airports:
        x_pos = map_x(ap)
        ap_rect = pygame.Rect(x_pos - 15, 485, 30, 15)
        pygame.draw.rect(screen, BLUE, ap_rect)
        ap_text = font.render(f"Dist: {ap:.0f}", True, BLACK)
        screen.blit(ap_text, (x_pos - 20, 510))

    # Aircraft
    current_pos = env.TOTAL_DISTANCE - env.distance_to_destination
    air_x = map_x(current_pos)
    air_y = map_y(twin.altitude)

    obs = env._get_obs()
    effective_rul = float(obs[2])
    current_weather = env._get_current_weather(current_pos, float(twin.altitude))

    shake_x = 0
    shake_y = 0
    if current_weather.zone_type == "turbulence":
        # Visual-only shake: physics still comes from AircraftEnv.
        shake_x = int(math.sin(step * 2.9) * 6)
        shake_y = int(math.cos(step * 4.1) * 4)
        pygame.draw.circle(screen, (240, 120, 50), (air_x, air_y), 17, 2)
        pygame.draw.line(screen, (240, 120, 50), (air_x - 18, air_y - 12), (air_x - 10, air_y - 5), 2)
        pygame.draw.line(screen, (240, 120, 50), (air_x + 18, air_y + 12), (air_x + 10, air_y + 5), 2)

    draw_x = air_x + shake_x
    draw_y = air_y + shake_y
    pygame.draw.circle(screen, BLACK, (draw_x, draw_y), 10)
    pygame.draw.polygon(screen, BLACK, [(draw_x + 10, draw_y), (draw_x - 10, draw_y - 5), (draw_x - 10, draw_y + 5)])

    # Status Panel
    status_y = 20
    status_text = f"Step: {step} | Phase: {env.flight_phase} | Total Reward: {total_reward:.0f}"
    screen.blit(font.render(status_text, True, BLACK), (20, status_y))

    info_text = (
        f"Alt: {twin.altitude:.0f} m | Fuel: {twin.fuel:.1f}/{env.FUEL_CAPACITY} | "
        f"Eff RUL: {effective_rul:.1f} cycles | Weather: {current_weather.zone_type}"
    )
    screen.blit(font.render(info_text, True, BLACK), (20, status_y + 25))

    weather_text = (
        f"Wind: {obs[11]:+.1f} | Fuel x{obs[12]:.1f} | RUL x{obs[13]:.1f} | "
        f"Next hazard: {obs[14]:.0f}m / type {int(obs[15])} | "
        f"Visual drift: ON"
    )
    screen.blit(font.render(weather_text, True, BLACK), (20, status_y + 50))

    act_labels = {0: "CRUISE", 1: "DESCEND", 2: "CLIMB"}
    act_str = act_labels.get(action, "NONE")
    act_color = RED if action == 1 else BLUE
    act_surf = font.render(f"Last Action: {act_str}", True, act_color)
    screen.blit(act_surf, (20, status_y + 75))

    # Fuel & RUL bars
    fuel_pct = max(0.0, min(1.0, twin.fuel / env.FUEL_CAPACITY))
    pygame.draw.rect(screen, GRAY, (600, 20, 200, 20))
    pygame.draw.rect(screen, YELLOW, (600, 20, int(200 * fuel_pct), 20))
    screen.blit(font.render("Fuel", True, BLACK), (560, 20))

    rul_pct = max(0.0, min(1.0, effective_rul / 150.0))
    rul_color = GREEN if rul_pct > 0.3 else RED
    pygame.draw.rect(screen, GRAY, (600, 50, 200, 20))
    pygame.draw.rect(screen, rul_color, (600, 50, int(200 * rul_pct), 20))
    screen.blit(font.render("RUL", True, BLACK), (560, 50))


def main():
    print("=" * 60)
    print("  Initializing 2D Simulation...")
    print("=" * 60)

    # 1. Load Data
    data_dir = os.path.join(PROJECT_ROOT, 'CMAPSSData')
    train_rolling, _, _, scaler = prepare_data(data_dir)

    # 2. Check LSTM model path. AircraftEnv loads the model internally.
    model_path = os.path.join(PROJECT_ROOT, 'models', 'lstm_rul_model.keras')
    if not os.path.exists(model_path):
        print("Error: Please train/format the model first. Missing lstm_rul_model.keras")
        return

    # 3. Create Environment
    env = AircraftEnv(
        fleet_data=train_rolling,
        model_path=model_path,
        scaler=scaler,
        sensor_list=KEY_SENSORS,
        features_list=FEATURES
    )

    # 4. Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Aircraft Predictive Maintenance 2D Sim - Static Weather + Visual Drift")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Consolas', 18)

    obs, info = env.reset()

    running = True
    done = False
    step_count = 0
    total_reward = 0.0
    last_action = -1
    auto_run = True

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_run = not auto_run  # Toggle play/pause
                elif event.key == pygame.K_RIGHT and not auto_run and not done:
                    last_action = 0  # Cruise
                    obs, r, done, _, _ = env.step(0)
                    total_reward += r
                    step_count += 1
                elif event.key == pygame.K_DOWN and not auto_run and not done:
                    last_action = 1  # Descend
                    obs, r, done, _, _ = env.step(1)
                    total_reward += r
                    step_count += 1
                elif event.key == pygame.K_UP and not auto_run and not done:
                    last_action = 2  # Climb
                    obs, r, done, _, _ = env.step(2)
                    total_reward += r
                    step_count += 1
                elif event.key == pygame.K_r:  # Reset
                    obs, info = env.reset()
                    done = False
                    total_reward = 0.0
                    step_count = 0
                    last_action = -1

        if auto_run and not done:
            # Very simple weather-aware heuristic for visualization only.
            current_obs = env._get_obs()
            current_rul = float(current_obs[2])
            current_fuel = float(current_obs[1])
            dist_closest = env._dist_to_nearest_airport()

            if env.flight_phase == "GROUNDED" or current_obs[0] <= 0:
                action = 2  # Climb after maintenance/ground stop
            elif (current_rul < 30 or current_fuel < 20) and dist_closest < env.LANDING_THRESHOLD:
                action = 1  # Descend when risk is high and an airport is reachable
            else:
                action = 0  # Cruise

            last_action = action
            obs, r, done, _, _ = env.step(action)
            total_reward += r
            step_count += 1

        draw_env(screen, env, font, step_count, last_action, total_reward)
        pygame.display.flip()

        # Slower framerate so user can see it
        clock.tick(FPS)

        if done:
            # Pause on end
            auto_run = False

    pygame.quit()


if __name__ == "__main__":
    main()
