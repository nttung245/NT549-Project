# -*- coding: utf-8 -*-
"""
Simple 2D Simulation for Aircraft Digital Twin (Pygame).
Visualizes the aircraft traveling towards the destination
and handles landing at sub-airports.
"""

import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pygame
from scripts.data_processor import prepare_data, FEATURES, KEY_SENSORS
from scripts.lstm_model import create_sequences, train_model
from scripts.aircraft_env import AircraftEnv

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


def draw_env(screen, env, font, step, action, total_reward):
    screen.fill(WHITE)
    
    # Coordinates mapping
    def map_x(distance_from_start):
        return int(50 + (distance_from_start / env.TOTAL_DISTANCE) * (SCREEN_WIDTH - 100))
        
    def map_y(altitude):
        # 10000m = top (100), 0m = bottom (500)
        return int(500 - (altitude / 10000.0) * 400)

    # Ground line
    pygame.draw.line(screen, GREEN, (50, 500), (SCREEN_WIDTH - 50, 500), 5)
    
    # Destination Airport
    dest_rect = pygame.Rect(map_x(env.TOTAL_DISTANCE) - 20, 480, 40, 20)
    pygame.draw.rect(screen, RED, dest_rect)
    dest_text = font.render(f"Dest", True, BLACK)
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
    air_y = map_y(env.twin.altitude)
    
    pygame.draw.circle(screen, BLACK, (air_x, air_y), 10)
    pygame.draw.polygon(screen, BLACK, [(air_x+10, air_y), (air_x-10, air_y-5), (air_x-10, air_y+5)])
    
    # Status Panel
    status_y = 20
    status_text = f"Step: {step} | Phase: {env.flight_phase} | Total Reward: {total_reward:.0f}"
    screen.blit(font.render(status_text, True, BLACK), (20, status_y))
    
    info_text = f"Alt: {env.twin.altitude:.0f} m | Fuel: {env.twin.fuel:.1f}/{env.FUEL_CAPACITY} | RUL: {env.twin.current_rul if env.twin.current_rul else 0:.1f} cycles"
    screen.blit(font.render(info_text, True, BLACK), (20, status_y + 25))
    
    act_labels = {0: "CRUISE", 1: "DESCEND", 2: "CLIMB"}
    act_str = act_labels.get(action, "NONE")
    act_color = RED if action == 1 else BLUE
    act_surf = font.render(f"Last Action: {act_str}", True, act_color)
    screen.blit(act_surf, (20, status_y + 50))
    
    # Fuel & RUL bars
    fuel_pct = max(0, env.twin.fuel / env.FUEL_CAPACITY)
    pygame.draw.rect(screen, GRAY, (600, 20, 200, 20))
    pygame.draw.rect(screen, YELLOW, (600, 20, int(200 * fuel_pct), 20))
    screen.blit(font.render("Fuel", True, BLACK), (560, 20))
    
    try:
        rul_pct = max(0, min(1.0, env.twin.current_rul / 150.0)) if env.twin.current_rul else 1.0
    except:
        rul_pct = 1.0
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

    # 2. Load LSTM
    model_path = os.path.join(PROJECT_ROOT, 'models', 'lstm_rul_model.keras')
    if os.path.exists(model_path):
        import tensorflow as tf
        lstm_model = tf.keras.models.load_model(model_path)
    else:
        print("Error: Please format model first. Missing lstm_rul_model.keras")
        return

    # 3. Create Environment
    env = AircraftEnv(
        fleet_data=train_rolling,
        model_path=lstm_model,
        scaler=scaler,
        sensor_list=KEY_SENSORS,
        features_list=FEATURES
    )

    # 4. Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Aircraft Predictive Maintenance 2D Sim")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Consolas', 18)

    obs, info = env.reset()
    
    running = True
    done = False
    step_count = 0
    total_reward = 0
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
                    # Step 1 tick manually
                    last_action = 0 # Fly
                    obs, r, done, t, i = env.step(0)
                    total_reward += r
                    step_count += 1
                elif event.key == pygame.K_DOWN and not auto_run and not done:
                    last_action = 1 # Land
                    obs, r, done, t, i = env.step(1)
                    total_reward += r
                    step_count += 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    done = False
                    total_reward = 0
                    step_count = 0
                    last_action = -1

        if auto_run and not done:
            # Policy: Dummy heuristic or random action for demo
            # Let's use a very basic heuristic:
            dist_closest = env._dist_to_nearest_airport()
            # env.twin is set after reset(), so we can safely access it
            current_rul = env.twin.current_rul if env.twin and env.twin.current_rul else 150  # type: ignore
            action = 0  # fly
            
            # If RUL or fuel is dangerously low and we are close to an airport
            if (current_rul < 30 or (env.twin and env.twin.fuel < 20)) and env.flight_phase == "CRUISING":  # type: ignore
                if dist_closest < env.LANDING_THRESHOLD:
                    action = 1
            
            # If we are already descending, action doesn't matter much
            
            last_action = action
            obs, r, done, t, i = env.step(action)
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
