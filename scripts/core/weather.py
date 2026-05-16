# -*- coding: utf-8 -*-
"""
Static weather system for the aircraft reinforcement-learning environment.

The first version intentionally keeps weather zones static during an episode.
Each zone is defined by a route interval and an altitude band. If the aircraft
is inside both, the zone modifies fuel burn, horizontal speed, and effective
engine-risk signals exposed to the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


WEATHER_TYPE_IDS: dict[str, int] = {
    "clear": 0,
    "headwind": 1,
    "tailwind": 2,
    "storm": 3,
    "turbulence": 4,
}


@dataclass(frozen=True)
class WeatherEffect:
    """Weather effect observed at one aircraft position and altitude."""

    zone_type: str = "clear"
    type_id: int = WEATHER_TYPE_IDS["clear"]
    fuel_multiplier: float = 1.0
    speed_multiplier: float = 1.0
    rul_multiplier: float = 1.0
    wind_strength: float = 0.0


@dataclass(frozen=True)
class WeatherZone:
    """A static weather zone with a route interval and altitude band."""

    zone_type: str
    start: float
    end: float
    alt_min: float
    alt_max: float
    fuel_multiplier: float
    speed_multiplier: float
    rul_multiplier: float
    wind_strength: float

    @property
    def type_id(self) -> int:
        return WEATHER_TYPE_IDS[self.zone_type]

    def contains(self, distance: float, altitude: float) -> bool:
        return self.start <= distance <= self.end and self.alt_min <= altitude <= self.alt_max

    def distance_from(self, distance: float) -> float:
        """Return signed distance from an aircraft position to this zone."""
        if distance < self.start:
            return self.start - distance
        if distance > self.end:
            return self.end - distance
        return 0.0

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "type": self.zone_type,
            "type_id": self.type_id,
            "start": float(self.start),
            "end": float(self.end),
            "alt_min": float(self.alt_min),
            "alt_max": float(self.alt_max),
            "fuel_multiplier": float(self.fuel_multiplier),
            "speed_multiplier": float(self.speed_multiplier),
            "rul_multiplier": float(self.rul_multiplier),
            "wind_strength": float(self.wind_strength),
        }


class WeatherMap:
    """A collection of static weather zones for one episode."""

    EFFECTS: dict[str, WeatherEffect] = {
        "headwind": WeatherEffect(
            zone_type="headwind",
            type_id=WEATHER_TYPE_IDS["headwind"],
            fuel_multiplier=1.5,
            speed_multiplier=0.80,
            rul_multiplier=1.0,
            wind_strength=-1.0,
        ),
        "tailwind": WeatherEffect(
            zone_type="tailwind",
            type_id=WEATHER_TYPE_IDS["tailwind"],
            fuel_multiplier=0.7,
            speed_multiplier=1.15,
            rul_multiplier=1.0,
            wind_strength=1.0,
        ),
        "storm": WeatherEffect(
            zone_type="storm",
            type_id=WEATHER_TYPE_IDS["storm"],
            fuel_multiplier=2.0,
            speed_multiplier=0.90,
            rul_multiplier=1.5,
            wind_strength=-0.6,
        ),
        "turbulence": WeatherEffect(
            zone_type="turbulence",
            type_id=WEATHER_TYPE_IDS["turbulence"],
            fuel_multiplier=1.0,
            speed_multiplier=0.95,
            rul_multiplier=1.3,
            wind_strength=0.0,
        ),
    }

    SPAWN_TYPES: tuple[str, ...] = ("headwind", "tailwind", "storm", "turbulence")
    SPAWN_WEIGHTS = np.array([0.35, 0.35, 0.20, 0.10], dtype=np.float64)

    def __init__(self, zones: list[WeatherZone] | None = None):
        self.zones = sorted(zones or [], key=lambda zone: zone.start)

    @classmethod
    def generate(
        cls,
        route_length: float,
        max_altitude: float,
        rng: Any,
        min_zones: int = 2,
        max_zones: int = 5,
    ) -> "WeatherMap":
        """Generate non-overlapping static weather zones for one episode."""
        zone_count = int(rng.integers(min_zones, max_zones + 1))
        candidate_centers = np.linspace(route_length * 0.12, route_length * 0.88, zone_count)
        jitter_limit = route_length * 0.035
        zones: list[WeatherZone] = []

        for center in candidate_centers:
            zone_type = str(rng.choice(cls.SPAWN_TYPES, p=cls.SPAWN_WEIGHTS))
            width = float(rng.uniform(route_length * 0.035, route_length * 0.075))
            jitter = float(rng.uniform(-jitter_limit, jitter_limit))
            start = max(0.0, center + jitter - width / 2.0)
            end = min(route_length, center + jitter + width / 2.0)

            band_height = float(rng.uniform(max_altitude * 0.22, max_altitude * 0.42))
            alt_min = float(rng.uniform(0.0, max_altitude - band_height))
            alt_max = alt_min + band_height

            effect = cls.EFFECTS[zone_type]
            zones.append(
                WeatherZone(
                    zone_type=zone_type,
                    start=start,
                    end=end,
                    alt_min=alt_min,
                    alt_max=alt_max,
                    fuel_multiplier=effect.fuel_multiplier,
                    speed_multiplier=effect.speed_multiplier,
                    rul_multiplier=effect.rul_multiplier,
                    wind_strength=effect.wind_strength,
                )
            )

        return cls(zones)

    def get_weather_at(self, distance: float, altitude: float) -> WeatherEffect:
        """Return current weather effect at the aircraft route position."""
        for zone in self.zones:
            if zone.contains(distance, altitude):
                return WeatherEffect(
                    zone_type=zone.zone_type,
                    type_id=zone.type_id,
                    fuel_multiplier=zone.fuel_multiplier,
                    speed_multiplier=zone.speed_multiplier,
                    rul_multiplier=zone.rul_multiplier,
                    wind_strength=zone.wind_strength,
                )
        return WeatherEffect()

    def get_next_zone(self, distance: float) -> WeatherZone | None:
        """Return the nearest weather zone strictly ahead, ignoring altitude."""
        ahead = [zone for zone in self.zones if zone.start > distance]
        if not ahead:
            return None
        return min(ahead, key=lambda zone: zone.start - distance)


    def to_dicts(self) -> list[dict[str, float | int | str]]:
        return [zone.to_dict() for zone in self.zones]
