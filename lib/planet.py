from dataclasses import dataclass, field
from typing import List, Tuple
import math
from lib.celestialBody import MyCelestialBody
from game_types.planet import PlanetType
from lib.sun import SUN_X, SUN_Y, angle_from_sun, distance_from_sun


@dataclass
class MyPlanet(MyCelestialBody):
    owner: int
    production: int
    ships: int
    angular_velocity: float
    planet_type: PlanetType = field(init=False)

    init_x: float = field(init=False)
    init_y: float = field(init=False)
    init_angle: float = field(init=False)
    init_orbital_radius: float = field(init=False)

    def __post_init__(self):
        self.init_x = self.x
        self.init_y = self.y
        
        self.init_orbital_radius = distance_from_sun(self.init_x, self.init_y)
        self.init_angle = angle_from_sun(self.init_x, self.init_y)
        
        self.planet_type = (
            PlanetType.STATIC
            if self.init_orbital_radius + self.radius < 50
            else PlanetType.ORBITAL
        )

    @classmethod
    def from_obs(cls, raw_data: List, angular_velocity: float = 0.0) -> "MyPlanet":
        if len(raw_data) != 7:
            raise ValueError(f"Expected 7 fields for planet observation, got {len(raw_data)}")

        planet_id, owner, x, y, radius, ships, production = raw_data
        return cls(
            id=int(planet_id),
            x=float(x),
            y=float(y),
            radius=float(radius),
            owner=int(owner),
            production=int(production),
            ships=int(ships),
            angular_velocity=float(angular_velocity),
        )

    @property
    def is_neutral(self) -> bool:
        return self.owner == -1

    def _angle_to_position(self, angle: float) -> Tuple[float, float]:
        return (
            SUN_X + math.cos(angle) * self.init_orbital_radius,
            SUN_Y + math.sin(angle) * self.init_orbital_radius,
        )

    def _calculate_position_from_turn(self, turn: int) -> Tuple[float, float]:
        if self.planet_type == PlanetType.STATIC:
            return (self.init_x, self.init_y)

        if self.planet_type == PlanetType.ORBITAL:
            angle = self.init_angle + self.angular_velocity * turn
            return self._angle_to_position(angle)

        return (self.init_x, self.init_y)
    
    