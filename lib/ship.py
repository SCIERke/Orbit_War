import math
from config.const import MAX_SHIP_SPEED
from dataclasses import dataclass, field
from lib.celestialBody import MyCelestialBody
from typing import List
from lib.planet import MyPlanet
from game_types.planet import PlanetType

@dataclass
class MyFleet(MyCelestialBody):
    owner: int
    from_planet_id: int
    ships: int
    radius: float = field(default=0.0, init=False)

    @classmethod
    def from_obs(cls, raw_data: List) -> "MyFleet":
        if len(raw_data) != 7:
            raise ValueError(f"Expected 7 fields for fleet observation, got {len(raw_data)}")
        fleet_id, owner, x, y, _angle, from_planet_id, ships = raw_data
        return cls(
            id=int(fleet_id),
            x=float(x),
            y=float(y),
            owner=int(owner),
            from_planet_id=int(from_planet_id),
            ships=int(ships),
        )

    @classmethod
    def from_planet(cls, planet: MyPlanet, ships: int) -> "MyFleet":
        return cls(
            id=planet.id,
            x=planet.x,
            y=planet.y,
            owner=planet.owner,
            from_planet_id=planet.id,
            ships=ships,
        )

    @classmethod
    def calculate_speed(cls, n_ship: int) -> float:
        return 1.0 + (MAX_SHIP_SPEED - 1.0) * (math.log(n_ship) / math.log(1000)) ** 1.5

    def _shoot_at_planet(self, planet: MyPlanet, current_turn: int) -> float:
        """
        Returns:
            float: angle to shoot at
        """
        if planet.planet_type == PlanetType.STATIC:
            return self._calculate_angle(planet)

        distance = self._calculate_distance(planet)
        speed = self.calculate_speed(self.ships)
        travel_turns = int(distance / speed)
        pred_x, pred_y = planet._calculate_position_from_turn(current_turn + travel_turns)
        return math.atan2(pred_y - self.y, pred_x - self.x)
