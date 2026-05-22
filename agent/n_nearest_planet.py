from lib.planet import MyPlanet
from lib.ship import MyFleet
from typing import Any, Dict, List, Tuple


def agent(obs: Dict, config: Any = None) -> List[Tuple[int, float, int]]:
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    current_turn = obs.get("step", 0) if isinstance(obs, dict) else obs.step
    angular_velocity = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else obs.angular_velocity
    max_ship_speed = getattr(config, "shipSpeed", 6.0) if config is not None else 6.0

    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [MyPlanet.from_obs(p, angular_velocity=angular_velocity) for p in raw_planets]

    my_planets = [p for p in planets if p.owner == player]
    neutral_targets = [p for p in planets if p.owner == -1]
    enemy_targets = [p for p in planets if p.owner not in (player, -1)]
    targets = neutral_targets if neutral_targets else enemy_targets

    if not targets:
        return moves

    for mine in my_planets:
        nearest = min(targets, key=lambda t: mine._calculate_distance(t))

        ships_needed = nearest.ships + 1
        if mine.ships < ships_needed:
            continue

        fleet = MyFleet.from_planet(mine, ships_needed)
        angle = fleet._shoot_at_planet(nearest, current_turn, max_ship_speed=max_ship_speed)
        moves.append([fleet.id, angle, ships_needed])

    return moves