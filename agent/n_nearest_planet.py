from lib.planet import MyPlanet
from lib.ship import MyFleet
from typing import Dict, List, Tuple


def agent(obs: Dict) -> List[Tuple[int, float, int]]:
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    
    current_turn = obs.get("step", 0) if isinstance(obs, dict) else obs.step
    
    initial_planets = [MyPlanet.from_obs(p) for p in obs.initial_planets]
    planets = [MyPlanet.from_obs(p) for p in obs.planets]
    
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    
    if not targets:
        return moves
    
    for mine in my_planets:
        nearest = min(targets, key=lambda t: mine._calculate_distance(t))
        
        ships_needed = max(nearest.ships + 1, 20)
        
        if mine.ships >= ships_needed:
            fleet = MyFleet.from_planet(mine, ships_needed)
            angle = fleet._shoot_at_planet(nearest, current_turn)
            moves.append([fleet.id, angle, ships_needed])
            
    return moves