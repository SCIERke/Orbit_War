from dataclasses import dataclass
import math

@dataclass
class MyCelestialBody:
    id: int
    x: float
    y: float
    radius: float
    
    def _calculate_distance(self, other: 'MyCelestialBody') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def _calculate_angle(self, other: 'MyCelestialBody') -> float:
        return math.atan2(other.y - self.y, other.x - self.x)