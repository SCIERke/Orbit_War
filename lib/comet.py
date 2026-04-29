from dataclasses import dataclass
from lib.celestialBody import MyCelestialBody
import math

@dataclass
class MyComet(MyCelestialBody):
    cometSpeed: float
    
    def __post_init__(self):
        super().__post_init__()
        self.init_angle = self._calculate_angle(self)