import math

SUN_X: float = 50.0
SUN_Y: float = 50.0

def angle_from_sun(x: float, y: float) -> float:
    return math.atan2(y - SUN_Y, x - SUN_X)


def distance_from_sun(x: float, y: float) -> float:
    return math.hypot(x - SUN_X, y - SUN_Y)
