import math
from ..models.location import Location


def calculate_distance(loc1: Location, loc2: Location) -> float:
    """Calculate straight-line distance between two locations"""
    return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)
