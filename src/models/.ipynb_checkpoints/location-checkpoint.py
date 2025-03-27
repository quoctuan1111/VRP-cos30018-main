from dataclasses import dataclass


@dataclass
class Location:
    x: float
    y: float

    def distance_to(self, other: 'Location') -> float:
        """calculate the distance between this location and another location"""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
