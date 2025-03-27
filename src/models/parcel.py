from typing import Optional
from .location import Location

class Parcel:
    def __init__(self, id: int, destination: Location, weight: float, source: Optional[Location] = None):
        self.id = id
        self.destination = destination
        self.source = source
        self.weight = weight

    def __str__(self):
        return f"Parcel {self.id} to {self.destination}, Weight: {self.weight} kg"