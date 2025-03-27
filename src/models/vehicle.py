from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class TruckSpecification:
    length: float          # Length in meters
    inner_size: Tuple[float, float]  # Length x Width in m²
    weight_capacity: float # Maximum weight in kg
    cost_per_km: float    # Cost per kilometer
    speed: float          # Speed in km/h

class Vehicle:
    def __init__(self, vehicle_id: str, truck_type: str):
        self.vehicle_id = vehicle_id
        self.truck_type = truck_type
        self.specification = self._get_truck_specification(truck_type)

    def _get_truck_specification(self, truck_type: str) -> TruckSpecification:
        """Get the specification for a truck type"""
        specs = {
            '16.5': TruckSpecification(
                length=16.5,
                inner_size=(16.1, 2.5),
                weight_capacity=10000,
                cost_per_km=3,
                speed=40
            ),
            '12.5': TruckSpecification(
                length=12.5,
                inner_size=(12.1, 2.5),
                weight_capacity=5000,
                cost_per_km=2,
                speed=40
            ),
            '9.6': TruckSpecification(
                length=9.6,
                inner_size=(9.1, 2.3),
                weight_capacity=2000,
                cost_per_km=1,
                speed=40
            )
        }
        return specs.get(truck_type)

    def calculate_route_cost(self, distance: float) -> float:
        """Calculate cost for a given route distance"""
        return self.specification.cost_per_km * distance

    def can_handle_weight(self, weight: float) -> bool:
        """Check if vehicle can handle the given weight"""
        return weight <= self.specification.weight_capacity

    def get_travel_time(self, distance: float) -> float:
        """Calculate travel time in hours for a given distance"""
        return distance / self.specification.speed