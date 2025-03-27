from typing import List, Optional
from .location import Location
from .parcel import Parcel
from ..data.data_processor import DataProcessor

class Route:
    def __init__(self, vehicle_id: str, locations: List[Location], parcels: List[Parcel], 
                 data_processor: Optional[DataProcessor] = None, vehicle_capacity: float = 50000.0):
        self.vehicle_id = vehicle_id
        self.vehicle_capacity = vehicle_capacity
        self.data_processor = data_processor
        self.is_feasible = True
        self.violation_reason = ""
        self.total_distance = 0.0
        self.total_cost = 0.0
        
        # Validate and set locations
        if not locations:
            raise ValueError("Route must have at least one location")
            
        # If data_processor is provided, check warehouse location
        if data_processor is not None:
            if locations[0].city_name != data_processor.warehouse_location:
                locations.insert(0, Location(city_name=data_processor.warehouse_location))
            if locations[-1].city_name != data_processor.warehouse_location:
                locations.append(Location(city_name=data_processor.warehouse_location))
        
        self.locations = locations
        self.parcels = parcels
        
        # Calculate initial distance
        self.calculate_total_distance()
        # Validate route after initialization
        self.validate()

    def calculate_total_distance(self) -> float:
        """Calculate total route distance"""
        total = 0.0
        if self.data_processor and len(self.locations) > 1:
            for i in range(len(self.locations) - 1):
                source_city = self.locations[i].city_name
                dest_city = self.locations[i + 1].city_name
                source_idx = self.data_processor.city_to_idx.get(source_city)
                dest_idx = self.data_processor.city_to_idx.get(dest_city)
                
                if source_idx is not None and dest_idx is not None:
                    distance = self.data_processor.distance_matrix[source_idx][dest_idx]
                    total += distance
                    print(f"Distance from {source_city} to {dest_city}: {distance:.2f} km")
                else:
                    print(f"Warning: Could not find indices for {source_city} -> {dest_city}")
                    print(f"Available cities: {list(self.data_processor.city_to_idx.keys())}")
        
        self.total_distance = total
        return total

    def get_total_weight(self) -> float:
        """Calculate total weight of all parcels"""
        return sum(parcel.weight for parcel in self.parcels)

    def validate(self) -> bool:
        """
        Validate route feasibility.
        Returns:
            bool: True if route is feasible, False otherwise
        """
        # Check capacity constraint
        total_weight = self.get_total_weight()
        if total_weight > self.vehicle_capacity:
            self.is_feasible = False
            self.violation_reason = f"Capacity exceeded: {total_weight} > {self.vehicle_capacity}"
            return False

        # Check if route starts and ends at warehouse
        if self.data_processor:
            if (self.locations[0].city_name != self.data_processor.warehouse_location or 
                self.locations[-1].city_name != self.data_processor.warehouse_location):
                self.is_feasible = False
                self.violation_reason = "Route must start and end at warehouse"
                return False

        # Check maximum distance constraint (5000 km as example)
        if self.total_distance > 5000:
            self.is_feasible = False
            self.violation_reason = f"Distance exceeded: {self.total_distance} > 5000"
            return False

        # Check if all parcels have valid source and destination
        for parcel in self.parcels:
            if not (parcel.source and parcel.destination):
                self.is_feasible = False
                self.violation_reason = f"Invalid parcel locations for parcel {parcel.id}"
                return False

        self.is_feasible = True
        self.violation_reason = ""
        return True

    def __str__(self) -> str:
        return (f"Route {self.vehicle_id}: {len(self.parcels)} parcels, "
                f"Distance: {self.total_distance:.2f} km, "
                f"Cost: ${self.total_cost:.2f}")