import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.data_processor import DataProcessor
from src.optimization.ga_or_optimizer import GAOROptimizer

def main():
    # Initialize DataProcessor and load data
    processor = DataProcessor()
    processor.load_data(os.path.join('distance.csv'), os.path.join('order_large.csv'))
    
    # Create optimizer instance
    optimizer = GAOROptimizer(processor)
    print(f"Optimizer created, _location_cache exists: {'_location_cache' in vars(optimizer)}")  # Debug
    
    # Run optimization
    routes = optimizer.optimize()
    print(f"Number of routes found: {len(routes)}")
    for route in routes:
        print(f"Route {route.vehicle_id}: {route.locations}, Distance: {route.total_distance}, Cost: {route.total_cost}")

if __name__ == "__main__":
    main()