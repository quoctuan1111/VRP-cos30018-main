import os
import sys
import time
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import DataProcessor
from src.optimization.or_ml_optimizer import ORToolsMLOptimizer
from src.ml.route_predictor import RoutePredictor

def main():
    """Main function to train the ML model"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ML model for route prediction')
    parser.add_argument('--distance', type=str, default='distance.csv',
                      help='Distance CSV file')
    parser.add_argument('--orders', type=str, default='order_large.csv',
                      help='Orders CSV file')
    parser.add_argument('--variations', type=int, default=20,
                      help='Number of OR-Tools solution variations to generate')
    parser.add_argument('--output', type=str, default='models/route_predictor.joblib',
                      help='Output model file')
    args = parser.parse_args()
    
    print("=== Training ML Model for Route Prediction ===")
    start_time = time.time()
    
    # Initialize data processor
    print("\nLoading data...")
    try:
        data_processor = DataProcessor()
        data_processor.load_data(args.distance, args.orders)
        print(f"Loaded {len(data_processor.order_data)} orders and {len(data_processor.cities)} cities")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create OR-Tools optimizer to generate training data
    optimizer = ORToolsMLOptimizer(data_processor)
    
    # Generate training data from OR-Tools
    print(f"\nGenerating {args.variations} OR-Tools solutions for training data...")
    or_solutions = optimizer._generate_or_tools_solutions(num_variations=args.variations)
    all_routes = [route for solution in or_solutions for route in solution if route.parcels]
    
    if not all_routes:
        print("Error: Could not generate any routes for training")
        return
    
    print(f"Generated {len(all_routes)} routes for training")
    
    # Convert to training data format
    training_data = []
    for route in all_routes:
        route_data = {
            'parcels': [{'destination': p.destination.city_name} for p in route.parcels],
            'total_weight': route.get_total_weight(),
            'total_distance': route.total_distance,
            'vehicle_capacity': route.vehicle_capacity,
            'total_cost': route.total_cost,
            'truck_type': route.vehicle_id.split('_')[1]
        }
        training_data.append(route_data)
    
    #