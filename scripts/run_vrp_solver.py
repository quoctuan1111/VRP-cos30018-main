import os
import sys
import time
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import DataProcessor
from src.optimization.vrp_solver import VRPSolver
from src.visualization.route_visualizer import RouteVisualizer
import tkinter as tk
import numpy as np

def generate_city_coordinates(data_processor):
    """
    Generate 2D coordinates for cities in a circular pattern for visualization.
    
    Args:
        data_processor: DataProcessor with city information
        
    Returns:
        Dictionary mapping city names to (x, y) coordinates
    """
    cities = list(data_processor.cities)
    n_cities = len(cities)
    
    # Generate coordinates in a circular pattern
    angles = np.linspace(0, 2*np.pi, n_cities-1, endpoint=False)
    radius = 100
    coordinates = {}
    
    # Place warehouse at center
    coordinates["WAREHOUSE"] = (0, 0)
    
    # Place other cities in a circle around warehouse
    remaining_cities = [city for city in cities if city != "WAREHOUSE"]
    for city, angle in zip(remaining_cities, angles):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        coordinates[city] = (x, y)
    
    return coordinates

def visualize_solution(solution, data_processor):
    """
    Visualize a solution using RouteVisualizer.
    
    Args:
        solution: List of Routes to visualize
        data_processor: DataProcessor with city information
    """
    # Generate city coordinates
    city_coordinates = generate_city_coordinates(data_processor)
    
    # Create visualization window
    root = tk.Tk()
    root.title("VRP Solution Visualization")
    root.geometry("1200x800")
    
    # Create visualizer and plot routes
    visualizer = RouteVisualizer(root, city_coordinates)
    visualizer.plot_routes(solution)
    
    # Start main loop
    root.mainloop()

def main():
    """Main function to run the VRP solver"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run VRP Solver')
    parser.add_argument('--method', type=str, default='all', 
                      choices=['all', 'ga', 'or_ml', 'ga_or', 'ga_or_mod'],
                      help='Optimization method to use')
    parser.add_argument('--distance', type=str, default='distance_1.csv',
                      help='Distance CSV file')
    parser.add_argument('--orders', type=str, default='order_large.csv',
                      help='Orders CSV file')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize solution after optimization')
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("=== VRP Solver ===")
    start_time = time.time()
    
    # Initialize data processor
    print("\nLoading data...")
    try:
        data_processor = DataProcessor()
        print(f"Loading distance data from: {args.distance}")
        print(f"Loading order data from: {args.orders}")
        data_processor.load_data(args.distance, args.orders)
        
        # Print data statistics
        print("\nData Statistics:")
        print(f"Number of cities: {len(data_processor.cities)}")
        print(f"Number of orders: {len(data_processor.order_data)}")
        print(f"Distance matrix shape: {data_processor.distance_matrix.shape}")
        print(f"Sample distance: {data_processor.distance_matrix[0][1] if data_processor.distance_matrix is not None else 'None'}")
        print(f"Sample order weight: {data_processor.order_data['Weight'].iloc[0] if data_processor.order_data is not None else 'None'}")
        print(f"Total orders weight: {data_processor.order_data['Weight'].sum() if data_processor.order_data is not None else 'None'}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create VRP solver
    print("\nCreating VRP solver...")
    solver = VRPSolver(data_processor)
    
    # Solve using specified method
    print(f"\nRunning optimization using method: {args.method}")
    try:
        solution = solver.solve(method=args.method)
        if solution:
            print(f"\nSolution found with {len(solution)} routes")
            for i, route in enumerate(solution):
                print(f"Route {i+1}: {len(route.parcels)} parcels, distance: {route.total_distance:.2f} km")
        else:
            print("\nNo solution found!")
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        solver.stop()
        return
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    # Visualize if requested
    if args.visualize and solution:
        print("\nVisualizing solution...")
        visualize_solution(solution, data_processor)

if __name__ == "__main__":
    main()