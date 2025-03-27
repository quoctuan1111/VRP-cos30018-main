import os
import sys
import tkinter as tk
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import DataProcessor
from src.optimization.or_ml_optimizer import ORToolsMLOptimizer
from src.visualization.route_visualizer import RouteVisualizer
import numpy as np

def generate_city_coordinates(data_processor: DataProcessor) -> dict:
    """Generate 2D coordinates for cities"""
    cities = list(data_processor.cities)
    n_cities = len(cities)
    
    # Generate coordinates in a circular pattern
    angles = np.linspace(0, 2*np.pi, n_cities-1, endpoint=False)  # -1 to leave space for warehouse
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

def main():
    # Initialize data and optimizer
    print("Loading data...")
    data_processor = DataProcessor()
    data_processor.load_data('distance.csv', 'order_large.csv')
    
    print("Running optimization...")
    optimizer = ORToolsMLOptimizer(data_processor)
    routes = optimizer.optimize()
    
    if not routes:
        print("No routes found!")
        return
        
    # Generate city coordinates
    city_coordinates = generate_city_coordinates(data_processor)
    
    # Set window size and position
    root = tk.Tk()
    root.geometry("1200x800")
    root.minsize(800, 600)
    
    # Create and run visualization
    visualizer = RouteVisualizer(root, city_coordinates)
    visualizer.plot_routes(routes)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()
