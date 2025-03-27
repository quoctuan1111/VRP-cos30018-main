import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from typing import List, Dict
from src.models.route import Route

class RouteVisualizer:
    def __init__(self, root: tk.Tk, cities_coordinates: dict):
        self.root = root
        self.root.title("VRP Route Visualization")
        self.cities_coordinates = cities_coordinates
        
        # Create main frame with better padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create figures for route and fitness plots
        self.route_fig, self.route_ax = plt.subplots(figsize=(12, 6), dpi=100)
        self.fitness_fig, self.fitness_ax = plt.subplots(figsize=(12, 4), dpi=100)
        
        # Create canvases
        self.route_canvas = FigureCanvasTkAgg(self.route_fig, master=self.main_frame)
        self.fitness_canvas = FigureCanvasTkAgg(self.fitness_fig, master=self.main_frame)
        
        # Create toolbar frame
        self.toolbar_frame = ttk.Frame(self.main_frame)
        self.route_toolbar = NavigationToolbar2Tk(self.route_canvas, self.toolbar_frame)
        
        # Create info panel
        self.info_frame = ttk.LabelFrame(self.main_frame, text="Route Information", padding="5")
        self.info_text = tk.Text(self.info_frame, height=8, width=60)
        self.route_listbox = tk.Listbox(self.info_frame, height=5)
        
        # Layout using grid
        self.toolbar_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.route_toolbar.grid(row=0, column=0, sticky="w")
        
        self.route_canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.fitness_canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.info_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=5)
        self.info_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.route_listbox.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure weights
        self.main_frame.grid_rowconfigure(0, weight=2)  # Route plot gets more space
        self.main_frame.grid_rowconfigure(2, weight=1)  # Fitness plot
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.info_frame.grid_columnconfigure(0, weight=3)
        self.info_frame.grid_columnconfigure(1, weight=1)
        
        # Store routes for later reference
        self.routes = []
        
        # Initialize the fitness plot
        self._setup_fitness_plot()

    def _setup_fitness_plot(self):
        """Setup the fitness evolution plot"""
        self.fitness_ax.clear()
        self.fitness_ax.set_title("Score Evolution", pad=10, fontsize=12)
        self.fitness_ax.set_xlabel("Generation")
        self.fitness_ax.set_ylabel("Score")
        self.fitness_ax.grid(True, linestyle='--', alpha=0.7)
        self.fitness_canvas.draw()

    def update_score_plot(self, score_history):
        """Update the score evolution plot with new data"""
        self.fitness_ax.clear()
        
        generations = score_history['generations']
        scores = score_history['scores']
        
        # Plot each score type with different colors and labels
        self.fitness_ax.plot(generations, scores['composite_score'], 'b-', 
                           label='Overall Score', linewidth=2)
        self.fitness_ax.plot(generations, scores['distance_score'], 'g--', 
                           label='Distance', alpha=0.7)
        self.fitness_ax.plot(generations, scores['cost_efficiency_score'], 'r--',
                           label='Cost Efficiency', alpha=0.7)
        self.fitness_ax.plot(generations, scores['capacity_utilization_score'], 'c--',
                           label='Capacity Util.', alpha=0.7)
        self.fitness_ax.plot(generations, scores['parcel_efficiency_score'], 'm--',
                           label='Parcel Efficiency', alpha=0.7)
        self.fitness_ax.plot(generations, scores['route_structure_score'], 'y--',
                           label='Route Structure', alpha=0.7)
        
        self.fitness_ax.set_title("Score Evolution", pad=10, fontsize=12)
        self.fitness_ax.set_xlabel("Generation")
        self.fitness_ax.set_ylabel("Score")
        self.fitness_ax.grid(True, linestyle='--', alpha=0.7)
        self.fitness_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        self.fitness_ax.set_ylim(0, 1)
        
        # Adjust layout to prevent label cutoff
        self.fitness_fig.tight_layout()
        self.fitness_canvas.draw()

    def plot_routes(self, routes: List[Route]):
        """Plot routes with improved styling"""
        self.routes = routes
        
        # Schedule the actual plotting in the main thread
        self.root.after(0, self._do_plot_routes)

    def _do_plot_routes(self):
        """Perform the actual route plotting in the main thread"""
        self.route_ax.clear()
        
        # Use a better color scheme
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.routes)))
        
        # Plot cities with improved markers
        self._plot_cities()
        
        # Plot routes with better styling
        self._plot_route_paths(colors)
        
        # Customize plot appearance
        self.route_ax.set_title("Vehicle Routing Solution", pad=20, fontsize=14)
        self.route_ax.grid(True, linestyle='--', alpha=0.7)
        self.route_ax.set_aspect('equal')
        
        # Add route selection handler
        self.route_listbox.delete(0, tk.END)
        for i, route in enumerate(self.routes):
            self.route_listbox.insert(tk.END, f"Route {route.vehicle_id}")
            
        self.route_listbox.bind('<<ListboxSelect>>', self._on_route_select)
        
        # Update canvas
        self.route_canvas.draw()
        
        # Update info panel
        self.update_info(self.routes)

    def _plot_cities(self):
        """Plot cities with improved styling"""
        cities = set()
        for route in self.routes:
            for loc in route.locations:
                cities.add(loc.city_name)
        
        for city in cities:
            x, y = self.cities_coordinates[city]
            if city == "WAREHOUSE":
                self.route_ax.scatter(x, y, c='red', s=200, marker='*', zorder=5)
                self.route_ax.annotate('WAREHOUSE', (x, y), 
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=12, fontweight='bold')
            else:
                self.route_ax.scatter(x, y, c='lightgray', s=100, edgecolor='black', zorder=4)
                self.route_ax.annotate(city, (x, y), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8)

    def _plot_route_paths(self, colors):
        """Plot route paths with improved styling"""
        for route, color in zip(self.routes, colors):
            route_coordinates = []
            for loc in route.locations:
                route_coordinates.append(self.cities_coordinates[loc.city_name])
            
            route_coordinates = np.array(route_coordinates)
            self.route_ax.plot(route_coordinates[:, 0], route_coordinates[:, 1], 
                        c=color, linewidth=2, label=f"Route {route.vehicle_id}",
                        zorder=3)

    def _on_route_select(self, event):
        """Handle route selection"""
        selection = self.route_listbox.curselection()
        if selection:
            route = self.routes[selection[0]]
            self.update_route_details(route)

    def update_route_details(self, route: Route):
        """Update info panel with selected route details"""
        self.info_text.delete('1.0', tk.END)
        info = f"Route Details: {route.vehicle_id}\n"
        info += "=" * 50 + "\n"
        info += f"Number of Parcels: {len(route.parcels)}\n"
        info += f"Total Distance: {route.total_distance:.2f} km\n"
        info += f"Total Cost: ${route.total_cost:.2f}\n"
        info += f"Vehicle Capacity: {route.vehicle_capacity:.2f} kg\n"
        info += f"Current Load: {route.get_total_weight():.2f} kg\n"
        info += f"Capacity Utilization: {(route.get_total_weight()/route.vehicle_capacity)*100:.1f}%\n"
        self.info_text.insert(tk.END, info)

    def update_info(self, routes: List[Route]):
        """Update overall solution information"""
        total_cost = sum(route.total_cost for route in routes)
        total_distance = sum(route.total_distance for route in routes)
        total_parcels = sum(len(route.parcels) for route in routes)
        
        # Calculate comprehensive scores
        comparison_scores = self.optimizer.calculate_route_comparison_score(routes)
        
        info = "Solution Summary\n"
        info += "=" * 50 + "\n"
        info += f"Comprehensive Scores:\n"
        info += f"  • Overall Score: {comparison_scores['composite_score']:.4f}\n"
        info += f"  • Distance Efficiency: {comparison_scores['distance_score']:.4f}\n"
        info += f"  • Cost Efficiency: {comparison_scores['cost_efficiency_score']:.4f}\n"
        info += f"  • Capacity Utilization: {comparison_scores['capacity_utilization_score']:.4f}\n"
        info += f"  • Parcel Efficiency: {comparison_scores['parcel_efficiency_score']:.4f}\n"
        info += f"  • Route Structure: {comparison_scores['route_structure_score']:.4f}\n"
        info += "\nRoute Metrics:\n"
        info += f"  • Total Routes: {len(routes)}\n"
        info += f"  • Total Distance: {total_distance:.2f} km\n"
        info += f"  • Total Cost: ${total_cost:.2f}\n"
        info += f"  • Total Parcels Delivered: {total_parcels}\n"
        
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert(tk.END, info)
