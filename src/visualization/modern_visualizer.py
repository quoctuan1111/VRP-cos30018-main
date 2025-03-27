import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for thread safety
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from typing import List, Dict
from src.models.route import Route
from src.visualization.fitness_plotter import FitnessPlotter
import queue
import threading
import folium
from folium import plugins
import webbrowser
import os
from tkinter import scrolledtext
import tkinterweb

class ModernVisualizer:
    """Modern visualization class for VRP solutions"""
    
    def __init__(self, parent, cities_coordinates):
        """Initialize the visualizer"""
        self.parent = parent
        self.cities_coordinates = cities_coordinates
        self._is_destroyed = False
        
        # Initialize update queue for thread-safe operations
        self.update_queue = queue.Queue()
        
        # Clear any existing widgets
        for widget in self.parent.winfo_children():
            widget.destroy()
        
        # Create main frame with proper weights
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for route list and controls
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create route listbox
        self.route_list_frame = ttk.LabelFrame(self.left_panel, text="Routes")
        self.route_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.route_listbox = tk.Listbox(self.route_list_frame, width=40)
        self.route_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar to route listbox
        self.route_scrollbar = ttk.Scrollbar(self.route_list_frame, orient=tk.VERTICAL)
        self.route_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.route_listbox.config(yscrollcommand=self.route_scrollbar.set)
        self.route_scrollbar.config(command=self.route_listbox.yview)
        
        # Create right panel for visualizations
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create frames for route visualization
        self.route_frame = ttk.LabelFrame(self.right_panel, text="Route Visualization")
        self.route_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create score comparison section
        self.score_frame = ttk.LabelFrame(self.right_panel, text="Method Comparison Scores")
        self.score_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create table for method scores
        columns = ("method", "composite", "distance", "cost", "capacity", "parcel", "route")
        self.score_table = ttk.Treeview(self.score_frame, columns=columns, show="headings", height=4)
        
        # Configure columns
        self.score_table.heading("method", text="Method")
        self.score_table.heading("composite", text="Composite")
        self.score_table.heading("distance", text="Distance")
        self.score_table.heading("cost", text="Cost")
        self.score_table.heading("capacity", text="Capacity")
        self.score_table.heading("parcel", text="Parcel")
        self.score_table.heading("route", text="Route")
        
        # Set column widths
        self.score_table.column("method", width=150, anchor="w")
        self.score_table.column("composite", width=100, anchor="center")
        self.score_table.column("distance", width=100, anchor="center")
        self.score_table.column("cost", width=100, anchor="center")
        self.score_table.column("capacity", width=100, anchor="center")
        self.score_table.column("parcel", width=100, anchor="center")
        self.score_table.column("route", width=100, anchor="center")
        
        self.score_table.pack(fill=tk.X, padx=5, pady=5)
        
        # Create figures and canvases in the main thread
        self.parent.after(0, self._create_figures)
        
        # Store routes for later reference
        self.routes = []
        
        # Bind destruction event
        self.parent.bind("<Destroy>", self._on_destroy)
        
        # Start update checker
        self._start_update_checker()
        
    def _create_figures(self):
        """Create matplotlib figures in the main thread"""
        if self._is_destroyed:
            return
            
        # Create route figure and canvas
        self.route_fig = plt.Figure(figsize=(10, 6))
        self.route_canvas = FigureCanvasTkAgg(self.route_fig, master=self.route_frame)
        self.route_canvas.draw()
        self.route_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.route_toolbar = NavigationToolbar2Tk(self.route_canvas, self.route_frame)
        self.route_toolbar.update()
        
        # Initialize plots
        self._init_plots()
        
    def _on_destroy(self, event):
        """Handle widget destruction"""
        if event.widget is self.parent:
            self._is_destroyed = True
            
            def cleanup():
                # Clean up resources in the main thread
                if hasattr(self, 'route_canvas'):
                    self.route_canvas.get_tk_widget().destroy()
                if hasattr(self, 'fitness_canvas'):
                    self.fitness_canvas.get_tk_widget().destroy()
                plt.close('all')
            
            # Schedule cleanup in main thread
            if threading.current_thread() is threading.main_thread():
                cleanup()
            else:
                self.parent.after(0, cleanup)

    def _thread_safe_update(self, func):
        """Execute function in the main thread"""
        if self._is_destroyed:
            return
            
        if threading.current_thread() is threading.main_thread():
            try:
                return func()
            except Exception as e:
                print(f"Error in visualization update: {str(e)}")
        else:
            self.update_queue.put(func)
            
    def _start_update_checker(self):
        """Start checking for updates from other threads"""
        if self._is_destroyed:
            return
            
        def check_queue():
            if self._is_destroyed:
                return
            try:
                while True:
                    func = self.update_queue.get_nowait()
                    if not self._is_destroyed:
                        try:
                            func()
                        except Exception as e:
                            print(f"Error in queued update: {str(e)}")
                    self.update_queue.task_done()
            except queue.Empty:
                pass
            finally:
                if not self._is_destroyed:
                    self.parent.after(100, check_queue)
        
        self.parent.after(100, check_queue)
        
    def _init_plots(self):
        """Initialize the plots"""
        # Route plot
        self.route_ax = self.route_fig.add_subplot(111)
        self.route_ax.set_title('Vehicle Routes')
        self.route_ax.set_xlabel('X Coordinate')
        self.route_ax.set_ylabel('Y Coordinate')
        self.route_ax.grid(True)
        
        # Plot initial cities
        x_coords = [coord[0] for coord in self.cities_coordinates.values()]
        y_coords = [coord[1] for coord in self.cities_coordinates.values()]
        self.route_ax.scatter(x_coords, y_coords, c='blue', marker='o', s=50, label='Cities')
        
        # Plot warehouse at (0,0)
        self.route_ax.scatter([0], [0], c='red', marker='s', s=100, label='Warehouse')
        
        self.route_ax.legend()
        
        # Set axis limits with padding
        x_min = min(min(x_coords), 0) - 5
        x_max = max(max(x_coords), 0) + 5
        y_min = min(min(y_coords), 0) - 5
        y_max = max(max(y_coords), 0) + 5
        
        self.route_ax.set_xlim(x_min, x_max)
        self.route_ax.set_ylim(y_min, y_max)
        
        # Update canvases
        self.route_fig.tight_layout()
        self.route_canvas.draw()
        
    def _create_left_panel(self):
        """Create the left panel with controls and information"""
        panel = ttk.Frame(self.main_container)
        
        # Info section
        info_frame = ttk.LabelFrame(panel, text="Solution Information", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=8, width=40, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Route selection
        route_frame = ttk.LabelFrame(panel, text="Route Selection", padding="10")
        route_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.route_listbox = tk.Listbox(route_frame, height=10)
        self.route_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.route_listbox.bind('<<ListboxSelect>>', self._on_route_select)
        
        # Route details
        details_frame = ttk.LabelFrame(panel, text="Route Details", padding="10")
        details_frame.pack(fill=tk.X)
        
        self.details_text = tk.Text(details_frame, height=8, width=40, wrap=tk.WORD)
        self.details_text.pack(fill=tk.X, padx=5, pady=5)
        
        return panel
        
    def _create_right_panel(self):
        """Create the right panel with visualizations"""
        panel = ttk.Frame(self.main_container)
        
        # Route visualization with larger size
        self.route_viz_frame = ttk.LabelFrame(panel, text="Route Visualization", padding="10")
        self.route_viz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Fitness plot
        self.fitness_frame = ttk.LabelFrame(panel, text="Fitness Evolution", padding="10")
        self.fitness_frame.pack(fill=tk.BOTH, expand=True)
        
        return panel
        
    def _create_map(self, routes: List[Route]):
        """Create an interactive map with routes"""
        if not self.map_viewer:
            return
            
        def _do_create_map():
            try:
                # Calculate center point
                lats = []
                lons = []
                for city, coords in self.cities_coordinates.items():
                    lat, lon = self._convert_xy_to_latlon(coords[0], coords[1])
                    lats.append(lat)
                    lons.append(lon)
                
                center_lat = np.mean(lats)
                center_lon = np.mean(lons)
                
                # Create map with a larger zoom level
                m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
                
                # Add warehouse marker
                warehouse_coords = self._convert_xy_to_latlon(
                    self.cities_coordinates["WAREHOUSE"][0],
                    self.cities_coordinates["WAREHOUSE"][1]
                )
                folium.Marker(
                    warehouse_coords,
                    popup="WAREHOUSE",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
                
                # Add routes with different colors
                colors = ['#%02x%02x%02x' % tuple(np.random.randint(0, 256, 3)) for _ in routes]
                for route, color in zip(routes, colors):
                    route_coords = []
                    for loc in route.locations:
                        x, y = self.cities_coordinates[loc.city_name]
                        lat, lon = self._convert_xy_to_latlon(x, y)
                        route_coords.append([lat, lon])
                        
                        # Add city marker
                        if loc.city_name != "WAREHOUSE":
                            folium.CircleMarker(
                                [lat, lon],
                                radius=8,  # Increased marker size
                                popup=loc.city_name,
                                color=color,
                                fill=True,
                                fill_opacity=0.7
                            ).add_to(m)
                    
                    # Add route line with increased width
                    folium.PolyLine(
                        route_coords,
                        weight=3,  # Increased line width
                        color=color,
                        opacity=0.8
                    ).add_to(m)
                
                # Save map to HTML
                map_path = os.path.join(os.path.dirname(__file__), "route_map.html")
                m.save(map_path)
                
                # Load map in viewer and force update
                self.map_viewer.load_file(map_path)
                self.map_viewer.update()
                
            except Exception as e:
                print(f"Warning: Could not create map: {str(e)}")
        
        # Schedule map creation in the main thread
        self.parent.after(0, _do_create_map)
        
    def _convert_xy_to_latlon(self, x: float, y: float) -> tuple:
        """Convert x,y coordinates to latitude and longitude
        This is a simple conversion for demonstration. In a real system,
        you would use actual GPS coordinates from your data."""
        # Center point (adjust these values based on your region)
        center_lat = 40.7128  # Example: New York City
        center_lon = -74.0060
        
        # Scale factors (adjust based on your coordinate system)
        lat_scale = 0.01
        lon_scale = 0.01
        
        lat = center_lat + (y * lat_scale)
        lon = center_lon + (x * lon_scale)
        
        return lat, lon
        
    def plot_fitness(self, generation: int, fitness_scores: List[float], best_fitness: float):
        """Plot fitness evolution with modern styling"""
        if self._is_destroyed:
            return
        def _update():
            if self._is_destroyed:
                return
            self.fitness_plotter.add_generation_data(generation, fitness_scores, best_fitness)
            
            self.fitness_ax.clear()
            
            # Plot with modern styling and real-time updates
            self.fitness_ax.plot(self.fitness_plotter.generation_numbers, 
                               self.fitness_plotter.avg_fitness_history,
                               label='Average Fitness', color='#2ecc71', alpha=0.6)
            
            self.fitness_ax.plot(self.fitness_plotter.generation_numbers,
                               self.fitness_plotter.best_fitness_history,
                               label='Best Fitness', color='#e74c3c', linewidth=2)
            
            # Add current generation marker
            if len(self.fitness_plotter.generation_numbers) > 0:
                self.fitness_ax.scatter(generation, best_fitness, 
                                      color='#e74c3c', s=100, zorder=5)
                self.fitness_ax.text(generation, best_fitness, f'Gen {generation}',
                                   xytext=(10, 10), textcoords='offset points')
            
            # Set axis properties
            self.fitness_ax.set_title('Fitness Evolution (Real-time)', pad=20, fontsize=14)
            self.fitness_ax.set_xlabel('Generation', fontsize=12)
            self.fitness_ax.set_ylabel('Fitness Score', fontsize=12)
            self.fitness_ax.legend(fontsize=10)
            self.fitness_ax.grid(True, linestyle='--', alpha=0.3)
            
            # Set y-axis limits with some padding
            if self.fitness_plotter.best_fitness_history:
                y_min = min(min(self.fitness_plotter.avg_fitness_history), 
                           min(self.fitness_plotter.best_fitness_history))
                y_max = max(max(self.fitness_plotter.avg_fitness_history),
                           max(self.fitness_plotter.best_fitness_history))
                padding = (y_max - y_min) * 0.1  # 10% padding
                self.fitness_ax.set_ylim(y_min - padding, y_max + padding)
            
            # Update canvas with proper layout
            self.fitness_fig.tight_layout()
            self.fitness_canvas.draw()
            
        self._thread_safe_update(_update)
        
    def clear_routes(self):
        """Clear existing routes from the plot"""
        if hasattr(self, 'route_ax'):
            self.route_ax.clear()
            self._plot_cities()  # Replot the cities after clearing
            
    def plot_routes(self, routes):
        """Plot the best route on the map with enhanced visualization"""
        if not routes:
            return
            
        # Clear previous routes and initialize plot
        self.route_ax.clear()
        self.route_ax.set_title('Best Route Visualization', pad=20, fontsize=14)
        self.route_ax.set_xlabel('X Coordinate', fontsize=12)
        self.route_ax.set_ylabel('Y Coordinate', fontsize=12)
        self.route_ax.grid(True, linestyle='--', alpha=0.3)
        
        # Find the best route based on cost per distance efficiency
        best_route = min(routes, key=lambda r: r.total_cost / (r.total_distance if r.total_distance > 0 else 1))
        
        # Plot only the best route
        route_coords = []
        depot_coords = (0, 0)  # Assuming depot is at origin
        
        # Start from depot
        route_coords.append(depot_coords)
        
        # Add each city in the route
        for parcel in best_route.parcels:
            city_name = parcel.destination.city_name
            if city_name in self.cities_coordinates:
                route_coords.append(self.cities_coordinates[city_name])
        
        # Return to depot
        route_coords.append(depot_coords)
        
        # Convert to numpy arrays for plotting
        route_coords = np.array(route_coords)
        
        # Plot route with arrows and gradient color
        for i in range(len(route_coords) - 1):
            start = route_coords[i]
            end = route_coords[i + 1]
            
            # Calculate arrow properties
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            arrow_length = np.sqrt(dx**2 + dy**2)
            
            # Use gradient color based on position in route
            color = plt.cm.viridis(i / (len(route_coords) - 1))
            
            # Draw arrow with improved visibility
            self.route_ax.arrow(
                start[0], start[1], dx, dy,
                head_width=arrow_length*0.05,
                head_length=arrow_length*0.1,
                fc=color,
                ec=color,
                length_includes_head=True,
                alpha=0.8,
                zorder=3
            )
            
            # Add distance annotation for each segment
            mid_point = (start + end) / 2
            distance = np.sqrt(dx**2 + dy**2)
            self.route_ax.annotate(
                f'{distance:.1f}km',
                xy=mid_point,
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8),
                fontsize=8,
                zorder=4
            )
            
            # Add segment number
            self.route_ax.annotate(
                f'#{i+1}',
                xy=mid_point,
                xytext=(0, -15),
                textcoords='offset points',
                ha='center',
                va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8),
                fontsize=8,
                zorder=4
            )
        
        # Plot cities with enhanced visibility
        cities_x = []
        cities_y = []
        city_labels = []
        
        # Add depot with special marker
        cities_x.append(depot_coords[0])
        cities_y.append(depot_coords[1])
        city_labels.append('Depot')
        self.route_ax.scatter([depot_coords[0]], [depot_coords[1]], 
                            c='#e74c3c',  # Red color for depot
                            marker='*', 
                            s=200, 
                            zorder=5,
                            label='Depot')
        
        # Add cities in the route with parcel information
        visited_cities = set()
        for i, parcel in enumerate(best_route.parcels):
            city_name = parcel.destination.city_name
            if city_name not in visited_cities and city_name in self.cities_coordinates:
                coords = self.cities_coordinates[city_name]
                cities_x.append(coords[0])
                cities_y.append(coords[1])
                city_labels.append(f"{city_name}\n({parcel.weight:.1f}kg)")
                visited_cities.add(city_name)
        
        # Plot cities with modern style and size based on parcel weight
        for i, (x, y, label) in enumerate(zip(cities_x[1:], cities_y[1:], city_labels[1:])):
            weight = float(label.split('\n')[1].strip('()kg'))
            size = 100 + (weight * 10)  # Scale size based on parcel weight
            self.route_ax.scatter(x, y, 
                                c='#27ae60',  # Green color for cities
                                s=size, 
                                zorder=4,
                                label='Delivery Points')
        
        # Add city labels with improved visibility and parcel information
        for i, (x, y, label) in enumerate(zip(cities_x, cities_y, city_labels)):
            self.route_ax.annotate(
                label,
                (x, y),
                xytext=(8, 8),
                textcoords='offset points',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    fc='white',
                    ec='gray',
                    alpha=0.8
                ),
                zorder=6,
                fontsize=9
            )
        
        # Add comprehensive route information
        info_text = f"Best Route Details:\n"
        info_text += f"Total Distance: {best_route.total_distance:.2f} km\n"
        info_text += f"Total Cost: ${best_route.total_cost:.2f}\n"
        info_text += f"Vehicle Load: {best_route.get_total_weight():.1f}/{best_route.vehicle_capacity:.1f} kg\n"
        info_text += f"Number of Stops: {len(best_route.parcels)}\n"
        info_text += f"Efficiency: {(best_route.get_total_weight()/best_route.vehicle_capacity)*100:.1f}%\n"
        info_text += f"Cost per km: ${best_route.total_cost/best_route.total_distance:.2f}"
        
        # Add text box with route information
        self.route_ax.text(
            0.02, 0.98, info_text,
            transform=self.route_ax.transAxes,
            bbox=dict(
                boxstyle='round,pad=0.5',
                fc='white',
                ec='gray',
                alpha=0.9
            ),
            verticalalignment='top',
            fontsize=10,
            zorder=7
        )
        
        # Add legend with improved visibility
        self.route_ax.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=10)
        
        # Set axis limits with padding
        x_coords = [coord[0] for coord in self.cities_coordinates.values()]
        y_coords = [coord[1] for coord in self.cities_coordinates.values()]
        x_min = min(min(x_coords), 0) - 5
        x_max = max(max(x_coords), 0) + 5
        y_min = min(min(y_coords), 0) - 5
        y_max = max(max(y_coords), 0) + 5
        self.route_ax.set_xlim(x_min, x_max)
        self.route_ax.set_ylim(y_min, y_max)
        
        # Update the plot with proper layout
        self.route_fig.tight_layout()
        self.route_canvas.draw()
        
    def _plot_cities(self):
        """Plot cities with enhanced visibility"""
        cities = set()
        for route in self.routes:
            for loc in route.locations:
                cities.add(loc.city_name)
        
        for city in cities:
            x, y = self.cities_coordinates[city]
            if city == "WAREHOUSE":
                # Make warehouse more prominent
                self.route_ax.scatter(x, y, c='#e74c3c', s=300, marker='*', zorder=5)
                self.route_ax.annotate('WAREHOUSE', (x, y),
                                     xytext=(15, 15), textcoords='offset points',
                                     fontsize=14, fontweight='bold',
                                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e74c3c'))
            else:
                # Make delivery points more visible
                self.route_ax.scatter(x, y, c='#3498db', s=150, edgecolor='white', zorder=4)
                self.route_ax.annotate(city, (x, y),
                                     xytext=(8, 8), textcoords='offset points',
                                     fontsize=10,
                                     bbox=dict(facecolor='white', alpha=0.8))
                
    def _plot_route_paths(self, colors):
        """Plot route paths with enhanced visibility"""
        for route, color in zip(self.routes, colors):
            route_coordinates = []
            for loc in route.locations:
                route_coordinates.append(self.cities_coordinates[loc.city_name])
            
            route_coordinates = np.array(route_coordinates)
            
            # Draw route path with increased width and better visibility
            self.route_ax.plot(route_coordinates[:, 0], route_coordinates[:, 1],
                             c=color, linewidth=3, label=f"Route {route.vehicle_id}",
                             zorder=3, alpha=0.8)
            
            # Add arrows to show direction
            for i in range(len(route_coordinates)-1):
                mid_point = (route_coordinates[i] + route_coordinates[i+1]) / 2
                direction = route_coordinates[i+1] - route_coordinates[i]
                direction = direction / np.linalg.norm(direction)
                self.route_ax.arrow(mid_point[0], mid_point[1],
                                  direction[0]*5, direction[1]*5,
                                  head_width=2, head_length=3,
                                  fc=color, ec=color, alpha=0.8)
            
    def _on_route_select(self, event):
        """Handle route selection"""
        selection = self.route_listbox.curselection()
        if selection:
            route = self.routes[selection[0]]
            self.update_route_details(route)
            
    def update_route_details(self, route: Route):
        """Update route details with modern formatting"""
        self.details_text.delete('1.0', tk.END)
        
        # Create a formatted string with modern styling
        info = f"Route Details: {route.vehicle_id}\n"
        info += "═" * 40 + "\n\n"
        
        # Add metrics with icons
        info += f"📦 Parcels: {len(route.parcels)}\n"
        info += f"🛣️ Distance: {route.total_distance:.2f} km\n"
        info += f"💰 Cost: ${route.total_cost:.2f}\n"
        info += f"🚛 Capacity: {route.vehicle_capacity:.2f} kg\n"
        info += f"⚖️ Current Load: {route.get_total_weight():.2f} kg\n"
        info += f"📊 Utilization: {(route.get_total_weight()/route.vehicle_capacity)*100:.1f}%\n"
        
        self.details_text.insert(tk.END, info)
        
    def update_info(self, routes: List[Route]):
        """Update solution information with modern formatting"""
        if self._is_destroyed:
            return
        def _update():
            if self._is_destroyed:
                return
            total_cost = sum(route.total_cost for route in routes)
            total_distance = sum(route.total_distance for route in routes)
            total_parcels = sum(len(route.parcels) for route in routes)
            
            info = "Solution Summary\n"
            info += "═" * 40 + "\n\n"
            
            # Add metrics with icons
            info += f"🚛 Total Routes: {len(routes)}\n"
            info += f"🛣️ Total Distance: {total_distance:.2f} km\n"
            info += f"💰 Total Cost: ${total_cost:.2f}\n"
            info += f"📦 Total Parcels: {total_parcels}\n"
            
            # Add efficiency metrics
            avg_utilization = np.mean([route.get_total_weight()/route.vehicle_capacity 
                                     for route in routes]) * 100
            info += f"📊 Average Utilization: {avg_utilization:.1f}%\n"
            
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert(tk.END, info)
            
        self._thread_safe_update(_update)
        
    def update_fitness_plot(self, generation: int, best_fitness_history: List[float], avg_fitness_history: List[float]):
        """Update the evolution plot with route quality metrics"""
        try:
            def _update():
                if self._is_destroyed:
                    return
                    
                # Clear previous plot
                self.fitness_ax.clear()
                
                # Create generation points
                generations = list(range(len(best_fitness_history)))
                
                # Plot best route quality
                self.fitness_ax.plot(generations, best_fitness_history, 'r-', 
                                   label='Best Route Quality', linewidth=2)
                
                # Plot average route quality
                self.fitness_ax.plot(generations, avg_fitness_history, 'g--', 
                                   label='Average Route Quality', alpha=0.7)
                
                # Add markers for latest points
                if generations:
                    self.fitness_ax.plot(generations[-1], best_fitness_history[-1], 'ro', 
                                       markersize=8, label='Latest Best')
                    self.fitness_ax.plot(generations[-1], avg_fitness_history[-1], 'go', 
                                       markersize=8, label='Latest Avg')
                
                # Customize plot
                self.fitness_ax.set_title('Route Quality Evolution', pad=10, fontsize=12)
                self.fitness_ax.set_xlabel('Generation', fontsize=10)
                self.fitness_ax.set_ylabel('Route Quality Score', fontsize=10)
                self.fitness_ax.grid(True, linestyle='--', alpha=0.3)
                
                # Set y-axis limits with padding
                if best_fitness_history:
                    ymin = min(min(best_fitness_history), min(avg_fitness_history))
                    ymax = max(max(best_fitness_history), max(avg_fitness_history))
                    padding = (ymax - ymin) * 0.1
                    self.fitness_ax.set_ylim(ymin - padding, ymax + padding)
                
                # Add legend
                self.fitness_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                
                # Add text box with current metrics
                if best_fitness_history:
                    current_best = best_fitness_history[-1]
                    current_avg = avg_fitness_history[-1]
                    improvement = ((current_best - current_avg) / current_avg * 100) if current_avg > 0 else 0
                    
                    metrics_text = f"Current Metrics:\n"
                    metrics_text += f"Best Quality: {current_best:.4f}\n"
                    metrics_text += f"Avg Quality: {current_avg:.4f}\n"
                    metrics_text += f"Improvement: {improvement:.1f}%"
                    
                    self.fitness_ax.text(0.02, 0.98, metrics_text,
                                       transform=self.fitness_ax.transAxes,
                                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                                       verticalalignment='top',
                                       fontsize=9)
                
                # Adjust layout
                self.fitness_fig.tight_layout()
                
                # Update canvas
                self.fitness_canvas.draw()
                self.fitness_canvas.get_tk_widget().update()
            
            # Execute update in main thread
            if threading.current_thread() is threading.main_thread():
                _update()
            else:
                self.parent.after(0, _update)
                
        except Exception as e:
            print(f"Error updating evolution plot: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def update_score_plot(self, score_history):
        """Update the score table with new data"""
        try:
            # Clear existing items
            for item in self.score_table.get_children():
                self.score_table.delete(item)
            
            # Add scores for each method
            for method_name, scores in score_history.items():
                self.score_table.insert("", "end", values=(
                    method_name,
                    f"{scores['composite_score']:.4f}",
                    f"{scores['distance_score']:.4f}",
                    f"{scores['cost_efficiency_score']:.4f}",
                    f"{scores['capacity_utilization_score']:.4f}",
                    f"{scores['parcel_efficiency_score']:.4f}",
                    f"{scores['route_structure_score']:.4f}"
                ))
                
        except Exception as e:
            print(f"Error updating score table: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def destroy(self):
        """Clean up resources"""
        self._is_destroyed = True
        plt.close(self.route_fig)
        plt.close(self.fitness_fig) 