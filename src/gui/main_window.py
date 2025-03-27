import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import yaml
import os
import sys
import numpy as np
import threading
import time
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.data_processor import DataProcessor
from src.optimization.vrp_solver import VRPSolver
from src.visualization.modern_visualizer import ModernVisualizer

class MainWindow:
    """Main GUI window for the VRP system"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the main window.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.root = tk.Tk()
        self.root.title("Vehicle Routing System")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Set theme
        style = ttk.Style()
        style.theme_use('clam')  # Use a modern theme
        
        # Load configuration if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Default parameters
        self.default_params = {
            'ga_settings': {
                'population_size': 200,
                'generations': 50,
                'mutation_rate': 0.5,
                'crossover_rate': 0.85,
                'elitism_count': 2
            },
            'data_files': {
                'distance': 'distance.csv',
                'orders': 'order_large.csv'
            }
        }
        
        # Use config values or defaults
        self.params = self.config or self.default_params
        
        # Initialize components
        self.data_processor = None
        self.solver = None
        self.optimization_thread = None
        self.is_running = False
        self.visualizer = None
        
        # Initialize score variables
        self.score_vars = {
            'composite_score': tk.StringVar(value="0.0000"),
            'distance_score': tk.StringVar(value="0.0000"),
            'cost_efficiency_score': tk.StringVar(value="0.0000"),
            'capacity_utilization_score': tk.StringVar(value="0.0000"),
            'parcel_efficiency_score': tk.StringVar(value="0.0000"),
            'route_structure_score': tk.StringVar(value="0.0000")
        }
        
        # Setup UI
        self.setup_ui()
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def setup_ui(self):
        """Set up the user interface"""
        # Main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure grid weights
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=1)
        
        # Left panel for controls
        self.control_panel = self._create_control_panel()
        self.control_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Right panel for visualization
        self.viz_panel = self._create_viz_panel()
        self.viz_panel.grid(row=0, column=1, sticky="nsew")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky="ew")
    
    def _create_control_panel(self):
        """Create the control panel with modern styling"""
        panel = ttk.Frame(self.main_container)
        
        # Data files section
        data_frame = ttk.LabelFrame(panel, text="Data Files", padding="10")
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Distance file
        ttk.Label(data_frame, text="Distance CSV:").pack(fill=tk.X, padx=5, pady=2)
        self.distance_var = tk.StringVar(value=self.params['data_files']['distance'])
        ttk.Entry(data_frame, textvariable=self.distance_var).pack(fill=tk.X, padx=5, pady=2)
        
        # Orders file
        ttk.Label(data_frame, text="Orders CSV:").pack(fill=tk.X, padx=5, pady=2)
        self.orders_var = tk.StringVar(value=self.params['data_files']['orders'])
        ttk.Entry(data_frame, textvariable=self.orders_var).pack(fill=tk.X, padx=5, pady=2)
        
        # Load data button
        ttk.Button(data_frame, text="📂 Load Data", command=self.load_data).pack(fill=tk.X, padx=5, pady=5)
        
        # Optimization method section
        method_frame = ttk.LabelFrame(panel, text="Optimization Method", padding="10")
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Method radiobuttons with icons
        self.method_var = tk.StringVar(value="all")
        methods = [
            ("🔄 All Methods (Compare)", "all"),
            ("🧬 Pure GA", "ga"),
            ("🤖 OR-Tools with ML", "or_ml"),
            ("⚡ GA + OR (No Fitness Mod)", "ga_or"),
            ("🎯 GA + OR (Modified Fitness)", "ga_or_mod")
        ]
        
        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, value=value,
                          variable=self.method_var).pack(fill=tk.X, padx=5, pady=2)
        
        # GA Parameters section
        ga_frame = ttk.LabelFrame(panel, text="GA Parameters", padding="10")
        ga_frame.pack(fill=tk.X, pady=(0, 10))
        
        # GA parameters with modern styling
        self.ga_params = {}
        for param, value in self.params['ga_settings'].items():
            frame = ttk.Frame(ga_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(frame, text=param.replace('_', ' ').title() + ":").pack(side=tk.LEFT)
            self.ga_params[param] = tk.StringVar(value=str(value))
            ttk.Entry(frame, textvariable=self.ga_params[param], width=10).pack(side=tk.RIGHT)
        
        # Action buttons section
        action_frame = ttk.Frame(panel, padding="10")
        action_frame.pack(fill=tk.X)
        
        # Start button with icon
        self.start_button = ttk.Button(action_frame, text="▶️ Start Optimization",
                                     command=self.start_optimization)
        self.start_button.pack(fill=tk.X, pady=2)
        
        # Stop button with icon
        self.stop_button = ttk.Button(action_frame, text="⏹️ Stop",
                                    command=self.stop_optimization, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        return panel
    
    def _create_viz_panel(self):
        """Create the visualization panel"""
        panel = ttk.Frame(self.main_container)
        
        # Placeholder for visualization
        self.viz_label = ttk.Label(panel, text="Run optimization to see visualization")
        self.viz_label.pack(fill=tk.BOTH, expand=True)
        
        return panel
    
    def load_data(self):
        """Load data from specified files"""
        self.log_message("Loading data...")
        self.status_var.set("Loading data...")
        
        distance_file = self.distance_var.get()
        orders_file = self.orders_var.get()
        
        try:
            self.data_processor = DataProcessor()
            self.data_processor.load_data(distance_file, orders_file)
            
            self.log_message(f"📦 Loaded {len(self.data_processor.order_data)} orders")
            self.log_message(f"🏙️ Loaded {len(self.data_processor.cities)} cities")
            self.log_message("✅ Data loaded successfully")
            self.status_var.set("Data loaded successfully")
            
        except Exception as e:
            self.log_message(f"❌ Error loading data: {str(e)}", error=True)
            self.status_var.set("Error loading data")
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def start_optimization(self):
        """Start the optimization process"""
        if not self.data_processor:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        if self.is_running:
            messagebox.showwarning("Warning", "Optimization is already running")
            return
        
        # Get selected method
        method = self.method_var.get()
        if method == 'all':
            method = None  # None means run all methods
        
        # Get GA parameters
        ga_settings = {}
        for param, var in self.ga_params.items():
            try:
                # Convert to appropriate type
                if param in ['population_size', 'generations', 'elitism_count']:
                    ga_settings[param] = int(var.get())
                else:
                    ga_settings[param] = float(var.get())
            except ValueError:
                messagebox.showerror("Error", f"Invalid value for {param}: {var.get()}")
                return
        
        # Clear results
        self.clear_results()
        
        # Create solver
        self.solver = VRPSolver(self.data_processor)
        self.solver.set_main_window(self)  # Connect solver to main window
        
        # Update GA parameters for all optimizers that use GA
        optimizers_with_ga = [
            self.solver.ga_optimizer,
            self.solver.ga_or_optimizer,
            self.solver.ga_or_mod_optimizer
        ]
        
        for optimizer in optimizers_with_ga:
            if optimizer:
                optimizer.population_size = ga_settings['population_size']
                optimizer.max_generations = ga_settings['generations']
                optimizer.base_mutation_rate = ga_settings['mutation_rate']
                optimizer.crossover_rate = ga_settings['crossover_rate']
                optimizer.elitism_count = ga_settings['elitism_count']
                
                # If optimizer has a nested GA optimizer, update that too
                if hasattr(optimizer, 'ga_optimizer'):
                    optimizer.ga_optimizer.population_size = ga_settings['population_size']
                    optimizer.ga_optimizer.max_generations = ga_settings['generations']
                    optimizer.ga_optimizer.base_mutation_rate = ga_settings['mutation_rate']
                    optimizer.ga_optimizer.crossover_rate = ga_settings['crossover_rate']
                    optimizer.ga_optimizer.elitism_count = ga_settings['elitism_count']
        
        # Update UI state
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Optimization in progress...")
        
        # Start optimization in a separate thread
        self.optimization_thread = threading.Thread(
            target=self._run_optimization,
            args=(method,)
        )
        self.optimization_thread.start()
        
        # Start monitoring thread
        self._monitor_optimization()
    
    def _run_optimization(self, method):
        """Run the optimization process"""
        try:
            # Generate city coordinates for visualization
            cities_coordinates = self._generate_city_coordinates()
            
            # Create visualizer if not exists
            if not self.visualizer:
                # Schedule visualizer creation in main thread
                self.root.after(0, self._create_visualizer, cities_coordinates)
            
            # Connect visualizer to all optimizers
            if hasattr(self.solver, 'visualizer'):
                self.solver.visualizer = self.visualizer
            if hasattr(self.solver, 'ga_optimizer'):
                self.solver.ga_optimizer.visualizer = self.visualizer
            if hasattr(self.solver, 'or_ml_optimizer'):
                self.solver.or_ml_optimizer.visualizer = self.visualizer
            if hasattr(self.solver, 'ga_or_optimizer'):
                self.solver.ga_or_optimizer.visualizer = self.visualizer
                if hasattr(self.solver.ga_or_optimizer, 'ga_optimizer'):
                    self.solver.ga_or_optimizer.ga_optimizer.visualizer = self.visualizer
            if hasattr(self.solver, 'ga_or_mod_optimizer'):
                self.solver.ga_or_mod_optimizer.visualizer = self.visualizer
                if hasattr(self.solver.ga_or_mod_optimizer, 'ga_optimizer'):
                    self.solver.ga_or_mod_optimizer.ga_optimizer.visualizer = self.visualizer
            
            # Run optimization
            solution = self.solver.optimize(method)
            
            # Schedule visualization update in main thread
            self.root.after(0, self._update_visualization, solution)
            
        except Exception as e:
            # Schedule error handling in main thread
            self.root.after(0, self._handle_optimization_error, str(e))
        finally:
            self.is_running = False
            # Schedule UI updates in main thread
            self.root.after(0, self._update_ui_after_optimization)
    
    def _create_visualizer(self, cities_coordinates):
        """Create visualizer in main thread"""
        self.visualizer = ModernVisualizer(self.viz_panel, cities_coordinates)
    
    def _update_visualization(self, solution):
        """Update visualization in main thread"""
        if self.visualizer and solution:
            self.visualizer.plot_routes(solution)
            self._display_solution(solution)
    
    def _handle_optimization_error(self, error_msg):
        """Handle optimization error in main thread"""
        self.log_message(f"❌ Error during optimization: {error_msg}", error=True)
        messagebox.showerror("Error", f"Optimization failed: {error_msg}")
    
    def _update_ui_after_optimization(self):
        """Update UI elements after optimization in main thread"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Optimization completed")
    
    def _monitor_optimization(self):
        """Monitor optimization progress"""
        if self.is_running and self.solver:
            try:
                # Get current optimizer based on selected method
                method = self.method_var.get()
                current_optimizer = None
                
                if method == 'ga':
                    current_optimizer = self.solver.ga_optimizer
                elif method == 'ga_or':
                    current_optimizer = self.solver.ga_or_optimizer
                elif method == 'ga_or_mod':
                    current_optimizer = self.solver.ga_or_mod_optimizer
                
                # Update status if optimizer is available
                if current_optimizer and hasattr(current_optimizer, 'current_generation'):
                    generation = current_optimizer.current_generation
                    self.status_var.set(f"Generation {generation} in progress...")
            except Exception as e:
                print(f"Warning: Error in monitoring: {str(e)}")
            
            # Schedule next update
            if self.is_running:
                self.root.after(100, self._monitor_optimization)
    
    def stop_optimization(self):
        """Stop the optimization process"""
        if self.is_running:
            self.is_running = False
            if self.solver:
                self.solver.ga_optimizer.stop()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("Optimization stopped")
    
    def _display_solution(self, solution):
        """Display detailed solution with route information"""
        if not solution:
            return
            
        total_cost = sum(route.total_cost for route in solution)
        total_distance = sum(route.total_distance for route in solution)
        total_parcels = sum(len(route.parcels) for route in solution)
        
        info = "🎯 BEST SOLUTION FOUND\n"
        info += "═" * 50 + "\n\n"
        
        # Overall Summary
        info += "📊 OVERALL SUMMARY\n"
        info += "─" * 30 + "\n"
        info += f"🚛 Total Routes: {len(solution)}\n"
        info += f"🛣️ Total Distance: {total_distance:.2f} km\n"
        info += f"💰 Total Cost: ${total_cost:.2f}\n"
        info += f"📦 Total Parcels: {total_parcels}\n"
        
        # Calculate overall efficiency
        avg_utilization = np.mean([route.get_total_weight()/route.vehicle_capacity 
                                 for route in solution]) * 100
        info += f"📈 Average Vehicle Utilization: {avg_utilization:.1f}%\n\n"
        
        # Detailed Route Information
        info += "📍 DETAILED ROUTE INFORMATION\n"
        info += "═" * 50 + "\n\n"
        
        for i, route in enumerate(solution, 1):
            info += f"🚚 Route #{i}\n"
            info += "─" * 30 + "\n"
            info += f"Vehicle: {route.vehicle_id}\n"
            info += f"Distance: {route.total_distance:.2f} km\n"
            info += f"Cost: ${route.total_cost:.2f}\n"
            info += f"Load: {route.get_total_weight():.1f}/{route.vehicle_capacity:.1f} kg "
            info += f"({(route.get_total_weight()/route.vehicle_capacity*100):.1f}%)\n"
            
            # Delivery sequence
            info += "\n📦 Delivery Sequence:\n"
            current_city = "Depot"
            for parcel in route.parcels:
                info += f"   {current_city} → {parcel.destination.city_name}"
                info += f" ({parcel.weight:.1f}kg)\n"
                current_city = parcel.destination.city_name
            info += f"   {current_city} → Depot\n\n"
        
        self.log_message("Best solution found!")
        
        # Create a popup window to display the detailed solution
        solution_window = tk.Toplevel(self.root)
        solution_window.title("Best Solution Details")
        solution_window.geometry("800x600")
        
        # Add scrolled text widget
        text_widget = scrolledtext.ScrolledText(solution_window, wrap=tk.WORD, 
                                              width=80, height=40)
        text_widget.pack(expand=True, fill='both', padx=10, pady=10)
        text_widget.insert(tk.END, info)
        text_widget.configure(state='disabled')  # Make read-only
    
    def _generate_city_coordinates(self):
        """Generate coordinates for cities"""
        if not self.data_processor:
            return {}
            
        coordinates = {}
        cities = list(self.data_processor.cities)
        
        # Generate coordinates in a circular layout
        n = len(cities)
        for i, city in enumerate(cities):
            angle = 2 * np.pi * i / n
            radius = 100
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            coordinates[city] = (x, y)
            
        return coordinates
    
    def log_message(self, message: str, error: bool = False):
        """Log a message with modern formatting"""
        timestamp = time.strftime("%H:%M:%S")
        prefix = "❌ " if error else "ℹ️ "
        self.status_var.set(f"{prefix}{message}")
    
    def clear_results(self):
        """Clear previous results"""
        if self.visualizer:
            self.visualizer = None
        self.viz_label.config(text="Run optimization to see visualization")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

    def _create_routes_section(self):
        """Create the routes section with scores display"""
        routes_frame = ttk.LabelFrame(self.main_container, text="Routes", padding="5")
        routes_frame.grid(row=2, column=0, rowspan=2, sticky="nsew", padx=5)
        
        # Routes list
        self.routes_text = tk.Text(routes_frame, width=40, height=10)
        self.routes_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(routes_frame, orient="vertical", command=self.routes_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.routes_text.configure(yscrollcommand=scrollbar.set)
        
        # Create scores frame
        scores_frame = ttk.LabelFrame(routes_frame, text="Route Scores", padding="5")
        scores_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        # Add score labels and values with improved styling
        score_labels = {
            'composite_score': "Overall Score:",
            'distance_score': "Distance Efficiency:",
            'cost_efficiency_score': "Cost Efficiency:",
            'capacity_utilization_score': "Capacity Utilization:",
            'parcel_efficiency_score': "Parcel Efficiency:",
            'route_structure_score': "Route Structure:"
        }
        
        for i, (var_name, label_text) in enumerate(score_labels.items()):
            # Create label with right alignment
            label = ttk.Label(scores_frame, text=label_text, anchor="e")
            label.grid(row=i, column=0, sticky="e", padx=(5,10), pady=2)
            
            # Create value label with modern styling
            value_frame = ttk.Frame(scores_frame, style="Score.TFrame")
            value_frame.grid(row=i, column=1, sticky="ew", pady=2)
            
            value_label = ttk.Label(value_frame, 
                                  textvariable=self.score_vars[var_name],
                                  width=8,
                                  anchor="center",
                                  style="Score.TLabel")
            value_label.pack(fill="x", padx=5)
            
            # Configure special styling for composite score
            if var_name == 'composite_score':
                value_label.configure(style="CompositeScore.TLabel")
                ttk.Separator(scores_frame).grid(row=i+1, column=0, columnspan=2, 
                                               sticky="ew", pady=5)
        
        # Configure weights for routes section
        routes_frame.grid_columnconfigure(0, weight=1)
        routes_frame.grid_rowconfigure(0, weight=1)
        scores_frame.grid_columnconfigure(1, weight=1)

    def update_scores(self, scores):
        """Update the score display with new values for all methods"""
        if self.visualizer:
            self.visualizer.update_score_plot(scores)

    def _setup_styles(self):
        """Setup custom styles for the score display"""
        style = ttk.Style()
        
        # Configure frame style
        style.configure("Score.TFrame", background="#f0f0f0")
        
        # Configure label styles
        style.configure("Score.TLabel",
                       background="#f0f0f0",
                       font=("Arial", 9))
        
        style.configure("CompositeScore.TLabel",
                       background="#e1e1ff",
                       font=("Arial", 10, "bold"))

def main():
    """Main entry point"""
    app = MainWindow()
    app.run()

if __name__ == "__main__":
    main()