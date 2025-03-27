import time
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

from src.models.route import Route
from src.optimization.base_optimizer import BaseOptimizer
from src.optimization.ga_optimizer import GAOptimizer
from src.optimization.or_ml_optimizer import ORToolsMLOptimizer
from src.optimization.ga_or_optimizer import GAOROptimizer
from src.optimization.ga_or_mod_optimizer import GAORModifiedOptimizer

class VRPSolver:
    """
    Main solver that coordinates all four optimization methods:
    1. Pure GA (Genetic Algorithm)
    2. OR-Tools with ML
    3. GA + OR without fitness modification
    4. GA + OR with modified fitness
    
    It compares and selects the best solution based on a consistent fitness evaluation.
    """
    
    def __init__(self, data_processor):
        """
        Initialize the VRP Solver with a data processor.
        
        Args:
            data_processor: DataProcessor containing problem data
        """
        self.data_processor = data_processor
        
        # Initialize the four optimizers
        self.ga_optimizer = GAOptimizer(data_processor)
        self.or_ml_optimizer = ORToolsMLOptimizer(data_processor)
        self.ga_or_optimizer = GAOROptimizer(data_processor)
        self.ga_or_mod_optimizer = GAORModifiedOptimizer(data_processor)
        
        # Solver results
        self.results = {
            'ga': {'solution': None, 'fitness': 0, 'metrics': None},
            'or_ml': {'solution': None, 'fitness': 0, 'metrics': None},
            'ga_or': {'solution': None, 'fitness': 0, 'metrics': None},
            'ga_or_mod': {'solution': None, 'fitness': 0, 'metrics': None}
        }
        
        # Best solution
        self.best_solution = None
        self.best_fitness = 0.0
        self.best_method = None
    
    def optimize(self, method: str = None):
        """
        Execute optimization with specified method or all methods if none specified.
        
        Args:
            method: Optional method to run ('ga', 'or_ml', 'ga_or', or 'ga_or_mod')
        """
        if method:
            # Run single method
            optimizer = self._get_optimizer(method)
            if optimizer:
                solution = optimizer.optimize()
                self._update_results(method, solution, optimizer)
                
                # Update main window scores if available
                if hasattr(self, 'main_window'):
                    scores = optimizer.calculate_route_comparison_score(solution)
                    self.main_window.update_scores(scores)
        else:
            # Run all methods
            print("\nRunning all optimization methods...")
            
            # Method 1: Pure GA
            print("\n--- Method 1: Pure GA ---")
            self.results['ga']['solution'] = self.ga_optimizer.optimize()
            self.results['ga']['fitness'] = self.ga_optimizer.calculate_fitness(self.results['ga']['solution'])
            self.results['ga']['metrics'] = self.ga_optimizer.evaluate_solution(self.results['ga']['solution'])
            self.results['ga']['scores'] = self.ga_optimizer.calculate_route_comparison_score(self.results['ga']['solution'])
            
            # Method 2: OR-Tools with ML
            print("\n--- Method 2: OR-Tools with ML ---")
            self.results['or_ml']['solution'] = self.or_ml_optimizer.optimize()
            self.results['or_ml']['fitness'] = self.ga_optimizer.calculate_fitness(self.results['or_ml']['solution'])
            self.results['or_ml']['metrics'] = self.ga_optimizer.evaluate_solution(self.results['or_ml']['solution'])
            self.results['or_ml']['scores'] = self.ga_optimizer.calculate_route_comparison_score(self.results['or_ml']['solution'])
            
            # Method 3: GA + OR (no fitness modification)
            print("\n--- Method 3: GA + OR ---")
            self.results['ga_or']['solution'] = self.ga_or_optimizer.optimize()
            self.results['ga_or']['fitness'] = self.ga_optimizer.calculate_fitness(self.results['ga_or']['solution'])
            self.results['ga_or']['metrics'] = self.ga_optimizer.evaluate_solution(self.results['ga_or']['solution'])
            self.results['ga_or']['scores'] = self.ga_optimizer.calculate_route_comparison_score(self.results['ga_or']['solution'])
            
            # Method 4: GA + OR with modified fitness
            print("\n--- Method 4: GA + OR with modified fitness ---")
            self.results['ga_or_mod']['solution'] = self.ga_or_mod_optimizer.optimize()
            self.results['ga_or_mod']['fitness'] = self.ga_optimizer.calculate_fitness(self.results['ga_or_mod']['solution'])
            self.results['ga_or_mod']['metrics'] = self.ga_optimizer.evaluate_solution(self.results['ga_or_mod']['solution'])
            self.results['ga_or_mod']['scores'] = self.ga_optimizer.calculate_route_comparison_score(self.results['ga_or_mod']['solution'])
            
            # Find best method based on composite score
            best_method = max(self.results.keys(), 
                            key=lambda k: self.results[k]['scores']['composite_score'])
            
            # Update main window with all method scores if available
            if hasattr(self, 'main_window'):
                method_scores = {
                    self._get_method_display_name(method): result['scores']
                    for method, result in self.results.items()
                }
                self.main_window.update_scores(method_scores)
            
            # Print comparison
            self._print_comparison()
            
            return self.results[best_method]['solution']
    
    def _update_results(self, method: str, solution: List[Route], optimizer: BaseOptimizer):
        """
        Update results for a single method.
        
        Args:
            method: Method name
            solution: Optimized solution
            optimizer: Optimizer used
        """
        self.results[method]['solution'] = solution
        self.results[method]['fitness'] = optimizer.calculate_fitness(solution)
        self.results[method]['metrics'] = optimizer.evaluate_solution(solution)
        self.results[method]['scores'] = optimizer.calculate_route_comparison_score(solution)
    
    def _get_optimizer(self, method: str) -> BaseOptimizer:
        """
        Get optimizer for a specified method.
        
        Args:
            method: Method name
            
        Returns:
            Optimizer for the method
        """
        if method == 'ga':
            return self.ga_optimizer
        elif method == 'or_ml':
            return self.or_ml_optimizer
        elif method == 'ga_or':
            return self.ga_or_optimizer
        elif method == 'ga_or_mod':
            return self.ga_or_mod_optimizer
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _print_comparison(self):
        """Print comparison of all methods"""
        print("\n=== Results Comparison ===")
        
        # Calculate comparison scores for each method
        method_scores = {}
        for method_name, result in self.results.items():
            metrics = result['metrics']
            fitness = result['fitness']
            comparison_scores = self.ga_optimizer.calculate_route_comparison_score(result['solution'])
            
            print(f"\n{self._get_method_display_name(method_name)}:")
            print(f"  Composite Score: {comparison_scores['composite_score']:.4f}")
            print(f"  Detailed Scores:")
            print(f"    - Distance Efficiency: {comparison_scores['distance_score']:.4f}")
            print(f"    - Cost Efficiency: {comparison_scores['cost_efficiency_score']:.4f}")
            print(f"    - Capacity Utilization: {comparison_scores['capacity_utilization_score']:.4f}")
            print(f"    - Parcel Efficiency: {comparison_scores['parcel_efficiency_score']:.4f}")
            print(f"    - Route Structure: {comparison_scores['route_structure_score']:.4f}")
            print(f"\n  Route Metrics:")
            print(f"    - Parcels: {metrics['parcels_delivered']}")
            print(f"    - Cost: ${metrics['total_cost']:.2f}")
            print(f"    - Distance: {metrics['total_distance']:.2f} km")
            print(f"    - Routes: {metrics['num_routes']}")
            print(f"    - Avg Load Factor: {metrics['avg_load_factor']*100:.1f}%")
            
            method_scores[method_name] = comparison_scores['composite_score']
        
        # Determine best method based on composite score
        best_method = max(method_scores.items(), key=lambda x: x[1])[0]
        print(f"\nBest method: {self._get_method_display_name(best_method)} with composite score {method_scores[best_method]:.4f}")
    
    def _print_metrics(self, solution: List[Route]):
        """
        Print metrics for a solution.
        
        Args:
            solution: Solution to evaluate
        """
        metrics = self.ga_optimizer.evaluate_solution(solution)
        
        print(f"Parcels Delivered: {metrics['parcels_delivered']}")
        print(f"Total Cost: ${metrics['total_cost']:.2f}")
        print(f"Total Distance: {metrics['total_distance']:.2f} km")
        print(f"Number of Routes: {metrics['num_routes']}")
        print(f"Average Load Factor: {metrics['avg_load_factor']*100:.1f}%")
    
    def _save_comparison_chart(self):
        """Create and save comparison chart of all methods"""
        method_names = [self._get_method_display_name(m) for m in self.results.keys()]
        fitness_scores = [result['fitness'] for result in self.results.values()]
        parcels = [result['metrics']['parcels_delivered'] for result in self.results.values()]
        costs = [result['metrics']['total_cost'] for result in self.results.values()]
        distances = [result['metrics']['total_distance'] for result in self.results.values()]
        routes = [result['metrics']['num_routes'] for result in self.results.values()]
        
        # Create figure and axes
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('VRP Optimization Methods Comparison', fontsize=16)
        
        # Fitness scores
        axs[0, 0].bar(method_names, fitness_scores, color='skyblue')
        axs[0, 0].set_title('Fitness Scores')
        axs[0, 0].set_ylabel('Score (higher is better)')
        plt.setp(axs[0, 0].get_xticklabels(), rotation=45, ha='right')
        
        # Parcels delivered
        axs[0, 1].bar(method_names, parcels, color='lightgreen')
        axs[0, 1].set_title('Parcels Delivered')
        axs[0, 1].set_ylabel('Count')
        plt.setp(axs[0, 1].get_xticklabels(), rotation=45, ha='right')
        
        # Total cost
        axs[1, 0].bar(method_names, costs, color='salmon')
        axs[1, 0].set_title('Total Cost')
        axs[1, 0].set_ylabel('Cost ($)')
        plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha='right')
        
        # Number of routes
        axs[1, 1].bar(method_names, routes, color='plum')
        axs[1, 1].set_title('Number of Routes')
        axs[1, 1].set_ylabel('Count')
        plt.setp(axs[1, 1].get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save figure
        plt.savefig('results/methods_comparison.png')
        print("Comparison chart saved to 'results/methods_comparison.png'")
    
    def _get_method_display_name(self, method: str) -> str:
        """
        Get display name for a method.
        
        Args:
            method: Method key
            
        Returns:
            Display name
        """
        display_names = {
            'ga': 'Pure GA',
            'or_ml': 'OR-Tools + ML',
            'ga_or': 'GA + OR (no mod)',
            'ga_or_mod': 'GA + OR (mod fitness)'
        }
        return display_names.get(method, method)
    
    def stop(self):
        """Stop all optimization processes"""
        self.ga_optimizer.stop()
        self.ga_or_optimizer.stop()
        self.ga_or_mod_optimizer.stop()

    def set_main_window(self, main_window):
        """Set the main window reference for score updates"""
        self.main_window = main_window