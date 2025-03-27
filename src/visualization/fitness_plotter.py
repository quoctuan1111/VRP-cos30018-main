import matplotlib.pyplot as plt
import numpy as np
from typing import List
import os

class FitnessPlotter:
    """Class for plotting fitness evolution"""
    
    def __init__(self):
        """Initialize fitness plotter"""
        self.generation_numbers = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def add_generation_data(self, generation: int, fitness_scores: List[float], best_fitness: float):
        """Add data for a new generation"""
        self.generation_numbers.append(generation)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(np.mean(fitness_scores))
        
    def plot(self, save_path: str = None):
        """Plot fitness evolution"""
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot average fitness
        ax.plot(self.generation_numbers, self.avg_fitness_history,
                'b-', label='Average Fitness', alpha=0.6)
        
        # Plot best fitness
        ax.plot(self.generation_numbers, self.best_fitness_history,
                'g-', label='Best Fitness', linewidth=2)
        
        # Add markers for latest points
        if self.generation_numbers:
            ax.plot(self.generation_numbers[-1], self.best_fitness_history[-1],
                    'go', markersize=8)
            ax.plot(self.generation_numbers[-1], self.avg_fitness_history[-1],
                    'bo', markersize=8)
        
        # Customize plot
        ax.set_title('Fitness Evolution')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Score')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        
        # Adjust plot limits
        if self.generation_numbers:
            ax.set_xlim(-1, max(self.generation_numbers) + 1)
            all_fitness = self.best_fitness_history + self.avg_fitness_history
            if all_fitness:
                ymin, ymax = min(all_fitness), max(all_fitness)
                y_margin = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
                ax.set_ylim(ymin - y_margin, ymax + y_margin)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        # Close the figure to free memory
        plt.close(fig)
        
    def clear(self):
        """Clear all data"""
        self.generation_numbers.clear()
        self.best_fitness_history.clear()
        self.avg_fitness_history.clear()
