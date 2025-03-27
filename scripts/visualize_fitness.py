import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import DataProcessor
from src.optimization.ga_optimizer import GAOptimizer

def main():
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize data and optimizer
    print("Loading data...")
    data_processor = DataProcessor()
    data_processor.load_data('distance.csv', 'order_large.csv')
    
    print("Running optimization with fitness tracking...")
    optimizer = GAOptimizer(data_processor)
    optimizer.optimize()
    
    print("\nFitness plot has been saved to 'results/fitness_evolution.png'")

if __name__ == "__main__":
    main()
