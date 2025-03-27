#!/usr/bin/env python3
import os
import sys
import argparse
from scripts.check_system import check_system_requirements

def main():
    """Main entry point for the VRP system"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vehicle Routing Problem System")
    parser.add_argument('--gui', action='store_true', help='Run with graphical user interface')
    parser.add_argument('--method', type=str, choices=['all', 'ga', 'or_ml', 'ga_or', 'ga_or_mod'],
                      default='all', help='Optimization method to use')
    parser.add_argument('--distance', type=str, default='distance.csv',
                      help='Distance CSV file')
    parser.add_argument('--orders', type=str, default='order_large.csv',
                      help='Orders CSV file')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize solution after optimization')
    parser.add_argument('--train-ml', action='store_true',
                      help='Train ML model only')
    args = parser.parse_args()
    
    # First check system requirements
    print("\n=== Checking System Requirements ===")
    check_system_requirements()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    if args.train_ml:
        # Run ML training script
        print("\n=== Training ML Model ===")
        from scripts.train_ml_model import main as train_ml_main
        sys.argv = [sys.argv[0]]  # Reset sys.argv
        if args.distance:
            sys.argv.extend(['--distance', args.distance])
        if args.orders:
            sys.argv.extend(['--orders', args.orders])
        train_ml_main()
        
    elif args.gui:
        # Start the main GUI application
        print("\n=== Starting GUI Application ===")
        from src.gui.main_window import main as gui_main
        gui_main()
        
    else:
        # Run VRP solver from command line
        print("\n=== Running VRP Solver ===")
        from scripts.run_vrp_solver import main as solver_main
        sys.argv = [sys.argv[0]]  # Reset sys.argv
        if args.method:
            sys.argv.extend(['--method', args.method])
        if args.distance:
            sys.argv.extend(['--distance', args.distance])
        if args.orders:
            sys.argv.extend(['--orders', args.orders])
        if args.visualize:
            sys.argv.append('--visualize')
        solver_main()

if __name__ == "__main__":
    main()