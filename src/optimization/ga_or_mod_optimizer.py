import time
import copy
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set

from src.models.route import Route
from src.optimization.base_optimizer import BaseOptimizer
from src.optimization.ga_optimizer import GAOptimizer
from src.optimization.or_ml_optimizer import ORToolsMLOptimizer
from src.optimization.ga_or_optimizer import GAOROptimizer
from src.visualization.fitness_plotter import FitnessPlotter

class GAORModifiedOptimizer(GAOROptimizer):
    """
    GA + OR-Tools optimizer with modified fitness function.
    This is Method 4 in the four-method system.
    
    This uses OR-Tools solutions to modify the fitness function
    to favor solutions with characteristics similar to those
    from OR-Tools.
    """
    
    def __init__(self, data_processor):
        """
        Initialize the GA + OR-Tools optimizer with modified fitness.
        
        Args:
            data_processor: DataProcessor containing problem data
        """
        super().__init__(data_processor)
        
        # OR-Tools patterns that we extract and use for fitness modification
        self.or_patterns = {}
        
        # Weight for OR-Tools pattern similarity in fitness
        self.pattern_weight = 0.2  # 20% influence from patterns
        
        # Initialize tracking variables
        self.current_generation = 0
        self.best_fitness = float('-inf')
        self.best_fitness_history = []
        self.current_population_fitness = []
        self.avg_fitness_history = []
    
    def optimize(self) -> List[Route]:
        """
        Execute the GA + OR-Tools optimization with modified fitness.
        
        Returns:
            List[Route]: The best solution found
        """
        print("\nStarting GA + OR-Tools (with modified fitness) optimization...")
        start_time = time.time()
        
        # Reset tracking variables
        self.current_generation = 0
        self.best_fitness = float('-inf')
        self.best_fitness_history = []
        self.current_population_fitness = []
        self.avg_fitness_history = []
        
        # First, get OR-Tools solutions
        print("Getting OR-Tools solutions for initial population and pattern extraction...")
        or_solutions = self._get_or_tools_solutions()
        
        # Extract patterns from OR-Tools solutions
        self.or_patterns = self._extract_patterns(or_solutions)
        print(f"Extracted {len(self.or_patterns)} patterns from OR-Tools solutions")
        
        # Initialize population with mix of OR-Tools and random solutions
        population = self._initialize_hybrid_population(or_solutions)
        
        # Evaluate initial population with modified fitness
        fitness_scores = [self._calculate_modified_fitness(solution) for solution in population]
        self.current_population_fitness = fitness_scores
        
        # Track best solution and average fitness
        best_idx = fitness_scores.index(max(fitness_scores))
        self.best_solution = copy.deepcopy(population[best_idx])
        self.best_fitness = fitness_scores[best_idx]
        
        # Record initial fitness data
        self.best_fitness_history.append(self.best_fitness)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        self.avg_fitness_history.append(avg_fitness)
        print(f"Initial best fitness: {self.best_fitness:.4f}, Average fitness: {avg_fitness:.4f}")
        
        # Main GA loop
        for generation in range(self.generations):
            if not self._running:
                print("Optimization stopped by user.")
                break
                
            # Selection
            selected = self._selection(population, fitness_scores)
            
            # Create new population
            new_population = []
            
            # Elitism - keep best solutions
            sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
            for i in range(self.elitism_count):
                new_population.append(copy.deepcopy(population[sorted_indices[i]]))
            
            # Crossover and Mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = selected[random.randint(0, len(selected) - 1)]
                parent2 = selected[random.randint(0, len(selected) - 1)]
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring = self._crossover(parent1, parent2)
                else:
                    offspring = copy.deepcopy(parent1)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring = self._mutate(offspring)
                
                # Repair solution if needed
                offspring = self._repair_solution(offspring)
                
                # Add to new population
                new_population.append(offspring)
            
            # Evaluate new population
            fitness_scores = [self._calculate_modified_fitness(solution) for solution in new_population]
            self.current_population_fitness = fitness_scores
            
            # Update best solution if better found
            current_best_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[current_best_idx] > self.best_fitness:
                self.best_solution = copy.deepcopy(new_population[current_best_idx])
                self.best_fitness = fitness_scores[current_best_idx]
                print(f"Generation {self.current_generation+1}: New best fitness: {self.best_fitness:.4f}")
            
            # Record fitness data
            self.best_fitness_history.append(self.best_fitness)
            current_avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.avg_fitness_history.append(current_avg_fitness)
            
            # Print detailed progress
            if (self.current_generation + 1) % 10 == 0:
                print(f"Generation {self.current_generation+1}/{self.generations}:")
                print(f"  Best fitness: {self.best_fitness:.4f}")
                print(f"  Average fitness: {current_avg_fitness:.4f}")
            
            # Update population and increment generation
            population = new_population
            self.current_generation += 1
            
            # Gradually reduce pattern weight over generations
            if self.pattern_weight > 0.05:  # Don't go below 5%
                self.pattern_weight *= 0.98  # Decay rate
        
        # Finalize and report
        elapsed_time = time.time() - start_time
        print(f"\nGA + OR-Tools with modified fitness completed in {elapsed_time:.2f} seconds.")
        print(f"Final best fitness: {self.best_fitness:.4f}")
        
        evaluation = self.evaluate_solution(self.best_solution)
        print(f"Parcels delivered: {evaluation['parcels_delivered']}")
        print(f"Total cost: ${evaluation['total_cost']:.2f}")
        
        # Save fitness evolution plot
        self.fitness_plotter.plot(save_path='results/ga_or_mod_fitness_evolution.png')
        
        return self.best_solution
    
    def _extract_patterns(self, or_solutions: List[List[Route]]) -> Dict:
        """
        Extract patterns from OR-Tools solutions to guide fitness calculation.
        
        Args:
            or_solutions: List of solutions from OR-Tools
            
        Returns:
            Dictionary with pattern information
        """
        patterns = {}
        
        # Flatten all routes from all solutions
        all_routes = [route for solution in or_solutions for route in solution]
        
        if not all_routes:
            return patterns
        
        # 1. Average parcels per route
        avg_parcels = sum(len(route.parcels) for route in all_routes) / len(all_routes)
        patterns['avg_parcels_per_route'] = avg_parcels
        
        # 2. Average distance per route
        avg_distance = sum(route.total_distance for route in all_routes) / len(all_routes)
        patterns['avg_distance_per_route'] = avg_distance
        
        # 3. Truck type distribution
        truck_types = {}
        for route in all_routes:
            truck_type = route.vehicle_id.split('_')[1]
            truck_types[truck_type] = truck_types.get(truck_type, 0) + 1
        
        # Convert to percentages
        total_routes = len(all_routes)
        truck_distribution = {t: count / total_routes for t, count in truck_types.items()}
        patterns['truck_type_distribution'] = truck_distribution
        
        # 4. Load factors
        load_factors = [route.get_total_weight() / route.vehicle_capacity for route in all_routes 
                      if route.vehicle_capacity > 0]
        avg_load_factor = sum(load_factors) / len(load_factors) if load_factors else 0
        patterns['avg_load_factor'] = avg_load_factor
        
        # 5. Parcel clustering - how many parcels go to the same destination
        destination_counts = {}
        for route in all_routes:
            for parcel in route.parcels:
                dest = parcel.destination.city_name
                destination_counts[dest] = destination_counts.get(dest, 0) + 1
        
        patterns['destination_clustering'] = destination_counts
        
        return patterns
    
    def _calculate_modified_fitness(self, solution: List[Route]) -> float:
        """
        Calculate fitness with consideration of OR-Tools patterns.
        
        Args:
            solution: Solution to evaluate
            
        Returns:
            Modified fitness score
        """
        # Get base fitness
        base_fitness = self.calculate_fitness(solution)
        
        # If no patterns or no solution, return base fitness
        if not self.or_patterns or not solution:
            return base_fitness
        
        # Calculate pattern similarity
        pattern_score = self._calculate_pattern_similarity(solution)
        
        # Combine scores
        return (1 - self.pattern_weight) * base_fitness + self.pattern_weight * pattern_score
    
    def _calculate_pattern_similarity(self, solution: List[Route]) -> float:
        """
        Calculate how similar a solution is to observed OR-Tools patterns.
        
        Args:
            solution: Solution to evaluate
            
        Returns:
            Pattern similarity score (0-1)
        """
        similarity_scores = []
        
        # Skip if no solution
        if not solution:
            return 0
        
        # 1. Average parcels per route similarity
        if 'avg_parcels_per_route' in self.or_patterns:
            solution_avg_parcels = sum(len(route.parcels) for route in solution) / len(solution)
            or_avg_parcels = self.or_patterns['avg_parcels_per_route']
            
            # Calculate similarity (1 - normalized difference)
            parcels_diff = abs(solution_avg_parcels - or_avg_parcels) / max(or_avg_parcels, 1)
            parcels_similarity = max(0, 1 - min(parcels_diff, 1))
            similarity_scores.append(parcels_similarity)
        
        # 2. Load factor similarity
        if 'avg_load_factor' in self.or_patterns:
            solution_load_factors = [route.get_total_weight() / route.vehicle_capacity 
                                   for route in solution if route.vehicle_capacity > 0]
            solution_avg_load = sum(solution_load_factors) / len(solution_load_factors) if solution_load_factors else 0
            or_avg_load = self.or_patterns['avg_load_factor']
            
            # Calculate similarity
            load_diff = abs(solution_avg_load - or_avg_load) / max(or_avg_load, 0.1)
            load_similarity = max(0, 1 - min(load_diff, 1))
            similarity_scores.append(load_similarity)
        
        # 3. Truck type distribution similarity
        if 'truck_type_distribution' in self.or_patterns:
            # Count truck types in solution
            solution_truck_types = {}
            for route in solution:
                truck_type = route.vehicle_id.split('_')[1]
                solution_truck_types[truck_type] = solution_truck_types.get(truck_type, 0) + 1
            
            # Convert to percentages
            solution_total = len(solution)
            solution_distribution = {t: count / solution_total 
                                  for t, count in solution_truck_types.items()}
            
            # Calculate distribution difference
            or_distribution = self.or_patterns['truck_type_distribution']
            all_types = set(solution_distribution.keys()) | set(or_distribution.keys())
            
            difference = 0
            for truck_type in all_types:
                sol_pct = solution_distribution.get(truck_type, 0)
                or_pct = or_distribution.get(truck_type, 0)
                difference += abs(sol_pct - or_pct)
            
            # Normalize difference (max difference would be 2)
            truck_similarity = max(0, 1 - min(difference / 2, 1))
            similarity_scores.append(truck_similarity)
        
        # 4. Distance efficiency (relative to OR-Tools)
        if 'avg_distance_per_route' in self.or_patterns:
            solution_avg_distance = sum(route.total_distance for route in solution) / len(solution)
            or_avg_distance = self.or_patterns['avg_distance_per_route']
            
            # If solution has shorter distances, that's good
            if solution_avg_distance <= or_avg_distance:
                distance_similarity = 1
            else:
                # Penalize for longer distances
                distance_diff = (solution_avg_distance - or_avg_distance) / or_avg_distance
                distance_similarity = max(0, 1 - min(distance_diff, 1))
                
            similarity_scores.append(distance_similarity)
        
        # 5. Destination clustering similarity
        if 'destination_clustering' in self.or_patterns and solution:
            # Count destinations in solution
            solution_destinations = {}
            for route in solution:
                for parcel in route.parcels:
                    dest = parcel.destination.city_name
                    solution_destinations[dest] = solution_destinations.get(dest, 0) + 1
            
            # Calculate Jaccard similarity between destination sets
            or_dests = set(self.or_patterns['destination_clustering'].keys())
            solution_dests = set(solution_destinations.keys())
            
            if or_dests or solution_dests:
                jaccard = len(or_dests & solution_dests) / len(or_dests | solution_dests)
                similarity_scores.append(jaccard)
        
        # Average all similarity scores
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0