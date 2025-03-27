import random
import copy
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import aiohttp
import asyncio
import json

from src.models.route import Route
from src.models.location import Location
from src.models.parcel import Parcel
from src.optimization.base_optimizer import BaseOptimizer
from src.visualization.fitness_plotter import FitnessPlotter

# Define Solution type alias
Solution = List[Route]

class GAOptimizer(BaseOptimizer):
    """
    Pure Genetic Algorithm implementation for VRP.
    This is Method 1 in the four-method system.
    """
    
    def __init__(self, data_processor):
        """
        Initialize the GA optimizer with enhanced parameters.
        """
        super().__init__(data_processor)
        
        # Initialize caches first
        self._distance_cache = {}
        self._route_cache = {}
        self._fitness_cache = {}
        self._location_cache = {}
        self.MAX_CACHE_SIZE = 2000          # Increased cache size
        
        # Core parameters - adjusted for better exploration
        self.population_size = 200      # Increased for more diversity
        self.max_generations = 50          # More generations to find improvements
        self.base_mutation_rate = 0.5   # Base mutation rate
        self.crossover_rate = 0.85      # Slightly reduced to maintain good solutions
        self.elitism_count = 2          # Keep best solutions
        
        # Adaptive parameters
        self.min_mutation_rate = 0.3
        self.max_mutation_rate = 0.8
        self.diversity_weight = 0.4     # Weight for diversity in selection
        
        # Early stopping - more patient
        self.early_stopping_generations = 80  # Much more patient
        self.improvement_threshold = 5e-7     # More sensitive to tiny improvements
        
        # Route building parameters
        self.min_parcels_per_route = 4
        self.max_route_length = 20           # Allow longer routes
        self.merge_probability = 0.6         # More aggressive merging
        
        # Diversity parameters
        self.diversity_threshold = 0.01
        self.max_stagnation = 100           # More patience
        self.tournament_size = 4             # Increased selection pressure
        
        # Initialize
        self.warehouse_location = self._get_warehouse_location()
        self.fitness_plotter = FitnessPlotter()
        self._running = True
        self.best_fitness_history = []
        self.population_diversity_history = []
        
        # Initialize population
        self.population = None
        
        # Initialize stop flag and visualization
        self.stop_flag = False
        self.current_generation = 0
        self.current_population_fitness = []
        self.best_fitness = float('-inf')
        self.visualizer = None
        
        self.websocket_url = "ws://localhost:8000/ws"
        
    def set_visualizer(self, visualizer):
        """Set the visualizer for real-time updates"""
        self.visualizer = visualizer
        
    async def send_fitness_update(self, generation: int, best_fitness: float, avg_fitness: float):
        """Send fitness update to WebSocket server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(self.websocket_url) as ws:
                    await ws.send_json({
                        "type": "fitness_update",
                        "generation": generation,
                        "best_fitness": best_fitness,
                        "avg_fitness": avg_fitness
                    })
        except Exception as e:
            print(f"Failed to send fitness update: {str(e)}")
            
    async def send_route_update(self, routes: List[Route]):
        """Send route update to WebSocket server"""
        try:
            route_data = []
            for route in routes:
                route_data.append({
                    "vehicle_id": route.vehicle_id,
                    "locations": [
                        {
                            "city_name": loc.city_name,
                            "lat": loc.lat,
                            "lon": loc.lon
                        } for loc in route.locations
                    ],
                    "total_distance": route.total_distance,
                    "total_cost": route.total_cost
                })
                
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(self.websocket_url) as ws:
                    await ws.send_json({
                        "type": "route_update",
                        "routes": route_data
                    })
        except Exception as e:
            print(f"Failed to send route update: {str(e)}")
            
    def optimize(self) -> List[Route]:
        """Execute the genetic algorithm optimization with real-time updates"""
        print("\nStarting Genetic Algorithm optimization...")
        start_time = time.time()
        
        # Initialize population
        self.population = self._initialize_population()
        
        # Initialize tracking variables
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solution = None
        self.best_fitness = float('-inf')
        
        # Main evolution loop
        generation = 0
        no_improvement = 0
        
        while generation < self.max_generations and not self.stop_flag:
            # Evaluate population
            population_fitness = [(solution, self.calculate_fitness(solution))
                               for solution in self.population]
            
            # Sort by fitness
            population_fitness.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate statistics
            current_best_fitness = population_fitness[0][1]
            avg_fitness = sum(fitness for _, fitness in population_fitness) / len(population_fitness)
            
            # Track history
            self.best_fitness_history.append(current_best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            # Update visualization if available
            if self.visualizer:
                try:
                    self.visualizer.update_fitness_plot(
                        generation,
                        self.best_fitness_history,
                        self.avg_fitness_history
                    )
                except Exception as e:
                    print(f"Warning: Failed to update fitness plot: {str(e)}")
            
            # Update best solution if improved
            if current_best_fitness > self.best_fitness:
                self.best_solution = population_fitness[0][0]
                self.best_fitness = current_best_fitness
                no_improvement = 0
                print(f"\nNew best solution found!")
                print(f"Fitness Score: {self.best_fitness:.4f}")
                print(f"Number of routes: {len(self.best_solution)}")
                print(f"Total parcels: {sum(len(route.parcels) for route in self.best_solution)}")
                print(f"Total distance: {sum(route.total_distance for route in self.best_solution):.2f}")
                print(f"Total cost: ${sum(route.total_cost for route in self.best_solution):.2f}")
                
                # Update route visualization if available
                if self.visualizer:
                    try:
                        self.visualizer.plot_routes(self.best_solution)
                    except Exception as e:
                        print(f"Warning: Failed to update route plot: {str(e)}")
            else:
                no_improvement += 1
            
            # Early stopping
            if no_improvement >= self.early_stopping_generations:
                print(f"\nStopping early - No improvement for {self.early_stopping_generations} generations")
                break
            
            # Status update
            print(f"Generation {generation}: Best = {current_best_fitness:.4f}, Avg = {avg_fitness:.4f}")
            
            # Selection
            selected = self._selection(self.population, [fitness for _, fitness in population_fitness])
            
            # Create next generation
            next_generation = []
            
            # Elitism - keep best solutions
            next_generation.extend([population_fitness[i][0] for i in range(self.elitism_count)])
            
            # Crossover and Mutation
            while len(next_generation) < self.population_size:
                if len(selected) >= 2:
                    parent1, parent2 = random.sample(selected, 2)
                    child = self._crossover(parent1, parent2)
                    child = self._mutation(child)
                    next_generation.append(child)
            
            self.population = next_generation
            generation += 1
        
        elapsed_time = time.time() - start_time
        print(f"\nGA Optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best Fitness: {self.best_fitness:.4f}")
        print(f"Total Generations: {generation}")
        
        # Final visualization update
        if self.visualizer:
            try:
                self.visualizer.update_fitness_plot(
                    generation,
                    self.best_fitness_history,
                    self.avg_fitness_history
                )
                self.visualizer.plot_routes(self.best_solution)
            except Exception as e:
                print(f"Warning: Failed to update final visualization: {str(e)}")
        
        return self.best_solution
        
    def _initialize_population(self) -> List[List[Route]]:
        """Optimized population initialization with parallel processing"""
        population = [None] * self.population_size
        
        # Create initial solutions with different strategies
        random_count = int(self.population_size * 0.3)
        greedy_count = int(self.population_size * 0.3)
        hybrid_count = int(self.population_size * 0.3)
        optimized_count = self.population_size - random_count - greedy_count - hybrid_count
        
        # Initialize solutions by type
        for i in range(self.population_size):
            if i < random_count:
                population[i] = self._create_randomized_solution()
            elif i < random_count + greedy_count:
                population[i] = self._create_greedy_solution()
            elif i < random_count + greedy_count + hybrid_count:
                population[i] = self._create_hybrid_solution()
            else:
                population[i] = self._create_optimized_solution()
        
        return population
        
    def _create_randomized_solution(self) -> List[Route]:
        """
        Create a solution with randomized order assignments.
        
        Returns:
            List of routes constituting a complete solution
        """
        routes = []
        truck_types = ['9.6', '12.5', '16.5']
        
        # Get all order IDs and shuffle them
        all_orders = list(self.data_processor.time_windows.keys())
        random.shuffle(all_orders)
        
        unassigned_orders = all_orders.copy()
        
        # Create routes until all orders are assigned or no more can be assigned
        while unassigned_orders:
            # Randomly select truck type with probabilities favoring smaller trucks
            probabilities = [0.5, 0.3, 0.2]  # 50% small, 30% medium, 20% large
            truck_type = random.choices(truck_types, probabilities)[0]
            
            route = self._build_route(truck_type, unassigned_orders, randomize=True)
            
            if route and route.parcels:
                routes.append(route)
                # Remove assigned orders using mapped integer IDs
                assigned_ids = [parcel.id for parcel in route.parcels]
                unassigned_orders = [order_id for order_id in unassigned_orders 
                                    if self.data_processor.get_order_id_int(order_id) not in assigned_ids]
            else:
                # If we couldn't create a route, drop a random order
                if unassigned_orders:
                    idx = random.randint(0, len(unassigned_orders) - 1)
                    unassigned_orders.pop(idx)
        
        return routes
        
    def _create_greedy_solution(self) -> List[Route]:
        """
        Create a solution using a greedy approach.
        """
        routes = []
        unassigned_orders = list(self.data_processor.time_windows.keys())
        truck_types = ['9.6', '12.5', '16.5']
        
        while unassigned_orders:
            # Choose truck type based on remaining orders
            total_weight = sum(self.data_processor.time_windows[order]['weight'] 
                             for order in unassigned_orders)
            if total_weight > 5000:
                truck_type = '16.5'
            elif total_weight > 3000:
                truck_type = '12.5'
            else:
                truck_type = '9.6'
            
            # Sort orders by weight
            sorted_orders = sorted(unassigned_orders,
                                 key=lambda x: self.data_processor.time_windows[x]['weight'],
                                 reverse=True)
            
            # Try to create a route
            route = self._build_route(truck_type, sorted_orders, randomize=False)
            if route and route.parcels:
                routes.append(route)
                # Remove assigned orders
                assigned_ids = [parcel.id for parcel in route.parcels]
                unassigned_orders = [order_id for order_id in unassigned_orders 
                                   if self.data_processor.get_order_id_int(order_id) not in assigned_ids]
            else:
                # If we couldn't create a route, remove the largest order
                if unassigned_orders:
                    unassigned_orders.pop(0)
        
        return routes
        
    def _create_hybrid_solution(self) -> List[Route]:
        """
        Create a solution using a combination of random and greedy approaches.
        """
        routes = []
        unassigned_orders = list(self.data_processor.time_windows.keys())
        random.shuffle(unassigned_orders)  # Randomize initial order
        
        while unassigned_orders:
            # Randomly choose between greedy and random approach
            if random.random() < 0.5:
                # Greedy approach for truck selection
                total_weight = sum(self.data_processor.time_windows[order]['weight'] 
                                 for order in unassigned_orders[:10])  # Look at next 10 orders
                if total_weight > 5000:
                    truck_type = '16.5'
                elif total_weight > 3000:
                    truck_type = '12.5'
                else:
                    truck_type = '9.6'
            else:
                # Random truck selection
                truck_type = random.choices(['9.6', '12.5', '16.5'], weights=[0.5, 0.3, 0.2])[0]
            
            route = self._build_route(truck_type, unassigned_orders, randomize=True)
            if route and route.parcels:
                routes.append(route)
                # Remove assigned orders
                assigned_ids = [parcel.id for parcel in route.parcels]
                unassigned_orders = [order_id for order_id in unassigned_orders 
                                   if self.data_processor.get_order_id_int(order_id) not in assigned_ids]
            else:
                if unassigned_orders:
                    unassigned_orders.pop(0)
        
        return routes
        
    def _build_route(self, truck_type: str, available_orders: List[str], randomize: bool = False) -> Optional[Route]:
        """Build a route with multiple cities and proper capacity constraints"""
        if not available_orders:
            return None
            
        # Initialize route
        vehicle_id = f"V_{truck_type}_{random.randint(0, 999)}"
        capacity = self.data_processor.truck_specifications[truck_type]['weight_capacity']
        
        # Initialize with warehouse
        locations = [self.warehouse_location]
        parcels = []
        remaining_capacity = capacity
        
        # Process orders
        processed_orders = []
        for order_id in available_orders:
            if order_id not in self.data_processor.time_windows:
                continue
                
            order_data = self.data_processor.time_windows[order_id]
            weight = order_data['weight']
            
            # Check capacity constraint
            if weight > remaining_capacity:
                continue
                
            # Create locations
            source_loc = self._create_location(order_data['source'])
            dest_loc = self._create_location(order_data['destination'])
            
            # Get mapped ID
            mapped_id = self.data_processor.get_order_id_int(order_id)
            if mapped_id == -1:
                continue
                
            # Create parcel
            parcel = Parcel(
                id=mapped_id,
                source=source_loc,
                destination=dest_loc,
                weight=weight
            )
            
            # Add to route
            parcels.append(parcel)
            processed_orders.append(order_id)
            remaining_capacity -= weight
            
            # Add locations if not already present
            if source_loc not in locations:
                locations.append(source_loc)
            if dest_loc not in locations:
                locations.append(dest_loc)
            
            # Stop if we've reached minimum parcels and route is getting long
            if len(parcels) >= self.min_parcels_per_route and len(locations) >= 5:
                break
        
        # Return None if not enough parcels
        if len(parcels) < self.min_parcels_per_route:
            return None
            
        # Ensure route ends at warehouse
        if locations[-1] != self.warehouse_location:
            locations.append(self.warehouse_location)
            
        # Create route
        route = Route(
            vehicle_id=vehicle_id,
            locations=locations,
            parcels=parcels,
            data_processor=self.data_processor,
            vehicle_capacity=capacity
        )
        
        # Calculate costs
        route.calculate_total_distance()
        route.total_cost = route.total_distance * \
            self.data_processor.truck_specifications[truck_type]['cost_per_km']
            
        # Validate route and return only if feasible
        if route.validate():
            return route
        return None
        
    def _find_best_insertion_point(self, locations: List[Location], new_loc: Location) -> int:
        """
        Find the best position to insert a new location in a route.
        
        Args:
            locations: Current list of locations
            new_loc: Location to insert
            
        Returns:
            Best insertion position
        """
        if not locations:
            return 0
            
        min_increase = float('inf')
        best_pos = 0
        
        for i in range(len(locations)):
            # Calculate distance increase if inserted at position i
            prev_loc = locations[i-1] if i > 0 else locations[-1]
            next_loc = locations[i]
            
            old_distance = self._calculate_distance(prev_loc, next_loc)
            new_distance = (self._calculate_distance(prev_loc, new_loc) +
                          self._calculate_distance(new_loc, next_loc))
            
            increase = new_distance - old_distance
            
            if increase < min_increase:
                min_increase = increase
                best_pos = i
        
        return best_pos
        
    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """Cached distance calculation"""
        cache_key = (loc1.city_name, loc2.city_name)
        if cache_key not in self._distance_cache:
            self._distance_cache[cache_key] = abs(float(self.data_processor.city_to_idx[loc1.city_name]) - 
                                                float(self.data_processor.city_to_idx[loc2.city_name]))
        return self._distance_cache[cache_key]

    def calculate_fitness(self, solution: List[Route]) -> float:
        """Cached fitness calculation"""
        # Create a cache key based on route properties
        cache_key = tuple((route.vehicle_id, tuple(p.id for p in route.parcels)) for route in solution)
        if cache_key in self._fitness_cache:
            return self._fitness_cache[cache_key]
        
        fitness = super().calculate_fitness(solution)
        self._fitness_cache[cache_key] = fitness
        return fitness
        
    def _create_optimized_solution(self) -> List[Route]:
        """
        Create a solution using advanced optimization techniques.
        """
        routes = []
        unassigned_orders = list(self.data_processor.time_windows.keys())
        
        # Sort orders by weight and time windows
        sorted_orders = sorted(
            unassigned_orders,
            key=lambda x: (
                -self.data_processor.time_windows[x]['weight'],  # Heaviest first
                self.data_processor.city_to_idx[self.data_processor.time_windows[x]['source']]  # Closer sources first
            )
        )
        
        while sorted_orders:
            # Choose best truck type based on remaining orders
            total_weight = sum(self.data_processor.time_windows[order]['weight'] 
                             for order in sorted_orders[:5])  # Look ahead 5 orders
            avg_weight = total_weight / min(5, len(sorted_orders))
            
            if avg_weight > 2000:
                truck_type = '16.5'
            elif avg_weight > 1000:
                truck_type = '12.5'
            else:
                truck_type = '9.6'
            
            # Build route with optimization
            route = self._build_route(truck_type, sorted_orders, randomize=False)
            
            if route and route.parcels:
                routes.append(route)
                # Remove assigned orders
                assigned_ids = [parcel.id for parcel in route.parcels]
                sorted_orders = [order_id for order_id in sorted_orders 
                               if self.data_processor.get_order_id_int(order_id) not in assigned_ids]
            else:
                # If we couldn't create a route, remove the first order
                if sorted_orders:
                    sorted_orders.pop(0)
        
        return routes
        
    def _selection(self, population: List[List[Route]], fitness_scores: List[float] = None) -> List[List[Route]]:
        """
        Tournament selection for parent selection.
        
        Args:
            population: Current population
            fitness_scores: Optional pre-calculated fitness scores. If not provided, will calculate.
            
        Returns:
            Selected parents
        """
        selected = []
        tournament_size = 3
        
        # Calculate fitness scores if not provided
        if fitness_scores is None:
            fitness_scores = [self.calculate_fitness(solution) for solution in population]
        
        for _ in range(self.population_size):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select winner (highest fitness)
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            selected.append(copy.deepcopy(population[winner_idx]))
        
        return selected
        
    def _crossover(self, parent1: List[Route], parent2: List[Route]) -> List[Route]:
        """Optimized route-based crossover with minimal overhead"""
        if not parent1 or not parent2:
            return parent1 if parent1 else parent2
            
        # Fast path for single-route solutions
        if len(parent1) == 1 and len(parent2) == 1:
            return parent1 if self.calculate_fitness(parent1) > self.calculate_fitness(parent2) else parent2
            
        offspring = []
        assigned_orders = set()
        
        # Pre-calculate route scores
        route_scores = []
        for route in parent1 + parent2:
            # Quick load factor calculation
            load_factor = route.get_total_weight() / route.vehicle_capacity
            
            # Simplified scoring
            if 0.4 <= load_factor <= 0.8:
                score = 1.0 - (route.total_cost / (1000 * len(route.parcels) if route.parcels else 1))  # Normalize cost per parcel
            else:
                score = 0.5 - abs(0.6 - load_factor)  # Penalize sub-optimal load factors
                
            route_scores.append((route, score))
            
        # Sort routes by score (best first)
        route_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Process routes in batches for efficiency
        batch_size = min(4, len(route_scores))
        for i in range(0, len(route_scores), batch_size):
            batch = route_scores[i:i + batch_size]
            
            # Try to add each route in the batch
            for route_tuple in batch:
                # Ensure route_tuple is a tuple with exactly 2 elements
                if not isinstance(route_tuple, tuple) or len(route_tuple) != 2:
                    print(f"Warning: Invalid route tuple: {route_tuple}")
                    continue
                
                route, score = route_tuple  # Explicitly unpack the tuple
                
                # Skip if route is invalid
                if not hasattr(route, 'parcels') or not route.parcels:
                    continue
                
                # Skip if all orders are already assigned
                route_orders = {p.id for p in route.parcels}
                if route_orders.issubset(assigned_orders):
                    continue
                    
                # Get unassigned orders
                unassigned = [str(order_id) for order_id in route_orders 
                             if order_id not in assigned_orders]
                             
                if not unassigned:
                    continue
                    
                # Create new route with unassigned orders
                truck_type = route.vehicle_id.split('_')[1]
                new_route = self._create_route_with_orders(truck_type, unassigned)
                
                if new_route and new_route.parcels:
                    offspring.append(new_route)
                    assigned_orders.update(p.id for p in new_route.parcels)
                    
            # Early exit if we have enough routes
            if len(offspring) >= max(len(parent1), len(parent2)):
                break
                
        # Quick local search if offspring is promising
        if offspring and len(offspring) <= max(len(parent1), len(parent2)):
            offspring = self._quick_local_search(offspring)
            
        return offspring if offspring else parent1
        
    def _quick_local_search(self, solution: List[Route]) -> List[Route]:
        """Fast local search for crossover improvement"""
        if not solution or len(solution) < 2:
            return solution
            
        # Try to improve the worst route
        worst_idx = max(range(len(solution)), 
                       key=lambda i: solution[i].total_cost / len(solution[i].parcels))
        worst_route = solution[worst_idx]
        
        # Try one random permutation
        if len(worst_route.parcels) >= 2:
            pairs = [(p.source, p.destination) for p in worst_route.parcels]
            random.shuffle(pairs)
            
            new_locations = [self.warehouse_location]
            for source, dest in pairs:
                new_locations.extend([source, dest])
            new_locations.append(self.warehouse_location)
            
            new_route = Route(
                vehicle_id=worst_route.vehicle_id,
                locations=new_locations,
                parcels=worst_route.parcels.copy(),
                data_processor=self.data_processor,
                vehicle_capacity=worst_route.vehicle_capacity
            )
            
            if new_route.total_cost < worst_route.total_cost:
                solution[worst_idx] = new_route
                
        return solution
        
    def _create_route_with_orders(self, truck_type: str, order_ids: List[str]) -> Optional[Route]:
        """Optimized route creation with minimal overhead"""
        # Skip processing if no orders
        if not order_ids:
            return None
            
        # Initialize route
        vehicle_id = f"V_{truck_type}_{random.randint(0, 999)}"
        capacity = self.data_processor.truck_specifications[truck_type]['weight_capacity']
        
        route = Route(
            vehicle_id=vehicle_id,
            locations=[self.warehouse_location],
            parcels=[],
            data_processor=self.data_processor,
            vehicle_capacity=capacity
        )
        
        # Process orders in sequence
        remaining_capacity = capacity
        current_location = self.warehouse_location
        
        for order_id in order_ids:
            if order_id not in self.data_processor.time_windows:
                continue
                
            order_data = self.data_processor.time_windows[order_id]
            weight = order_data['weight']
            
            if weight > remaining_capacity:
                continue
                
            # Create locations and parcel
            source_loc = self._create_location(order_data['source'])
            dest_loc = self._create_location(order_data['destination'])
            
            mapped_id = self.data_processor.get_order_id_int(order_id)
            if mapped_id == -1:
                continue
                
            parcel = Parcel(
                id=mapped_id,
                destination=dest_loc,
                source=source_loc,
                weight=weight
            )
            
            # Update route
            route.parcels.append(parcel)
            remaining_capacity -= weight
            
            # Add locations
            if route.locations[-1] != source_loc:
                route.locations.append(source_loc)
            route.locations.append(dest_loc)
            
            current_location = dest_loc
        
        # Complete route if parcels were added
        if route.parcels:
            if route.locations[-1] != self.warehouse_location:
                route.locations.append(self.warehouse_location)
                
            route.calculate_total_distance()
            route.total_cost = route.total_distance * \
                self.data_processor.truck_specifications[truck_type]['cost_per_km']
            
            route.is_feasible, route.violation_reason = self._check_route_feasibility(
                route, truck_type
            )
            
            return route if route.is_feasible else None
        
        return None
        
    def _mutation(self, solution: List[Route]) -> List[Route]:
        """Enhanced mutation with adaptive rates and more operators"""
        if not solution:
            return solution
            
        # Calculate current diversity and adjust mutation rate
        if self.population_diversity_history:
            current_diversity = self.population_diversity_history[-1]
            # Adjust mutation rate based on diversity
            mutation_rate = self.base_mutation_rate * (1 + (self.diversity_threshold - current_diversity))
            mutation_rate = max(self.min_mutation_rate, min(self.max_mutation_rate, mutation_rate))
        else:
            mutation_rate = self.base_mutation_rate
            
        # Apply multiple mutations with probability based on diversity
        num_mutations = random.choices([1, 2, 3, 4], 
                                     weights=[0.3, 0.3, 0.2, 0.2])[0]
        
        for _ in range(num_mutations):
            if random.random() > mutation_rate:
                continue
                
            # Select mutation operator based on solution characteristics
            total_parcels = sum(len(route.parcels) for route in solution)
            avg_parcels_per_route = total_parcels / max(1, len(solution))
            
            # Adjust operator probabilities based on current solution
            if avg_parcels_per_route < 4:
                # Favor merging for solutions with small routes
                probs = [0.4, 0.3, 0.2, 0.1]  # merge, swap, split, reorder
                ops = [self._merge_routes_mutation, self._swap_orders_mutation,
                      self._split_route_mutation, self._reorder_stops_mutation]
            else:
                # Favor splitting and reordering for solutions with large routes
                probs = [0.3, 0.3, 0.2, 0.2]  # split, reorder, merge, swap
                ops = [self._split_route_mutation, self._reorder_stops_mutation,
                      self._merge_routes_mutation, self._swap_orders_mutation]
            
            # Apply selected mutation
            op = random.choices(ops, weights=probs)[0]
            new_solution = op(solution)
            
            # Validate and repair
            if new_solution and len(new_solution) > 0:
                solution = self._repair_solution(new_solution)
                
                # Apply local search with small probability
                if random.random() < 0.1:
                    solution = self._quick_local_search(solution)
        
        return solution
        
    def _swap_orders_mutation(self, solution: List[Route]) -> List[Route]:
        """
        Enhanced swap orders mutation with more aggressive changes.
        """
        if len(solution) < 2:
            return solution
            
        # Select multiple routes for swapping
        num_routes = min(3, len(solution))  # Up to 3 routes
        route_indices = random.sample(range(len(solution)), num_routes)
        
        # Create pairs of routes for swapping
        for i in range(len(route_indices)):
            for j in range(i + 1, len(route_indices)):
                route1 = solution[route_indices[i]]
                route2 = solution[route_indices[j]]
                
                if not route1.parcels or not route2.parcels:
                    continue
                
                # Determine number of parcels to swap (up to 50% of the smaller route)
                max_swap = max(1, min(len(route1.parcels), len(route2.parcels)) // 2)
                swap_count = random.randint(1, max_swap)
                
                # Select random parcels to swap
                parcels1 = random.sample(route1.parcels, min(swap_count, len(route1.parcels)))
                parcels2 = random.sample(route2.parcels, min(swap_count, len(route2.parcels)))
                
                # Try to create new routes with swapped parcels
                new_route1 = self._create_route_without_parcels(route1, parcels1)
                new_route2 = self._create_route_without_parcels(route2, parcels2)
                
                if new_route1 and new_route2:
                    # Add swapped parcels
                    success1 = self._add_parcels_to_route(new_route1, parcels2)
                    success2 = self._add_parcels_to_route(new_route2, parcels1)
                    
                    if success1 and success2:
                        solution[route_indices[i]] = new_route1
                        solution[route_indices[j]] = new_route2
        
        return solution
        
    def _split_route_mutation(self, solution: List[Route]) -> List[Route]:
        """
        Enhanced split route mutation with multiple splits.
        """
        if not solution:
            return solution
            
        # Select multiple routes to split
        num_splits = min(2, len(solution))  # Try to split up to 2 routes
        candidates = [i for i, route in enumerate(solution) if len(route.parcels) > 3]
        
        if not candidates:
            return solution
            
        # Perform splits
        for _ in range(num_splits):
            if not candidates:
                break
                
            route_idx = random.choice(candidates)
            candidates.remove(route_idx)
            
            route = solution[route_idx]
            
            # Try different split points
            split_points = sorted(random.sample(range(1, len(route.parcels)), 
                                  min(2, len(route.parcels) - 1)))
            
            current_parcels = route.parcels
            new_routes = []
            
            # Split into multiple parts
            start_idx = 0
            for split_point in split_points:
                parcels_segment = current_parcels[start_idx:split_point]
                truck_type = random.choice(['9.6', '12.5', '16.5'])
                new_route = self._create_route_with_orders(truck_type, 
                                                         [str(p.id) for p in parcels_segment])
                if new_route and new_route.is_feasible:
                    new_routes.append(new_route)
                start_idx = split_point
            
            # Handle remaining parcels
            if start_idx < len(current_parcels):
                remaining_parcels = current_parcels[start_idx:]
                truck_type = random.choice(['9.6', '12.5', '16.5'])
                new_route = self._create_route_with_orders(truck_type,
                                                         [str(p.id) for p in remaining_parcels])
                if new_route and new_route.is_feasible:
                    new_routes.append(new_route)
            
            # Replace original route with new routes if any were created
            if new_routes:
                solution.pop(route_idx)
                solution.extend(new_routes)
        
        return solution
        
    def _merge_routes_mutation(self, solution: List[Route]) -> List[Route]:
        """Enhanced merge operation that tries harder to combine routes"""
        if len(solution) < 2:
            return solution
            
        # Try multiple merges
        num_attempts = min(5, len(solution) // 2)
        for _ in range(num_attempts):
            # Select routes that might be good candidates for merging
            candidates = [(i, route) for i, route in enumerate(solution)
                         if route.get_total_weight() < route.vehicle_capacity * 0.7]
            
            if len(candidates) < 2:
                continue
                
            # Select two routes with compatible locations
            best_pair = None
            best_distance = float('inf')
            
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    idx1, route1 = candidates[i]
                    idx2, route2 = candidates[j]
                    
                    # Check if routes can be merged based on capacity
                    total_weight = route1.get_total_weight() + route2.get_total_weight()
                    if total_weight > self.data_processor.truck_specifications['16.5']['weight_capacity']:
                        continue
                        
                    # Calculate connection distance
                    dist = min(
                        self._calculate_distance(route1.locations[-2], route2.locations[1]),
                        self._calculate_distance(route2.locations[-2], route1.locations[1])
                    )
                    
                    if dist < best_distance:
                        best_distance = dist
                        best_pair = (idx1, idx2)
            
            if best_pair and best_distance < 500:  # Only merge if routes are reasonably close
                idx1, idx2 = best_pair
                route1 = solution[idx1]
                route2 = solution[idx2]
                
                # Create merged route
                merged_parcels = route1.parcels + route2.parcels
                new_route = self._create_route_with_orders('16.5', [str(p.id) for p in merged_parcels])
                
                if new_route and new_route.is_feasible:
                    # Remove old routes and add merged route
                    if idx1 < idx2:
                        solution.pop(idx2)
                        solution.pop(idx1)
                    else:
                        solution.pop(idx1)
                        solution.pop(idx2)
                    solution.append(new_route)
        
        return solution
        
    def _create_route_without_parcels(self, route: Route, parcels_to_remove: List[Parcel]) -> Optional[Route]:
        """
        Create a new route excluding specific parcels.
        """
        remaining_parcels = [p for p in route.parcels if p not in parcels_to_remove]
        if not remaining_parcels:
            return None
            
        truck_type = route.vehicle_id.split('_')[1]
        return self._create_route_with_orders(truck_type, [str(p.id) for p in remaining_parcels])
        
    def _add_parcels_to_route(self, route: Route, parcels: List[Parcel]) -> bool:
        """
        Add parcels to an existing route if possible.
        """
        if not route or not parcels:
            return False
            
        # Check capacity
        total_weight = route.get_total_weight() + sum(p.weight for p in parcels)
        if total_weight > route.vehicle_capacity:
            return False
            
        # Add parcels and update route
        for parcel in parcels:
            if parcel.source not in route.locations:
                route.locations.insert(-1, parcel.source)
            if parcel.destination not in route.locations:
                route.locations.insert(-1, parcel.destination)
            route.parcels.append(parcel)
            
        route.calculate_total_distance()
        truck_type = route.vehicle_id.split('_')[1]
        route.total_cost = route.total_distance * \
            self.data_processor.truck_specifications[truck_type]['cost_per_km']
            
        route.is_feasible, route.violation_reason = self._check_route_feasibility(
            route, truck_type
        )
        
        return route.is_feasible
        
    def _change_truck_mutation(self, solution: List[Route]) -> List[Route]:
        """
        Change truck type for a random route.
        
        Args:
            solution: Solution to mutate
            
        Returns:
            Mutated solution
        """
        if not solution:
            return solution
            
        # Select a random route
        route_idx = random.randint(0, len(solution) - 1)
        route = solution[route_idx]
        
        # Get current truck type
        current_type = route.vehicle_id.split('_')[1]
        
        # Select a different truck type
        truck_types = ['9.6', '12.5', '16.5']
        truck_types.remove(current_type)
        new_type = random.choice(truck_types)
        
        # Create a new route with the same orders but different truck type
        order_ids = [str(parcel.id) for parcel in route.parcels]
        new_route = self._create_route_with_orders(new_type, order_ids)
        
        if new_route and new_route.is_feasible:
            solution[route_idx] = new_route
        
        return solution
        
    def _reorder_stops_mutation(self, solution: List[Route]) -> List[Route]:
        """
        Reorder the stops within a route.
        
        Args:
            solution: Solution to mutate
            
        Returns:
            Mutated solution
        """
        if not solution:
            return solution
            
        # Select a random route
        route_idx = random.randint(0, len(solution) - 1)
        route = solution[route_idx]
        
        if len(route.parcels) < 2:
            return solution
            
        # Get the order IDs
        order_ids = [str(parcel.id) for parcel in route.parcels]
        random.shuffle(order_ids)
        
        # Recreate the route with reordered stops
        truck_type = route.vehicle_id.split('_')[1]
        new_route = self._create_route_with_orders(truck_type, order_ids)
        
        if new_route and new_route.is_feasible:
            solution[route_idx] = new_route
        
        return solution
        
    def _repair_solution(self, solution: List[Route]) -> List[Route]:
        """
        Repair infeasible solutions.
        
        Args:
            solution: Solution to repair
            
        Returns:
            Repaired solution
        """
        # Remove any infeasible routes
        solution = [route for route in solution if route.is_feasible]
        
        # Check for duplicate orders
        assigned_orders = set()
        repaired_solution = []
        
        for route in solution:
            # Create a new route with only orders not yet assigned
            new_order_ids = []
            for parcel in route.parcels:
                if parcel.id not in assigned_orders:
                    new_order_ids.append(str(parcel.id))
                    assigned_orders.add(parcel.id)
            
            if new_order_ids:
                truck_type = route.vehicle_id.split('_')[1]
                new_route = self._create_route_with_orders(truck_type, new_order_ids)
                if new_route and new_route.is_feasible and new_route.parcels:
                    repaired_solution.append(new_route)
        
        return repaired_solution
        
    def _check_route_feasibility(self, route: Route, truck_type: str) -> Tuple[bool, str]:
        """
        Check if route meets all constraints.
        
        Args:
            route: Route to check
            truck_type: Type of truck
            
        Returns:
            Tuple of (is_feasible, reason)
        """
        # Check capacity
        total_weight = route.get_total_weight()
        capacity = self.data_processor.truck_specifications[truck_type]['weight_capacity']
        
        if total_weight > capacity:
            return False, f"Capacity exceeded: {total_weight} > {capacity}"
        
        # Check maximum distance constraint
        max_distance = 5000  # Example value
        if route.total_distance > max_distance:
            return False, f"Distance exceeded: {route.total_distance} > {max_distance}"
        
        return True, "Route is feasible"
        
    def _get_warehouse_location(self) -> Location:
        """
        Get warehouse location (assuming first city is warehouse).
        
        Returns:
            Location object for warehouse
        """
        warehouse_city = list(self.data_processor.cities)[0]
        return self._create_location(warehouse_city)
        
    def _create_location(self, city_name: str) -> Location:
        """
        Create Location object from city name with caching.
        
        Args:
            city_name: Name of the city
            
        Returns:
            Location object
        """
        if city_name not in self._location_cache:
            idx = self.data_processor.city_to_idx.get(city_name, 0)
            self._location_cache[city_name] = Location(
                city_name=city_name,
                lat=float(idx),  # Use index as coordinate for visualization
                lon=float(idx)   # Use index as coordinate for visualization
            )
        return self._location_cache[city_name]
        
    def stop(self):
        """Stop the optimization process"""
        self.stop_flag = True
        self._running = False

    def _calculate_population_diversity(self, population: List[List[Route]]) -> float:
        """Simplified diversity calculation focusing on key metrics"""
        if not population:
            return 0.0
        
        # Calculate only essential features
        features = []
        for solution in population:
            if not solution:
                continue
            
            # Basic solution characteristics
            num_routes = len(solution)
            total_parcels = sum(len(route.parcels) for route in solution)
            total_cost = sum(route.total_cost for route in solution)
            
            # Simplified feature vector
            feature_vector = [
                num_routes / self.population_size,
                total_parcels / len(self.data_processor.time_windows),
                total_cost / 1000000
            ]
            features.append(feature_vector)
        
        if not features:
            return 0.0
        
        # Convert to numpy array
        features = np.array(features)
        
        # Calculate average pairwise distance (simplified)
        n = len(features)
        if n < 2:
            return 0.0
        
        # Use only first 10 solutions for diversity calculation
        n = min(n, 10)
        features = features[:n]
        
        # Calculate mean and std for normalization
        mean_features = np.mean(features, axis=0)
        std_features = np.std(features, axis=0) + 1e-6  # Avoid division by zero
        
        # Normalize features
        normalized_features = (features - mean_features) / std_features
        
        # Calculate diversity score
        diversity = np.mean(np.std(normalized_features, axis=0))
        
        return float(diversity)

    def _should_restart_population(self, diversity: float, generations_without_improvement: int) -> bool:
        """
        Enhanced decision making for population restart.
        """
        # Multiple criteria for restart
        diversity_too_low = diversity < self.diversity_threshold
        stagnation = generations_without_improvement >= self.max_stagnation
        
        # Adaptive diversity threshold
        if self.best_fitness_history:
            avg_diversity = sum(self.best_fitness_history[-10:]) / min(10, len(self.best_fitness_history))
            diversity_declining = diversity < avg_diversity * 0.8  # 20% below recent average
        else:
            diversity_declining = False
        
        # Combined decision
        should_restart = (
            (diversity_too_low and generations_without_improvement > self.max_stagnation // 2) or
            (stagnation) or
            (diversity_declining and generations_without_improvement > self.max_stagnation // 3)
        )
        
        return should_restart

    def _tournament_selection(self, population: List[List[Route]], fitness_scores: List[float]) -> List[Route]:
        """Enhanced tournament selection considering both fitness and diversity"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        
        # Calculate diversity contribution for tournament participants
        diversity_scores = []
        for idx in tournament_indices:
            solution = population[idx]
            diversity_score = self._calculate_solution_diversity(solution, population)
            diversity_scores.append(diversity_score)
            
        # Normalize scores
        max_fitness = max(fitness_scores[i] for i in tournament_indices)
        max_diversity = max(diversity_scores)
        
        if max_fitness == 0:
            max_fitness = 1
        if max_diversity == 0:
            max_diversity = 1
            
        # Combined score using weighted sum
        combined_scores = []
        for i, idx in enumerate(tournament_indices):
            fitness_norm = fitness_scores[idx] / max_fitness
            diversity_norm = diversity_scores[i] / max_diversity
            combined_score = (1 - self.diversity_weight) * fitness_norm + \
                           self.diversity_weight * diversity_norm
            combined_scores.append(combined_score)
            
        # Select winner based on combined score
        winner_idx = tournament_indices[combined_scores.index(max(combined_scores))]
        return copy.deepcopy(population[winner_idx])

    def _calculate_solution_diversity(self, solution: List[Route], population: List[List[Route]]) -> float:
        """Calculate how different a solution is from the population"""
        if not solution or not population:
            return 0.0
            
        # Extract features
        solution_features = self._extract_solution_features(solution)
        
        # Calculate average distance to other solutions
        distances = []
        sample_size = min(10, len(population))  # Limit computation
        for other in random.sample(population, sample_size):
            if other:
                other_features = self._extract_solution_features(other)
                distance = sum((a - b) ** 2 for a, b in zip(solution_features, other_features))
                distances.append(distance ** 0.5)
                
        return sum(distances) / len(distances) if distances else 0.0

    def _extract_solution_features(self, solution: List[Route]) -> List[float]:
        """Extract normalized features from a solution"""
        if not solution:
            return [0.0] * 5
            
        total_routes = len(solution)
        total_parcels = sum(len(route.parcels) for route in solution)
        total_cost = sum(route.total_cost for route in solution)
        avg_parcels = total_parcels / total_routes if total_routes > 0 else 0
        max_parcels = max(len(route.parcels) for route in solution)
        
        # Normalize features
        return [
            total_routes / self.population_size,
            total_parcels / len(self.data_processor.time_windows),
            total_cost / 1000000,
            avg_parcels / 10,
            max_parcels / 20
        ]

    def calculate_route_quality(self, solution: List[Route]) -> float:
        """Calculate a quality score for a solution based on route metrics"""
        if not solution:
            return 0.0
            
        # Calculate key metrics
        total_distance = sum(route.total_distance for route in solution)
        total_cost = sum(route.total_cost for route in solution)
        total_parcels = sum(len(route.parcels) for route in solution)
        avg_utilization = np.mean([route.get_total_weight()/route.vehicle_capacity 
                                 for route in solution]) * 100
        
        # Normalize metrics (lower is better for distance and cost)
        normalized_distance = 1000 / (total_distance + 1)  # Add 1 to avoid division by zero
        normalized_cost = 1000 / (total_cost + 1)
        normalized_parcels = total_parcels / len(self.data_processor.time_windows)
        normalized_utilization = avg_utilization / 100
        
        # Combine metrics with weights
        quality_score = (
            0.3 * normalized_distance +    # 30% weight on distance
            0.3 * normalized_cost +        # 30% weight on cost
            0.2 * normalized_parcels +     # 20% weight on parcel coverage
            0.2 * normalized_utilization   # 20% weight on vehicle utilization
        )
        
        return quality_score