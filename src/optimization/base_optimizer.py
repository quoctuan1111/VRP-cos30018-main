from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.models.route import Route

class BaseOptimizer(ABC):
    """
    Abstract base class for all VRP optimization methods.
    Defines the common interface that all optimizers must implement.
    """
    
    def __init__(self, data_processor):
        """
        Initialize the optimizer with a data processor.
        
        Args:
            data_processor: DataProcessor instance containing problem data
        """
        self.data_processor = data_processor
        self.best_solution: List[Route] = []
        self.best_fitness = 0.0
        self.score_history = {
            'composite_score': [],
            'distance_score': [],
            'cost_efficiency_score': [],
            'capacity_utilization_score': [],
            'parcel_efficiency_score': [],
            'route_structure_score': []
        }
        self.generation_count = 0
        
    @abstractmethod
    def optimize(self) -> List[Route]:
        """
        Execute the optimization algorithm and return the best solution found.
        
        Returns:
            List[Route]: The best solution found
        """
        pass
    
    def evaluate_solution(self, routes: List[Route]) -> Dict[str, Any]:
        """
        Evaluate a solution with various metrics.
        
        Args:
            routes: List of Routes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Count total parcels delivered
        parcels_delivered = sum(len(route.parcels) for route in routes if route.is_feasible)
        
        # Calculate total cost
        total_cost = sum(route.total_cost for route in routes if route.is_feasible)
        
        # Calculate total distance
        total_distance = sum(route.total_distance for route in routes if route.is_feasible)
        
        # Count infeasible routes
        infeasible_routes = sum(1 for route in routes if not route.is_feasible)
        
        # Calculate load factors
        if routes:
            weights = [route.get_total_weight() for route in routes if route.is_feasible]
            capacities = [route.vehicle_capacity for route in routes if route.is_feasible]
            load_factors = [w/c for w, c in zip(weights, capacities) if c > 0]
            avg_load_factor = sum(load_factors) / len(load_factors) if load_factors else 0
        else:
            avg_load_factor = 0
        
        return {
            'parcels_delivered': parcels_delivered,
            'total_cost': total_cost,
            'total_distance': total_distance,
            'num_routes': len(routes),
            'infeasible_routes': infeasible_routes,
            'avg_load_factor': avg_load_factor
        }
    
    def calculate_fitness(self, solution: List[Route]) -> float:
        """
        Calculate fitness score for a solution with improved differentiation.
        """
        if not solution:
            return 0.0

        # Get solution metrics
        evaluation = self.evaluate_solution(solution)
        total_parcels = evaluation['parcels_delivered']
        total_cost = evaluation['total_cost']
        total_distance = sum(route.total_distance for route in solution)
        
        # Calculate normalized metrics with non-linear scaling
        if total_parcels > 0:
            # Exponential reward for more parcels delivered
            parcels_score = (total_parcels / len(self.data_processor.time_windows)) ** 1.2
            
            # Progressive penalty for cost with more granular scaling
            cost_per_parcel = total_cost / total_parcels
            max_acceptable_cost = 5000  # Example threshold
            cost_score = 1.0 - min(1.0, (cost_per_parcel / max_acceptable_cost) ** 0.6)
            
            # Route efficiency with smoother scaling
            optimal_routes = max(1, len(self.data_processor.time_windows) // 5)  # Assume average 5 parcels per route
            route_ratio = len(solution) / optimal_routes
            route_score = 1.0 - abs(1.0 - route_ratio) ** 0.7
            
            # Load balancing score with improved scaling
            weights = [route.get_total_weight() for route in solution]
            if weights:
                avg_weight = sum(weights) / len(weights)
                weight_variations = [abs(w - avg_weight) / avg_weight if avg_weight > 0 else 1.0 for w in weights]
                balance_score = 1.0 - min(1.0, sum(weight_variations) / len(weights)) ** 0.8
            else:
                balance_score = 0.0
            
            # Distance efficiency with more granular scaling
            avg_distance_per_parcel = total_distance / total_parcels
            max_acceptable_distance = 100  # Example threshold per parcel
            distance_score = 1.0 - min(1.0, (avg_distance_per_parcel / max_acceptable_distance) ** 0.5)
            
            # Combine scores with adjusted weights
            weights = {
                'parcels': 0.40,  # Increased importance of delivering parcels
                'cost': 0.25,     # Balanced cost consideration
                'routes': 0.15,   # Route count importance
                'balance': 0.10,  # Load balancing importance
                'distance': 0.10  # Distance efficiency
            }
            
            fitness = (weights['parcels'] * parcels_score +
                      weights['cost'] * cost_score +
                      weights['routes'] * route_score +
                      weights['balance'] * balance_score +
                      weights['distance'] * distance_score)
            
            # Apply penalty for infeasible solutions with smoother scaling
            infeasible_routes = sum(1 for route in solution if not route.is_feasible)
            if infeasible_routes > 0:
                penalty = (infeasible_routes / len(solution)) ** 1.5
                fitness *= (1.0 - penalty)
            
            return max(0.0, min(1.0, fitness))
        return 0.0
    
    def get_best_solution(self) -> List[Route]:
        """Return the best solution found during optimization"""
        return self.best_solution
    
    def get_best_fitness(self) -> float:
        """Return the fitness of the best solution"""
        return self.best_fitness

    def calculate_route_comparison_score(self, routes: List[Route]) -> Dict[str, float]:
        """
        Calculate a comprehensive comparison score for routes.
        
        Args:
            routes: List of routes to evaluate
            
        Returns:
            Dictionary containing various scoring metrics and final composite score
        """
        # Validate routes first
        feasible_routes = [route for route in routes if route.validate()]
        
        if not feasible_routes:
            print("Warning: No feasible routes found for scoring")
            return {
                'distance_score': 0.0,
                'cost_efficiency_score': 0.0,
                'capacity_utilization_score': 0.0,
                'parcel_efficiency_score': 0.0,
                'route_structure_score': 0.0,
                'composite_score': 0.0
            }

        # 1. Distance Score (lower is better)
        total_distance = sum(route.total_distance for route in feasible_routes)
        total_parcels = sum(len(route.parcels) for route in feasible_routes)
        avg_distance_per_parcel = total_distance / total_parcels if total_parcels > 0 else float('inf')
        distance_score = max(0, 1 - (avg_distance_per_parcel / 2000))  # Normalize to 0-1 with 2000km as max

        # 2. Cost Efficiency Score
        total_cost = sum(route.total_cost for route in feasible_routes)
        cost_per_parcel = total_cost / total_parcels if total_parcels > 0 else float('inf')
        cost_efficiency_score = max(0, 1 - (cost_per_parcel / 5000))  # Normalize to 0-1 with $5000 as max

        # 3. Capacity Utilization Score
        utilization_scores = []
        for route in feasible_routes:
            capacity = route.vehicle_capacity
            if capacity > 0:
                utilization = route.get_total_weight() / capacity
                # Penalize both under and over-utilization
                utilization_score = 1 - abs(0.85 - utilization)  # 85% utilization is ideal
                utilization_scores.append(utilization_score)
        capacity_utilization_score = sum(utilization_scores) / len(utilization_scores) if utilization_scores else 0

        # 4. Parcel Efficiency Score
        avg_parcels_per_route = total_parcels / len(feasible_routes)
        # Assume 5 parcels per route is minimum efficient, 15 is optimal
        parcel_efficiency_score = min(1.0, avg_parcels_per_route / 15)

        # 5. Route Structure Score
        structure_scores = []
        for route in feasible_routes:
            if len(route.locations) > 2:  # More than just depot-return
                # Calculate average distance between consecutive stops
                consecutive_distances = []
                for i in range(len(route.locations) - 1):
                    if self.data_processor:
                        source_idx = self.data_processor.city_to_idx.get(route.locations[i].city_name, 0)
                        dest_idx = self.data_processor.city_to_idx.get(route.locations[i + 1].city_name, 0)
                        consecutive_distances.append(self.data_processor.distance_matrix[source_idx][dest_idx])
                
                if consecutive_distances:
                    avg_distance = sum(consecutive_distances) / len(consecutive_distances)
                    # Lower average distance between consecutive stops is better
                    structure_score = max(0, 1 - (avg_distance / 1000))  # Normalize to 0-1 with 1000km as max
                    structure_scores.append(structure_score)
        
        route_structure_score = sum(structure_scores) / len(structure_scores) if structure_scores else 0

        # Calculate composite score with weighted components
        weights = {
            'distance': 0.25,
            'cost_efficiency': 0.25,
            'capacity_utilization': 0.2,
            'parcel_efficiency': 0.15,
            'route_structure': 0.15
        }

        composite_score = (
            weights['distance'] * distance_score +
            weights['cost_efficiency'] * cost_efficiency_score +
            weights['capacity_utilization'] * capacity_utilization_score +
            weights['parcel_efficiency'] * parcel_efficiency_score +
            weights['route_structure'] * route_structure_score
        )

        # Print debug information
        print(f"\nScoring Summary for {len(feasible_routes)} feasible routes out of {len(routes)} total routes:")
        print(f"Distance Score: {distance_score:.4f}")
        print(f"Cost Efficiency Score: {cost_efficiency_score:.4f}")
        print(f"Capacity Utilization Score: {capacity_utilization_score:.4f}")
        print(f"Parcel Efficiency Score: {parcel_efficiency_score:.4f}")
        print(f"Route Structure Score: {route_structure_score:.4f}")
        print(f"Composite Score: {composite_score:.4f}")

        return {
            'distance_score': distance_score,
            'cost_efficiency_score': cost_efficiency_score,
            'capacity_utilization_score': capacity_utilization_score,
            'parcel_efficiency_score': parcel_efficiency_score,
            'route_structure_score': route_structure_score,
            'composite_score': composite_score
        }

    def update_score_history(self, solution: List[Route]):
        """
        Update the score history with the current solution's scores.
        
        Args:
            solution: Current solution to evaluate
        """
        scores = self.calculate_route_comparison_score(solution)
        for key in self.score_history:
            self.score_history[key].append(scores[key])
        self.generation_count += 1

    def get_score_history(self):
        """
        Get the history of scores.
        
        Returns:
            Dict containing lists of scores and the number of generations
        """
        return {
            'scores': self.score_history,
            'generations': list(range(self.generation_count))
        }