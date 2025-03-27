import numpy as np
import os
import joblib
from typing import List, Dict, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

class RoutePredictor:
    """
    Machine Learning model for route cost prediction.
    Uses Random Forest Regression to predict route costs based on features.
    """
    
    def __init__(self):
        """Initialize the Route Predictor"""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'num_parcels', 
            'total_weight', 
            'total_distance', 
            'unique_destinations',
            'vehicle_capacity', 
            'load_factor', 
            'avg_distance_per_parcel',
            'truck_type',
            'parcels_per_destination'
        ]
    
    def train(self, routes_data: List[Dict]) -> float:
        """
        Train the model on historical route data.
        
        Args:
            routes_data: List of route data dictionaries
            
        Returns:
            Validation mean squared error
        """
        print(f"Training ML model on {len(routes_data)} routes")
        
        # Convert route data to features and targets
        X = self._extract_features(routes_data)
        y = self._extract_targets(routes_data)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Define parameter grid for cross-validation
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 15, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=0
        )
        
        # Fit model
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        # Print best parameters
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val_scaled)
        mse = np.mean((y_pred - y_val) ** 2)
        rmse = np.sqrt(mse)
        print(f"Validation RMSE: ${rmse:.2f}")
        
        # Calculate feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Feature importances:")
        for i in range(min(len(indices), len(self.feature_names))):
            print(f"  {self.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        return mse
    
    def predict_route_cost(self, route_data: Dict) -> float:
        """
        Predict the cost of a route based on its features.
        
        Args:
            route_data: Dictionary with route features
            
        Returns:
            Predicted cost
        """
        if not self.is_trained:
            # Return an estimate based on distance if not trained
            return route_data.get('total_distance', 0) * 2
        
        # Extract features from single route
        X = self._extract_features_single(route_data)
        
        # Scale features
        X_scaled = self.scaler.transform([X])
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        # Ensure prediction is positive
        return max(0.0, prediction)
    
    def _extract_features(self, routes_data: List[Dict]) -> np.ndarray:
        """
        Extract features from multiple routes.
        
        Args:
            routes_data: List of route data dictionaries
            
        Returns:
            Feature matrix
        """
        features = []
        
        for route_data in routes_data:
            features.append(self._extract_features_single(route_data))
        
        return np.array(features)
    
    def _extract_features_single(self, route_data: Dict) -> List[float]:
        """
        Extract features from a single route.
        
        Args:
            route_data: Dictionary with route data
            
        Returns:
            Feature vector
        """
        # Number of parcels
        num_parcels = len(route_data['parcels'])
        
        # Total weight
        total_weight = route_data.get('total_weight', 0)
        
        # Total distance
        total_distance = route_data.get('total_distance', 0)
        
        # Number of unique destinations
        unique_destinations = len(set(p['destination'] for p in route_data['parcels']))
        
        # Vehicle capacity
        vehicle_capacity = route_data.get('vehicle_capacity', 0)
        
        # Load factor (total weight / capacity)
        load_factor = total_weight / max(vehicle_capacity, 1)
        
        # Average distance per parcel
        avg_distance_per_parcel = total_distance / max(num_parcels, 1)
        
        # Truck type (numeric)
        truck_type = self._extract_truck_type(route_data.get('truck_type', '9.6'))
        
        # Parcels per destination (density)
        parcels_per_destination = num_parcels / max(unique_destinations, 1)
        
        # Return feature vector
        return [
            num_parcels,
            total_weight,
            total_distance,
            unique_destinations,
            vehicle_capacity,
            load_factor,
            avg_distance_per_parcel,
            truck_type,
            parcels_per_destination
        ]
    
    def _extract_truck_type(self, truck_type: str) -> float:
        """
        Extract numeric truck type from string.
        
        Args:
            truck_type: String representation of truck type
            
        Returns:
            Numeric representation of truck type
        """
        # Handle formats like "V_12.5_123" or "12.5"
        try:
            if '_' in str(truck_type):
                parts = str(truck_type).split('_')
                for part in parts:
                    try:
                        return float(part)
                    except ValueError:
                        continue
            return float(truck_type)
        except (ValueError, TypeError):
            # Default to smallest truck if parsing fails
            return 9.6
    
    def _extract_targets(self, routes_data: List[Dict]) -> np.ndarray:
        """
        Extract target values (costs) from routes.
        
        Args:
            routes_data: List of route data dictionaries
            
        Returns:
            Target vector
        """
        targets = [route_data.get('total_cost', 0) for route_data in routes_data]
        return np.array(targets)
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and scaler
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = data['is_trained']
        
        # Load feature names if available
        if 'feature_names' in data:
            self.feature_names = data['feature_names']
    
    def analyze_or_tools_patterns(self, routes: List[Any]) -> Dict:
        """
        Analyze patterns in OR-Tools routes to extract characteristics.
        
        Args:
            routes: List of routes from OR-Tools
            
        Returns:
            Dictionary with pattern information
        """
        if not routes:
            return {}
        
        patterns = {}
        
        # Average parcels per route
        parcels_per_route = [len(route.parcels) for route in routes]
        patterns['avg_parcels_per_route'] = np.mean(parcels_per_route)
        
        # Average distance per route
        distances = [route.total_distance for route in routes]
        patterns['avg_distance_per_route'] = np.mean(distances)
        
        # Truck type distribution
        truck_types = {}
        for route in routes:
            truck_type = route.vehicle_id.split('_')[1]
            truck_types[truck_type] = truck_types.get(truck_type, 0) + 1
        patterns['truck_type_distribution'] = truck_types
        
        # Load factors
        load_factors = []
        for route in routes:
            if hasattr(route, 'get_total_weight') and route.vehicle_capacity > 0:
                load_factor = route.get_total_weight() / route.vehicle_capacity
                load_factors.append(load_factor)
        
        patterns['avg_load_factor'] = np.mean(load_factors) if load_factors else 0
        
        return patterns