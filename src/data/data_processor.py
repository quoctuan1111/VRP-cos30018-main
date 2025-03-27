import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from src.models.location import Location

class DataProcessor:
    def __init__(self):
        """Initialize DataProcessor with empty data structures and truck specifications."""
        print("Initializing DataProcessor")  # Debug print
        self.distance_data: Optional[pd.DataFrame] = None
        self.order_data: Optional[pd.DataFrame] = None
        self.distance_matrix: Optional[np.ndarray] = None
        self.cities: set = {"WAREHOUSE"}  # Initialize with warehouse
        self.city_to_idx: Dict[str, int] = {}
        self.truck_specifications: Dict[str, Dict] = {
            '9.6': {'weight_capacity': 2.0, 'cost_per_km': 1, 'speed': 40},  # 2 tons = 2000 kg
            '12.5': {'weight_capacity': 5.0, 'cost_per_km': 2, 'speed': 40}, # 5 tons = 5000 kg
            '16.5': {'weight_capacity': 10.0, 'cost_per_km': 3, 'speed': 40}, # 10 tons = 10000 kg
            '50.0': {'weight_capacity': 800.0, 'cost_per_km': 10, 'speed': 40} # 800 tons = 800000 kg
        }
        self.warehouse_location = "WAREHOUSE"
        self.warehouse_lat = 0.0
        self.warehouse_lon = 0.0
        self.vehicle_capacities: Dict[str, float] = {}  # vehicle_id -> capacity in kg
        self.order_id_map: Dict[str, int] = {}  # Add mapping for order IDs

    def set_warehouse(self, location: str, lat: float, lon: float) -> None:
        self.warehouse_location = location
        self.warehouse_lat = lat
        self.warehouse_lon = lon

    def add_vehicle_capacity(self, vehicle_id: str, capacity: float) -> None:
        """Add or update vehicle capacity in kg"""
        self.vehicle_capacities[vehicle_id] = capacity

    def get_vehicle_capacity(self, vehicle_id: str) -> float:
        """Get vehicle capacity in kg"""
        return self.vehicle_capacities.get(vehicle_id, 0.0)

    def load_data(self, distance_file: str, order_file: str) -> None:
        """Load and process distance and order data from CSV files."""
        try:
            # Load distance data
            self.distance_data = pd.read_csv(distance_file)
            if self.distance_data.empty:
                raise ValueError("Distance CSV is empty")
            
            # Add warehouse to distance data if not present
            if "WAREHOUSE" not in set(self.distance_data['Source'].unique()):
                # Add warehouse with zero distance to itself
                warehouse_row = pd.DataFrame({
                    'Source': ['WAREHOUSE'],
                    'Destination': ['WAREHOUSE'],
                    'Distance(M)': [0.0]
                })
                self.distance_data = pd.concat([warehouse_row, self.distance_data], ignore_index=True)
            
            self._validate_distance_data()
            self._extract_cities()
            self._create_distance_matrix()

            # Load order data
            self.order_data = pd.read_csv(order_file)
            if self.order_data.empty:
                raise ValueError("Order CSV is empty")
                
            # Create numeric mapping for order IDs
            unique_order_ids = self.order_data['Order_ID'].unique()
            self.order_id_map = {str(oid): idx for idx, oid in enumerate(unique_order_ids)}
            
            # Add numeric Order_ID_Int column
            self.order_data['Order_ID_Int'] = self.order_data['Order_ID'].map(lambda x: self.order_id_map[str(x)])
            
            print(f"Order data columns: {self.order_data.columns.tolist()}")
            print(f"Sample Available_Time: {self.order_data['Available_Time'].head().tolist()}")
            print(f"Sample Deadline: {self.order_data['Deadline'].head().tolist()}")
            print(f"Sample Weight: {self.order_data['Weight'].head().tolist()}")

            # Parse datetime columns
            try:
                # First try ISO format
                self.order_data['Available_Time'] = pd.to_datetime(self.order_data['Available_Time'])
                self.order_data['Deadline'] = pd.to_datetime(self.order_data['Deadline'])
            except:
                # Fall back to original format if ISO fails
                expected_format = '%d/%m/%Y %H:%M'
                self.order_data['Available_Time'] = pd.to_datetime(self.order_data['Available_Time'], format=expected_format)
                self.order_data['Deadline'] = pd.to_datetime(self.order_data['Deadline'], format=expected_format)
            
            invalid_rows = self.order_data[self.order_data['Available_Time'].isna() | self.order_data['Deadline'].isna()]
            if not invalid_rows.empty:
                raise ValueError(f"Failed to parse datetime in rows: {invalid_rows[['Order_ID', 'Available_Time', 'Deadline']]}")

            self._validate_order_data()
            
            # Convert weights from grams to kg and scale them down
            self.order_data['Weight'] = self.order_data['Weight'] / 1000  # Convert grams to kg
            
            # Scale down weights to make the problem more manageable
            # Find the maximum weight that would allow at least 2 orders per truck
            max_truck_capacity = min(spec['weight_capacity'] for spec in self.truck_specifications.values())
            max_single_order = max_truck_capacity / 2  # Allow at least 2 orders per truck
            
            # Scale down weights if they exceed the maximum
            if self.order_data['Weight'].max() > max_single_order:
                print(f"Scaling down weights to maximum of {max_single_order} kg")
                self.order_data['Weight'] = self.order_data['Weight'].apply(lambda x: min(x, max_single_order))
            
            # Initialize time windows after loading data
            self._process_time_windows()
            
            print(f"Loaded {len(self.order_data)} orders and {len(self.distance_data)} distance entries")
            print(f"Final weight statistics:")
            print(f"  Min weight: {self.order_data['Weight'].min():.2f} kg")
            print(f"  Max weight: {self.order_data['Weight'].max():.2f} kg")
            print(f"  Mean weight: {self.order_data['Weight'].mean():.2f} kg")
            print(f"  Total weight: {self.order_data['Weight'].sum():.2f} kg")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {e}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV: {e}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def _validate_distance_data(self) -> None:
        """Validate the structure and content of distance data."""
        required_columns = ['Source', 'Destination', 'Distance(M)']
        missing_columns = [col for col in required_columns if col not in self.distance_data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in distance data: {missing_columns}")
        if self.distance_data['Distance(M)'].dtype not in [np.float64, np.int64]:
            raise ValueError("Distance(M) column must be numeric")
        if self.distance_data['Distance(M)'].min() < 0:
            raise ValueError("Distance values cannot be negative")

    def _validate_order_data(self) -> None:
        """Validate the structure and content of order data."""
        required_columns = ['Order_ID', 'Material_ID', 'Item_ID', 'Source', 'Destination', 
                           'Available_Time', 'Deadline', 'Danger_Type', 'Area', 'Weight']
        missing_columns = [col for col in required_columns if col not in self.order_data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in order data: {missing_columns}")
        order_cities = set(self.order_data['Destination'].unique())
        distance_cities = set(self.distance_data['Destination'].unique()).union(set(self.distance_data['Source'].unique()))
        if not order_cities.issubset(distance_cities):
            raise ValueError(f"Order destinations {order_cities - distance_cities} not in distance data")
        if self.order_data['Weight'].dtype not in [np.float64, np.int64]:
            raise ValueError("Weight column must be numeric")
        if self.order_data['Weight'].min() < 0:
            raise ValueError("Weight values cannot be negative")
        start_times = (self.order_data['Available_Time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        end_times = (self.order_data['Deadline'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        invalid_windows = self.order_data[start_times > end_times]
        if not invalid_windows.empty:
            raise ValueError(f"Invalid time windows found: {invalid_windows[['Order_ID', 'Available_Time', 'Deadline']]}. Start must be <= End")

    def _extract_cities(self) -> None:
        """Extract unique cities from distance data."""
        source_cities = set(self.distance_data['Source'].unique())
        dest_cities = set(self.distance_data['Destination'].unique())
        self.cities = source_cities.union(dest_cities).union({"WAREHOUSE"})
        self.city_to_idx = {city: idx for idx, city in enumerate(sorted(self.cities))}
        print(f"Extracted {len(self.cities)} unique cities (including WAREHOUSE)")

    def _create_distance_matrix(self) -> None:
        """Create a square distance matrix based on city indices."""
        n_cities = len(self.cities)
        self.distance_matrix = np.zeros((n_cities, n_cities), dtype=float)  # Use float for precision
        
        # First pass: fill in the matrix with actual distances
        for _, row in self.distance_data.iterrows():
            source_idx = self.city_to_idx[row['Source']]
            dest_idx = self.city_to_idx[row['Destination']]
            distance = row['Distance(M)'] / 1000  # Convert to km
            self.distance_matrix[source_idx][dest_idx] = distance
            self.distance_matrix[dest_idx][source_idx] = distance  # Ensure symmetry
        
        # Second pass: set warehouse distances to nearest city
        warehouse_idx = self.city_to_idx[self.warehouse_location]
        for i in range(n_cities):
            if i != warehouse_idx:
                # Find minimum distance to warehouse from this city
                min_distance = float('inf')
                for j in range(n_cities):
                    if j != warehouse_idx and self.distance_matrix[i][j] > 0:
                        min_distance = min(min_distance, self.distance_matrix[i][j])
                
                if min_distance != float('inf'):
                    # Set warehouse distance to half of the minimum distance to any city
                    self.distance_matrix[i][warehouse_idx] = min_distance / 2
                    self.distance_matrix[warehouse_idx][i] = min_distance / 2
        
        # Ensure diagonal is zero
        np.fill_diagonal(self.distance_matrix, 0)
        
        print(f"Distance matrix shape: {self.distance_matrix.shape}")
        print(f"Distance statistics:")
        print(f"  Min distance: {np.min(self.distance_matrix[self.distance_matrix > 0]):.2f} km")
        print(f"  Max distance: {np.max(self.distance_matrix):.2f} km")
        print(f"  Mean distance: {np.mean(self.distance_matrix[self.distance_matrix > 0]):.2f} km")

    def _process_time_windows(self):
        """Process and store time windows for all orders"""
        self.time_windows = {}
        
        for _, row in self.order_data.iterrows():
            order_id = str(row['Order_ID'])
            self.time_windows[order_id] = {
                'source': row['Source'],
                'destination': row['Destination'],
                'available_time': row['Available_Time'],
                'deadline': row['Deadline'],
                'weight': row['Weight']
            }
        
        print(f"Processed time windows for {len(self.time_windows)} orders")
        return True

    def get_time_window(self, order_id: str) -> Tuple[datetime, datetime]:
        """Get the time window for a specific order."""
        if not hasattr(self, 'time_windows'):
            self._process_time_windows()
        if order_id not in self.time_windows:
            raise ValueError(f"Invalid order ID: {order_id}")
        time_window = self.time_windows[order_id]
        return (time_window['available_time'], time_window['deadline'])

    def get_total_demand(self, city: str) -> int:
        """Get the total demand (weight) for a city based on orders."""
        if city not in self.city_to_idx:
            return 0
        city_idx = self.city_to_idx[city]
        demand = self.order_data[self.order_data['Destination'] == city]['Weight'].sum()
        return int(demand) if not np.isnan(demand) else 0

    def get_order_details(self, order_id: str) -> Dict:
        """Get detailed information for a specific order."""
        order = self.order_data[self.order_data['Order_ID'] == order_id]
        if order.empty:
            raise ValueError(f"Invalid order ID: {order_id}")
        return order.iloc[0].to_dict()

    def process_data(self, order_file: str, distance_file: str, truck_file: str):
        self.load_data(distance_file, order_file)
        
        # Add data validation and preprocessing
        self.validate_and_clean_data()
        
        # Set warehouse location as the first city
        self.warehouse_location = list(self.cities)[0]
        return True

    def validate_and_clean_data(self):
        """Validate and clean the loaded data"""
        if self.order_data is not None:
            # Remove orders with invalid weights
            self.order_data = self.order_data[self.order_data['Weight'] > 0]
            
            # Ensure datetime columns are properly formatted
            self.order_data['Available_Time'] = pd.to_datetime(self.order_data['Available_Time'])
            self.order_data['Deadline'] = pd.to_datetime(self.order_data['Deadline'])
            
            # Remove orders where deadline is before available time
            self.order_data = self.order_data[
                self.order_data['Deadline'] > self.order_data['Available_Time']
            ]

        # Ensure distance matrix is symmetric and non-negative
        if self.distance_matrix is not None:
            self.distance_matrix = np.maximum(self.distance_matrix, 0)
            self.distance_matrix = (self.distance_matrix + self.distance_matrix.T) / 2

        # Ensure truck specifications are valid
        if self.truck_specifications is not None:
            for truck_type in list(self.truck_specifications.keys()):
                if self.truck_specifications[truck_type]['weight_capacity'] <= 0:
                    del self.truck_specifications[truck_type]

    def preprocess_demands(self):
        """Preprocess demands to make them more manageable for the solver"""
        if self.order_data is not None:
            # Cap individual order weights
            max_single_order = 10000  # 10 tons
            self.order_data.loc[self.order_data['Weight'] > max_single_order, 'Weight'] = max_single_order
            
            # Group orders by destination to reduce problem size
            grouped = self.order_data.groupby('Destination').agg({
                'Weight': 'sum',
                'Available_Time': 'min',
                'Deadline': 'max'
            }).reset_index()
            
            # Update order data with grouped orders
            if len(grouped) < len(self.order_data):
                print(f"Reduced orders from {len(self.order_data)} to {len(grouped)} by grouping")
                self.order_data = grouped

        return True

    def preprocess_data(self):
        """Preprocess data for OR-Tools optimization"""
        if self.order_data is not None:
            # Remove orders with excessive weights (in kg)
            self.order_data = self.order_data[self.order_data['Weight'] <= 5.0]  # 5 kg limit
            
            # Consolidate orders to same destination
            self.order_data = self.order_data.groupby(['Destination', 'Source']).agg({
                'Weight': 'sum',
                'Order_ID': 'first',
                'Available_Time': 'min',
                'Deadline': 'max'
            }).reset_index()
            
            # Sort by weight descending to prioritize larger orders
            self.order_data = self.order_data.sort_values('Weight', ascending=False)

        if self.distance_matrix is not None:
            # Ensure non-negative distances
            self.distance_matrix = np.maximum(self.distance_matrix, 0)
            
            # Make matrix symmetric
            size = min(50, len(self.distance_matrix))  # Limit size
            self.distance_matrix = self.distance_matrix[:size,:size]
            self.distance_matrix = (self.distance_matrix + self.distance_matrix.T) / 2

        return True

    def get_order_id_int(self, order_id: str) -> int:
        """Convert string order ID to integer"""
        return self.order_id_map.get(str(order_id), -1)

    def get_original_order_id(self, int_id: int) -> str:
        """Convert integer ID back to original order ID"""
        reverse_map = {v: k for k, v in self.order_id_map.items()}
        return reverse_map.get(int_id, '')