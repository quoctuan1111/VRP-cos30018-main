from typing import Dict, List
from .message_protocol import Message
from .message_queue import MessageQueue  # Use your MessageQueue
from src.utils.performance_metrics import MessageTimeTracker
from src.utils.queue_metrics import QueueRateTracker
from src.utils.memory_metrics import MemoryTracker
from src.optimization.or_ml_optimizer import ORToolsMLOptimizer
from src.data.data_processor import DataProcessor
from src.agents.master_routing_agent import MasterRoutingAgent
from src.agents.delivery_agent import DeliveryAgent
from src.protocols.message_protocol import MessageType
from src.models.route import Route

class CommunicationManager:
    def __init__(self, data_processor: DataProcessor):
        self.message_tracker = MessageTimeTracker()
        self.queue_tracker = QueueRateTracker()
        self.memory_tracker = MemoryTracker()
        
        # Initialize agents dictionary first
        self.agents = {}
        self.message_queue = MessageQueue()
        self._running = False

        # Only create optimizer if data_processor is not None
        self.optimizer = None
        self.master_agent = None
        
        if data_processor is not None:
            self.optimizer = ORToolsMLOptimizer(data_processor)
            self.master_agent = MasterRoutingAgent("MASTER")
            self.register_agent(self.master_agent)

    def register_agent(self, agent):
        self.agents[agent.agent_id] = agent

    def send_message(self, message: Message):
        self.message_queue.enqueue(message)  # Use enqueue instead of put
        self.queue_tracker.record_message()

    def start(self):
        self._running = True
        self.memory_tracker.take_snapshot()
        self._process_messages()

    def stop(self):
        self._running = False

    def process_message(self, message):
        """Process a single message"""
        start_time = self.message_tracker.start_tracking()

        #Take memory snapshot
        self.memory_tracker.take_snapshot()

        #Process message
        receiver = self.agents[message.receiver_id]
        response = receiver.process_message(message)

        #Record metrics
        self.message_tracker.stop_tracking(start_time, message.msg_type.value)
        self.queue_tracker.record_message()

        return response

    def _process_messages(self):
        """Internal method to process messages from queue"""
        while self._running:
            if not self.message_queue.is_empty():  # Use is_empty instead of empty()
                message = self.message_queue.dequeue()  # Use dequeue instead of get
                response = self.process_message(message)
                if response:
                    self.send_message(response)
                    
    def optimize_routes(self):
        routes = self.optimizer.optimize()
        
        # Save route data for ML training
        self._save_route_data(routes)
        
        for route in routes:
            delivery_agent = DeliveryAgent(
                agent_id=route.vehicle_id,
                capacity=self.optimizer.data_processor.truck_specifications[route.vehicle_id.split('_')[1]]['weight_capacity'],
                max_distance=5000  # Example max distance
            )
            self.register_agent(delivery_agent)
            self.send_message(Message(
                msg_type=MessageType.ROUTE_ASSIGNMENT,
                sender_id="MASTER",
                receiver_id=route.vehicle_id,
                content={"route": route}
            ))

    def _save_route_data(self, routes: List[Route]):
        """Save route data for ML training"""
        route_data = []
        for route in routes:
            data = {
                'parcels': [{'destination': p.delivery_location.city_name} for p in route.parcels],
                'total_weight': route.get_total_weight(),
                'total_distance': route.total_distance,
                'vehicle_capacity': route.vehicle_capacity,
                'total_cost': route.total_cost
            }
            route_data.append(data)
            
        # Save to file
        import json
        import os
        
        data_dir = 'data/ml'
        os.makedirs(data_dir, exist_ok=True)
        
        with open(f'{data_dir}/route_data.json', 'a') as f:
            json.dump(route_data, f)
            f.write('\n')

    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        current_rate, avg_rate, peak_rate = self.queue_tracker.get_rate_statistics()
        memory_stats = self.memory_tracker.get_memory_statistics()
        memory_trend = self.memory_tracker.get_memory_trend()
        
        return {
            'message_processing': {
                'average_time': self.message_tracker.get_average_processing_time(),
                'by_type': self.message_tracker.get_metrics_by_type()
            },
            'queue_processing': {
                'current_rate': current_rate,
                'average_rate': avg_rate,
                'peak_rate': peak_rate,
                'queue_size': self.message_queue.size()  # Added queue size metric
            },
            'memory_usage': {
                'statistics': memory_stats,
                'trend': memory_trend
            }
        }

    def print_performance_metrics(self):
        """Print current performance metrics"""
        metrics = self.get_performance_metrics()
        print("\n=== Performance Metrics ===")
        print(f"Message Processing:")
        print(f"  Average Time: {metrics['message_processing']['average_time']:.2f}ms")
        print(f"\nQueue Processing:")
        print(f"  Current Rate: {metrics['queue_processing']['current_rate']:.2f} msgs/sec")
        print(f"  Average Rate: {metrics['queue_processing']['average_rate']:.2f} msgs/sec")
        print(f"  Queue Size: {metrics['queue_processing']['queue_size']}")
        print(f"\nMemory Usage:")
        print(f"  Current: {metrics['memory_usage']['statistics']['rss']['current']:.2f} MB")
        print(f"  Peak: {metrics['memory_usage']['statistics']['rss']['peak']:.2f} MB")