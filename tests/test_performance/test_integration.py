import unittest
from src.protocols.communication_manager import CommunicationManager
from src.protocols.message_protocol import Message, MessageType
from src.agents.delivery_agent import DeliveryAgent
from src.agents.master_routing_agent import MasterRoutingAgent

class TestPerformanceIntegration(unittest.TestCase):
    def setUp(self):
        self.comm_manager = CommunicationManager()
        
        # Create test agents
        self.test_sender = MasterRoutingAgent("test_sender")
        self.test_receiver = DeliveryAgent("test_receiver", capacity=100, max_distance=1000)
        
        # Register agents with communication manager
        self.comm_manager.register_agent(self.test_sender)
        self.comm_manager.register_agent(self.test_receiver)

    def test_complete_performance_monitoring(self):
        # Process some messages
        for _ in range(5):
            message = Message(
                msg_type=MessageType.CAPACITY_REQUEST,
                sender_id="test_sender",
                receiver_id="test_receiver",
                content={}
            )
            self.comm_manager.process_message(message)

        # Get complete metrics
        metrics = self.comm_manager.get_performance_metrics()

        # Check message processing metrics
        self.assertIn('message_processing', metrics)
        self.assertGreater(
            metrics['message_processing']['average_time'],
            0
        )

        # Check queue metrics
        self.assertIn('queue_processing', metrics)
        self.assertGreater(
            metrics['queue_processing']['current_rate'],
            0
        )

        # Check memory metrics
        self.assertIn('memory_usage', metrics)
        self.assertIn('statistics', metrics['memory_usage'])

    def tearDown(self):
        # Clean up
        self.comm_manager.stop()