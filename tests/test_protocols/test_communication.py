import unittest
from src.protocols.communication_manager import CommunicationManager
from src.agents.master_routing_agent import MasterRoutingAgent
from src.agents.delivery_agent import DeliveryAgent
from src.protocols.message_protocol import Message, MessageType


class TestCommunication(unittest.TestCase):
    def setUp(self):
        print("\n=== Setting up Communication Test ===")
        self.comm_manager = CommunicationManager()
        self.mra = MasterRoutingAgent("MRA_1")
        self.da1 = DeliveryAgent("DA_1", capacity=10, max_distance=100)

        print(f"Created MRA: {self.mra.agent_id}")
        print(f"Created DA: {self.da1.agent_id} (Capacity: {self.da1.capacity}, Max Distance: {self.da1.max_distance})")

        self.comm_manager.register_agent(self.mra)
        self.comm_manager.register_agent(self.da1)
        print("Registered agents with Communication Manager")

    def test_agent_registration(self):
        print("\n=== Testing Agent Registration ===")
        print(f"Checking if {self.mra.agent_id} is registered...")
        self.assertIn(self.mra.agent_id, self.comm_manager.agents)
        print(f"Checking if {self.da1.agent_id} is registered...")
        self.assertIn(self.da1.agent_id, self.comm_manager.agents)
        print("Agent registration successful!")

    def test_message_sending(self):
        print("\n=== Testing Message Sending ===")
        request = Message(
            msg_type=MessageType.CAPACITY_REQUEST,
            sender_id=self.mra.agent_id,
            receiver_id=self.da1.agent_id,
            content={}
        )
        print(f"MRA ({request.sender_id}) sending capacity request to DA ({request.receiver_id})")
        self.comm_manager.send_message(request)
        print("Message sent successfully!")
        self.assertFalse(self.comm_manager.message_queue.is_empty())

    def test_message_processing(self):
        print("\n=== Testing Message Processing ===")
        print("Creating capacity request message...")
        request = Message(
            msg_type=MessageType.CAPACITY_REQUEST,
            sender_id=self.mra.agent_id,
            receiver_id=self.da1.agent_id,
            content={}
        )

        print(f"MRA -> DA: Requesting capacity information")
        response = self.da1.process_message(request)

        print("DA -> MRA: Sending capacity response")
        print(f"Response type: {response.msg_type}")
        print(f"Capacity: {response.content['capacity']}")
        print(f"Max Distance: {response.content['max_distance']}")

        self.assertEqual(response.msg_type, MessageType.CAPACITY_RESPONSE)
        print("Message processing test completed successfully!")

    def tearDown(self):
        print("\nCleaning up test environment...")
        self.comm_manager.stop()
        print("Test completed")


if __name__ == '__main__':
    unittest.main(verbosity=2)