import pytest
from src.agents.master_routing_agent import MasterRoutingAgent
from src.protocols.message_protocol import Message, MessageType

class TestMasterRoutingAgent:
    @pytest.fixture
    def setup_agent(self):
        """Setup test master routing agent"""
        return MasterRoutingAgent("MRA_1")

    def test_capacity_response_handling(self, setup_agent):
        """Test handling capacity response"""
        mra = setup_agent
        
        message = Message(
            msg_type=MessageType.CAPACITY_RESPONSE,
            sender_id="DA_1",
            receiver_id="MRA_1",
            content={
                "capacity": 10.0,
                "max_distance": 100.0
            }
        )
        
        response = mra.process_message(message)
        
        assert "DA_1" in mra.delivery_agents
        assert mra.delivery_agents["DA_1"]["capacity"] == 10.0
        assert mra.delivery_agents["DA_1"]["max_distance"] == 100.0

    def test_route_confirmation_handling(self, setup_agent):
        """Test handling route confirmation"""
        mra = setup_agent
        
        message = Message(
            msg_type=MessageType.ROUTE_CONFIRMATION,
            sender_id="DA_1",
            receiver_id="MRA_1",
            content={"status": "ACCEPTED"}
        )
        
        response = mra.process_message(message)
        # Add assertions based on your implementation