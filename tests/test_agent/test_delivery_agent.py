import pytest
from src.agents.delivery_agent import DeliveryAgent
from src.protocols.message_protocol import Message, MessageType
from src.models.route import Route
from src.models.location import Location
from src.models.parcel import Parcel

class TestDeliveryAgent:
    @pytest.fixture
    def setup_agent(self):
        """Setup test delivery agent"""
        return DeliveryAgent("DA_1", capacity=10, max_distance=100)

    def test_capacity_request_handling(self, setup_agent):
        """Test handling capacity request"""
        da = setup_agent
        
        message = Message(
            msg_type=MessageType.CAPACITY_REQUEST,
            sender_id="MRA_1",
            receiver_id="DA_1",
            content={}
        )
        
        response = da.process_message(message)
        
        assert response.msg_type == MessageType.CAPACITY_RESPONSE
        assert response.content["capacity"] == 10
        assert response.content["max_distance"] == 100

    def test_route_assignment_handling(self, setup_agent):
        """Test handling route assignment"""
        da = setup_agent
        
        # Create test route
        route = Route(
            vehicle_id="DA_1",
            locations=[Location(0, 0), Location(1, 1)],
            parcels=[Parcel(1, Location(1, 1), weight=5.0)],
            total_distance=50.0
        )
        
        message = Message(
            msg_type=MessageType.ROUTE_ASSIGNMENT,
            sender_id="MRA_1",
            receiver_id="DA_1",
            content={"route": route}
        )
        
        response = da.process_message(message)
        
        assert response.msg_type == MessageType.ROUTE_CONFIRMATION
        assert response.content["status"] == "accepted"
        assert da.current_route == route