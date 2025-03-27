import pytest
from src.protocols.message_protocol import Message, MessageType

class TestMessageProtocol:
    def test_message_creation(self):
        """Test creating a message"""
        message = Message(
            msg_type=MessageType.CAPACITY_REQUEST,
            sender_id="MRA_1",
            receiver_id="DA_1",
            content={}
        )
        
        assert message.msg_type == MessageType.CAPACITY_REQUEST
        assert message.sender_id == "MRA_1"
        assert message.receiver_id == "DA_1"
        assert isinstance(message.content, dict)
        assert message.timestamp is not None

    def test_message_types(self):
        """Test all message types are working"""
        assert MessageType.CAPACITY_REQUEST.value == "CAPACITY_REQUEST"
        assert MessageType.CAPACITY_RESPONSE.value == "CAPACITY_RESPONSE"
        assert MessageType.ROUTE_ASSIGNMENT.value == "ROUTE_ASSIGNMENT"
        assert MessageType.ROUTE_CONFIRMATION.value == "ROUTE_CONFIRMATION"
        assert MessageType.ERROR.value == "ERROR"