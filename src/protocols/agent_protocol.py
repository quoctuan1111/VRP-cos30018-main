from enum import Enum
from dataclasses import dataclass
from typing import List, Any
from datetime import datetime

class MessageType(Enum):
    REQUEST_ROUTE = "REQUEST_ROUTE"
    ROUTE_ASSIGNED = "ROUTE_ASSIGNED"
    ROUTE_ACCEPTED = "ROUTE_ACCEPTED"
    ROUTE_REJECTED = "ROUTE_REJECTED"
    DELIVERY_STATUS = "DELIVERY_STATUS"
    ROUTE_COMPLETED = "ROUTE_COMPLETED"

@dataclass
class Message:
    msg_type: MessageType
    sender_id: str
    receiver_id: str
    content: Any
    timestamp: datetime = datetime.now()

class AgentProtocol:
    """FIPA Contract Net Protocol implementation"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.conversations = {}
        
    def create_request(self, receiver_id: str, content: Any) -> Message:
        """Create a route request message"""
        return Message(
            msg_type=MessageType.REQUEST_ROUTE,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=content
        )
        
    def create_response(self, request: Message, accept: bool, content: Any) -> Message:
        """Create a response message"""
        return Message(
            msg_type=MessageType.ROUTE_ACCEPTED if accept else MessageType.ROUTE_REJECTED,
            sender_id=self.agent_id,
            receiver_id=request.sender_id,
            content=content
        )
        
    def create_status_update(self, receiver_id: str, content: Any) -> Message:
        """Create a status update message"""
        return Message(
            msg_type=MessageType.DELIVERY_STATUS,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=content
        )
