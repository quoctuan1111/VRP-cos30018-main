from typing import Optional
from .base_agent import BaseAgent
from src.protocols.message_protocol import Message, MessageType
from src.models.route import Route

class DeliveryAgent(BaseAgent):
    def __init__(self, agent_id: str, capacity: float, max_distance: float):
        super().__init__(agent_id)
        self.capacity = capacity
        self.max_distance = max_distance
        self.current_route = None
        self.message_handler = self._setup_handlers()

    def _setup_handlers(self):
        return {
            MessageType.CAPACITY_REQUEST: self._handle_capacity_request,
            MessageType.ROUTE_ASSIGNMENT: self._handle_route_assignment
        }

    def process_message(self, message: Message) -> Optional[Message]:
        if message.msg_type in self.message_handler:
            return self.message_handler[message.msg_type](message)
        return None

    def _handle_capacity_request(self, message: Message) -> Message:
        return Message(
            msg_type=MessageType.CAPACITY_RESPONSE,
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            content={
                "capacity": self.capacity,
                "max_distance": self.max_distance
            }
        )
        
    def _validate_route(self, route: Route) -> bool:
        """
        Validate if route meets capacity and distance constraints
        """
        # Check capacity constraint
        total_weight = sum(parcel.weight for parcel in route.parcels)
        if total_weight > self.capacity:
            return False

        # Check distance constraint
        if route.total_distance > self.max_distance:
            return False

        return True

    def _handle_route_assignment(self, message: Message) -> Message:
        route = message.content["route"]
        if self._validate_route(route):
            self.current_route = route
            return Message(
                msg_type=MessageType.ROUTE_CONFIRMATION,
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content={"status": "accepted"}
            )
        return Message(
            msg_type=MessageType.ROUTE_CONFIRMATION,
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            content={"status": "rejected"}
        )