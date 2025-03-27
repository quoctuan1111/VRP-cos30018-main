from typing import Dict, List, Optional, Any  # Added Any import
from .base_agent import BaseAgent
from src.protocols.message_protocol import Message, MessageType

class MasterRoutingAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.delivery_agents = {}
        self.message_handler = {
            MessageType.CAPACITY_RESPONSE: self._handle_capacity_response,
            MessageType.ROUTE_CONFIRMATION: self._handle_route_confirmation,
            MessageType.OPTIMIZATION_RESPONSE: self._handle_optimization_response,
            MessageType.STATUS_UPDATE: self._handle_status_update
        }
        
    def set_optimization_agent(self, agent_id: str):
        """Register the optimization agent"""
        self.optimization_agent_id = agent_id
        
    def request_optimization(self, params: Dict) -> Message:
        """Request route optimization"""
        if not self.optimization_agent_id:
            raise ValueError("No optimization agent registered")
            
        return Message(
            msg_type=MessageType.OPTIMIZATION_REQUEST,
            sender_id=self.agent_id,
            receiver_id=self.optimization_agent_id,
            content={"parameters": params}
        )
        
    def _handle_optimization_response(self, message: Message) -> Optional[Message]:
        """Handle optimization response messages"""
        # Store optimization results
        optimization_result = message.content
        
        # If successful, assign routes to delivery agents
        if optimization_result.get("status") == "success":
            routes = optimization_result.get("routes", [])
            
            # Assign routes to delivery agents
            for route in routes:
                vehicle_id = route.get("vehicle_id")
                
                # Create route assignment message
                route_msg = Message(
                    msg_type=MessageType.ROUTE_ASSIGNMENT,
                    sender_id=self.agent_id,
                    receiver_id=vehicle_id,
                    content={"route": route}
                )
                
                # This message would be sent via your communication manager
                # For now, we'll just return it as part of the optimization response
                return route_msg
        
        return None

    def _setup_handlers(self):
        handlers = {
            MessageType.CAPACITY_RESPONSE: self._handle_capacity_response,
            MessageType.ROUTE_CONFIRMATION: self._handle_route_confirmation
        }
        return handlers

    def process_message(self, message: Message) -> Optional[Message]:
        return self.message_handler.get(message.msg_type, lambda x: None)(message)

    def _handle_capacity_response(self, message: Message) -> Optional[Message]:
        agent_id = message.sender_id
        self.delivery_agents[agent_id] = {
            "capacity": message.content["capacity"],
            "max_distance": message.content["max_distance"]
        }
        return None
    
    def _handle_status_update(self, message: Message) -> Optional[Message]:
        """Handle status update messages"""
        # Process status updates from various agents
        # This could be extended based on your needs
        return None
    
    def _handle_route_confirmation(self, message: Message) -> Optional[Message]:
        """Handle route confirmation messages from Delivery Agents"""
        agent_id = message.sender_id
        status = message.content["status"]
        if status == "accepted":
            self.delivery_agents[agent_id]["current_route"] = message.content.get("route")
