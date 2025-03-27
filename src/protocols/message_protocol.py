from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
import time

class MessageType(Enum):
    CAPACITY_REQUEST = "CAPACITY_REQUEST"
    CAPACITY_RESPONSE = "CAPACITY_RESPONSE"
    ROUTE_ASSIGNMENT = "ROUTE_ASSIGNMENT"
    ROUTE_CONFIRMATION = "ROUTE_CONFIRMATION"
    STATUS_UPDATE = "STATUS_UPDATE"
    ERROR = "ERROR"
    
    
    OPTIMIZATION_REQUEST = "OPTIMIZATION_REQUEST"
    OPTIMIZATION_RESPONSE = "OPTIMIZATION_RESPONSE"
    OPTIMIZATION_STATUS = "OPTIMIZATION_STATUS"
    PARAMETERS_UPDATE = "PARAMETERS_UPDATE"

@dataclass
class Message:
    msg_type: MessageType
    sender_id: str
    receiver_id: str
    content: Dict[str, Any]
    conversation_id: Optional[str] = None
    timestamp: float = time.time()