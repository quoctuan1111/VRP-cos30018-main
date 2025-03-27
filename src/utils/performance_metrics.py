import time
from typing import List, Dict
from dataclasses import dataclass
from statistics import mean

@dataclass
class MessageMetrics:
    start_time: float
    end_time: float
    processing_time: float
    message_type: str

class MessageTimeTracker:
    def __init__(self):
        self.message_metrics: List[MessageMetrics] = []
        self.total_messages = 0
        self.start_time = None

    def start_tracking(self):
        """Start tracking time for a message"""
        return time.perf_counter()

    def stop_tracking(self, start_time: float, message_type: str):
        """Stop tracking and record metrics"""
        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        metrics = MessageMetrics(
            start_time=start_time,
            end_time=end_time,
            processing_time=processing_time,
            message_type=message_type
        )
        self.message_metrics.append(metrics)
        self.total_messages += 1

    def get_average_processing_time(self) -> float:
        """Calculate average processing time in milliseconds"""
        if not self.message_metrics:
            return 0.0
        return mean(m.processing_time for m in self.message_metrics)

    def get_metrics_by_type(self) -> Dict[str, float]:
        """Get average processing time by message type"""
        metrics_by_type = {}
        for msg_type in set(m.message_type for m in self.message_metrics):
            type_metrics = [m.processing_time for m in self.message_metrics 
                          if m.message_type == msg_type]
            metrics_by_type[msg_type] = mean(type_metrics)
        return metrics_by_type