
import time
from collections import deque
from typing import Deque, Tuple

class QueueRateTracker:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size  # Window size in seconds
        self.message_timestamps: Deque[float] = deque()
        self.start_time = time.time()

    def record_message(self):
        """Record a message being processed"""
        current_time = time.time()
        self.message_timestamps.append(current_time)
        
        # Remove timestamps older than window_size
        while (self.message_timestamps and 
               current_time - self.message_timestamps[0] > self.window_size):
            self.message_timestamps.popleft()

    def get_current_rate(self) -> float:
        """Calculate current processing rate (messages/second)"""
        if not self.message_timestamps:
            return 0.0

        current_time = time.time()
        window_start = current_time - self.window_size

        # Count messages in current window
        messages_in_window = sum(1 for t in self.message_timestamps
                               if t > window_start)

        return messages_in_window / self.window_size

    def get_rate_statistics(self) -> Tuple[float, float, float]:
        """Get current, average, and peak rates"""
        # Protect against division by zero
        if not self.message_timestamps:
            return 0.0, 0.0, 0.0

        current_time = time.time()
        total_time = max(current_time - self.start_time, 0.001)  # Ensure non-zero
        total_messages = len(self.message_timestamps)

        # Calculate current rate
        window_start = current_time - self.window_size
        messages_in_window = sum(1 for t in self.message_timestamps if t > window_start)
        current_rate = messages_in_window / max(self.window_size, 0.001)

        # Calculate average rate
        average_rate = total_messages / total_time

        # Calculate peak rate
        window_time = min(total_time, self.window_size)
        peak_rate = total_messages / max(window_time, 0.001)

        return current_rate, average_rate, peak_rate