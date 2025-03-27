# tests/test_performance/test_message_processing.py
import unittest
import time
from src.utils.performance_metrics import MessageTimeTracker
from src.protocols.message_protocol import MessageType

class TestMessageProcessing(unittest.TestCase):
    def setUp(self):
        self.tracker = MessageTimeTracker()

    def test_message_timing(self):
        # Simulate message processing
        start_time = self.tracker.start_tracking()
        time.sleep(0.1)  # Simulate work
        self.tracker.stop_tracking(start_time, MessageType.CAPACITY_REQUEST.value)

        # Check metrics
        avg_time = self.tracker.get_average_processing_time()
        self.assertGreater(avg_time, 0)
        self.assertLess(avg_time, 200)  # Should be around 100ms

    def test_multiple_messages(self):
        # Process multiple messages
        for _ in range(3):
            start = self.tracker.start_tracking()
            time.sleep(0.05)
            self.tracker.stop_tracking(start, MessageType.CAPACITY_REQUEST.value)

        metrics = self.tracker.get_metrics_by_type()
        self.assertIn(MessageType.CAPACITY_REQUEST.value, metrics)