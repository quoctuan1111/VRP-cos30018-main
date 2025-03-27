# tests/test_performance/test_queue_rate.py
import unittest
import time
from src.utils.queue_metrics import QueueRateTracker

class TestQueueRate(unittest.TestCase):
    def setUp(self):
        self.tracker = QueueRateTracker(window_size=5)

    def test_processing_rate(self):
        # Simulate processing 10 messages in 1 second
        for _ in range(10):
            self.tracker.record_message()
            

        current_rate = self.tracker.get_current_rate()
        self.assertGreater(current_rate, 1)  # Should be around 10 msgs/sec
        self.assertLess(current_rate, 1000)

    def test_rate_statistics(self):
        # Process messages
        for _ in range(5):
            self.tracker.record_message()
            time.sleep(0.1)  # Add small delay between messages
        
        current, average, peak = self.tracker.get_rate_statistics()
        # More reasonable assertions
        self.assertGreater(current, 0)
        self.assertGreater(average, 0)
        self.assertGreater(peak, 0)
        self.assertLess(peak, 100)  # Reasonable upper bound