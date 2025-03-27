
import unittest
from src.utils.memory_metrics import MemoryTracker

class TestMemoryTracking(unittest.TestCase):
    def setUp(self):
        self.tracker = MemoryTracker()

    def test_memory_snapshot(self):
        snapshot = self.tracker.take_snapshot()
        self.assertGreater(snapshot.rss, 0)
        self.assertGreater(snapshot.vms, 0)

    def test_memory_statistics(self):
        # Take multiple snapshots
        for _ in range(3):
            self.tracker.take_snapshot()
            # Create some memory usage
            _ = [i for i in range(1000000)]

        stats = self.tracker.get_memory_statistics()
        self.assertIn('rss', stats)
        self.assertIn('vms', stats)
        self.assertGreater(stats['rss']['current'], 0)

    def test_memory_trend(self):
        self.tracker.take_snapshot()
        # Create some memory usage
        _ = [i for i in range(1000000)]
        self.tracker.take_snapshot()

        trend = self.tracker.get_memory_trend()
        self.assertIn('trend', trend)
        self.assertIn('total_change', trend)