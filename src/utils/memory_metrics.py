import psutil
import os
from dataclasses import dataclass
from typing import List, Dict
from statistics import mean
import time

@dataclass
class MemorySnapshot:
    timestamp: float
    rss: float  # Resident Set Size in MB
    vms: float  # Virtual Memory Size in MB
    shared: float  # Shared Memory in MB
    data: float  # Data Memory in MB

class MemoryTracker:
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.snapshots: List[MemorySnapshot] = []
        self.process = psutil.Process(os.getpid())

    def take_snapshot(self) -> MemorySnapshot:
        """Take a snapshot of current memory usage"""
        meminfo = self.process.memory_info()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss=meminfo.rss / 1024 / 1024,  # Convert to MB
            vms=meminfo.vms / 1024 / 1024,
            shared=getattr(meminfo, 'shared', 0) / 1024 / 1024,
            data=getattr(meminfo, 'data', 0) / 1024 / 1024
        )
        
        self.snapshots.append(snapshot)
        return snapshot

    def get_memory_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get detailed memory statistics"""
        if not self.snapshots:
            return {}

        stats = {
            'rss': {
                'current': self.snapshots[-1].rss,
                'average': mean(s.rss for s in self.snapshots),
                'peak': max(s.rss for s in self.snapshots)
            },
            'vms': {
                'current': self.snapshots[-1].vms,
                'average': mean(s.vms for s in self.snapshots),
                'peak': max(s.vms for s in self.snapshots)
            }
        }
        
        return stats

    def get_memory_trend(self) -> Dict[str, float]:
        """Calculate memory usage trend"""
        if len(self.snapshots) < 2:
            return {'trend': 0.0, 'total_change': 0.0, 'duration': 0.0}
            
        first, last = self.snapshots[0], self.snapshots[-1]
        time_diff = max(last.timestamp - first.timestamp, 0.001)  # Ensure non-zero
        memory_diff = last.rss - first.rss
        
        return {
            'trend': memory_diff / time_diff,  # MB per second
            'total_change': memory_diff,
            'duration': time_diff
        }