from queue import Queue
from typing import Optional
from .message_protocol import Message

class MessageQueue:
    """
    FIFO Queue Implementation
    - Enqueue: O(1)
    - Dequeue: O(1)
    - Space Complexity: O(n) where n is queue size
    """
    def __init__(self):
        self.queue = Queue()

    def enqueue(self, message: Message):
        self.queue.put(message)

    def dequeue(self) -> Optional[Message]:
        if not self.queue.empty():
            return self.queue.get()
        return None

    def is_empty(self) -> bool:
        return self.queue.empty()

    def size(self) -> int:
        return self.queue.qsize()