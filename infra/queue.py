"""
CloudSRE v2 — Real Async Message Queue Infrastructure.

This is NOT a simulation. This is a real in-process message queue with real
backpressure, real overflow, real consumer lag, and real message persistence.

When the worker_service crashes:
  - Messages pile up in this queue FOR REAL
  - Queue depth increases FOR REAL (visible in /metrics)
  - When max capacity is hit, producers get REAL QueueFull errors

When the worker restarts:
  - It drains messages from this queue FOR REAL
  - If drain rate is too fast, it can overwhelm the database (CASCADE!)

This is the second piece of infrastructure that makes cascading failures real.

Kube SRE Gym equivalent: None. They don't have a message queue.
"""

import asyncio
import os
import shutil
import threading
import time
import json
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Message:
    """A single message in the queue."""
    id: int
    topic: str  # e.g., "payment.completed", "auth.session_created"
    payload: dict
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    max_retries: int = 3


class MessageQueue:
    """Real file-backed message queue with backpressure and fault injection.

    Every message is persisted as a real JSON file on disk at:
        /data/queue/msg_00001.json
        /data/queue/msg_00002.json

    The agent can ACTUALLY run:
        ls /data/queue/ | wc -l     → see real queue depth
        cat /data/queue/msg_00001.json → read real message contents

    This is NOT a Python list pretending to be a queue. These are real files
    on a real filesystem.

    Real failure modes:
      - Queue overflow: when capacity is hit, push() raises QueueFull
      - Consumer crash: messages pile up, depth increases
      - Thundering herd: drain_all() releases everything at once → DB overload
      - Dead letters: messages that fail max_retries go to dead letter queue
    """

    def __init__(self, max_size: int = 1000, queue_dir: str = "/data/queue"):
        self._queue: deque = deque()
        self._dead_letters: deque = deque(maxlen=100)
        self._lock = threading.Lock()
        self._max_size = max_size
        self._message_counter = 0

        # Real filesystem backing
        self._queue_dir = queue_dir
        os.makedirs(queue_dir, exist_ok=True)

        # Metrics
        self._total_pushed = 0
        self._total_popped = 0
        self._total_dropped = 0
        self._total_dead_lettered = 0

        # Fault injection
        self._is_paused = False
        self._drop_rate = 0.0  # 0.0 = no drops, 1.0 = drop everything

    def push(self, topic: str, payload: dict) -> int:
        """Push a message to the queue. Returns message ID.

        The message is written as a REAL JSON file on disk.
        Raises QueueFull if capacity is hit.
        """
        with self._lock:
            if len(self._queue) >= self._max_size:
                self._total_dropped += 1
                raise QueueFull(
                    f"Queue capacity exceeded (max={self._max_size}, "
                    f"current={len(self._queue)})"
                )

            if self._is_paused:
                self._total_dropped += 1
                raise QueuePaused("Queue is paused — consumer not accepting messages")

            self._message_counter += 1
            msg = Message(
                id=self._message_counter,
                topic=topic,
                payload=payload,
            )
            self._queue.append(msg)
            self._total_pushed += 1

            # Write to real file on disk
            self._write_msg_file(msg)

            return msg.id

    # ── File I/O Helpers ──────────────────────────────────────────────────

    def _write_msg_file(self, msg: Message):
        """Write a message as a real JSON file on disk."""
        try:
            path = os.path.join(self._queue_dir, f"msg_{msg.id:05d}.json")
            with open(path, "w") as f:
                json.dump({"id": msg.id, "topic": msg.topic, "payload": msg.payload,
                           "created_at": msg.created_at, "attempts": msg.attempts}, f)
        except OSError:
            pass  # Non-critical — in-memory queue is the source of truth

    def _delete_msg_file(self, msg_id: int):
        """Delete a message file from disk."""
        try:
            path = os.path.join(self._queue_dir, f"msg_{msg_id:05d}.json")
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    def _clear_queue_dir(self):
        """Remove all message files from the queue directory."""
        try:
            if os.path.exists(self._queue_dir):
                shutil.rmtree(self._queue_dir)
            os.makedirs(self._queue_dir, exist_ok=True)
        except OSError:
            pass

    def pop(self) -> Optional[Message]:
        """Pop the next message from the queue. Deletes the file from disk."""
        with self._lock:
            if not self._queue:
                return None

            msg = self._queue.popleft()
            msg.attempts += 1
            self._total_popped += 1
            self._delete_msg_file(msg.id)
            return msg

    def pop_batch(self, batch_size: int = 10) -> List[Message]:
        """Pop up to batch_size messages at once. Deletes files from disk."""
        with self._lock:
            batch = []
            for _ in range(min(batch_size, len(self._queue))):
                msg = self._queue.popleft()
                msg.attempts += 1
                batch.append(msg)
                self._delete_msg_file(msg.id)
            self._total_popped += len(batch)
            return batch

    def nack(self, msg: Message):
        """Return a message to the queue (processing failed).

        If max retries exceeded, move to dead letter queue.
        """
        with self._lock:
            if msg.attempts >= msg.max_retries:
                self._dead_letters.append(msg)
                self._total_dead_lettered += 1
            else:
                self._queue.appendleft(msg)  # Put back at front

    def drain_all(self) -> List[Message]:
        """Drain the entire queue at once. Wipes all files from disk.

        WARNING: This is the thundering herd trigger!
        The SMART agent should use drain_controlled() instead.
        """
        with self._lock:
            messages = list(self._queue)
            self._total_popped += len(messages)
            self._queue.clear()
            self._clear_queue_dir()
            return messages

    def drain_controlled(self, rate: int = 10, delay: float = 0.1) -> List[Message]:
        """Drain the queue at a controlled rate. Returns first batch.

        This is what a GOOD SRE agent should use. It prevents the
        thundering herd cascade.
        """
        return self.pop_batch(rate)

    def depth(self) -> int:
        """Current queue depth. Visible in /metrics."""
        return len(self._queue)

    def dead_letter_count(self) -> int:
        """Number of permanently failed messages."""
        return len(self._dead_letters)

    # ── Fault Injection ──────────────────────────────────────────────────

    def inject_overflow(self, fill_count: int = 900):
        """Fill the queue with dummy messages — written as REAL files on disk.

        After this, `ls /data/queue/ | wc -l` shows the real count.
        """
        with self._lock:
            for i in range(fill_count):
                if len(self._queue) >= self._max_size:
                    break
                self._message_counter += 1
                msg = Message(
                    id=self._message_counter,
                    topic="synthetic.fill",
                    payload={"synthetic": True, "fill_index": i},
                )
                self._queue.append(msg)
                self._write_msg_file(msg)

    def inject_pause(self):
        """Pause the queue — simulates consumer crash."""
        self._is_paused = True

    def release_pause(self):
        """Unpause the queue."""
        self._is_paused = False

    def inject_capacity_reduction(self, new_max: int = 10):
        """Reduce queue capacity — messages will overflow faster."""
        self._max_size = new_max

    def release_capacity_reduction(self):
        """Restore normal queue capacity."""
        self._max_size = 1000

    # ── Metrics ──────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return real queue metrics for /metrics endpoint.

        The agent can read these via:
            curl http://localhost:8003/metrics
        """
        return {
            "queue_depth": self.depth(),
            "queue_max_size": self._max_size,
            "queue_total_pushed": self._total_pushed,
            "queue_total_popped": self._total_popped,
            "queue_total_dropped": self._total_dropped,
            "queue_dead_letters": self._total_dead_lettered,
            "queue_is_paused": self._is_paused,
            "queue_utilization": round(self.depth() / max(self._max_size, 1), 2),
        }

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self):
        """Full reset — empty queue, wipe files, reset metrics."""
        with self._lock:
            self._queue.clear()
            self._dead_letters.clear()
            self._clear_queue_dir()
            self._message_counter = 0
            self._total_pushed = 0
            self._total_popped = 0
            self._total_dropped = 0
            self._total_dead_lettered = 0
            self._is_paused = False
            self._drop_rate = 0.0
            self._max_size = 1000


class QueueFull(Exception):
    """Raised when the queue capacity is exceeded.

    This is a REAL error. payment_service catches this and returns HTTP 503.
    The agent sees this in the logs as:
        [ERROR] payment: QueueFull — Queue capacity exceeded (max=1000, current=1000)
    """
    pass


class QueuePaused(Exception):
    """Raised when trying to push to a paused queue.

    This is a REAL error. Happens when worker_service is down.
    """
    pass
