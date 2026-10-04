"""Shared backoff and request pacing for API retries."""

import threading
import time

MAX_ATTEMPTS = 8


class SharedBackoff:
    def __init__(self, clock=time.monotonic, sleep=time.sleep, spacing=0.25):
        self.clock = clock
        self.sleep = sleep
        self.spacing = spacing
        self.lock = threading.Lock()
        self.ready_at = 0.0
        self.next_start = 0.0

    def wait(self):
        start = self.clock()
        while True:
            with self.lock:
                now = self.clock()
                delay = max(self.ready_at, self.next_start) - now
                if delay <= 0:
                    self.next_start = now + self.spacing
                    return now - start
            self.sleep(min(delay, 30.0))

    def defer(self, attempt):
        delay = min(15.0 * 2**attempt, 120.0)
        with self.lock:
            self.ready_at = max(self.ready_at, self.clock() + delay)
        return delay


def transient(error):
    message = str(error).lower()
    return any(
        x in message
        for x in (
            "vertex http 429:",
            "vertex http 500:",
            "vertex http 502:",
            "vertex http 503:",
            "timed out",
            "connection",
        )
    )
