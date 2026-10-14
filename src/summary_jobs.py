"""Process-owned summary jobs; UI reruns never own or cancel inference."""

from dataclasses import dataclass, replace
from threading import RLock, Thread
from typing import Callable
from uuid import uuid4


@dataclass
class SummaryState:
    id: str
    running: bool = True
    text: str = ""
    progress: str = "작업을 준비하고 있습니다."
    error: str = ""
    fraction: float = 0.0


class SummaryJobs:
    def __init__(self):
        self._lock = RLock()
        self._states = {}

    def is_running(self):
        with self._lock:
            return any(state.running for state in self._states.values())

    def snapshot(self, key):
        with self._lock:
            state = self._states.get(key)
            return replace(state) if state else None

    def start(self, key, work: Callable) -> bool:
        with self._lock:
            previous = self._states.get(key)
            if previous and previous.running:
                return False
            state = self._states[key] = SummaryState(uuid4().hex)

        def progress(message, fraction=None):
            with self._lock:
                state.progress = message
                if fraction is not None:
                    state.fraction = max(0.0, min(1.0, fraction))

        def token(value):
            with self._lock:
                state.text += value

        def run():
            try:
                work(progress, token)
                with self._lock:
                    state.progress = "작업을 완료했습니다."
            except Exception as exc:
                with self._lock:
                    state.error = str(exc)
                    state.progress = "작업에 실패했습니다. 완료된 결과는 유지됩니다."
            finally:
                with self._lock:
                    state.running = False
                    # Completed results live in SQLite, not a second memory copy.
                    state.text = ""

        Thread(target=run, name=f"lass-summary-{state.id}", daemon=True).start()
        return True


summary_jobs = SummaryJobs()
