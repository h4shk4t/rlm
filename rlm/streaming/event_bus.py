import queue
import threading
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventSubscription:
    token: str
    run_id: str | None
    queue: queue.Queue[dict[str, Any]]


class EventBus:
    """
    Thread-safe in-process pub/sub bus with per-run replay.

    - Subscribers can listen to all runs (`run_id=None`) or one run.
    - Event history is retained per run so clients can subscribe after a run
      starts and still receive initial events.
    """

    def __init__(self, max_history_per_run: int = 1000, queue_size: int = 1024):
        self._max_history_per_run = max_history_per_run
        self._queue_size = queue_size
        self._lock = threading.Lock()
        self._subscriptions: dict[str, EventSubscription] = {}
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._event_counter: dict[str, int] = defaultdict(int)

    def publish(self, event: dict[str, Any]) -> dict[str, Any]:
        run_id = event.get("run_id")
        if not run_id:
            raise ValueError("Event must include a non-empty 'run_id'")

        with self._lock:
            self._event_counter[run_id] += 1
            enriched = {
                "event_id": self._event_counter[run_id],
                **event,
            }

            history = self._history[run_id]
            history.append(enriched)
            if len(history) > self._max_history_per_run:
                del history[:-self._max_history_per_run]

            subscribers = list(self._subscriptions.values())

        for subscription in subscribers:
            if subscription.run_id is None or subscription.run_id == run_id:
                self._enqueue(subscription.queue, enriched)

        return enriched

    def subscribe(self, run_id: str | None = None) -> EventSubscription:
        subscription = EventSubscription(
            token=str(uuid.uuid4()),
            run_id=run_id,
            queue=queue.Queue(maxsize=self._queue_size),
        )
        with self._lock:
            self._subscriptions[subscription.token] = subscription
            replay_events = self._get_replay_events_locked(run_id)

        self._enqueue_many(subscription.queue, replay_events)
        return subscription

    def unsubscribe(self, token: str) -> None:
        with self._lock:
            self._subscriptions.pop(token, None)

    def _get_replay_events_locked(self, run_id: str | None) -> list[dict[str, Any]]:
        if run_id is not None:
            return list(self._history.get(run_id, []))
        replay_events: list[dict[str, Any]] = []
        for events in self._history.values():
            replay_events.extend(events)
        return replay_events

    @staticmethod
    def _enqueue_many(
        q: queue.Queue[dict[str, Any]],
        events: Iterable[dict[str, Any]],
    ) -> None:
        for event in events:
            EventBus._enqueue(q, event)

    @staticmethod
    def _enqueue(q: queue.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
        while True:
            try:
                q.put_nowait(event)
                return
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    return
