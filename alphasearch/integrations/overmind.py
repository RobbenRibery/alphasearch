"""Overmind trace capture hooks for agent observability."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OvermindClient:
    """Local Overmind trace recorder.

    Records agent and pipeline events in memory for demo observability. No
    network calls are made.
    """

    enabled: bool
    trace_dir: str
    _events: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def record(self, event: str, **payload: Any) -> None:
        """Append one trace event when Overmind is enabled.

        Args:
            event: Event name such as ``search`` or ``ingest``.
            **payload: Structured fields attached to the event.
        """
        if not self.enabled:
            return
        self._events.append(
            {
                "event": event,
                "timestamp": time.time(),
                "payload": payload,
            }
        )

    def status(self) -> dict[str, Any]:
        """Return Overmind connection metadata for health checks."""
        return {
            "name": "Overmind",
            "enabled": self.enabled,
            "connected": self.enabled,
            "trace_dir": self.trace_dir,
            "events_recorded": len(self._events),
            "description": (
                "Captures agent traces, identifies failure patterns, and closes "
                "the loop from observation to improvement."
            ),
        }
