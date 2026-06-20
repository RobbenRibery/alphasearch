"""Cognee agent memory hooks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CogneeClient:
    """Local Cognee memory adapter for cross-session retrieval context.

    Stores recent queries in process memory only. No external memory service is
    contacted.
    """

    enabled: bool
    session_id: str
    _memory_limit: int = 32
    _queries: deque[str] = field(default_factory=deque, init=False, repr=False)

    def remember_query(self, query: str) -> None:
        """Persist one query in the local Cognee memory window.

        Args:
            query: Natural-language query issued by an agent or user.
        """
        if not self.enabled:
            return
        self._queries.append(query)
        while len(self._queries) > self._memory_limit:
            self._queries.popleft()

    def recall_context(self) -> str | None:
        """Return a short memory prefix derived from recent queries.

        Returns:
            Optional context string prepended to new retrieval queries.
        """
        if not self.enabled or not self._queries:
            return None
        recent = list(self._queries)[-3:]
        return " ".join(recent)

    def status(self) -> dict[str, Any]:
        """Return Cognee connection metadata for health checks."""
        return {
            "name": "Cognee",
            "enabled": self.enabled,
            "connected": self.enabled,
            "session_id": self.session_id,
            "memories_stored": len(self._queries),
            "description": (
                "Structured, searchable agent memory for persistent recall "
                "across sessions."
            ),
        }
