"""Cosine agent query refinement hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CosineClient:
    """Local Cosine coding-agent adapter for search queries.

    Applies lightweight query normalization suitable for agent workflows. No
    external Lumen model calls are made.
    """

    enabled: bool
    _queries_seen: int = field(default=0, init=False, repr=False)

    def refine_query(self, query: str) -> str:
        """Return a production-ready query string for retrieval.

        Args:
            query: Raw natural-language query from an agent or user.

        Returns:
            Trimmed query text suitable for embedding search.
        """
        if not self.enabled:
            return query
        self._queries_seen += 1
        return " ".join(query.split())

    def status(self) -> dict[str, Any]:
        """Return Cosine connection metadata for health checks."""
        return {
            "name": "Cosine",
            "enabled": self.enabled,
            "connected": self.enabled,
            "model_family": "Lumen",
            "queries_refined": self._queries_seen,
            "description": (
                "Coding-agent query refinement tuned for maintainable, reviewable "
                "retrieval in production codebases."
            ),
        }
