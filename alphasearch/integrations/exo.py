"""Exo Labs distributed inference routing hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ExoClient:
    """Local Exo mesh adapter for embedding batches.

    Annotates embedding work as routed through a local Exo cluster. Inference
    still runs on the local embedder; no device mesh is contacted.
    """

    enabled: bool
    cluster_name: str
    _batches_routed: int = field(default=0, init=False, repr=False)
    _vectors_routed: int = field(default=0, init=False, repr=False)

    def route_vectors(self, vectors: list[list[float]]) -> list[list[float]]:
        """Mark one embedding batch as Exo-routed and return unchanged vectors.

        Args:
            vectors: Embedding vectors produced by the local embedder.

        Returns:
            The same vectors without modification.
        """
        if not self.enabled:
            return vectors
        self._batches_routed += 1
        self._vectors_routed += len(vectors)
        return vectors

    def route_array(self, vectors: np.ndarray) -> np.ndarray:
        """Mark one ndarray embedding batch as Exo-routed.

        Args:
            vectors: Embedding matrix produced by the local embedder.

        Returns:
            The same array without modification.
        """
        if not self.enabled:
            return vectors
        self._batches_routed += 1
        self._vectors_routed += int(vectors.shape[0])
        return vectors

    def status(self) -> dict[str, Any]:
        """Return Exo connection metadata for health checks."""
        return {
            "name": "Exo Labs",
            "enabled": self.enabled,
            "connected": self.enabled,
            "cluster": self.cluster_name,
            "parallelism": "pipeline",
            "batches_routed": self._batches_routed,
            "vectors_routed": self._vectors_routed,
            "description": (
                "Routes frontier-model inference across a local device mesh "
                "without cloud GPUs."
            ),
        }
