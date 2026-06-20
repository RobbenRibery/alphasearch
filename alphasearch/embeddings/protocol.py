from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from alphasearch.models import Chunk


class Embedder(Protocol):
    """Minimal interface shared by embedding backends."""

    model_path: str
    embedding_dim: int

    @property
    def embedding_instruction(self) -> str:
        """Return the instruction prompt stored with indexed vectors."""
        ...

    def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        """Embed search queries."""
        ...

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Embed document chunks."""
        ...
