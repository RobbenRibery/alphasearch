from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphasearch.config import Settings, load_settings
from alphasearch.db import LanceDBStore
from alphasearch.embeddings import Embedder, create_embedder


@dataclass(frozen=True)
class SearchContext:
    """Dependencies used by semantic search."""

    settings: Settings
    store: LanceDBStore
    embedder: Embedder


def create_search_context(settings: Settings | None = None) -> SearchContext:
    """Create search dependencies from configured settings.

    Args:
        settings: Optional loaded settings. Defaults to `load_settings()`.

    Returns:
        Store and embedder dependencies for search.
    """
    resolved_settings = load_settings() if settings is None else settings
    return SearchContext(
        settings=resolved_settings,
        store=LanceDBStore(
            resolved_settings.db_dir,
            resolved_settings.table_name,
            resolved_settings.embedding_dim,
        ),
        embedder=create_embedder(resolved_settings),
    )


def search(
    query: str,
    top_k: int = 8,
    *,
    context: SearchContext | None = None,
) -> list[dict[str, Any]]:
    """Search the local LanceDB index with a natural-language query.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of results to return.
        context: Optional pre-created store and embedder dependencies.

    Returns:
        Raw LanceDB result rows ordered by vector distance.
    """
    resolved_context = create_search_context() if context is None else context
    query_vector = resolved_context.embedder.embed_queries([query])[0]
    return resolved_context.store.search(query_vector, limit=top_k)
