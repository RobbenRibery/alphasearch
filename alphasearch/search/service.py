from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphasearch.config import Settings, load_qwen_settings, load_settings
from alphasearch.db import LanceDBStore
from alphasearch.embeddings import Embedder, create_embedder
from alphasearch.search.mapping import row_to_retrieved_item
from alphasearch.search.models import RetrievedItem


@dataclass(frozen=True)
class SearchContext:
    """Dependencies used by semantic search."""

    settings: Settings
    store: LanceDBStore
    embedder: Embedder


def create_qwen_search_context(settings: Settings | None = None) -> SearchContext:
    """Create search dependencies for the Qwen LanceDB index.

    Args:
        settings: Optional Qwen settings. Defaults to ``load_qwen_settings()``.

    Returns:
        Store and embedder dependencies for Qwen search.
    """
    resolved_settings = load_qwen_settings() if settings is None else settings
    return SearchContext(
        settings=resolved_settings,
        store=LanceDBStore(
            resolved_settings.db_dir,
            resolved_settings.table_name,
            resolved_settings.embedding_dim,
        ),
        embedder=create_embedder(resolved_settings),
    )


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


def search_retrieved_items(
    query: str,
    top_k: int = 8,
    *,
    context: SearchContext | None = None,
) -> list[RetrievedItem]:
    """Search the LanceDB index and return API-ready retrieved items.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of results to return.
        context: Optional pre-created store and embedder dependencies.

    Returns:
        Retrieved items ordered by vector similarity.
    """
    resolved_context = create_search_context() if context is None else context
    if resolved_context.store.row_count() == 0:
        return []
    rows = search(query, top_k=top_k, context=resolved_context)
    return [row_to_retrieved_item(row) for row in rows]
