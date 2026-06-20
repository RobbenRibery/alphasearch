"""MCP server exposing Qwen LanceDB semantic search."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from alphasearch.search.models import SearchResponse
from alphasearch.search.service import SearchContext, create_qwen_search_context, search_retrieved_items

mcp = FastMCP(
    "alphasearch-qwen",
    instructions=(
        "Search local PDF and image files indexed with Qwen3-VL embeddings in LanceDB. "
        "Use the search tool for natural-language retrieval over the Qwen chunks table."
    ),
)

_search_context: SearchContext | None = None


def get_qwen_search_context() -> SearchContext:
    """Return a lazily initialized Qwen search context."""
    global _search_context
    if _search_context is None:
        _search_context = create_qwen_search_context()
    return _search_context


def reset_qwen_search_context() -> None:
    """Clear the cached Qwen search context."""
    global _search_context
    _search_context = None


@mcp.tool(
    name="qwen_search",
    description=(
        "Search the local Qwen LanceDB index with a natural-language query. "
        "Returns ranked PDF text chunks and image matches with absolute file paths."
    ),
)
def qwen_search(query: str, top_k: int = 8) -> dict[str, Any]:
    """Search indexed chunks using the Qwen embedding index.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of results to return.

    Returns:
        Search results in the same shape as the HTTP ``/search`` endpoint.
    """
    context = get_qwen_search_context()
    results = search_retrieved_items(query, top_k=top_k, context=context)
    return SearchResponse(results=results).model_dump()


@mcp.tool(
    name="qwen_search_index_status",
    description="Return row count and LanceDB location for the Qwen search index.",
)
def qwen_search_index_status() -> dict[str, Any]:
    """Describe the configured Qwen LanceDB index."""
    context = get_qwen_search_context()
    settings = context.settings
    return {
        "embedder": settings.embedder,
        "table_name": settings.table_name,
        "db_dir": str(settings.db_dir),
        "model_path": settings.model_path,
        "embedding_dim": settings.embedding_dim,
        "row_count": context.store.row_count(),
    }


def main() -> None:
    """Run the Qwen search MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
