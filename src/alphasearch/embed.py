"""Embedding client for LM Studio."""

import httpx

from alphasearch.config import (
    EMBEDDING_MODEL,
    EMBEDDINGS_URL,
    EMBED_TIMEOUT_SECONDS,
)


async def embed_query(client: httpx.AsyncClient, query: str) -> list[float]:
    """Embed a search query via the LM Studio embeddings API.

    Args:
        client: Shared async HTTP client.
        query: Natural-language text to embed.

    Returns:
        Embedding vector for the query.

    Raises:
        httpx.HTTPStatusError: If the embeddings API returns an error status.
        KeyError: If the response body is missing the embedding vector.
    """
    response = await client.post(
        EMBEDDINGS_URL,
        json={"model": EMBEDDING_MODEL, "input": query},
        timeout=EMBED_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]
