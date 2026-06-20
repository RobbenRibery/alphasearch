"""Async search orchestration."""

import asyncio

import httpx

from alphasearch.embed import embed_query
from alphasearch.mapping import row_to_retrieved_item
from alphasearch.models import RetrievedItem
from alphasearch.store import ChunkIndex


class SearchService:
    """Coordinates embedding and vector retrieval."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        index: ChunkIndex,
    ) -> None:
        """Initialize the search service.

        Args:
            http_client: Shared async HTTP client for embedding requests.
            index: LanceDB chunk index.
        """
        self._http_client = http_client
        self._index = index

    async def search(self, query: str, top_k: int = 1) -> list[RetrievedItem]:
        """Search indexed chunks by cosine similarity.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results to return.

        Returns:
            Retrieved chunks ordered by descending similarity.
        """
        query_vector = await embed_query(self._http_client, query)
        rows = await asyncio.to_thread(self._index.search, query_vector, top_k)
        return [row_to_retrieved_item(row) for row in rows]
