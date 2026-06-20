"""Tests for embedding, mapping, and API layers."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from alphasearch.api import app, get_search_service
from alphasearch.config import EMBEDDING_MODEL, EMBEDDINGS_URL
from alphasearch.embed import embed_query
from alphasearch.mapping import cosine_similarity, file_link, row_to_retrieved_item
from alphasearch.models import RetrievedItem
from alphasearch.service import SearchService


@pytest.mark.asyncio
async def test_embed_query_posts_to_embeddings_url() -> None:
    """Embedding requests forward to the LM Studio embeddings endpoint."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    mock_response.raise_for_status.return_value = None

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    vector = await embed_query(mock_client, "hello")

    mock_client.post.assert_awaited_once_with(
        EMBEDDINGS_URL,
        json={"model": EMBEDDING_MODEL, "input": "hello"},
        timeout=60.0,
    )
    assert vector == [0.1, 0.2, 0.3]


def test_file_link_uses_file_uri() -> None:
    """Original files are linked via file:// URIs."""
    link = file_link("/tmp/example.pdf")
    assert link.startswith("file://")
    assert link.endswith("example.pdf")


def test_cosine_similarity_converts_distance() -> None:
    """Identical vectors have similarity 1.0."""
    assert cosine_similarity(0.0) == 1.0
    assert cosine_similarity(0.25) == 0.75


def test_row_to_retrieved_item_maps_fields() -> None:
    """LanceDB rows map into RetrievedItem with a file link."""
    item = row_to_retrieved_item(
        {
            "id": "chunk-1",
            "source_id": "source-1",
            "absolute_path": "/tmp/docs/paper.pdf",
            "relative_path": "docs/paper.pdf",
            "filename": "paper.pdf",
            "chunk_text": "hello world",
            "chunk_index": 0,
            "page_number": 3,
            "_distance": 0.1,
        }
    )

    assert item.id == "chunk-1"
    assert item.file_link.startswith("file://")
    assert item.score == 0.9


@pytest.mark.asyncio
async def test_search_service_composes_embed_and_index() -> None:
    """SearchService embeds the query and maps index rows."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_index = MagicMock()
    mock_index.search.return_value = [
        {
            "id": "chunk-1",
            "source_id": "source-1",
            "absolute_path": "/tmp/docs/paper.pdf",
            "relative_path": "docs/paper.pdf",
            "filename": "paper.pdf",
            "chunk_text": "hello world",
            "chunk_index": 0,
            "page_number": 3,
            "_distance": 0.1,
        }
    ]

    with patch("alphasearch.service.embed_query", new=AsyncMock(return_value=[0.5, 0.5])):
        service = SearchService(mock_http, mock_index)
        results = await service.search("hello", top_k=2)

    mock_index.search.assert_called_once_with([0.5, 0.5], 2)
    assert len(results) == 1
    assert results[0].chunk_text == "hello world"


def test_search_endpoint_returns_results() -> None:
    """POST /search returns retrieved items."""
    mock_service = MagicMock(spec=SearchService)
    mock_service.search = AsyncMock(
        return_value=[
            RetrievedItem(
                id="chunk-1",
                source_id="source-1",
                file_link="file:///tmp/docs/paper.pdf",
                absolute_path="/tmp/docs/paper.pdf",
                relative_path="docs/paper.pdf",
                filename="paper.pdf",
                chunk_text="hello world",
                chunk_index=0,
                page_number=3,
                score=0.9,
            )
        ]
    )
    app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        client = TestClient(app)
        response = client.post("/search", json={"query": "hello", "top_k": 1})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["filename"] == "paper.pdf"
    mock_service.search.assert_awaited_once_with("hello", 1)


def test_health_endpoint() -> None:
    """GET /health returns ok."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
