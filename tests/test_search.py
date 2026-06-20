"""Tests for mapping, search, and API layers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from alphasearch.api.app import app, get_ingest_context, get_search_context
from alphasearch.ingestion.pipeline import IngestResult
from alphasearch.search.mapping import cosine_similarity, file_link, row_to_retrieved_item
from alphasearch.search.service import SearchContext, search


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


def test_search_composes_embedder_and_store() -> None:
    """The search function embeds the query and searches LanceDB."""
    mock_store = MagicMock()
    mock_store.search.return_value = [
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
    mock_embedder = MagicMock()
    mock_embedder.embed_queries.return_value = [[0.5, 0.5]]
    context = SearchContext(
        settings=MagicMock(),
        store=mock_store,
        embedder=mock_embedder,
    )

    results = search("hello", top_k=2, context=context)

    mock_embedder.embed_queries.assert_called_once_with(["hello"])
    mock_store.search.assert_called_once_with([0.5, 0.5], limit=2)
    assert len(results) == 1
    assert results[0]["chunk_text"] == "hello world"


def test_search_endpoint_returns_results() -> None:
    """POST /search returns retrieved items."""
    mock_store = MagicMock()
    mock_store.row_count.return_value = 1
    mock_store.search.return_value = [
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
    mock_embedder = MagicMock()
    mock_embedder.embed_queries.return_value = [[0.5, 0.5]]
    context = SearchContext(
        settings=MagicMock(),
        store=mock_store,
        embedder=mock_embedder,
    )
    app.dependency_overrides[get_search_context] = lambda: context

    try:
        client = TestClient(app)
        response = client.post("/search", json={"query": "hello", "top_k": 1})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["filename"] == "paper.pdf"
    mock_store.search.assert_called_once_with([0.5, 0.5], limit=1)


def test_ingest_endpoint_returns_summary() -> None:
    """POST /ingest returns an ingestion summary."""
    context = MagicMock()
    app.dependency_overrides[get_ingest_context] = lambda: context
    result = IngestResult(
        data_dir=Path("/tmp/data"),
        files_scanned=3,
        files_indexed=2,
        files_already_indexed=1,
        chunks_inserted=5,
    )

    try:
        with patch("alphasearch.api.app.run_ingest", return_value=result) as mock_ingest:
            client = TestClient(app)
            response = client.post(
                "/ingest",
                json={"folder": "./data", "reset": True, "limit": 2},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "data_dir": "/tmp/data",
        "files_scanned": 3,
        "files_indexed": 2,
        "files_already_indexed": 1,
        "chunks_inserted": 5,
    }
    mock_ingest.assert_called_once_with(
        "./data",
        reset=True,
        limit=2,
        context=context,
        show_progress=False,
    )


def test_health_endpoint() -> None:
    """GET /health returns ok."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
