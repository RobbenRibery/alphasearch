"""Tests for the Qwen MCP search server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from alphasearch.mcp.server import qwen_search, qwen_search_index_status, reset_qwen_search_context
from alphasearch.search.models import RetrievedItem


def _sample_item() -> RetrievedItem:
    """Build a sample retrieved item for tests."""
    return RetrievedItem(
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


def test_qwen_search_returns_api_shape() -> None:
    """The MCP search tool returns the same payload shape as the HTTP endpoint."""
    reset_qwen_search_context()
    mock_context = MagicMock()

    with patch("alphasearch.mcp.server.get_qwen_search_context", return_value=mock_context):
        with patch(
            "alphasearch.mcp.server.search_retrieved_items",
            return_value=[_sample_item()],
        ) as mock_search:
            payload = qwen_search("hello", top_k=3)

    mock_search.assert_called_once_with("hello", top_k=3, context=mock_context)
    assert payload == {
        "results": [
            {
                "id": "chunk-1",
                "source_id": "source-1",
                "file_link": "file:///tmp/docs/paper.pdf",
                "absolute_path": "/tmp/docs/paper.pdf",
                "relative_path": "docs/paper.pdf",
                "filename": "paper.pdf",
                "chunk_text": "hello world",
                "chunk_index": 0,
                "page_number": 3,
                "score": 0.9,
            }
        ]
    }


def test_qwen_search_index_status_reports_table_metadata() -> None:
    """The MCP status tool exposes Qwen index metadata."""
    reset_qwen_search_context()
    mock_store = MagicMock()
    mock_store.row_count.return_value = 42
    mock_settings = MagicMock(
        embedder="qwen",
        table_name="chunks",
        db_dir="/tmp/lancedb",
        model_path="/tmp/models/qwen",
        embedding_dim=2048,
    )
    mock_context = MagicMock(store=mock_store, settings=mock_settings)

    with patch("alphasearch.mcp.server.get_qwen_search_context", return_value=mock_context):
        payload = qwen_search_index_status()

    assert payload == {
        "embedder": "qwen",
        "table_name": "chunks",
        "db_dir": "/tmp/lancedb",
        "model_path": "/tmp/models/qwen",
        "embedding_dim": 2048,
        "row_count": 42,
    }
