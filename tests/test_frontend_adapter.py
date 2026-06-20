"""Tests for shortcut frontend result adaptation."""

from unittest.mock import MagicMock, patch

from alphasearch.frontend.adapter import search_frontend


def test_search_frontend_splits_dedupes_and_sorts_results() -> None:
    """LanceDB rows are shaped into image and text frontend buckets."""
    rows = [
        {
            "id": "text-low",
            "source_id": "source-text",
            "absolute_path": "/tmp/docs/paper.pdf",
            "relative_path": "docs/paper.pdf",
            "filename": "paper.pdf",
            "modality": "pdf_text",
            "chunk_text": "less relevant",
            "chunk_index": 1,
            "page_number": 2,
            "_distance": 0.4,
        },
        {
            "id": "image-high",
            "source_id": "source-image",
            "absolute_path": "/tmp/data/photo.jpg",
            "relative_path": "photo.jpg",
            "filename": "photo.jpg",
            "modality": "image",
            "chunk_text": None,
            "chunk_index": 0,
            "page_number": None,
            "_distance": 0.05,
        },
        {
            "id": "text-high",
            "source_id": "source-text",
            "absolute_path": "/tmp/docs/paper.pdf",
            "relative_path": "docs/paper.pdf",
            "filename": "paper.pdf",
            "modality": "pdf_text",
            "chunk_text": "more relevant",
            "chunk_index": 0,
            "page_number": 1,
            "_distance": 0.1,
        },
    ]

    with patch("alphasearch.frontend.adapter.search", return_value=rows) as mock_search:
        response = search_frontend("memory systems", context=MagicMock())

    mock_search.assert_called_once()
    assert response["images"] == [
        {
            "path": "/tmp/data/photo.jpg",
            "name": "photo.jpg",
            "score": 0.95,
            "taken_at": None,
        }
    ]
    assert response["texts"] == [
        {
            "path": "/tmp/docs/paper.pdf",
            "name": "paper.pdf",
            "score": 0.9,
            "snippet": "Page 1: more relevant",
        }
    ]


def test_search_frontend_returns_empty_for_blank_query() -> None:
    """Blank queries do not call LanceDB search."""
    with patch("alphasearch.frontend.adapter.search") as mock_search:
        response = search_frontend("   ", context=MagicMock())

    assert response == {"images": [], "texts": []}
    mock_search.assert_not_called()
