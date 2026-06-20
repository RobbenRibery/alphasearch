"""Tests for the shortcut service API."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from service import app, get_search_context


def test_api_search_returns_frontend_shape() -> None:
    """GET /api/search returns image and text result buckets."""
    context = MagicMock()
    context.store.row_count.return_value = 1
    expected = {
        "images": [
            {
                "path": "/tmp/data/photo.jpg",
                "name": "photo.jpg",
                "score": 0.91,
                "taken_at": None,
            }
        ],
        "texts": [
            {
                "path": "/tmp/data/paper.pdf",
                "name": "paper.pdf",
                "score": 0.88,
                "snippet": "Page 1: hello",
            }
        ],
    }
    app.dependency_overrides[get_search_context] = lambda: context

    try:
        with patch("service.search_frontend", return_value=expected) as mock_search:
            client = TestClient(app)
            response = client.get("/api/search?q=hello&k=3")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == expected
    mock_search.assert_called_once_with(
        "hello",
        image_limit=3,
        text_limit=5,
        context=context,
    )


def test_api_search_returns_empty_when_index_is_empty() -> None:
    """GET /api/search returns an empty response when LanceDB has no rows."""
    context = MagicMock()
    context.store.row_count.return_value = 0
    app.dependency_overrides[get_search_context] = lambda: context

    try:
        with patch("service.search_frontend") as mock_search:
            client = TestClient(app)
            response = client.get("/api/search?q=hello")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"images": [], "texts": []}
    mock_search.assert_not_called()
