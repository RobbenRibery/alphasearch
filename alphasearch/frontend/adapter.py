"""Map LanceDB search results into the shortcut frontend contract."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

from alphasearch.search.mapping import row_to_retrieved_item
from alphasearch.search.models import RetrievedItem
from alphasearch.search.service import SearchContext, search

DEFAULT_CANDIDATE_LIMIT = 50


class ImageResult(TypedDict):
    """Image result returned by the shortcut frontend API."""

    path: str
    name: str
    score: float
    taken_at: str | None


class TextResult(TypedDict):
    """Text result returned by the shortcut frontend API."""

    path: str
    name: str
    score: float
    snippet: str


class FrontendSearchResponse(TypedDict):
    """Shortcut frontend search response."""

    images: list[ImageResult]
    texts: list[TextResult]


def search_frontend(
    query: str,
    *,
    image_limit: int = 9,
    text_limit: int = 5,
    candidate_limit: int | None = None,
    context: SearchContext | None = None,
) -> FrontendSearchResponse:
    """Search LanceDB and shape results for the shortcut frontend.

    Args:
        query: Natural-language search query.
        image_limit: Maximum image results to return.
        text_limit: Maximum text results to return.
        candidate_limit: Optional LanceDB candidate count before modality splitting.
        context: Optional pre-created search dependencies.

    Returns:
        JSON-compatible search results with separate image and text lists.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return {"images": [], "texts": []}

    resolved_candidate_limit = candidate_limit or max(
        DEFAULT_CANDIDATE_LIMIT,
        image_limit + text_limit,
    )
    rows = search(cleaned_query, top_k=resolved_candidate_limit, context=context)
    items = [row_to_retrieved_item(row) for row in rows]

    image_items = _best_items_by_path(item for item in items if item.modality == "image")
    text_items = _best_items_by_path(item for item in items if item.modality == "pdf_text")

    return {
        "images": [_image_result(item) for item in image_items[:image_limit]],
        "texts": [_text_result(item) for item in text_items[:text_limit]],
    }


def _best_items_by_path(items: Iterable[RetrievedItem]) -> list[RetrievedItem]:
    """Return best-scoring items by absolute path.

    Args:
        items: Iterable of retrieved chunks.

    Returns:
        Best item for each file path sorted by score descending.
    """
    best_by_path: dict[str, RetrievedItem] = {}
    for item in items:
        previous = best_by_path.get(item.absolute_path)
        if previous is None or item.score > previous.score:
            best_by_path[item.absolute_path] = item
    return sorted(best_by_path.values(), key=lambda item: item.score, reverse=True)


def _image_result(item: RetrievedItem) -> ImageResult:
    """Convert an image item into frontend JSON.

    Args:
        item: Retrieved image item.

    Returns:
        Image result for the shortcut frontend.
    """
    return {
        "path": item.absolute_path,
        "name": item.filename,
        "score": round(item.score, 3),
        "taken_at": None,
    }


def _text_result(item: RetrievedItem) -> TextResult:
    """Convert a text item into frontend JSON.

    Args:
        item: Retrieved text item.

    Returns:
        Text result for the shortcut frontend.
    """
    return {
        "path": item.absolute_path,
        "name": item.filename,
        "score": round(item.score, 3),
        "snippet": _snippet(item),
    }


def _snippet(item: RetrievedItem) -> str:
    """Build a compact text snippet for a retrieved chunk.

    Args:
        item: Retrieved text item.

    Returns:
        Snippet with a page prefix when page metadata is available.
    """
    text = " ".join((item.chunk_text or "").split())
    if item.page_number is None:
        return text
    if not text:
        return f"Page {item.page_number}"
    return f"Page {item.page_number}: {text}"
