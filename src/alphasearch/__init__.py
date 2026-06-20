"""Semantic search over a LanceDB embedding index."""

from alphasearch.api import app, create_app
from alphasearch.models import RetrievedItem, SearchRequest, SearchResponse
from alphasearch.service import SearchService

__all__ = [
    "RetrievedItem",
    "SearchRequest",
    "SearchResponse",
    "SearchService",
    "app",
    "create_app",
]
