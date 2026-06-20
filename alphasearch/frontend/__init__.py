"""Frontend adapters for local search surfaces."""

from alphasearch.frontend.adapter import (
    FrontendSearchResponse,
    ImageResult,
    TextResult,
    search_frontend,
)
from alphasearch.frontend.defaults import configure_qwen_lancedb_defaults

__all__ = [
    "FrontendSearchResponse",
    "ImageResult",
    "TextResult",
    "configure_qwen_lancedb_defaults",
    "search_frontend",
]
