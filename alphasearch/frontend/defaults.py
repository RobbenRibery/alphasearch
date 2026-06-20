"""Default configuration for shortcut frontend search surfaces."""

from __future__ import annotations

import os

QWEN_MODEL_PATH = "Qwen/Qwen3-VL-Embedding-2B"


def configure_qwen_lancedb_defaults() -> None:
    """Set deterministic Qwen LanceDB defaults for shortcut frontends."""
    os.environ.setdefault("ALPHASEARCH_EMBEDDER", "qwen")
    os.environ.setdefault("ALPHASEARCH_TABLE", "chunks")
    os.environ.setdefault("ALPHASEARCH_DB_DIR", "./var/lancedb")
    os.environ.setdefault("ALPHASEARCH_EMBEDDING_DIM", "2048")
    os.environ.setdefault("ALPHASEARCH_MODEL_PATH", QWEN_MODEL_PATH)
