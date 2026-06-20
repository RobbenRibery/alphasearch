"""Default configuration for shortcut frontend search surfaces."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

QWEN_MODEL_PATH = "Qwen/Qwen3-VL-Embedding-2B"


def configure_qwen_lancedb_defaults() -> None:
    """Set deterministic Qwen LanceDB defaults for shortcut frontends."""
    root_dir = Path(__file__).resolve().parents[2]
    load_dotenv(root_dir / ".env")
    os.environ.setdefault("ALPHASEARCH_EMBEDDER", "qwen")
    os.environ.setdefault("ALPHASEARCH_TABLE", "chunks")
    os.environ.setdefault("ALPHASEARCH_DB_DIR", "./var/lancedb")
    os.environ.setdefault("ALPHASEARCH_DATA_DIR", "./data")
    os.environ.setdefault("ALPHASEARCH_EMBEDDING_DIM", "2048")
    os.environ.setdefault("ALPHASEARCH_MODEL_PATH", QWEN_MODEL_PATH)
