from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


EmbedderKind = Literal["qwen", "clip"]

DEFAULT_INSTRUCTION = (
    "Retrieve local files, PDF passages, and images relevant to the user's search query."
)

QWEN_DEFAULTS = {
    "model_path": "Qwen/Qwen3-VL-Embedding-2B",
    "table_name": "chunks",
    "embedding_dim": 2048,
    "embedding_instruction": DEFAULT_INSTRUCTION,
}

CLIP_DEFAULTS = {
    "model_path": "sentence-transformers/clip-ViT-B-32",
    "table_name": "chunks_clip",
    "embedding_dim": 512,
    "embedding_instruction": "",
}


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    data_dir: Path
    db_dir: Path
    embedder: EmbedderKind
    table_name: str
    model_path: str
    embedding_dim: int
    batch_size: int
    embedding_instruction: str
    offline: bool


def _resolve_embedder(value: str | None) -> EmbedderKind:
    """Parse the configured embedder kind."""
    embedder = (value or "qwen").strip().lower()
    if embedder == "qwen":
        return "qwen"
    if embedder == "clip":
        return "clip"
    raise ValueError(
        f"Unsupported ALPHASEARCH_EMBEDDER={embedder!r}. Expected 'qwen' or 'clip'."
    )


def _embedder_defaults(embedder: EmbedderKind) -> dict[str, str | int]:
    """Return default model, table, and dimension settings for an embedder."""
    return CLIP_DEFAULTS if embedder == "clip" else QWEN_DEFAULTS


def _resolve_model_path(model_path: str, root_dir: Path) -> str:
    """Resolve a local or remote model path."""
    if model_path.startswith((".", "/", "~")):
        resolved_model_path = Path(model_path).expanduser()
        if not resolved_model_path.is_absolute():
            resolved_model_path = root_dir / resolved_model_path
        return str(resolved_model_path.resolve())
    return model_path


def load_settings() -> Settings:
    """Load settings from .env and environment variables."""
    root_dir = Path(__file__).resolve().parents[1]
    load_dotenv(root_dir / ".env")

    data_dir = Path(os.getenv("ALPHASEARCH_DATA_DIR", "./data")).expanduser()
    db_dir = Path(os.getenv("ALPHASEARCH_DB_DIR", "./var/lancedb")).expanduser()

    if not data_dir.is_absolute():
        data_dir = root_dir / data_dir
    if not db_dir.is_absolute():
        db_dir = root_dir / db_dir

    embedder = _resolve_embedder(os.getenv("ALPHASEARCH_EMBEDDER"))
    defaults = _embedder_defaults(embedder)

    model_path = _resolve_model_path(
        os.getenv("ALPHASEARCH_MODEL_PATH", str(defaults["model_path"])),
        root_dir,
    )

    offline = _as_bool(os.getenv("ALPHASEARCH_OFFLINE"), False)
    if offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    return Settings(
        root_dir=root_dir,
        data_dir=data_dir.resolve(),
        db_dir=db_dir.resolve(),
        embedder=embedder,
        table_name=os.getenv("ALPHASEARCH_TABLE", str(defaults["table_name"])),
        model_path=model_path,
        embedding_dim=int(
            os.getenv("ALPHASEARCH_EMBEDDING_DIM", str(defaults["embedding_dim"]))
        ),
        batch_size=int(os.getenv("ALPHASEARCH_BATCH_SIZE", "4")),
        embedding_instruction=os.getenv(
            "ALPHASEARCH_EMBEDDING_INSTRUCTION",
            str(defaults["embedding_instruction"]),
        ),
        offline=offline,
    )
