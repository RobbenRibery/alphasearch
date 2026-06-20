from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_INSTRUCTION = (
    "Retrieve local files, PDF passages, and images relevant to the user's search query."
)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    data_dir: Path
    db_dir: Path
    table_name: str
    model_path: str
    embedding_dim: int
    batch_size: int
    embedding_instruction: str
    offline: bool


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

    model_path = os.getenv("ALPHASEARCH_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-2B")
    if model_path.startswith((".", "/", "~")):
        resolved_model_path = Path(model_path).expanduser()
        if not resolved_model_path.is_absolute():
            resolved_model_path = root_dir / resolved_model_path
        model_path = str(resolved_model_path.resolve())

    offline = _as_bool(os.getenv("ALPHASEARCH_OFFLINE"), False)
    if offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    return Settings(
        root_dir=root_dir,
        data_dir=data_dir.resolve(),
        db_dir=db_dir.resolve(),
        table_name=os.getenv("ALPHASEARCH_TABLE", "chunks"),
        model_path=model_path,
        embedding_dim=int(os.getenv("ALPHASEARCH_EMBEDDING_DIM", "2048")),
        batch_size=int(os.getenv("ALPHASEARCH_BATCH_SIZE", "4")),
        embedding_instruction=os.getenv(
            "ALPHASEARCH_EMBEDDING_INSTRUCTION", DEFAULT_INSTRUCTION
        ),
        offline=offline,
    )
