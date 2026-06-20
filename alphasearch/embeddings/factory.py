from __future__ import annotations

from alphasearch.config import EmbedderKind, Settings
from alphasearch.embeddings.clip import CLIPEmbedder
from alphasearch.embeddings.protocol import Embedder
from alphasearch.embeddings.qwen_vl import QwenVLEmbedder


def create_embedder(settings: Settings) -> Embedder:
    """Create the configured embedding backend.

    Args:
        settings: Loaded application settings.

    Returns:
        An embedder matching ``settings.embedder``.
    """
    if settings.embedder == "clip":
        return CLIPEmbedder(
            model_path=settings.model_path,
            embedding_dim=settings.embedding_dim,
        )
    return QwenVLEmbedder(
        model_path=settings.model_path,
        instruction=settings.embedding_instruction,
        embedding_dim=settings.embedding_dim,
    )
