from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from alphasearch.models import Chunk


class CLIPEmbedder:
    """Thin wrapper around CLIP via SentenceTransformers."""

    DEFAULT_MODEL_PATH = "sentence-transformers/clip-ViT-B-32"
    DEFAULT_EMBEDDING_DIM = 512

    def __init__(
        self,
        model_path: str,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        """Initialize the CLIP embedder.

        Args:
            model_path: Hugging Face model id or local model directory.
            embedding_dim: Expected embedding vector length.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.model = SentenceTransformer(
            model_path,
            device="mps",
            model_kwargs={"torch_dtype": torch.float16},
        )

    @property
    def embedding_instruction(self) -> str:
        """Return the instruction prompt used during indexing."""
        return ""

    def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        """Embed search queries."""
        return self._encode(list(queries))

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Embed document chunks."""
        inputs: list[Any] = []
        for chunk in chunks:
            if chunk.modality == "image":
                inputs.append(_load_image(Path(chunk.source.absolute_path)))
            elif chunk.chunk_text:
                inputs.append(chunk.chunk_text)
            else:
                inputs.append("")
        return self._encode(inputs)

    def _encode(self, inputs: Sequence[Any]) -> list[list[float]]:
        """Encode text or image inputs into normalized vectors."""
        embeddings = self.model.encode(
            list(inputs),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D embedding matrix, got shape {array.shape}")
        if array.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected {self.embedding_dim}-d embeddings, got {array.shape[1]}. "
                "Update ALPHASEARCH_EMBEDDING_DIM if you intentionally changed CLIP output size."
            )
        return array.tolist()


def _load_image(path: Path) -> Image.Image:
    """Load an image from disk and return an independent RGB copy."""
    with Image.open(path) as image:
        return image.convert("RGB").copy()
