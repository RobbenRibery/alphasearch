from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alphasearch.models import Chunk


class QwenVLEmbedder:
    """Thin wrapper around Qwen3-VL-Embedding via SentenceTransformers."""

    def __init__(
        self,
        model_path: str,
        instruction: str,
        embedding_dim: int = 2048,
    ) -> None:
        """Initialize the Qwen3-VL embedder.

        Args:
            model_path: Hugging Face model id or local model directory.
            instruction: Prompt prepended to each input before embedding.
            embedding_dim: Expected embedding vector length.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_path = model_path
        self.instruction = instruction
        self.embedding_dim = embedding_dim
        self.model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            device="mps",
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

    def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        """Embed search queries."""
        return self._encode(list(queries))

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Embed document chunks."""
        inputs: list[Any] = []
        for chunk in chunks:
            if chunk.modality == "image":
                inputs.append({"image": str(Path(chunk.source.absolute_path))})
            elif chunk.chunk_text:
                inputs.append({"text": chunk.chunk_text})
            else:
                inputs.append({"text": ""})
        return self._encode(inputs)

    def _encode(self, inputs: Sequence[Any]) -> list[list[float]]:
        """Encode text or multimodal inputs into normalized vectors."""
        embeddings = self.model.encode(
            list(inputs),
            prompt=self.instruction,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D embedding matrix, got shape {array.shape}")
        if array.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected {self.embedding_dim}-d embeddings, got {array.shape[1]}. "
                "Update ALPHASEARCH_EMBEDDING_DIM if you intentionally changed Qwen output size."
            )
        return array.tolist()
