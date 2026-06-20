"""Tests for embedder selection and CLIP chunk preparation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alphasearch.config import load_settings
from alphasearch.embeddings.clip import CLIPEmbedder
from alphasearch.embeddings.factory import create_embedder
from alphasearch.embeddings.qwen_vl import QwenVLEmbedder
from alphasearch.models import Chunk, SourceFile


def _source_file(path: Path) -> SourceFile:
    """Build a minimal source file record for tests."""
    return SourceFile(
        absolute_path=path,
        relative_path=path.name,
        filename=path.name,
        mime_type="image/png",
        time_created=0,
        time_modified=0,
        file_size=1,
        file_hash="hash",
    )


def test_create_embedder_selects_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLIP settings create a CLIP embedder and separate LanceDB table defaults."""
    monkeypatch.setenv("ALPHASEARCH_EMBEDDER", "clip")
    monkeypatch.delenv("ALPHASEARCH_TABLE", raising=False)
    monkeypatch.delenv("ALPHASEARCH_MODEL_PATH", raising=False)
    monkeypatch.delenv("ALPHASEARCH_EMBEDDING_DIM", raising=False)

    settings = load_settings()

    assert settings.embedder == "clip"
    assert settings.table_name == "chunks_clip"
    assert settings.embedding_dim == 512
    assert settings.model_path == "sentence-transformers/clip-ViT-B-32"

    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        embedder = create_embedder(settings)

    assert isinstance(embedder, CLIPEmbedder)
    mock_st.assert_called_once()


def test_create_embedder_selects_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen remains the default embedder."""
    monkeypatch.setenv("ALPHASEARCH_EMBEDDER", "qwen")
    monkeypatch.delenv("ALPHASEARCH_TABLE", raising=False)

    settings = load_settings()

    assert settings.embedder == "qwen"
    assert settings.table_name == "chunks"

    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        embedder = create_embedder(settings)

    assert isinstance(embedder, QwenVLEmbedder)
    mock_st.assert_called_once()


def test_clip_embedder_uses_text_and_image_inputs(tmp_path: Path) -> None:
    """CLIP chunk embedding routes text and image modalities differently."""
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"png")

    source = _source_file(image_path)
    chunks = [
        Chunk(source=source, modality="pdf_text", chunk_index=0, chunk_text="hello"),
        Chunk(source=source, modality="image", chunk_index=1),
    ]

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]

    embedder = CLIPEmbedder.__new__(CLIPEmbedder)
    embedder.model_path = CLIPEmbedder.DEFAULT_MODEL_PATH
    embedder.embedding_dim = 2
    embedder.model = mock_model

    with patch("alphasearch.embeddings.clip._load_image", return_value="image-input"):
        vectors = embedder.embed_chunks(chunks)

    mock_model.encode.assert_called_once_with(
        ["hello", "image-input"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    assert len(vectors) == 2
    assert vectors[0] == pytest.approx([0.1, 0.2])
    assert vectors[1] == pytest.approx([0.3, 0.4])
