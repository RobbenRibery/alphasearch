"""
Core offline search engine.

Everything here runs on-device:
  - CLIP (clip-ViT-B-32) embeds images AND text queries into ONE shared space,
    so a typed query like "my cat on a sofa" can be matched against photos.
  - all-MiniLM-L6-v2 embeds text-file contents for document search.

Models are downloaded once (while online) and cached under ~/.cache/huggingface.
Set HF_HUB_OFFLINE=1 to prove there is zero network access during the demo.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

# HEIC/HEIF support (iPhone photos) -- registers a Pillow opener.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pass

from PIL import Image, ExifTags

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".heic", ".heif", ".tiff"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".py", ".js", ".ts", ".csv", ".log", ".json", ".html"}

CLIP_MODEL_NAME = "clip-ViT-B-32"
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"

# ----------------------------------------------------------------------------
# Model loading (lazy + cached)
# ----------------------------------------------------------------------------
_clip_model = None
_text_model = None


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_clip():
    global _clip_model
    if _clip_model is None:
        from sentence_transformers import SentenceTransformer

        _clip_model = SentenceTransformer(CLIP_MODEL_NAME, device=_device())
    return _clip_model


def get_text_model():
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer

        _text_model = SentenceTransformer(TEXT_MODEL_NAME, device=_device())
    return _text_model


# ----------------------------------------------------------------------------
# Metadata helpers
# ----------------------------------------------------------------------------
_EXIF_DATE_TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def _image_taken_at(path: Path) -> str | None:
    """Best-effort capture time from EXIF, falling back to file mtime."""
    try:
        img = Image.open(path)
        exif = img.getexif()
        for tag_name in ("DateTimeOriginal", "DateTime"):
            tag_id = _EXIF_DATE_TAGS.get(tag_name)
            if tag_id and tag_id in exif:
                raw = str(exif[tag_id])
                try:
                    dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
                    return dt.isoformat()
                except ValueError:
                    pass
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Index data structures
# ----------------------------------------------------------------------------
@dataclass
class ImageRecord:
    path: str
    taken_at: str | None


@dataclass
class TextRecord:
    path: str
    snippet: str


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ----------------------------------------------------------------------------
# Indexing
# ----------------------------------------------------------------------------
def find_files(root: Path):
    images, texts = [], []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(p)
        elif ext in TEXT_EXTS:
            texts.append(p)
    return images, texts


def build_index(root: str, out_dir: str, progress=None) -> dict:
    """Walk `root`, embed images + text files, save to `out_dir`.

    `progress` is an optional callable(stage:str, done:int, total:int).
    """
    root_path = Path(root).expanduser()
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    images, texts = find_files(root_path)

    # --- Images via CLIP ---
    img_records: list[ImageRecord] = []
    img_embeddings: list[np.ndarray] = []
    if images:
        clip = get_clip()
        batch = 16
        valid_imgs = []
        loaded = []
        for i, p in enumerate(images):
            try:
                im = Image.open(p).convert("RGB")
                loaded.append(im)
                valid_imgs.append(p)
            except Exception:
                continue
            if len(loaded) == batch or i == len(images) - 1:
                emb = clip.encode(loaded, convert_to_numpy=True, show_progress_bar=False)
                img_embeddings.append(emb)
                loaded = []
                if progress:
                    progress("images", len(valid_imgs), len(images))
        for p in valid_imgs:
            img_records.append(ImageRecord(path=str(p), taken_at=_image_taken_at(p)))

    if img_embeddings:
        img_mat = _normalize(np.vstack(img_embeddings).astype("float32"))
    else:
        img_mat = np.zeros((0, 512), dtype="float32")

    # --- Text files via MiniLM ---
    txt_records: list[TextRecord] = []
    txt_embeddings: list[np.ndarray] = []
    if texts:
        tm = get_text_model()
        contents = []
        for i, p in enumerate(texts):
            try:
                raw = p.read_text(errors="ignore")[:4000]
            except Exception:
                continue
            if not raw.strip():
                continue
            contents.append(raw)
            txt_records.append(TextRecord(path=str(p), snippet=raw[:200].replace("\n", " ")))
            if progress:
                progress("texts", i + 1, len(texts))
        if contents:
            txt_embeddings = tm.encode(contents, convert_to_numpy=True, show_progress_bar=False)

    if len(txt_embeddings):
        txt_mat = _normalize(np.asarray(txt_embeddings).astype("float32"))
    else:
        txt_mat = np.zeros((0, 384), dtype="float32")

    np.save(out / "image_emb.npy", img_mat)
    np.save(out / "text_emb.npy", txt_mat)
    (out / "image_meta.json").write_text(json.dumps([asdict(r) for r in img_records]))
    (out / "text_meta.json").write_text(json.dumps([asdict(r) for r in txt_records]))
    (out / "index_info.json").write_text(
        json.dumps(
            {
                "root": str(root_path),
                "num_images": len(img_records),
                "num_texts": len(txt_records),
                "built_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
    )
    return {"num_images": len(img_records), "num_texts": len(txt_records)}


# ----------------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------------
class SearchIndex:
    def __init__(self, index_dir: str):
        d = Path(index_dir).expanduser()
        self.image_emb = np.load(d / "image_emb.npy")
        self.text_emb = np.load(d / "text_emb.npy")
        self.image_meta = json.loads((d / "image_meta.json").read_text())
        self.text_meta = json.loads((d / "text_meta.json").read_text())
        self.info = json.loads((d / "index_info.json").read_text())

    def search_images(self, query: str, k: int = 12):
        if len(self.image_emb) == 0:
            return []
        clip = get_clip()
        q = clip.encode([query], convert_to_numpy=True)
        q = _normalize(q.astype("float32"))[0]
        scores = self.image_emb @ q
        idx = np.argsort(-scores)[:k]
        return [
            {**self.image_meta[i], "score": float(scores[i])}
            for i in idx
        ]

    def search_texts(self, query: str, k: int = 8):
        if len(self.text_emb) == 0:
            return []
        tm = get_text_model()
        q = tm.encode([query], convert_to_numpy=True)
        q = _normalize(q.astype("float32"))[0]
        scores = self.text_emb @ q
        idx = np.argsort(-scores)[:k]
        return [
            {**self.text_meta[i], "score": float(scores[i])}
            for i in idx
        ]


DEFAULT_INDEX_DIR = str(Path(__file__).parent / "index_data")
