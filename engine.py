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
    mtime: float = 0.0


@dataclass
class TextRecord:
    path: str
    snippet: str
    mtime: float = 0.0


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _mtime(p) -> float:
    try:
        return Path(p).stat().st_mtime
    except Exception:
        return 0.0


IMG_DIM = 512
TXT_DIM = 384

# Skip noisy / irrelevant folders so we don't index junk.
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".Trash",
             "Library", "index_data"}


# ----------------------------------------------------------------------------
# Indexing
# ----------------------------------------------------------------------------
def find_files(root: Path):
    images, texts = [], []
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(p)
        elif ext in TEXT_EXTS:
            texts.append(p)
    return images, texts


def _embed_image_paths(paths, progress=None):
    """Return (normalized embeddings (N, IMG_DIM), list[ImageRecord])."""
    if not paths:
        return np.zeros((0, IMG_DIM), "float32"), []
    clip = get_clip()
    batch, loaded, valid, embs = 16, [], [], []
    for i, p in enumerate(paths):
        try:
            loaded.append(Image.open(p).convert("RGB"))
            valid.append(Path(p))
        except Exception:
            continue
        if len(loaded) == batch or i == len(paths) - 1:
            embs.append(clip.encode(loaded, convert_to_numpy=True, show_progress_bar=False))
            loaded = []
            if progress:
                progress("images", len(valid), len(paths))
    mat = np.vstack(embs).astype("float32") if embs else np.zeros((0, IMG_DIM), "float32")
    records = [ImageRecord(str(p), _image_taken_at(p), _mtime(p)) for p in valid]
    return (_normalize(mat) if len(mat) else mat), records


def _embed_text_paths(paths, progress=None):
    """Return (normalized embeddings (N, TXT_DIM), list[TextRecord])."""
    contents, records = [], []
    for i, p in enumerate(paths):
        p = Path(p)
        try:
            raw = p.read_text(errors="ignore")[:4000]
        except Exception:
            continue
        if not raw.strip():
            continue
        contents.append(raw)
        records.append(TextRecord(str(p), raw[:200].replace("\n", " "), _mtime(p)))
        if progress:
            progress("texts", i + 1, len(paths))
    if contents:
        mat = _normalize(get_text_model().encode(
            contents, convert_to_numpy=True, show_progress_bar=False).astype("float32"))
    else:
        mat = np.zeros((0, TXT_DIM), "float32")
    return mat, records


def _to_dicts(items):
    return [m if isinstance(m, dict) else asdict(m) for m in items]


def _save(out: Path, root_path: Path, img_mat, img_meta, txt_mat, txt_meta):
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "image_emb.npy", img_mat)
    np.save(out / "text_emb.npy", txt_mat)
    (out / "image_meta.json").write_text(json.dumps(_to_dicts(img_meta)))
    (out / "text_meta.json").write_text(json.dumps(_to_dicts(txt_meta)))
    (out / "index_info.json").write_text(json.dumps({
        "root": str(root_path),
        "num_images": len(img_meta),
        "num_texts": len(txt_meta),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }))


def build_index(root: str, out_dir: str, progress=None) -> dict:
    """Full (re)build: embed every image + text file under `root`."""
    root_path = Path(root).expanduser()
    out = Path(out_dir).expanduser()
    images, texts = find_files(root_path)
    img_mat, img_records = _embed_image_paths(images, progress)
    txt_mat, txt_records = _embed_text_paths(texts, progress)
    _save(out, root_path, img_mat, img_records, txt_mat, txt_records)
    return {"num_images": len(img_records), "num_texts": len(txt_records)}


def _merge(current_paths, emb, meta, embed_fn, dim, progress):
    """Keep unchanged rows, drop deleted/changed, embed new/changed.

    Returns (final_emb, final_meta_dicts, added, removed).
    """
    cur = {str(Path(p)): _mtime(p) for p in current_paths}
    existing_mtime = {m["path"]: m.get("mtime", 0.0) for m in meta}

    keep_rows, keep_meta = [], []
    for i, m in enumerate(meta):
        p = m["path"]
        if p in cur and abs(cur[p] - m.get("mtime", 0.0)) < 1e-6:
            keep_rows.append(i)
            keep_meta.append(m)

    to_embed = [
        p for p in current_paths
        if str(Path(p)) not in existing_mtime
        or abs(cur[str(Path(p))] - existing_mtime[str(Path(p))]) >= 1e-6
    ]
    removed = sum(1 for m in meta if m["path"] not in cur)

    new_mat, new_records = embed_fn(to_embed, progress)
    kept_mat = emb[keep_rows] if keep_rows else np.zeros((0, dim), "float32")
    final_mat = np.vstack([kept_mat, new_mat]) if (len(kept_mat) or len(new_mat)) \
        else np.zeros((0, dim), "float32")
    final_meta = keep_meta + _to_dicts(new_records)
    return final_mat, final_meta, len(to_embed), removed


def update_index(root: str, out_dir: str, progress=None) -> dict:
    """Incremental update: only embed new/changed files; drop deleted ones."""
    root_path = Path(root).expanduser()
    out = Path(out_dir).expanduser()
    if not (out / "index_info.json").exists():
        return build_index(root, out_dir, progress)

    img_emb = np.load(out / "image_emb.npy")
    txt_emb = np.load(out / "text_emb.npy")
    img_meta = json.loads((out / "image_meta.json").read_text())
    txt_meta = json.loads((out / "text_meta.json").read_text())

    images, texts = find_files(root_path)
    img_mat, img_meta2, ai, ri = _merge(images, img_emb, img_meta, _embed_image_paths, IMG_DIM, progress)
    txt_mat, txt_meta2, at, rt = _merge(texts, txt_emb, txt_meta, _embed_text_paths, TXT_DIM, progress)

    _save(out, root_path, img_mat, img_meta2, txt_mat, txt_meta2)
    return {
        "num_images": len(img_meta2),
        "num_texts": len(txt_meta2),
        "added": ai + at,
        "removed": ri + rt,
    }


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
