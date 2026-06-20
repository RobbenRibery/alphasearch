from __future__ import annotations

import mimetypes
from pathlib import Path


SUPPORTED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
SUPPORTED_MIMES = {"application/pdf", *SUPPORTED_IMAGE_MIMES}


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type:
        return mime_type

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".heic":
        return "image/heic"
    if suffix == ".heif":
        return "image/heif"
    return "application/octet-stream"


def is_supported_mime(mime_type: str) -> bool:
    return mime_type == "application/pdf" or mime_type.startswith("image/")

