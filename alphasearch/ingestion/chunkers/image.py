from __future__ import annotations

import base64
from typing import Any

from PIL import ExifTags, Image

from alphasearch.models import Chunk, SourceFile

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass


def _safe_exif(image: Image.Image) -> dict[str, Any]:
    """Extract simple EXIF values safe to serialize as JSON.

    Args:
        image: Open Pillow image.

    Returns:
        EXIF fields with primitive values.
    """
    try:
        exif = image.getexif()
    except Exception:
        return {}

    result: dict[str, Any] = {}
    for key, value in exif.items():
        tag = ExifTags.TAGS.get(key, str(key))
        if isinstance(value, bytes):
            continue
        if isinstance(value, (str, int, float)):
            result[tag] = value
    return result


def chunk_image(source: SourceFile) -> Chunk:
    """Represent one image as one chunk for the hackathon MVP.

    Args:
        source: Image file metadata.

    Returns:
        A single image chunk with metadata and base64-encoded bytes.
    """
    with Image.open(source.absolute_path) as image:
        metadata = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "exif": _safe_exif(image),
        }

    chunk_b64 = base64.b64encode(source.absolute_path.read_bytes()).decode("ascii")
    return Chunk(
        source=source,
        modality="image",
        chunk_index=0,
        chunk_text=None,
        chunk_b64=chunk_b64,
        page_number=None,
        metadata=metadata,
    )
