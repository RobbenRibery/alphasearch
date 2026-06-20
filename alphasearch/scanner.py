from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from alphasearch.models import SourceFile
from alphasearch.utils.hashing import file_sha256
from alphasearch.utils.mime import guess_mime_type, is_supported_mime


IGNORED_FILENAMES = {".DS_Store"}


def scan_data_dir(data_dir: Path) -> Iterator[SourceFile]:
    """Yield supported source files under data_dir in stable path order."""
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name in IGNORED_FILENAMES:
            continue

        mime_type = guess_mime_type(path)
        if not is_supported_mime(mime_type):
            continue

        stat = path.stat()
        yield SourceFile(
            absolute_path=path.resolve(),
            relative_path=str(path.relative_to(data_dir)),
            filename=path.name,
            mime_type=mime_type,
            time_created=int(stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_ctime),
            time_modified=int(stat.st_mtime),
            file_size=stat.st_size,
            file_hash=file_sha256(path),
        )

