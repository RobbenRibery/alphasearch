from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from alphasearch.models import Chunk


class LanceDBStore:
    def __init__(self, db_dir: Path, table_name: str, embedding_dim: int) -> None:
        self.db_dir = db_dir
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_dir))
        self.table = self._open_or_create_table()

    def _schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("source_id", pa.string(), nullable=False),
                pa.field("absolute_path", pa.string(), nullable=False),
                pa.field("relative_path", pa.string(), nullable=False),
                pa.field("filename", pa.string(), nullable=False),
                pa.field("mime_type", pa.string(), nullable=False),
                pa.field("modality", pa.string(), nullable=False),
                pa.field("time_created", pa.int64(), nullable=False),
                pa.field("time_modified", pa.int64(), nullable=False),
                pa.field("indexed_at", pa.int64(), nullable=False),
                pa.field("file_size", pa.int64(), nullable=False),
                pa.field("file_hash", pa.string(), nullable=False),
                pa.field("metadata", pa.string(), nullable=False),
                pa.field("chunk_b64", pa.string(), nullable=True),
                pa.field("chunk_text", pa.string(), nullable=True),
                pa.field("chunk_index", pa.int64(), nullable=False),
                pa.field("page_number", pa.int64(), nullable=True),
                pa.field("embedding_model", pa.string(), nullable=False),
                pa.field("embedding_instruction", pa.string(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_dim), nullable=False),
            ]
        )

    def _open_or_create_table(self):
        table_names = set(self.db.table_names())
        if self.table_name in table_names:
            return self.db.open_table(self.table_name)
        return self.db.create_table(self.table_name, schema=self._schema())

    def reset(self) -> None:
        if self.table_name in set(self.db.table_names()):
            self.db.drop_table(self.table_name)
        self.table = self.db.create_table(self.table_name, schema=self._schema())

    def row_count(self) -> int:
        return self.table.count_rows()

    def indexed_files(self) -> dict[str, str]:
        if self.table.count_rows() == 0:
            return {}
        rows = self.table.to_arrow().select(["relative_path", "file_hash"]).to_pylist()
        return {row["relative_path"]: row["file_hash"] for row in rows}

    def delete_relative_path(self, relative_path: str) -> None:
        escaped = relative_path.replace("'", "''")
        self.table.delete(f"relative_path = '{escaped}'")

    def add_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        embedding_model: str,
        embedding_instruction: str,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts do not match")
        if not chunks:
            return 0

        now = int(time.time())
        rows = [
            self._row_from_chunk(
                chunk=chunk,
                vector=vector,
                indexed_at=now,
                embedding_model=embedding_model,
                embedding_instruction=embedding_instruction,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.table.add(rows)
        return len(rows)

    def search(self, vector: list[float], limit: int = 8) -> list[dict[str, Any]]:
        return self.table.search(vector).limit(limit).to_list()

    def _row_from_chunk(
        self,
        chunk: Chunk,
        vector: list[float],
        indexed_at: int,
        embedding_model: str,
        embedding_instruction: str,
    ) -> dict[str, Any]:
        source = chunk.source
        return {
            "id": f"{source.file_hash}:{chunk.modality}:{chunk.chunk_index}",
            "source_id": chunk.source_id,
            "absolute_path": str(source.absolute_path),
            "relative_path": source.relative_path,
            "filename": source.filename,
            "mime_type": source.mime_type,
            "modality": chunk.modality,
            "time_created": source.time_created,
            "time_modified": source.time_modified,
            "indexed_at": indexed_at,
            "file_size": source.file_size,
            "file_hash": source.file_hash,
            "metadata": json.dumps(chunk.metadata, sort_keys=True),
            "chunk_b64": chunk.chunk_b64,
            "chunk_text": chunk.chunk_text,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "embedding_model": embedding_model,
            "embedding_instruction": embedding_instruction,
            "vector": vector,
        }
