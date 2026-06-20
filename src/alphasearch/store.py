"""LanceDB chunk index access."""

from typing import Any

import lancedb

from alphasearch.config import LANCE_URI, TABLE_NAME


class ChunkIndex:
    """Read-only vector index over indexed chunks."""

    def __init__(
        self,
        uri: str = LANCE_URI,
        table_name: str = TABLE_NAME,
    ) -> None:
        """Open a LanceDB table handle.

        Args:
            uri: LanceDB database URI.
            table_name: Name of the chunks table.
        """
        self._table = lancedb.connect(uri).open_table(table_name)

    def search(self, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """Run cosine similarity search over stored embeddings.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            Matching chunk rows ordered by ascending cosine distance.
        """
        return (
            self._table.search(query_vector)
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )
