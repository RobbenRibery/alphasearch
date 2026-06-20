from __future__ import annotations

from alphasearch.config import load_settings
from alphasearch.db import LanceDBStore


def main() -> None:
    settings = load_settings()
    store = LanceDBStore(settings.db_dir, settings.table_name, settings.embedding_dim)
    store.reset()
    print(f"Reset LanceDB table {settings.table_name} at {settings.db_dir}")


if __name__ == "__main__":
    main()

