"""Temporary script to recover integration source from pyc."""
from __future__ import annotations

import dis
import marshal
import types
from pathlib import Path


def load_pyc(path: Path):
    with path.open("rb") as handle:
        handle.read(16)
        return marshal.loads(handle.read())


def main() -> None:
    base = Path("alphasearch/integrations/__pycache__")
    for name in ["config", "overmind", "cosine", "exo", "cognee", "captur", "hub", "__init__"]:
        code = load_pyc(base / f"{name}.cpython-312.pyc")
        print(f"\n===== {name}.py =====")
        for const in code.co_consts:
            if isinstance(const, types.CodeType) and not const.co_name.startswith("<"):
                print(f"--- {const.co_name} {const.co_varnames} ---")
                dis.dis(const)


if __name__ == "__main__":
    main()
