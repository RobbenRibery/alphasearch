from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from alphasearch.config import load_settings


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Download Qwen embedding model for offline use.")
    parser.add_argument("--repo-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--local-dir", default=str(settings.root_dir / "models" / "Qwen3-VL-Embedding-2B"))
    args = parser.parse_args()

    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id} to {local_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print("Done.")
    print(f"Set ALPHASEARCH_MODEL_PATH={local_dir}")


if __name__ == "__main__":
    main()

