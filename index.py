"""CLI: build the search index from a folder.

Usage:
    python index.py /path/to/photos
    python index.py /path/to/photos --out index_data
"""

import argparse
import time

from engine import build_index, DEFAULT_INDEX_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder to index (images + text files)")
    ap.add_argument("--out", default=DEFAULT_INDEX_DIR, help="Index output directory")
    args = ap.parse_args()

    def progress(stage, done, total):
        print(f"  [{stage}] {done}/{total}", end="\r", flush=True)

    print(f"Indexing {args.folder} ...")
    t0 = time.time()
    stats = build_index(args.folder, args.out, progress=progress)
    dt = time.time() - t0
    print()
    print(f"Done in {dt:.1f}s -> {stats['num_images']} images, {stats['num_texts']} text files")
    print(f"Index saved to: {args.out}")


if __name__ == "__main__":
    main()
