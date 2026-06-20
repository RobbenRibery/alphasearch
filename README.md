# AlphaSearch

AlphaSearch is a local-first search prototype for the offline agent hackathon. It
turns files under `./data` into multimodal embeddings and stores them in a local
LanceDB database.

The current MVP supports:

- PDF text chunking with page numbers.
- One chunk per image for JPEG, PNG, WebP, HEIC, and HEIF files.
- Qwen/Qwen3-VL-Embedding-2B embeddings through SentenceTransformers.
- Local LanceDB storage under `./var/lancedb`.
- Idempotent re-indexing by file hash.
- A small CLI for indexing and natural-language search.

## Repository Layout

```text
alphasearch/
  data/                       # demo PDFs and images
  alphasearch/                # package code
  scripts/                    # runnable wrappers
  models/                     # optional local model snapshot, gitignored
  var/lancedb/                # generated LanceDB data, gitignored
```

## Stored Fields

Each LanceDB row represents one searchable chunk.

```text
id: string
source_id: string
absolute_path: string
relative_path: string
filename: string
mime_type: string
modality: pdf_text | image
time_created: int
time_modified: int
indexed_at: int
file_size: int
file_hash: string
metadata: JSON-encoded dict
chunk_b64: string | null
chunk_text: string | null
chunk_index: int
page_number: int | null
embedding_model: string
embedding_instruction: string
vector: float32[2048]
```

`metadata` is stored as a JSON string so PDF and image chunks can carry different
metadata shapes without changing the LanceDB schema.

## Fresh Mac Setup

Install `uv` first if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then bootstrap the project:

```bash
cd /path/to/alphasearch
./scripts/bootstrap.sh
cp .env.example .env
```

Download the embedding model before the offline demo:

```bash
uv run python scripts/download_model.py
```

Edit `.env` so the model path points at the local snapshot:

```bash
ALPHASEARCH_MODEL_PATH=./models/Qwen3-VL-Embedding-2B
```

## Index Files

Build or rebuild the local index:

```bash
uv run python scripts/index_data.py --reset
```

Re-run without `--reset` to skip unchanged files. If a file has changed at the
same relative path, stale rows are removed and the file is re-indexed:

```bash
uv run python scripts/index_data.py
```

For a faster smoke test:

```bash
uv run python scripts/index_data.py --reset --limit 2
```

## Search

Search the local index:

```bash
uv run python scripts/search.py "photos of people at an event"
uv run python scripts/search.py "papers about memory in transformers" -k 5
```

Print raw rows:

```bash
uv run python scripts/search.py "software as content" --json
```

## Offline Demo Mode

After the model has been downloaded and `.env` points at `./models/...`, set:

```bash
export ALPHASEARCH_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Then disconnect the internet and run:

```bash
uv run python scripts/index_data.py --reset
uv run python scripts/search.py "research papers about uncertainty"
```

## Configuration

Environment variables are loaded from `.env`.

```text
ALPHASEARCH_DATA_DIR=./data
ALPHASEARCH_DB_DIR=./var/lancedb
ALPHASEARCH_TABLE=chunks
ALPHASEARCH_MODEL_PATH=Qwen/Qwen3-VL-Embedding-2B
ALPHASEARCH_EMBEDDING_DIM=2048
ALPHASEARCH_BATCH_SIZE=4
ALPHASEARCH_EMBEDDING_INSTRUCTION=Retrieve local files, PDF passages, and images relevant to the user's search query.
```

Lower `ALPHASEARCH_BATCH_SIZE` if the Mac runs out of memory.

## Notes

- PDFs are chunked from extracted text, roughly 3,200 characters per chunk with
  overlap.
- Images are stored as one chunk each. This is intentionally simple for the MVP.
- `absolute_path` is useful for opening a result locally; `relative_path` keeps
  the index understandable after a fresh setup.
- `chunk_b64` stores the original image bytes for image chunks and UTF-8 text for
  PDF chunks. `chunk_text` is kept separately for readable previews and future
  hybrid search.
