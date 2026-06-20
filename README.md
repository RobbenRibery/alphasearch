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
    ingestion/                # scanning, PDF/image chunking, ingest pipeline
    search/                   # query search service and API response models
    api/                      # FastAPI app for /ingest and /search
    cli/                      # unified CLI and compatibility commands
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
uv run alphasearch ingest ./data --reset
```

Re-run without `--reset` to skip unchanged files. If a file has changed at the
same relative path, stale rows are removed and the file is re-indexed:

```bash
uv run alphasearch ingest ./data
```

For a faster smoke test:

```bash
uv run alphasearch ingest ./data --reset --limit 2
```

## LanceDB Index

After indexing `./data`, the local vector store is:

```text
URI:   ./var/lancedb
Table: chunks
```

On this machine, that resolves to:

```text
/Users/ericliu/startups/alphasearch/var/lancedb
```

A full ingest of the bundled `./data` folder produced **600 chunks** from **25 files** in the `chunks` table.

Both the CLI and API read from this LanceDB location via `ALPHASEARCH_DB_DIR` and `ALPHASEARCH_TABLE`.

## Search

Search the local index:

```bash
uv run alphasearch search "photos of people at an event"
uv run alphasearch search "papers about memory in transformers" -k 5
```

Print raw rows:

```bash
uv run alphasearch search "software as content" --json
```

## API

Run the local HTTP API:

```bash
uv run alphasearch serve
```

The API exposes two main endpoints:

- `POST /ingest` with `{"folder": "./data", "reset": false, "limit": null}`
- `POST /search` with `{"query": "research papers about uncertainty", "top_k": 5}`

## Offline Demo Mode

After the model has been downloaded and `.env` points at `./models/...`, set:

```bash
export ALPHASEARCH_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Then disconnect the internet and run:

```bash
uv run alphasearch ingest ./data --reset
uv run alphasearch search "research papers about uncertainty"
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
