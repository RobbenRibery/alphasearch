#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

step() {
  printf '\n%s%s%s\n' "$BOLD" "$1" "$RESET"
}

ok() {
  printf '%b✓ %s%b\n' "$GREEN" "$1" "$RESET"
}

warn() {
  printf '%b! %s%b\n' "$YELLOW" "$1" "$RESET"
}

fail() {
  printf '%b✗ %s%b\n' "$RED" "$1" "$RESET"
}

prompt_yes_no() {
  local question="$1"
  local default="${2:-n}"
  local hint="y/N"
  if [[ "$default" == "y" ]]; then
    hint="Y/n"
  fi

  while true; do
    read -r -p "$question [$hint] " reply
    reply="${reply:-$default}"
    case "$reply" in
      [Yy]|[Yy][Ee][Ss]) return 0 ;;
      [Nn]|[Nn][Oo]) return 1 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
}

read_lancedb_status() {
  uv run python - <<'PY'
from alphasearch.config import load_qwen_settings
from alphasearch.db import LanceDBStore

settings = load_qwen_settings()
db_dir = settings.db_dir
table_name = settings.table_name
db_exists = db_dir.exists()
table_exists = False
row_count = 0

if db_exists:
    store = LanceDBStore(settings.db_dir, settings.table_name, settings.embedding_dim)
    table_exists = table_name in set(store.db.table_names())
    if table_exists:
        row_count = store.row_count()

print(f"db_dir={db_dir}")
print(f"table_name={table_name}")
print(f"db_exists={db_exists}")
print(f"table_exists={table_exists}")
print(f"row_count={row_count}")
PY
}

verify_mcp_config() {
  uv run python - <<'PY'
import asyncio
import json
from pathlib import Path

from alphasearch.mcp.server import mcp

config_path = Path(".cursor/mcp.json")
config = json.loads(config_path.read_text())
server_names = sorted(config.get("mcpServers", {}))
if "alphasearch-qwen" not in server_names:
    raise SystemExit("alphasearch-qwen missing from .cursor/mcp.json")

tool_names = sorted(tool.name for tool in asyncio.run(mcp.list_tools()))
print(f"config_path={config_path.resolve()}")
print(f"server_names={','.join(server_names)}")
print(f"tool_names={','.join(tool_names)}")
PY
}

step "AlphaSearch interactive setup"
echo "This wizard checks your LanceDB index and wires up the Cursor MCP server."

step "1/4  Prerequisites"
if ! command -v uv >/dev/null 2>&1; then
  fail "uv is required. Install it from https://docs.astral.sh/uv/ and re-run: make setup"
  exit 1
fi
ok "uv is installed"

mkdir -p var/lancedb var/cache models
uv sync >/dev/null
ok "Python dependencies are synced"

if [[ ! -f .env ]]; then
  warn ".env not found"
  if prompt_yes_no "Create .env from .env.example?" "y"; then
    cp .env.example .env
    ok "Created .env from .env.example"
  else
    warn "Continuing without .env (defaults will be used)"
  fi
else
  ok ".env exists"
fi

step "2/4  Verify LanceDB"
eval "$(read_lancedb_status)"

echo "  Directory: $db_dir"
echo "  Table:     $table_name"

if [[ "$db_exists" != "True" ]]; then
  warn "LanceDB directory does not exist yet"
  mkdir -p "$db_dir"
  ok "Created $db_dir"
fi

if [[ "$table_exists" != "True" ]]; then
  warn "Table '$table_name' is missing"
  if prompt_yes_no "Create an empty LanceDB table now?" "y"; then
    uv run python - <<'PY'
from alphasearch.config import load_qwen_settings
from alphasearch.db import LanceDBStore

settings = load_qwen_settings()
LanceDBStore(settings.db_dir, settings.table_name, settings.embedding_dim)
print("created")
PY
    ok "Created table '$table_name'"
    eval "$(read_lancedb_status)"
  fi
fi

if [[ "${row_count:-0}" -eq 0 ]]; then
  warn "LanceDB table '$table_name' has 0 indexed chunks"
  if prompt_yes_no "Index ./data now? (downloads/uses the embedding model)" "n"; then
    uv run alphasearch ingest ./data --reset
    eval "$(read_lancedb_status)"
  fi
else
  ok "LanceDB table '$table_name' contains $row_count chunks"
fi

if [[ "${row_count:-0}" -eq 0 ]]; then
  warn "Search and MCP tools will work only after you ingest data:"
  echo "  uv run alphasearch ingest ./data --reset"
fi

step "3/4  Verify MCP server"
if [[ ! -f .cursor/mcp.json ]]; then
  fail ".cursor/mcp.json is missing"
  echo "Expected a project MCP config at .cursor/mcp.json"
  exit 1
fi
ok "Found .cursor/mcp.json"

if verify_mcp_config; then
  ok "MCP config is valid and exposes qwen_search tools"
else
  fail "MCP config check failed"
  exit 1
fi

step "4/4  Hook up Cursor MCP"
echo "Cursor reads MCP servers from .cursor/mcp.json in this workspace."
echo
echo "Enable the server in Cursor:"
echo "  1. Open Cursor Settings → MCP"
echo "  2. Confirm ${BOLD}alphasearch-qwen${RESET} appears and is enabled"
echo "  3. Reload the window if tools do not show up (Cmd+Shift+P → Developer: Reload Window)"
echo
echo "Quick test from a Cursor chat:"
echo "  Ask the agent to call qwen_search_index_status or search with qwen_search."
echo

if prompt_yes_no "Run a sample MCP search now?" "n"; then
  read -r -p "Search query: " sample_query
  if [[ -n "$sample_query" ]]; then
    SAMPLE_QUERY="$sample_query" uv run python - <<'PY'
import os

from alphasearch.mcp.server import qwen_search

query = os.environ["SAMPLE_QUERY"]
results = qwen_search(query, top_k=3)["results"]
if not results:
    print("No results returned.")
else:
    for index, item in enumerate(results, start=1):
        print(f"{index}. {item['filename']}  score={item['score']:.3f}")
        print(f"   {item['absolute_path']}")
PY
  fi
fi

step "Setup complete"
ok "LanceDB verified at $db_dir (table=$table_name, rows=${row_count:-0})"
ok "Cursor MCP config ready at .cursor/mcp.json"
echo
echo "Useful commands:"
echo "  make setup          # re-run this wizard"
echo "  uv run alphasearch search \"your query\""
echo "  uv run alphasearch-mcp"
