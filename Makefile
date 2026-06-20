.PHONY: setup bootstrap verify-index ingest smoke-test

setup:
	@./scripts/setup_interactive.sh

bootstrap:
	@./scripts/bootstrap.sh

verify-index:
	@uv run python -c "from alphasearch.mcp.server import qwen_search_index_status; import json; print(json.dumps(qwen_search_index_status(), indent=2))"

ingest:
	uv run alphasearch ingest ./data --reset

smoke-test:
	uv run alphasearch ingest ./data --reset --limit 2
	uv run alphasearch search "test query" -k 3
