"""Configuration for optional partner integrations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _as_bool(value: str | None, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class IntegrationSettings:
    """Feature flags for partner integrations."""

    overmind_enabled: bool
    cosine_enabled: bool
    exo_enabled: bool
    cognee_enabled: bool
    captur_enabled: bool
    overmind_trace_dir: str
    cognee_session_id: str
    exo_cluster_name: str


def load_integration_settings(root_dir: Path | None = None) -> IntegrationSettings:
    """Load integration settings from ``.env`` and the process environment.

    Args:
        root_dir: Optional project root used to locate ``.env``.

    Returns:
        Parsed integration feature flags and stub connection metadata.
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parents[2]
    load_dotenv(Path(root_dir) / ".env")

    return IntegrationSettings(
        overmind_enabled=_as_bool(os.getenv("ALPHASEARCH_OVERMIND_ENABLED"), True),
        cosine_enabled=_as_bool(os.getenv("ALPHASEARCH_COSINE_ENABLED"), True),
        exo_enabled=_as_bool(os.getenv("ALPHASEARCH_EXO_ENABLED"), True),
        cognee_enabled=_as_bool(os.getenv("ALPHASEARCH_COGNEE_ENABLED"), True),
        captur_enabled=_as_bool(os.getenv("ALPHASEARCH_CAPTUR_ENABLED"), True),
        overmind_trace_dir=os.getenv("ALPHASEARCH_OVERMIND_TRACE_DIR", "./var/overmind"),
        cognee_session_id=os.getenv("ALPHASEARCH_COGNEE_SESSION_ID", "alphasearch-local"),
        exo_cluster_name=os.getenv("ALPHASEARCH_EXO_CLUSTER", "alphasearch-local-mesh"),
    )
