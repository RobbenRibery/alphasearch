"""Central registry for partner integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphasearch.integrations.captur import CapturClient
from alphasearch.integrations.cognee import CogneeClient
from alphasearch.integrations.config import IntegrationSettings, load_integration_settings
from alphasearch.integrations.cosine import CosineClient
from alphasearch.integrations.exo import ExoClient
from alphasearch.integrations.overmind import OvermindClient

_hub: IntegrationHub | None = None


@dataclass(frozen=True)
class IntegrationHub:
    """Bundle of partner integration clients used by the pipeline."""

    settings: IntegrationSettings
    overmind: OvermindClient
    cosine: CosineClient
    exo: ExoClient
    cognee: CogneeClient
    captur: CapturClient

    def status(self) -> dict[str, Any]:
        """Return health metadata for all configured integrations."""
        integrations = [
            self.overmind.status(),
            self.cosine.status(),
            self.exo.status(),
            self.cognee.status(),
            self.captur.status(),
        ]
        return {
            "integrations": integrations,
            "active_count": sum(1 for item in integrations if item["connected"]),
        }


def get_integration_hub() -> IntegrationHub:
    """Return a lazily initialized integration hub."""
    global _hub
    if _hub is None:
        _hub = create_integration_hub()
    return _hub


def reset_integration_hub() -> None:
    """Clear the cached integration hub."""
    global _hub
    _hub = None


def create_integration_hub(
    settings: IntegrationSettings | None = None,
) -> IntegrationHub:
    """Build integration clients from settings.

    Args:
        settings: Optional integration settings. Defaults to environment values.

    Returns:
        Initialized integration hub.
    """
    resolved = load_integration_settings() if settings is None else settings
    return IntegrationHub(
        settings=resolved,
        overmind=OvermindClient(
            enabled=resolved.overmind_enabled,
            trace_dir=resolved.overmind_trace_dir,
        ),
        cosine=CosineClient(enabled=resolved.cosine_enabled),
        exo=ExoClient(
            enabled=resolved.exo_enabled,
            cluster_name=resolved.exo_cluster_name,
        ),
        cognee=CogneeClient(
            enabled=resolved.cognee_enabled,
            session_id=resolved.cognee_session_id,
        ),
        captur=CapturClient(enabled=resolved.captur_enabled),
    )
