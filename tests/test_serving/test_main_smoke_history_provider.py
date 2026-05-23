from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


def test_smoke_history_provider_env_helper_is_off_by_default(monkeypatch) -> None:
    from serving import main

    monkeypatch.delenv("DDI_SMOKE_HISTORY_PROVIDER", raising=False)

    assert main._dl_history_provider_from_env() is None


def test_smoke_history_provider_env_helper_builds_provider(monkeypatch, caplog) -> None:
    from scripts.ops.smoke_history_provider import SmokeHistoryProvider
    from serving import main

    monkeypatch.setenv("DDI_SMOKE_HISTORY_PROVIDER", "1")

    provider = main._dl_history_provider_from_env()

    assert isinstance(provider, SmokeHistoryProvider)
    assert "DDI_SMOKE_HISTORY_PROVIDER" in caplog.text


def test_lifespan_passes_smoke_history_provider_to_predictor(monkeypatch) -> None:
    from scripts.ops.smoke_history_provider import SmokeHistoryProvider
    from serving import main

    captured: dict = {}

    def fake_init_predictor(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setenv("DDI_SMOKE_HISTORY_PROVIDER", "true")
    monkeypatch.setattr(main, "init_predictor", fake_init_predictor)
    monkeypatch.setattr(main, "init_metrics_writer", lambda **kwargs: None)

    async def run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    asyncio.run(run_lifespan())

    assert isinstance(captured["dl_history_provider"], SmokeHistoryProvider)
