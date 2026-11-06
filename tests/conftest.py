from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "docs-rag-snapshot"


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call a real provider or read a developer's .env."""
    for var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from llm_distill.settings import get_settings

    get_settings.cache_clear()


@pytest.fixture(scope="session")
def pairs():
    from llm_distill.dataset import build_pairs

    return build_pairs(SNAPSHOT / "corpus", SNAPSHOT / "golden")
