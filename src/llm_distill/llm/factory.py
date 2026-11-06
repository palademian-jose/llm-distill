"""Provider factory.

Explicit LLM_PROVIDER wins; otherwise the first backend with a key is used, else fake.
"""

from __future__ import annotations

import logging

from llm_distill.llm.base import LLMClient
from llm_distill.llm.fake import FakeAnswerabilityClient
from llm_distill.settings import Settings, get_settings

log = logging.getLogger(__name__)


def get_client(settings: Settings | None = None, *, role: str = "cheap") -> LLMClient:
    """Build the LLM client for ``role`` ("generator", "judge" or "cheap")."""
    settings = settings or get_settings()
    provider = settings.resolved_llm_provider()
    model = {
        "generator": settings.llm_model,
        "judge": settings.judge_model,
        "cheap": settings.cheap_model,
    }.get(role, settings.cheap_model)

    if provider == "student":
        from llm_distill.client import student_client

        return student_client(settings)
    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise RuntimeError("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is not set")
        from llm_distill.llm.openai_compat import OpenAICompatClient

        return OpenAICompatClient(
            settings.openrouter_api_key,
            provider="openrouter",
            default_model=model,
            base_url=settings.openrouter_base_url,
            site_url=settings.openrouter_site_url,
            app_name=settings.openrouter_app_name,
        )
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        from llm_distill.llm.anthropic_client import AnthropicClient

        return AnthropicClient(settings.anthropic_api_key, default_model=model)
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        from llm_distill.llm.openai_compat import OpenAICompatClient

        return OpenAICompatClient(settings.openai_api_key, provider="openai", default_model=model)
    if settings.llm_provider is None:
        log.warning("No LLM API key found; using the deterministic fake provider.")
    return FakeAnswerabilityClient()
