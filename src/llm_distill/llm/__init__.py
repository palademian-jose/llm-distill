"""LLM provider abstraction (fake, openrouter, anthropic, openai)."""

from __future__ import annotations

from typing import Any

from llm_distill.llm.base import LLMClient, LLMResponse, Message, ToolCall, ToolSpec, Usage

__all__ = [
    "FakeAnswerabilityClient",
    "LLMClient",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "get_client",
]


def __getattr__(name: str) -> Any:
    if name == "get_client":
        from llm_distill.llm.factory import get_client

        return get_client
    if name == "FakeAnswerabilityClient":
        from llm_distill.llm.fake import FakeAnswerabilityClient

        return FakeAnswerabilityClient
    raise AttributeError(name)
