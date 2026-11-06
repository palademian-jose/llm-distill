"""Deterministic fake teacher.

For an answerability prompt (see ``llm_distill.prompts``) it answers "yes" when at least
``threshold`` of the distinct content words of the question appear in the passage, and "no"
otherwise. The threshold is fixed, not tuned on any split. Any other prompt gets a short
deterministic acknowledgement. Rows it produces are labelled ``fake-provider`` in every table.
"""

from __future__ import annotations

import time
from collections import deque

from llm_distill.llm.base import LLMResponse, Message, ToolSpec, Usage, approx_tokens
from llm_distill.prompts import parse_prompt
from llm_distill.textutil import content_tokens, overlap_ratio

FAKE_MODEL = "fake-lexical-overlap-v1"
DEFAULT_THRESHOLD = 0.6


def lexical_overlap(question: str, passage: str) -> float:
    return overlap_ratio(content_tokens(question), content_tokens(passage))


class FakeAnswerabilityClient:
    provider = "fake"

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        model: str = FAKE_MODEL,
    ) -> None:
        self._queue: deque[str] = deque(responses or [])
        self.threshold = threshold
        self.model = model
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        start = time.perf_counter()
        self.calls.append(messages)
        prompt_text = "\n".join(m.text() for m in messages)
        if self._queue:
            text = self._queue.popleft()
        else:
            parsed = parse_prompt(prompt_text)
            if parsed is None:
                last_user = next((m.text() for m in reversed(messages) if m.role == "user"), "")
                text = f"[fake] {last_user[:200]}"
            else:
                question, passage = parsed
                text = "yes" if lexical_overlap(question, passage) >= self.threshold else "no"
        return LLMResponse(
            text=text,
            usage=Usage(input_tokens=approx_tokens(prompt_text), output_tokens=1),
            model=self.model,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
