"""StudentClient: a drop-in ``LLMClient`` whose "model" is a distilled answerability classifier.

It reads the same prompt the teacher gets (``llm_distill.prompts.build_messages``) and replies
with the text ``yes`` or ``no``, so any code that asks an LLM "does this passage answer the
question?" can switch to the student with ``LLM_PROVIDER=student``.
"""

from __future__ import annotations

import re
import time

from llm_distill.llm.base import LLMResponse, Message, ToolSpec, Usage, approx_tokens
from llm_distill.prompts import parse_prompt
from llm_distill.settings import Settings, get_settings
from llm_distill.students.base import Student

_LOOSE_RE = re.compile(
    r"question\s*:\s*(?P<q>.+?)\s*\n+\s*(?:passage|context|chunk|document)\s*:\s*(?P<p>.+)",
    re.IGNORECASE | re.DOTALL,
)


def extract_pair(messages: list[Message]) -> tuple[str, str]:
    """Find (question, passage) in a tagged prompt or a ``Question: ... Passage: ...`` one."""
    text = "\n".join(m.text() for m in messages if m.role != "system")
    parsed = parse_prompt(text)
    if parsed:
        return parsed
    m = _LOOSE_RE.search(text)
    if m:
        return m.group("q").strip(), m.group("p").strip()
    raise ValueError(
        "StudentClient needs a prompt with <question>...</question> and <passage>...</passage> "
        "tags (see llm_distill.prompts.build_messages)"
    )


class StudentClient:
    provider = "student"

    def __init__(self, student: Student, threshold: float = 0.5, *, model: str | None = None):
        self.student = student
        self.threshold = threshold
        self.model = model or student.name

    def score(self, question: str, passage: str) -> float:
        return float(self.student.predict_proba([question], [passage])[0])

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
        question, passage = extract_pair(messages)
        proba = self.score(question, passage)
        text = "yes" if proba >= self.threshold else "no"
        return LLMResponse(
            text=text,
            usage=Usage(input_tokens=approx_tokens(question + passage), output_tokens=1),
            model=self.model,
            latency_ms=(time.perf_counter() - start) * 1000,
            stop_reason="end_turn",
            cost_usd=0.0,
        )


def load_student(settings: Settings | None = None) -> tuple[Student, float]:
    """Load the student named by ``STUDENT_BACKEND`` (onnx, torch or tfidf) and its threshold."""
    s = settings or get_settings()
    if s.student_backend == "tfidf":
        from llm_distill.students.tfidf import TfidfStudent

        return TfidfStudent.load(s.tfidf_path)
    if s.student_backend == "torch":
        from llm_distill.students.cross_encoder import CrossEncoderStudent

        return CrossEncoderStudent.from_dir(s.student_dir)
    from llm_distill.students.onnx_student import OnnxStudent

    return OnnxStudent.from_dir(s.student_dir, s.onnx_path)


def student_client(settings: Settings | None = None) -> StudentClient:
    student, threshold = load_student(settings)
    return StudentClient(student, threshold)
