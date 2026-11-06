"""The answerability prompt shared by the teacher, the fake teacher and the StudentClient."""

from __future__ import annotations

import re

from llm_distill.llm.base import Message

SYSTEM_PROMPT = (
    "You check retrieval results for a question-answering system. Given a question and one "
    "passage, decide whether the passage contains the information needed to answer the "
    "question. Reply with exactly one word: yes or no."
)

_QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL)
_PASSAGE_RE = re.compile(r"<passage>\s*(.*?)\s*</passage>", re.DOTALL)


def build_messages(question: str, passage: str) -> list[Message]:
    user = (
        f"<question>\n{question}\n</question>\n\n"
        f"<passage>\n{passage}\n</passage>\n\n"
        "Does the passage contain the information needed to answer the question? "
        "Answer yes or no."
    )
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=user),
    ]


def parse_prompt(text: str) -> tuple[str, str] | None:
    """Recover (question, passage) from a prompt built by ``build_messages``."""
    q = _QUESTION_RE.search(text)
    p = _PASSAGE_RE.search(text)
    if not q or not p:
        return None
    return q.group(1), p.group(1)


def parse_verdict(text: str) -> bool | None:
    """Map a model reply to True (yes), False (no) or None when it is neither."""
    head = re.sub(r"[^a-z ]", " ", text.lower()).split()
    if not head:
        return None
    first = head[0]
    if first in {"yes", "y", "true"}:
        return True
    if first in {"no", "n", "false"}:
        return False
    return None
