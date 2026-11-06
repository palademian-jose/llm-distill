"""Offline demo: retrieve with BM25 and let the teacher and the student judge each passage.

    make demo

Uses the fake teacher and whichever student exists (ONNX, else TF-IDF). No network needed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from rank_bm25 import BM25Okapi

from llm_distill.client import StudentClient, load_student
from llm_distill.corpus import chunk_documents, load_corpus
from llm_distill.llm import FakeAnswerabilityClient
from llm_distill.prompts import build_messages
from llm_distill.settings import get_settings
from llm_distill.textutil import content_tokens

QUESTIONS = [
    "How long does the canary receive traffic before it is promoted?",
    "What is the maximum size of a pull request?",
    "What is the capital of France?",
]


def main() -> None:
    s = get_settings()
    if not s.student_dir.exists() and s.student_backend != "tfidf":
        print("No S1 model found; using the TF-IDF student (run `make train` for S1).\n")
        os.environ["STUDENT_BACKEND"] = "tfidf"
        get_settings.cache_clear()
        s = get_settings()
    chunks = chunk_documents(
        load_corpus(s.snapshot_dir / "corpus"), size=s.chunk_size, overlap=s.chunk_overlap
    )
    bm25 = BM25Okapi([content_tokens(c.embed_text) or ["<empty>"] for c in chunks])
    teacher = FakeAnswerabilityClient()
    student_model, threshold = load_student(s)
    student = StudentClient(student_model, threshold)
    print(f"teacher: {teacher.model} (fake-provider)   student: {student.model}\n")

    for question in QUESTIONS:
        scores = np.asarray(bm25.get_scores(content_tokens(question)))
        print(f"Q: {question}")
        for idx in np.argsort(-scores, kind="stable")[:3]:
            c = chunks[int(idx)]
            msgs = build_messages(question, c.embed_text)
            t = teacher.complete(msgs)
            st = student.complete(msgs)
            print(
                f"   {c.doc_path}#L{c.start_line}-{c.end_line:<4} "
                f"teacher={t.text:<3} student={st.text:<3} ({st.latency_ms:.1f} ms)"
            )
        print()


if __name__ == "__main__":
    sys.exit(main())
