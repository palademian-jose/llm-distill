"""Build labelled (question, passage) -> answerable pairs from the docs-rag snapshot.

Positives   the golden question paired with its expected chunk(s).
Hard neg.   the same question paired with the highest-BM25 chunks that are not expected and do
            not contain the answer line (so a hard negative is never secretly a positive).
Easy neg.   random chunks from other documents.
Refusal questions (nothing in the corpus answers them) contribute negatives only.

Splits are 70/15/15 by question id, stratified by question group, so no question appears in
two splits.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from llm_distill.corpus import Chunk, chunk_documents, load_corpus
from llm_distill.textutil import content_tokens

Kind = Literal["positive", "hard_negative", "easy_negative"]
Split = Literal["train", "val", "test"]
SPLIT_FRACTIONS: tuple[float, float, float] = (0.70, 0.15, 0.15)


class Question(BaseModel):
    id: str
    text: str
    group: Literal["synthetic", "handwritten", "refusal"]
    refuse: bool
    doc_path: str | None = None
    chunk_ids: list[str] = []
    answer_line: int | None = None
    answer_phrase: str | None = None


class Pair(BaseModel):
    pair_id: str
    question_id: str
    question: str
    chunk_id: str
    doc_path: str
    passage: str
    label: int  # 1 = the passage answers the question
    kind: Kind
    split: Split


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_questions(golden_dir: Path) -> list[Question]:
    out: list[Question] = []
    for path in sorted(Path(golden_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            exp = row.get("expected", {})
            refuse = bool(exp.get("refuse"))
            if refuse:
                group = "refusal"
            elif row.get("source") == "human":
                group = "handwritten"
            else:
                group = "synthetic"
            out.append(
                Question(
                    id=row["id"],
                    text=row["input"]["question"],
                    group=group,
                    refuse=refuse,
                    doc_path=exp.get("doc_path"),
                    chunk_ids=list(exp.get("chunk_ids") or []),
                    answer_line=exp.get("answer_line"),
                    answer_phrase=exp.get("answer_phrase"),
                )
            )
    return out


def assign_splits(questions: list[Question], seed: int) -> dict[str, Split]:
    """Question id -> split, stratified by group (each group is shuffled and cut 70/15/15)."""
    rng = random.Random(seed)
    result: dict[str, Split] = {}
    for group in ("synthetic", "handwritten", "refusal"):
        ids = sorted(q.id for q in questions if q.group == group)
        rng.shuffle(ids)
        n = len(ids)
        n_train = round(n * SPLIT_FRACTIONS[0])
        n_val = round(n * SPLIT_FRACTIONS[1])
        for i, qid in enumerate(ids):
            result[qid] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return result


def _positive_chunks(q: Question, chunks: list[Chunk]) -> list[Chunk]:
    if q.refuse:
        return []
    by_id = {c.chunk_id: c for c in chunks}
    if q.chunk_ids:
        return [by_id[cid] for cid in q.chunk_ids if cid in by_id]
    if q.doc_path and q.answer_phrase:
        phrase = _norm(q.answer_phrase)
        found = [c for c in chunks if c.doc_path == q.doc_path and phrase in _norm(c.text)]
        return found[:1]
    return []


def _covers_answer(q: Question, chunk: Chunk) -> bool:
    """True when the chunk contains the answer, so it must not be used as a negative."""
    if q.refuse or chunk.doc_path != q.doc_path:
        return False
    if q.answer_line is not None and chunk.start_line <= q.answer_line <= chunk.end_line:
        return True
    return bool(q.answer_phrase) and _norm(q.answer_phrase or "") in _norm(chunk.text)


def build_pairs(
    corpus_dir: Path,
    golden_dir: Path,
    *,
    chunk_size: int = 128,
    chunk_overlap: int = 16,
    n_hard: int = 8,
    n_easy: int = 6,
    seed: int = 13,
) -> list[Pair]:
    chunks = chunk_documents(load_corpus(corpus_dir), size=chunk_size, overlap=chunk_overlap)
    questions = load_questions(golden_dir)
    splits = assign_splits(questions, seed)
    bm25 = BM25Okapi([content_tokens(c.embed_text) or ["<empty>"] for c in chunks])
    rng = random.Random(seed)
    pairs: list[Pair] = []

    for q in sorted(questions, key=lambda x: x.id):
        positives = _positive_chunks(q, chunks)
        pos_ids = {c.chunk_id for c in positives}
        pos_docs = {c.doc_path for c in positives}
        scores = np.asarray(bm25.get_scores(content_tokens(q.text)), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        hard: list[Chunk] = []
        for idx in order:
            c = chunks[int(idx)]
            if c.chunk_id in pos_ids or _covers_answer(q, c):
                continue
            hard.append(c)
            if len(hard) >= n_hard:
                break
        used = pos_ids | {c.chunk_id for c in hard}
        pool = [
            c
            for c in chunks
            if c.chunk_id not in used and not _covers_answer(q, c) and c.doc_path not in pos_docs
        ]
        easy = rng.sample(pool, min(n_easy, len(pool)))
        for kind, group in (
            ("positive", positives),
            ("hard_negative", hard),
            ("easy_negative", easy),
        ):
            for c in group:
                pairs.append(
                    Pair(
                        pair_id=f"{q.id}::{c.chunk_id}",
                        question_id=q.id,
                        question=q.text,
                        chunk_id=c.chunk_id,
                        doc_path=c.doc_path,
                        passage=c.embed_text,
                        label=1 if kind == "positive" else 0,
                        kind=kind,  # type: ignore[arg-type]
                        split=splits[q.id],
                    )
                )
    return pairs


def write_pairs(pairs: list[Pair], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for p in pairs:
            fh.write(p.model_dump_json() + "\n")


def read_pairs(path: Path, split: Split | None = None) -> list[Pair]:
    rows = [
        Pair.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [p for p in rows if split is None or p.split == split]


def summarize(pairs: list[Pair]) -> dict:
    by_split: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        sub = [p for p in pairs if p.split == split]
        by_split[split] = {
            "pairs": len(sub),
            "questions": len({p.question_id for p in sub}),
            "positives": sum(p.label for p in sub),
            "kinds": dict(Counter(p.kind for p in sub)),
        }
    return {
        "pairs": len(pairs),
        "questions": len({p.question_id for p in pairs}),
        "positives": sum(p.label for p in pairs),
        "splits": by_split,
    }
