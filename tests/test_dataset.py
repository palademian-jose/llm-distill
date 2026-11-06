from __future__ import annotations

from collections import defaultdict

from llm_distill.dataset import (
    assign_splits,
    build_pairs,
    load_questions,
    read_pairs,
    summarize,
    write_pairs,
)

from .conftest import SNAPSHOT


def test_pair_count_and_kinds(pairs):
    assert 3000 <= len(pairs) <= 6000
    assert {p.kind for p in pairs} == {"positive", "hard_negative", "easy_negative"}
    assert all(p.label == (1 if p.kind == "positive" else 0) for p in pairs)


def test_no_question_leaks_across_splits(pairs):
    splits = defaultdict(set)
    for p in pairs:
        splits[p.question_id].add(p.split)
    assert all(len(s) == 1 for s in splits.values())


def test_split_proportions(pairs):
    n_q = len({p.question_id for p in pairs})
    per = {s: len({p.question_id for p in pairs if p.split == s}) for s in ("train", "val", "test")}
    assert sum(per.values()) == n_q
    assert abs(per["train"] / n_q - 0.70) < 0.03
    assert abs(per["val"] / n_q - 0.15) < 0.03
    assert abs(per["test"] / n_q - 0.15) < 0.03


def test_every_split_has_positives_and_refusals(pairs):
    questions = {q.id: q for q in load_questions(SNAPSHOT / "golden")}
    for split in ("train", "val", "test"):
        sub = [p for p in pairs if p.split == split]
        assert any(p.label for p in sub)
        assert any(questions[p.question_id].refuse for p in sub)


def test_refusal_questions_have_only_negatives(pairs):
    questions = {q.id: q for q in load_questions(SNAPSHOT / "golden")}
    for p in pairs:
        if questions[p.question_id].refuse:
            assert p.label == 0


def test_negatives_never_contain_the_answer(pairs):
    questions = {q.id: q for q in load_questions(SNAPSHOT / "golden")}
    for p in pairs:
        q = questions[p.question_id]
        if p.kind != "positive" and q.answer_phrase:
            assert q.answer_phrase.lower() not in " ".join(p.passage.lower().split())


def test_easy_negatives_come_from_other_documents(pairs):
    questions = {q.id: q for q in load_questions(SNAPSHOT / "golden")}
    for p in pairs:
        q = questions[p.question_id]
        if p.kind == "easy_negative" and q.doc_path:
            assert p.doc_path != q.doc_path


def test_build_is_deterministic():
    a = build_pairs(SNAPSHOT / "corpus", SNAPSHOT / "golden", seed=5)
    b = build_pairs(SNAPSHOT / "corpus", SNAPSHOT / "golden", seed=5)
    assert a == b


def test_assign_splits_changes_with_seed():
    qs = load_questions(SNAPSHOT / "golden")
    assert assign_splits(qs, 1) != assign_splits(qs, 2)


def test_roundtrip_and_summary(tmp_path, pairs):
    path = tmp_path / "pairs.jsonl"
    write_pairs(pairs, path)
    assert read_pairs(path) == pairs
    assert read_pairs(path, "test") == [p for p in pairs if p.split == "test"]
    stats = summarize(pairs)
    assert stats["pairs"] == len(pairs)
    assert sum(v["pairs"] for v in stats["splits"].values()) == len(pairs)


def test_committed_pairs_file_matches_a_fresh_build(pairs):
    committed = read_pairs(SNAPSHOT.parent / "pairs.jsonl")
    assert [p.pair_id for p in committed] == [p.pair_id for p in pairs]
