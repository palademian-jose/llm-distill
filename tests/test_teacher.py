from __future__ import annotations

from llm_distill.client import extract_pair
from llm_distill.dataset import Pair
from llm_distill.llm import FakeAnswerabilityClient, get_client
from llm_distill.llm.base import LLMClient
from llm_distill.prompts import build_messages, parse_prompt, parse_verdict
from llm_distill.teacher import FAKE_LABEL, label_pairs, read_labels, write_labels


def _pair(i: int, q: str, passage: str, label: int) -> Pair:
    return Pair(
        pair_id=f"q{i}::c{i}",
        question_id=f"q{i}",
        question=q,
        chunk_id=f"c{i}",
        doc_path="d.md",
        passage=passage,
        label=label,
        kind="positive" if label else "hard_negative",
        split="test",
    )


Q = "How long does the canary receive traffic?"
PAIRS = [
    _pair(1, Q, "The canary receives traffic for ten minutes.", 1),
    _pair(2, Q, "Passwords rotate every ninety days.", 0),
]


def test_get_client_defaults_to_fake_without_keys():
    client = get_client()
    assert client.provider == "fake"
    assert isinstance(client, LLMClient)


def test_fake_teacher_is_a_deterministic_lexical_heuristic():
    client = FakeAnswerabilityClient()
    yes = build_messages(PAIRS[0].question, PAIRS[0].passage)
    no = build_messages(PAIRS[1].question, PAIRS[1].passage)
    assert client.complete(yes).text == "yes"
    assert client.complete(no).text == "no"
    assert client.complete(yes).text == client.complete(yes).text


def test_prompt_roundtrip_and_verdict_parsing():
    msgs = build_messages("Why?", "Because.")
    assert parse_prompt("\n".join(m.text() for m in msgs)) == ("Why?", "Because.")
    assert extract_pair(msgs) == ("Why?", "Because.")
    assert parse_verdict("Yes.") is True
    assert parse_verdict("  NO, it does not") is False
    assert parse_verdict("maybe") is None
    assert parse_verdict("") is None


def test_label_pairs_tags_fake_provider_and_flags_unparseable(tmp_path):
    labels = label_pairs(FakeAnswerabilityClient(), PAIRS)
    assert [lab.teacher for lab in labels] == [1, 0]
    assert all(lab.provider_label == FAKE_LABEL for lab in labels)
    bad = label_pairs(FakeAnswerabilityClient(responses=["perhaps", "yes"]), PAIRS)
    assert [lab.teacher for lab in bad] == [None, 1]
    path = tmp_path / "labels.jsonl"
    write_labels(labels, path)
    assert read_labels(path) == labels
    assert read_labels(path, "val") == []


def test_threaded_labelling_keeps_order():
    labels = label_pairs(FakeAnswerabilityClient(), PAIRS * 5, workers=4)
    assert [lab.pair_id for lab in labels] == [p.pair_id for p in PAIRS * 5]
