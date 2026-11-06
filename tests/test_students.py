"""Student tests. The transformer is a tiny randomly initialised BERT with a vocabulary written
to a temporary directory, so nothing here needs the network or the Hugging Face cache."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from llm_distill.client import StudentClient, extract_pair
from llm_distill.compare import (
    ModelResult,
    bench_latency,
    bootstrap_ci,
    evaluate_student,
    render_tables,
    summarize_model,
)
from llm_distill.dataset import Pair
from llm_distill.metrics import classification_metrics
from llm_distill.prompts import build_messages
from llm_distill.students.cross_encoder import (
    CrossEncoderStudent,
    TrainConfig,
    save_student,
    train_cross_encoder,
)
from llm_distill.students.onnx_student import OnnxStudent, export_onnx
from llm_distill.students.tfidf import TfidfStudent

GOOD = [
    ("How long does the canary run?", "The canary runs for ten minutes before promotion."),
    ("Who approves a release?", "The release manager approves each release."),
    ("When do passwords rotate?", "Passwords rotate every ninety days."),
    ("How big can a pull request be?", "A pull request stays under four hundred lines."),
]
BAD = [
    ("How long does the canary run?", "Passwords rotate every ninety days."),
    ("Who approves a release?", "A pull request stays under four hundred lines."),
    ("When do passwords rotate?", "The canary runs for ten minutes before promotion."),
    ("How big can a pull request be?", "The release manager approves each release."),
]


def _xy(reps: int = 6):
    qs = [q for q, _ in (GOOD + BAD) * reps]
    ps = [p for _, p in (GOOD + BAD) * reps]
    ys = ([1] * len(GOOD) + [0] * len(BAD)) * reps
    return qs, ps, ys


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    words = sorted({w.strip("?.").lower() for q, p in GOOD + BAD for w in (q + " " + p).split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    d = tmp_path_factory.mktemp("tiny")
    (d / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(d / "vocab.txt"), do_lower_case=True)
    torch.manual_seed(0)
    cfg = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=2,
    )
    return BertForSequenceClassification(cfg), tok


# -- S2 -----------------------------------------------------------------------------------------
def test_tfidf_separates_matching_from_mismatched_passages(tmp_path):
    qs, ps, ys = _xy()
    s = TfidfStudent().fit(qs, ps, ys)
    proba = s.predict_proba(qs, ps)
    assert proba.shape == (len(ys),)
    assert np.mean((proba >= 0.5) == np.asarray(ys)) > 0.95
    path = tmp_path / "s2.joblib"
    s.save(path, 0.42)
    loaded, thr = TfidfStudent.load(path)
    assert thr == pytest.approx(0.42)
    np.testing.assert_allclose(loaded.predict_proba(qs, ps), proba)


# -- S1 -----------------------------------------------------------------------------------------
def test_cross_encoder_trains_saves_loads_and_exports(tiny, tmp_path):
    model, tok = tiny
    qs, ps, ys = _xy(reps=8)
    cfg = TrainConfig(epochs=8, batch_size=8, lr=2e-3, max_length=48, seed=3)
    report = train_cross_encoder(
        model,
        tok,
        (qs, ps, ys),
        (qs[:16], ps[:16], ys[:16]),
        cfg,
        device=torch.device("cpu"),
        log=lambda *_: None,
    )
    assert report["cuda_used"] is False
    assert report["best_epoch"] >= 1
    assert report["val"]["f1"] > 0.6

    out = tmp_path / "s1"
    save_student(model, tok, out, report)
    student, thr = CrossEncoderStudent.from_dir(out, device=torch.device("cpu"))
    p_torch = student.predict_proba(qs[:8], ps[:8])
    assert p_torch.shape == (8,)
    assert thr == pytest.approx(report["threshold"])

    onnx_path = export_onnx(student.model, student.tokenizer, out / "model.onnx", max_length=48)
    ox, thr_ox = OnnxStudent.from_dir(out, onnx_path)
    p_onnx = ox.predict_proba(qs[:8], ps[:8])
    np.testing.assert_allclose(p_torch, p_onnx, atol=1e-4)
    assert thr_ox == pytest.approx(thr)
    # Batch size must not change the answer.
    np.testing.assert_allclose(ox.predict_proba(qs[:3], ps[:3]), p_onnx[:3], atol=1e-4)

    client = StudentClient(ox, thr)
    reply = client.complete(build_messages(*GOOD[0]))
    assert reply.text in {"yes", "no"}
    assert reply.cost_usd == 0.0 and reply.model == "s1-minilm-onnx"


def test_training_is_reproducible_for_a_fixed_seed(tiny):
    _, tok = tiny
    qs, ps, ys = _xy(reps=2)

    def run() -> float:
        torch.manual_seed(0)
        cfg = BertConfig(
            vocab_size=tok.vocab_size,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
            num_labels=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        m = BertForSequenceClassification(cfg)
        train_cross_encoder(
            m,
            tok,
            (qs, ps, ys),
            (qs, ps, ys),
            TrainConfig(epochs=2, batch_size=4, lr=1e-3, max_length=48, seed=5),
            device=torch.device("cpu"),
            log=lambda *_: None,
        )
        return float(
            CrossEncoderStudent(m, tok, max_length=48, device=torch.device("cpu")).predict_proba(
                qs[:1], ps[:1]
            )[0]
        )

    assert run() == pytest.approx(run(), abs=1e-6)


# -- StudentClient -------------------------------------------------------------------------------
class _Const:
    name = "const"

    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, questions, passages):
        return np.full(len(questions), self.p)


def test_student_client_yes_no_and_loose_prompt_format():
    msgs = build_messages("Why?", "Because.")
    assert StudentClient(_Const(0.9), 0.5).complete(msgs).text == "yes"
    assert StudentClient(_Const(0.1), 0.5).complete(msgs).text == "no"
    from llm_distill.llm.base import LLMClient, Message

    assert isinstance(StudentClient(_Const(0.9)), LLMClient)
    loose = [Message(role="user", content="Question: Why?\n\nPassage: Because.")]
    assert extract_pair(loose) == ("Why?", "Because.")
    with pytest.raises(ValueError):
        extract_pair([Message(role="user", content="hello")])


def test_factory_serves_the_student_with_one_env_var(monkeypatch, tmp_path):
    from llm_distill.llm import get_client
    from llm_distill.settings import get_settings

    qs, ps, ys = _xy()
    TfidfStudent().fit(qs, ps, ys).save(tmp_path / "s2.joblib", 0.5)
    monkeypatch.setenv("LLM_PROVIDER", "student")
    monkeypatch.setenv("STUDENT_BACKEND", "tfidf")
    monkeypatch.setenv("TFIDF_PATH", str(tmp_path / "s2.joblib"))
    get_settings.cache_clear()
    client = get_client()
    assert client.provider == "student"
    assert client.complete(build_messages(*GOOD[0])).text == "yes"
    assert client.complete(build_messages(*BAD[0])).text == "no"


# -- compare helpers ------------------------------------------------------------------------------
def _pairs(n_q: int = 10):
    out = []
    for i in range(n_q):
        for j, (kind, label) in enumerate(
            [("positive", 1), ("hard_negative", 0), ("easy_negative", 0)]
        ):
            out.append(
                Pair(
                    pair_id=f"q{i}::{j}",
                    question_id=f"q{i}",
                    question="q",
                    chunk_id=str(j),
                    doc_path="d",
                    passage="p",
                    label=label,
                    kind=kind,
                    split="test",
                )
            )
    return out


def test_bench_latency_counts_calls_and_scales_per_decision():
    calls: list[int] = []

    def fn(q, p):
        calls.append(len(q))
        return np.zeros(len(q))

    qs = ["q"] * 70
    out = bench_latency(fn, qs, qs, batch=32, warmup=1, repeats=2)
    assert len(out) == 4  # two full batches of 32 per repeat; the remainder is dropped
    assert calls[0] == 32  # warm-up call
    assert len(bench_latency(fn, qs, qs, batch=1, warmup=0)) == 70


def test_bootstrap_ci_brackets_the_point_estimate():
    test = _pairs()
    truth = [p.label for p in test]
    pred = list(truth)
    pred[1] = 1  # one false positive
    ci = bootstrap_ci(truth, pred, [p.question_id for p in test], n_boot=200)
    point = classification_metrics(truth, pred)
    assert ci["accuracy"][0] <= point["accuracy"] <= ci["accuracy"][1]
    assert 0 <= ci["f1"][0] <= ci["f1"][1] <= 1


def test_evaluate_student_and_render_tables_label_provenance():
    test = _pairs()
    res = evaluate_student(
        _Const(1.0), test, 0.5, usd_per_hour=0.10, size_mb=1.0, key="const", repeats_b32=1
    )
    assert res.cost_per_1k_b1 is not None and res.cost_per_1k_b1 >= 0
    teacher = ModelResult(
        key="teacher",
        name="teacher (x)",
        label="fake-provider",
        pred=[p.label for p in test],
        size_mb=None,
        latency_b1_ms=[0.01],
        cost_per_1k_b1=0.5,
        cost_basis="tokens x list price (projected)",
        is_teacher=True,
    )
    models = [summarize_model(r, test, teacher.pred, n_boot=50) for r in (teacher, res)]
    md = render_tables({"models": models})
    assert md.count("fake-provider") >= 3  # quality, latency and cost tables
    assert "trained on derived labels" in md
