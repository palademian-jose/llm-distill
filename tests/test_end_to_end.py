"""End-to-end checks that need trained weights or the Hugging Face cache; skipped otherwise."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_distill.dataset import read_pairs
from llm_distill.metrics import classification_metrics

from .conftest import ROOT, SNAPSHOT

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
BASE_CACHED = (HF_CACHE / "models--cross-encoder--ms-marco-MiniLM-L-6-v2").exists()
STUDENT_DIR = ROOT / "models" / "s1-minilm"
TRAINED = (STUDENT_DIR / "model.onnx").exists()


@pytest.mark.skipif(not TRAINED, reason="run `make train` first")
def test_trained_onnx_student_is_accurate_on_the_test_split():
    from llm_distill.students.onnx_student import OnnxStudent

    student, threshold = OnnxStudent.from_dir(STUDENT_DIR)
    test = read_pairs(SNAPSHOT.parent / "pairs.jsonl", "test")
    proba = student.predict_proba([p.question for p in test], [p.passage for p in test])
    m = classification_metrics([p.label for p in test], [int(x >= threshold) for x in proba])
    assert m["accuracy"] > 0.95
    assert m["f1"] > 0.80


@pytest.mark.skipif(not BASE_CACHED, reason="base model is not in the Hugging Face cache")
def test_base_model_loads_from_cache_and_scores_a_pair():
    from llm_distill.students.cross_encoder import CrossEncoderStudent, load_base

    model, tok = load_base("cross-encoder/ms-marco-MiniLM-L-6-v2")
    student = CrossEncoderStudent(model, tok, max_length=64)
    proba = student.predict_proba(["How long is the canary?"], ["The canary runs ten minutes."])
    assert proba.shape == (1,) and 0.0 <= float(proba[0]) <= 1.0
