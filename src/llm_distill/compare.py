"""Comparison of the teacher and the students on the held-out test split.

Everything here is computed from model outputs and wall-clock measurements; nothing is typed
in by hand. ``scripts/compare.py`` is the command-line wrapper.
"""

from __future__ import annotations

import platform
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llm_distill.dataset import Pair
from llm_distill.metrics import agreement, classification_metrics, confusion, percentile
from llm_distill.pricing import Price, cost_usd
from llm_distill.students.base import Student
from llm_distill.teacher import TeacherLabel

PredictFn = Callable[[Sequence[str], Sequence[str]], np.ndarray]


@dataclass
class ModelResult:
    """Predictions and measurements for one row of the comparison table."""

    key: str  # short id used in JSON
    name: str
    label: str  # provenance shown in every table, e.g. "fake-provider" or "trained, real"
    pred: list[int]
    size_mb: float | None
    latency_b1_ms: list[float] = field(default_factory=list)  # per decision
    latency_b32_ms: list[float] = field(default_factory=list)  # per decision
    cost_per_1k_b1: float | None = None
    cost_per_1k_b32: float | None = None
    cost_basis: str = ""
    is_teacher: bool = False


# -- measurement ---------------------------------------------------------------------------------
def bench_latency(
    predict: PredictFn,
    questions: Sequence[str],
    passages: Sequence[str],
    *,
    batch: int,
    warmup: int = 3,
    repeats: int = 1,
) -> list[float]:
    """Milliseconds per decision, one sample per call (``batch`` pairs per call)."""
    n = len(questions)
    for i in range(min(warmup, n)):
        predict(questions[i : i + batch], passages[i : i + batch])
    out: list[float] = []
    for _ in range(repeats):
        for i in range(0, n - batch + 1, batch):
            q, p = questions[i : i + batch], passages[i : i + batch]
            t0 = time.perf_counter()
            predict(q, p)
            out.append((time.perf_counter() - t0) * 1000 / batch)
    return out


def student_cost_per_1k(mean_ms_per_decision: float, usd_per_hour: float) -> float:
    return mean_ms_per_decision / 1000 * usd_per_hour / 3600 * 1000


def teacher_cost_per_1k(labels: Sequence[TeacherLabel], price: Price) -> tuple[float, str]:
    """USD per 1k decisions, from provider-reported cost when present, else tokens x list price."""
    reported = [lab.cost_usd for lab in labels if lab.cost_usd is not None]
    if reported and len(reported) == len(labels):
        return float(np.mean(reported)) * 1000, "provider-reported usage.cost"
    per = [cost_usd(price, lab.input_tokens, lab.output_tokens) for lab in labels]
    return float(np.mean(per)) * 1000 if per else 0.0, "tokens x list price (projected)"


def evaluate_student(
    student: Student,
    test: Sequence[Pair],
    threshold: float,
    *,
    usd_per_hour: float,
    size_mb: float | None,
    key: str,
    label: str = "trained on derived labels, measured on CPU",
    repeats_b32: int = 3,
) -> ModelResult:
    qs = [p.question for p in test]
    ps = [p.passage for p in test]
    proba = student.predict_proba(qs, ps)
    pred = [int(x >= threshold) for x in proba]
    b1 = bench_latency(student.predict_proba, qs, ps, batch=1)
    b32 = bench_latency(student.predict_proba, qs, ps, batch=32, repeats=repeats_b32)
    return ModelResult(
        key=key,
        name=student.name,
        label=label,
        pred=pred,
        size_mb=size_mb,
        latency_b1_ms=b1,
        latency_b32_ms=b32,
        cost_per_1k_b1=student_cost_per_1k(float(np.mean(b1)), usd_per_hour),
        cost_per_1k_b32=student_cost_per_1k(float(np.mean(b32)), usd_per_hour),
        cost_basis=f"measured CPU time at ${usd_per_hour:.2f}/h",
    )


def teacher_result(
    labels: Sequence[TeacherLabel], test: Sequence[Pair], price: Price, price_key: str
) -> ModelResult:
    by_id = {lab.pair_id: lab for lab in labels}
    sel = [by_id[p.pair_id] for p in test]
    cost, basis = teacher_cost_per_1k(sel, price)
    provider = sel[0].provider_label if sel else "unknown"
    return ModelResult(
        key="teacher",
        name=f"teacher ({sel[0].model if sel else '?'})",
        label=provider if provider == "fake-provider" else f"{provider}, real",
        pred=[lab.teacher or 0 for lab in sel],
        size_mb=None,
        latency_b1_ms=[lab.latency_ms for lab in sel],
        latency_b32_ms=[],
        cost_per_1k_b1=cost,
        cost_per_1k_b32=None,
        cost_basis=f"{basis}, priced as {price_key}",
        is_teacher=True,
    )


# -- statistics ----------------------------------------------------------------------------------
def bootstrap_ci(
    truth: Sequence[int],
    pred: Sequence[int],
    clusters: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 13,
) -> dict[str, tuple[float, float]]:
    """95% percentile intervals for accuracy and F1, resampling whole questions."""
    t = np.asarray(truth)
    p = np.asarray(pred)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    ids = sorted(groups)
    idx = [np.asarray(groups[c]) for c in ids]
    rng = np.random.default_rng(seed)
    acc: list[float] = []
    f1: list[float] = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        sel = np.concatenate([idx[i] for i in pick])
        m = classification_metrics(t[sel], p[sel])
        acc.append(m["accuracy"])
        f1.append(m["f1"])
    return {
        "accuracy": (float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))),
        "f1": (float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))),
    }


def summarize_model(
    res: ModelResult,
    test: Sequence[Pair],
    teacher_pred: Sequence[int],
    *,
    n_boot: int = 1000,
) -> dict[str, Any]:
    truth = [p.label for p in test]
    m = classification_metrics(truth, res.pred)
    ci = bootstrap_ci(truth, res.pred, [p.question_id for p in test], n_boot=n_boot)
    errors: dict[str, int] = defaultdict(int)
    for p, y in zip(test, res.pred, strict=True):
        if y != p.label:
            errors[p.kind] += 1
    return {
        "key": res.key,
        "name": res.name,
        "label": res.label,
        **m,
        "confusion": confusion(truth, res.pred),
        "accuracy_ci95": ci["accuracy"],
        "f1_ci95": ci["f1"],
        "agreement_with_teacher": None if res.is_teacher else agreement(res.pred, teacher_pred),
        "errors_by_kind": dict(errors),
        "size_mb": res.size_mb,
        "latency_b1_p50_ms": percentile(res.latency_b1_ms, 50),
        "latency_b1_p95_ms": percentile(res.latency_b1_ms, 95),
        "latency_b32_p50_ms": percentile(res.latency_b32_ms, 50) if res.latency_b32_ms else None,
        "latency_b32_p95_ms": percentile(res.latency_b32_ms, 95) if res.latency_b32_ms else None,
        "cost_per_1k_b1_usd": res.cost_per_1k_b1,
        "cost_per_1k_b32_usd": res.cost_per_1k_b32,
        "cost_basis": res.cost_basis,
    }


def slice_accuracy(
    results: Sequence[ModelResult], test: Sequence[Pair], groups: dict[str, str]
) -> list[dict[str, Any]]:
    """Accuracy per slice (question group and pair kind) for every model."""
    slices: dict[str, list[int]] = {}
    for g in ("synthetic", "handwritten", "refusal"):
        slices[f"questions: {g}"] = [
            i for i, p in enumerate(test) if groups.get(p.question_id) == g
        ]
    for k in ("positive", "hard_negative", "easy_negative"):
        slices[f"pairs: {k}"] = [i for i, p in enumerate(test) if p.kind == k]
    rows = []
    for name, idx in slices.items():
        row: dict[str, Any] = {"slice": name, "n": len(idx)}
        for r in results:
            row[r.key] = float(np.mean([r.pred[i] == test[i].label for i in idx])) if idx else None
        rows.append(row)
    return rows


def environment() -> dict[str, str]:
    import onnxruntime
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": str(__import__("os").cpu_count()),
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }


# -- rendering -----------------------------------------------------------------------------------
def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _ci(x: float, lo_hi: Sequence[float]) -> str:
    return f"{100 * x:.1f}% [{100 * lo_hi[0]:.1f}, {100 * lo_hi[1]:.1f}]"


def _ms(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.5f}"


def render_tables(summary: dict[str, Any]) -> str:
    """The three headline tables, without a title (embedded in the README)."""
    models = summary["models"]
    lines = ["**Quality** (held-out test split, 95% interval resampled by question)", ""]
    lines += [
        "| Model | Provenance | Accuracy [95% CI] | F1 (answerable) [95% CI] | Precision | Recall |"
        " Agreement with teacher | Size (MB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        agree = (
            "(is the teacher)"
            if m["agreement_with_teacher"] is None
            else _pct(m["agreement_with_teacher"])
        )
        size = "n/a (API)" if m["size_mb"] is None else f"{m['size_mb']:.1f}"
        lines.append(
            f"| {m['name']} | {m['label']} | {_ci(m['accuracy'], m['accuracy_ci95'])} "
            f"| {_ci(m['f1'], m['f1_ci95'])} | {_pct(m['precision'])} | {_pct(m['recall'])} "
            f"| {agree} | {size} |"
        )
    lines += ["", "**Latency per decision** (milliseconds; CPU; includes tokenisation)", ""]
    lines += [
        "| Model | Provenance | p50 batch 1 | p95 batch 1 | p50 batch 32 | p95 batch 32 |",
        "|---|---|---|---|---|---|",
    ]
    for m in models:
        suffix = " (heuristic, not an LLM call)" if m["label"] == "fake-provider" else ""
        lines.append(
            f"| {m['name']} | {m['label']} | {_ms(m['latency_b1_p50_ms'])}{suffix} "
            f"| {_ms(m['latency_b1_p95_ms'])} | {_ms(m['latency_b32_p50_ms'])} "
            f"| {_ms(m['latency_b32_p95_ms'])} |"
        )
    lines += ["", "**Cost per 1,000 decisions** (USD)", ""]
    lines += [
        "| Model | Provenance | Batch 1 | Batch 32 | Basis |",
        "|---|---|---|---|---|",
    ]
    for m in models:
        lines.append(
            f"| {m['name']} | {m['label']} | {_usd(m['cost_per_1k_b1_usd'])} "
            f"| {_usd(m['cost_per_1k_b32_usd'])} | {m['cost_basis']} |"
        )
    return "\n".join(lines)


def render_report(summary: dict[str, Any]) -> str:
    env = summary["environment"]
    ds = summary["dataset"]
    keys = [m["key"] for m in summary["models"]]
    names = {m["key"]: m["name"] for m in summary["models"]}
    out = [
        "# Distillation report",
        "",
        f"Generated by `scripts/compare.py` on {summary['timestamp']} "
        f"(commit `{summary['commit']}`). Every number below comes from that run.",
        "",
        "Rows labelled `fake-provider` come from the deterministic lexical-overlap teacher, "
        "not from a language model. They describe a heuristic. Student rows are real: the "
        "models were trained on the derived ground-truth labels and measured here.",
        "",
        render_tables(summary),
        "",
        "## Test split",
        "",
        f"{ds['test_pairs']} pairs from {ds['test_questions']} questions "
        f"({ds['test_positives']} answerable, {ds['test_pairs'] - ds['test_positives']} not). "
        f"The student threshold was chosen on the validation split, never on the test split. "
        f"Teacher price key: `{summary['price_key']}`"
        + (
            ""
            if summary["price_exact"]
            else " (nearest family member; the exact model is not in pricing.json)"
        )
        + ".",
        "",
        "## Where each model is right",
        "",
        "| Slice | n | " + " | ".join(names[k] for k in keys) + " |",
        "|---|---|" + "|".join("---" for _ in keys) + "|",
    ]
    for row in summary["slices"]:
        out.append(
            f"| {row['slice']} | {row['n']} | " + " | ".join(_pct(row[k]) for k in keys) + " |"
        )
    out += [
        "",
        "## Errors by pair kind",
        "",
        "| Model | Confusion (tp / fp / fn / tn) | Errors by kind |",
        "|---|---|---|",
    ]
    for m in summary["models"]:
        c = m["confusion"]
        errs = ", ".join(f"{k}: {v}" for k, v in sorted(m["errors_by_kind"].items())) or "none"
        out.append(f"| {m['name']} | {c['tp']} / {c['fp']} / {c['fn']} / {c['tn']} | {errs} |")
    out += [
        "",
        "## Teacher label noise on the full dataset",
        "",
        "| Split | Pairs | Teacher agreement with ground truth | Provenance |",
        "|---|---|---|---|",
    ]
    for row in summary["teacher_noise"]:
        out.append(
            f"| {row['split']} | {row['pairs']} | {_pct(row['agreement'])} | {row['label']} |"
        )
    out += [
        "",
        "## Environment",
        "",
        "| Key | Value |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in env.items()),
        "",
        "Training environment (from `models/s1-minilm/student_meta.json`): "
        f"device `{summary['training'].get('device', 'unknown')}`, "
        f"{summary['training'].get('train_seconds', 'n/a')} s, "
        f"best epoch {summary['training'].get('best_epoch', 'n/a')}.",
        "",
    ]
    return "\n".join(out)
