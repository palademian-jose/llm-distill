"""Compare the teacher with the S2 and S1 students on the held-out test split.

Writes reports/distillation.md (human readable), reports/distillation.json (all numbers),
reports/eval-<run_id>.json and reports/eval-latest.json (the evalkit result for the gate), and,
with --readme, refreshes the results block of README.md.

    uv run python scripts/compare.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from llm_distill.compare import (
    ModelResult,
    environment,
    evaluate_student,
    render_report,
    render_tables,
    slice_accuracy,
    summarize_model,
    teacher_result,
)
from llm_distill.dataset import load_questions, read_pairs
from llm_distill.metrics import agreement
from llm_distill.pricing import load_pricing, resolve_price
from llm_distill.settings import get_settings
from llm_distill.students.cross_encoder import META_NAME, CrossEncoderStudent
from llm_distill.students.onnx_student import OnnxStudent
from llm_distill.students.tfidf import TfidfStudent
from llm_distill.teacher import read_labels

README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _mb(*paths: Path) -> float:
    return sum(p.stat().st_size for p in paths if p.exists()) / 1e6


def _weights_files(d: Path) -> list[Path]:
    return [d / "model.safetensors"] if (d / "model.safetensors").exists() else []


def _eval_result(
    summary: dict, results: list[ModelResult], test: list, primary: str, run_id: str
) -> dict:
    by_key = {m["key"]: m for m in summary["models"]}
    p = by_key[primary]
    res = next(r for r in results if r.key == primary)
    metrics = {
        "accuracy": p["accuracy"],
        "f1": p["f1"],
        "precision": p["precision"],
        "recall": p["recall"],
        "agreement_with_teacher": p["agreement_with_teacher"],
        "p50_latency_ms": p["latency_b1_p50_ms"],
        "p95_latency_ms": p["latency_b1_p95_ms"],
        "p95_latency_ms_batch32": p["latency_b32_p95_ms"],
        "cost_per_1k_usd": p["cost_per_1k_b1_usd"],
        "size_mb": p["size_mb"],
        "teacher_accuracy": by_key["teacher"]["accuracy"],
        "teacher_f1": by_key["teacher"]["f1"],
    }
    for key in ("s2_tfidf", "s1_torch"):
        if key in by_key:
            metrics[f"{key}_accuracy"] = by_key[key]["accuracy"]
            metrics[f"{key}_f1"] = by_key[key]["f1"]
    n = len(test)
    cases = [
        {
            "id": pair.pair_id,
            "scores": {"correct": pred == pair.label},
            "passed": pred == pair.label,
            "latency_ms": res.latency_b1_ms[i] if i < len(res.latency_b1_ms) else 0.0,
            "cost_usd": (res.cost_per_1k_b1 or 0.0) / 1000,
        }
        for i, (pair, pred) in enumerate(zip(test, res.pred, strict=True))
    ]
    assert len(cases) == n
    return {
        "run_id": run_id,
        "commit": summary["commit"],
        "timestamp": summary["timestamp"],
        "provider": "fake",
        "model": res.name,
        "metrics": metrics,
        "cases": cases,
        "meta": {
            "primary_student": primary,
            "teacher_provenance": by_key["teacher"]["label"],
            "note": "Teacher metrics are fake-provider (lexical-overlap heuristic).",
        },
    }


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=s.pairs_path)
    ap.add_argument("--labels", type=Path, default=s.teacher_labels_path)
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--usd-per-hour", type=float, default=s.cpu_usd_per_hour)
    ap.add_argument(
        "--price-model", default=s.cheap_model, help="model id used to price the teacher"
    )
    ap.add_argument("--readme", type=Path, default=None, help="refresh the results block here")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    test = read_pairs(args.pairs, "test")
    labels = read_labels(args.labels)
    questions = load_questions(s.snapshot_dir / "golden")
    groups = {q.id: q.group for q in questions}

    table = load_pricing(str(s.pricing_path))
    price_key, price, exact = resolve_price(args.price_model, table)

    results: list[ModelResult] = [teacher_result(labels, test, price, price_key)]

    if s.tfidf_path.exists():
        tfidf, thr = TfidfStudent.load(s.tfidf_path)
        results.append(
            evaluate_student(
                tfidf,
                test,
                thr,
                usd_per_hour=args.usd_per_hour,
                size_mb=_mb(s.tfidf_path),
                key="s2_tfidf",
            )
        )
    training: dict = {}
    if (s.student_dir / META_NAME).exists():
        import torch

        training = json.loads((s.student_dir / META_NAME).read_text())["train_report"]
        torch.set_num_threads(max(1, torch.get_num_threads()))
        st, thr = CrossEncoderStudent.from_dir(s.student_dir, device=torch.device("cpu"))
        results.append(
            evaluate_student(
                st,
                test,
                thr,
                usd_per_hour=args.usd_per_hour,
                size_mb=_mb(*_weights_files(s.student_dir)),
                key="s1_torch",
            )
        )
        if s.onnx_path.exists():
            ox, thr = OnnxStudent.from_dir(s.student_dir, s.onnx_path)
            results.append(
                evaluate_student(
                    ox,
                    test,
                    thr,
                    usd_per_hour=args.usd_per_hour,
                    size_mb=_mb(s.onnx_path),
                    key="s1_onnx",
                )
            )

    teacher_pred = results[0].pred
    models = [summarize_model(r, test, teacher_pred, n_boot=args.boot) for r in results]
    noise = []
    for split in ("train", "val", "test"):
        sel = [lab for lab in labels if lab.split == split]
        noise.append(
            {
                "split": split,
                "pairs": len(sel),
                "agreement": agreement([x.truth for x in sel], [x.teacher or 0 for x in sel]),
                "label": sel[0].provider_label if sel else "n/a",
            }
        )
    now = datetime.now(UTC)
    summary = {
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _git_sha(),
        "dataset": {
            "test_pairs": len(test),
            "test_questions": len({p.question_id for p in test}),
            "test_positives": sum(p.label for p in test),
        },
        "price_key": price_key,
        "price_exact": exact,
        "usd_per_hour": args.usd_per_hour,
        "environment": environment(),
        "training": {
            k: training.get(k) for k in ("device", "cuda_used", "train_seconds", "best_epoch")
        },
        "models": models,
        "slices": slice_accuracy(results, test, groups),
        "teacher_noise": noise,
    }

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "distillation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.reports / "distillation.md").write_text(render_report(summary), encoding="utf-8")

    primary = "s1_onnx" if any(r.key == "s1_onnx" for r in results) else results[-1].key
    run_id = f"distill-{now.strftime('%Y%m%dT%H%M%S')}"
    eval_json = _eval_result(summary, results, test, primary, run_id)
    for name in (f"eval-{run_id}.json", "eval-latest.json"):
        (args.reports / name).write_text(json.dumps(eval_json, indent=2) + "\n", encoding="utf-8")

    if args.readme and args.readme.exists():
        text = args.readme.read_text(encoding="utf-8")
        if README_START in text and README_END in text:
            head, rest = text.split(README_START, 1)
            _, tail = rest.split(README_END, 1)
            block = f"{README_START}\n{render_tables(summary)}\n{README_END}"
            args.readme.write_text(head + block + tail, encoding="utf-8")

    print(render_tables(summary))
    print(f"\nwrote {args.reports / 'distillation.md'} and eval-latest.json (primary: {primary})")


if __name__ == "__main__":
    main()
