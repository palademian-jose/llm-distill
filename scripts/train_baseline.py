"""Fit the S2 TF-IDF + logistic-regression baseline on the train split.

The decision threshold is chosen on the validation split; the test split is only used by
scripts/compare.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from llm_distill.dataset import read_pairs
from llm_distill.metrics import classification_metrics
from llm_distill.settings import get_settings
from llm_distill.students.base import tune_threshold
from llm_distill.students.tfidf import TfidfStudent


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=s.pairs_path)
    ap.add_argument("--out", type=Path, default=s.tfidf_path)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    train = read_pairs(args.pairs, "train")
    val = read_pairs(args.pairs, "val")
    start = time.perf_counter()
    student = TfidfStudent(seed=args.seed).fit(
        [p.question for p in train], [p.passage for p in train], [p.label for p in train]
    )
    fit_seconds = time.perf_counter() - start
    proba = student.predict_proba([p.question for p in val], [p.passage for p in val])
    y_val = [p.label for p in val]
    threshold = tune_threshold(y_val, proba)
    student.save(args.out, threshold)
    report = {
        "student": student.name,
        "train_pairs": len(train),
        "fit_seconds": round(fit_seconds, 2),
        "threshold": threshold,
        "val": classification_metrics(y_val, [int(p >= threshold) for p in proba]),
    }
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
