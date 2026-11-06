"""Binary-classification metrics for the positive class ("answerable")."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    return {
        "tp": int(((t == 1) & (p == 1)).sum()),
        "fp": int(((t == 0) & (p == 1)).sum()),
        "fn": int(((t == 1) & (p == 0)).sum()),
        "tn": int(((t == 0) & (p == 0)).sum()),
    }


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    c = confusion(y_true, y_pred)
    n = sum(c.values())
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (c["tp"] + c["tn"]) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def agreement(a: Sequence[int], b: Sequence[int]) -> float:
    """Fraction of items on which two label sequences agree."""
    x = np.asarray(a, dtype=int)
    y = np.asarray(b, dtype=int)
    return float((x == y).mean()) if len(x) else 0.0


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    x = np.asarray(a, dtype=int)
    y = np.asarray(b, dtype=int)
    if not len(x):
        return 0.0
    po = float((x == y).mean())
    pe = sum(float((x == k).mean()) * float((y == k).mean()) for k in (0, 1))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else 0.0
