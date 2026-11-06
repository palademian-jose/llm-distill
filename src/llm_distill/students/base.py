"""Shared student interface and threshold selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from llm_distill.metrics import classification_metrics


class Student(Protocol):
    name: str

    def predict_proba(self, questions: Sequence[str], passages: Sequence[str]) -> np.ndarray:
        """Probability that each passage answers its question, shape (n,)."""
        ...


def tune_threshold(y_true: Sequence[int], proba: Sequence[float]) -> float:
    """Pick the decision threshold with the best F1 on the validation split.

    Ties go to the threshold closest to 0.5. The test split is never used.
    """
    best_t, best_key = 0.5, (-1.0, 0.0)
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = [int(p >= t) for p in proba]
        f1 = classification_metrics(y_true, pred)["f1"]
        key = (f1, -abs(float(t) - 0.5))
        if key > best_key:
            best_t, best_key = float(t), key
    return best_t
