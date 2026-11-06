"""S2: logistic regression on TF-IDF interaction terms plus lexical-overlap features.

This is the honest cheap baseline. It sees the same (question, passage) text as the
cross-encoder but only through bag-of-words statistics.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from llm_distill.textutil import content_tokens, overlap_ratio

N_DENSE = 7


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:], strict=False))


class TfidfStudent:
    name = "s2-tfidf-logreg"

    def __init__(self, c: float = 4.0, seed: int = 13) -> None:
        self.c = c
        self.seed = seed
        self.vectorizer = TfidfVectorizer(
            tokenizer=content_tokens, lowercase=False, token_pattern=None, sublinear_tf=True
        )
        self.model = LogisticRegression(
            C=c, class_weight="balanced", max_iter=2000, random_state=seed
        )
        self.idf: dict[str, float] = {}

    # -- features -------------------------------------------------------------------------
    def _dense(self, questions: Sequence[str], passages: Sequence[str]) -> np.ndarray:
        rows = []
        for q, p in zip(questions, passages, strict=True):
            qt, pt = content_tokens(q), content_tokens(p)
            qs, ps = set(qt), set(pt)
            matched = qs & ps
            idf_total = sum(self.idf.get(t, 1.0) for t in qs) or 1.0
            idf_hit = sum(self.idf.get(t, 1.0) for t in matched)
            qb, pb = _bigrams(qt), _bigrams(pt)
            rows.append(
                [
                    overlap_ratio(qt, pt),
                    idf_hit / idf_total,
                    len(matched) / max(1, len(qs | ps)),
                    (len(qb & pb) / len(qb)) if qb else 0.0,
                    min(len(matched), 12) / 12.0,
                    min(len(qs), 30) / 30.0,
                    min(len(ps), 200) / 200.0,
                ]
            )
        return np.asarray(rows, dtype=np.float64)

    def _features(self, questions: Sequence[str], passages: Sequence[str]) -> sparse.csr_matrix:
        qv = self.vectorizer.transform(questions)
        pv = self.vectorizer.transform(passages)
        interaction = qv.multiply(pv).tocsr()
        cosine = np.asarray(interaction.sum(axis=1)).ravel()[:, None]
        dense = np.hstack([self._dense(questions, passages), cosine])
        return sparse.hstack([interaction, sparse.csr_matrix(dense)], format="csr")

    # -- API ------------------------------------------------------------------------------
    def fit(
        self, questions: Sequence[str], passages: Sequence[str], labels: Sequence[int]
    ) -> TfidfStudent:
        self.vectorizer.fit(list(questions) + list(passages))
        self.idf = dict(
            zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_, strict=True)
        )
        self.model.fit(self._features(questions, passages), np.asarray(labels))
        return self

    def predict_proba(self, questions: Sequence[str], passages: Sequence[str]) -> np.ndarray:
        feats = self._features(questions, passages)
        return self.model.predict_proba(feats)[:, 1]

    def save(self, path: Path, threshold: float = 0.5) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"student": self, "threshold": threshold}, path)

    @staticmethod
    def load(path: Path) -> tuple[TfidfStudent, float]:
        blob = joblib.load(path)
        return blob["student"], float(blob["threshold"])
