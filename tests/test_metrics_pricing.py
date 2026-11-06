from __future__ import annotations

import pytest

from llm_distill.metrics import agreement, classification_metrics, cohens_kappa, percentile
from llm_distill.pricing import cost_usd, load_pricing, resolve_price
from llm_distill.students.base import tune_threshold

from .conftest import ROOT


def test_classification_metrics():
    m = classification_metrics([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_degenerate_metrics_do_not_divide_by_zero():
    assert classification_metrics([0, 0], [0, 0])["f1"] == 0.0
    assert agreement([], []) == 0.0


def test_kappa_and_percentile():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0


def test_tune_threshold_prefers_separating_value():
    t = tune_threshold([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.9])
    assert 0.3 <= t <= 0.7


def test_pricing_exact_and_family_fallback():
    table = load_pricing(str(ROOT / "pricing.json"))
    key, price, exact = resolve_price("anthropic/claude-haiku-4.5", table)
    assert exact and price.input_per_m == 1.0
    key2, price2, exact2 = resolve_price("anthropic/claude-haiku-5.5", table)
    assert not exact2 and key2 == key and price2 == price
    with pytest.raises(KeyError):
        resolve_price("nonexistent/zzz-model", table)


def test_cost_math():
    table = load_pricing(str(ROOT / "pricing.json"))
    _, price, _ = resolve_price("anthropic/claude-haiku-4.5", table)
    total = cost_usd(price, 1_000_000, 1_000_000)
    assert total == pytest.approx(price.input_per_m + price.output_per_m)
