"""Price lookup over pricing.json (copied from llm-evalops, which is the source of truth)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


class Price(BaseModel):
    input_per_m: float = 0.0
    output_per_m: float = 0.0


@lru_cache(maxsize=4)
def load_pricing(path: str = "pricing.json") -> dict[str, Price]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k.lower(): Price(**v) for k, v in raw.items() if not k.startswith("_")}


def _version_key(key: str) -> tuple[int, ...]:
    digits = "".join(ch if ch.isdigit() else " " for ch in key).split()
    return tuple(int(d) for d in digits)


def resolve_price(model: str, table: dict[str, Price]) -> tuple[str, Price, bool]:
    """Return ``(pricing key, price, exact)`` for ``model``.

    Order: exact key, vendor prefix stripped, dots and dashes normalised. When the exact model
    is not in the table (for example a newer minor version), fall back to the newest key of the
    same family (``anthropic/claude-haiku-5.5`` -> ``anthropic/claude-haiku-4.5``) and report
    ``exact=False`` so the report can say so.
    """
    m = model.lower().strip()
    bare = m.split("/", 1)[-1]
    for cand in (m, bare, m.replace(".", "-"), bare.replace(".", "-")):
        if cand in table:
            return cand, table[cand], True
    family = bare.rsplit("-", 1)[0] if "-" in bare else bare
    family = family.rstrip("-0123456789.")
    matches = [k for k in table if "/" in k and family and family in k]
    if not matches:
        matches = [k for k in table if family and family in k]
    if not matches:
        raise KeyError(f"no price for {model!r} in pricing table")
    best = max(matches, key=_version_key)
    return best, table[best], False


def cost_usd(price: Price, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * price.input_per_m + output_tokens * price.output_per_m) / 1_000_000
