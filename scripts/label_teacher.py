"""Label every pair with the teacher and report its agreement with the derived ground truth.

    LLM_PROVIDER=fake uv run python scripts/label_teacher.py          # deterministic heuristic
    OPENROUTER_API_KEY=... uv run python scripts/label_teacher.py     # CHEAP_MODEL via OpenRouter

With the fake provider every row is tagged ``fake-provider``; those numbers describe a lexical
heuristic, not a language model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_distill.dataset import read_pairs
from llm_distill.llm import get_client
from llm_distill.metrics import agreement, classification_metrics
from llm_distill.settings import get_settings
from llm_distill.teacher import label_pairs, write_labels


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=s.pairs_path)
    ap.add_argument("--out", type=Path, default=s.teacher_labels_path)
    ap.add_argument("--model", default=None, help="override CHEAP_MODEL")
    ap.add_argument("--limit", type=int, default=None, help="label only the first N pairs")
    args = ap.parse_args()

    pairs = read_pairs(args.pairs)[: args.limit]
    client = get_client(s, role="cheap")
    labels = label_pairs(
        client,
        pairs,
        model=args.model,
        progress=lambda i, n: print(f"\r{i}/{n}", end="", flush=True) if i % 200 == 0 else None,
    )
    print()
    write_labels(labels, args.out)

    unparsed = sum(1 for lab in labels if lab.teacher is None)
    truth = [lab.truth for lab in labels]
    pred = [lab.teacher or 0 for lab in labels]
    summary = {
        "provider": labels[0].provider_label if labels else None,
        "model": labels[0].model if labels else None,
        "pairs": len(labels),
        "unparseable_replies": unparsed,
        "agreement_with_truth": round(agreement(truth, pred), 4),
        **{k: round(v, 4) for k, v in classification_metrics(truth, pred).items()},
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
