"""Build data/pairs.jsonl from the committed docs-rag snapshot.

uv run python scripts/build_dataset.py                         # offline, from the snapshot
uv run python scripts/build_dataset.py --refresh ../docs-rag   # re-copy the snapshot first
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_distill.dataset import build_pairs, summarize, write_pairs
from llm_distill.settings import get_settings
from llm_distill.snapshot import refresh_snapshot


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", type=Path, help="docs-rag checkout to copy the snapshot from")
    ap.add_argument("--snapshot", type=Path, default=s.snapshot_dir)
    ap.add_argument("--out", type=Path, default=s.pairs_path)
    ap.add_argument("--hard", type=int, default=8, help="hard negatives per question")
    ap.add_argument("--easy", type=int, default=6, help="easy negatives per question")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if args.refresh:
        print("snapshot:", json.dumps(refresh_snapshot(args.refresh, args.snapshot)))
    pairs = build_pairs(
        args.snapshot / "corpus",
        args.snapshot / "golden",
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        n_hard=args.hard,
        n_easy=args.easy,
        seed=args.seed,
    )
    write_pairs(pairs, args.out)
    stats = summarize(pairs)
    (args.out.parent / "pairs_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    print(f"wrote {len(pairs)} pairs to {args.out}")


if __name__ == "__main__":
    main()
