"""Fine-tune the S1 MiniLM cross-encoder on the train split (`make train`).

CUDA is used when the installed torch build supports it; otherwise CPU. The default settings
(2 epochs, max length 192, batch 16) finish well inside 10 minutes on a laptop CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_distill.dataset import read_pairs
from llm_distill.settings import get_settings
from llm_distill.students.cross_encoder import (
    TrainConfig,
    load_base,
    save_student,
    train_cross_encoder,
)


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=s.pairs_path)
    ap.add_argument("--out", type=Path, default=s.student_dir)
    ap.add_argument("--base-model", default=s.student_base_model)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    cfg = TrainConfig(
        base_model=args.base_model,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    train = read_pairs(args.pairs, "train")
    val = read_pairs(args.pairs, "val")
    model, tokenizer = load_base(cfg.base_model)
    report = train_cross_encoder(
        model,
        tokenizer,
        ([p.question for p in train], [p.passage for p in train], [p.label for p in train]),
        ([p.question for p in val], [p.passage for p in val], [p.label for p in val]),
        cfg,
    )
    save_student(model, tokenizer, args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
