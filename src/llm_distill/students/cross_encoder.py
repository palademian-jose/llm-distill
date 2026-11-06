"""S1: cross-encoder answerability classifier (PyTorch) fine-tuned from MiniLM.

The question and the passage are encoded together as ``[CLS] question [SEP] passage [SEP]`` and
a two-way classification head predicts "answerable". Training uses fixed seeds, dynamic padding
and picks the epoch with the best validation F1. CUDA is used automatically when the installed
torch build supports it.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from llm_distill.metrics import classification_metrics
from llm_distill.students.base import tune_threshold

META_NAME = "student_meta.json"


def probs_from_logits(logits: Any) -> Any:
    """P(answerable) from either a 1-logit head (sigmoid) or a 2-logit head (softmax)."""
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits[:, 0])
    return torch.softmax(logits, dim=-1)[:, 1]


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 192
    epochs: int = 2
    batch_size: int = 16
    lr: float = 4e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 13


def _encode(
    tokenizer: Any, questions: Sequence[str], passages: Sequence[str], max_length: int
) -> dict[str, torch.Tensor]:
    return tokenizer(
        list(questions),
        list(passages),
        truncation="only_second",
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )


@torch.no_grad()
def _predict(
    model: nn.Module,
    tokenizer: Any,
    questions: Sequence[str],
    passages: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(questions), batch_size):
        enc = _encode(
            tokenizer, questions[i : i + batch_size], passages[i : i + batch_size], max_length
        )
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        out.append(probs_from_logits(logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


class CrossEncoderStudent:
    """Inference wrapper around a fine-tuned sequence-classification model."""

    name = "s1-minilm-torch"

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        max_length: int = 256,
        batch_size: int = 32,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or pick_device()
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def predict_proba(self, questions: Sequence[str], passages: Sequence[str]) -> np.ndarray:
        return _predict(
            self.model,
            self.tokenizer,
            questions,
            passages,
            max_length=self.max_length,
            batch_size=self.batch_size,
            device=self.device,
        )

    @classmethod
    def from_dir(cls, path: Path, *, device: torch.device | None = None) -> tuple[Any, float]:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        meta = json.loads((path / META_NAME).read_text(encoding="utf-8"))
        model = AutoModelForSequenceClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        student = cls(model, tokenizer, max_length=int(meta["max_length"]), device=device)
        return student, float(meta["threshold"])


def load_base(base_model: str) -> tuple[nn.Module, Any]:
    """Download (or read from the Hugging Face cache) the base encoder.

    A cross-encoder checkpoint such as ms-marco-MiniLM keeps its 1-logit relevance head; a plain
    sentence encoder gets a fresh 2-logit head.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)
    if model.config.num_labels not in (1, 2):
        raise ValueError("expected a 1- or 2-label classification head")
    return model, tokenizer


def train_cross_encoder(
    model: nn.Module,
    tokenizer: Any,
    train: tuple[Sequence[str], Sequence[str], Sequence[int]],
    val: tuple[Sequence[str], Sequence[str], Sequence[int]],
    cfg: TrainConfig,
    *,
    device: torch.device | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Fine-tune ``model`` in place and return a report. Restores the best-F1 epoch."""
    set_seed(cfg.seed)
    device = device or pick_device()
    model.to(device)
    q, p, y = (list(x) for x in train)
    n = len(y)
    pos = max(1, sum(y))
    neg = max(1, n - pos)
    # Square-root class weighting: the positive class is about 6% of the pairs.
    pos_weight = math.sqrt(neg / pos)
    weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=weights)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] == 1:
            return bce(logits[:, 0], labels.float())
        return ce(logits, labels)

    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {
            "params": [
                prm for nm, prm in model.named_parameters() if not any(nd in nm for nd in no_decay)
            ],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [
                prm for nm, prm in model.named_parameters() if any(nd in nm for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optim = torch.optim.AdamW(groups, lr=cfg.lr)
    steps_per_epoch = math.ceil(n / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    warm = max(1, int(total * cfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)),
    )

    rng = random.Random(cfg.seed)
    best: dict[str, Any] = {"f1": -1.0, "state": None, "epoch": 0, "threshold": 0.5}
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = list(range(n))
        rng.shuffle(order)
        running = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            enc = _encode(tokenizer, [q[j] for j in idx], [p[j] for j in idx], cfg.max_length)
            labels = torch.tensor([y[j] for j in idx], dtype=torch.long, device=device)
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            running += float(loss.item()) * len(idx)
        proba = _predict(
            model,
            tokenizer,
            val[0],
            val[1],
            max_length=cfg.max_length,
            batch_size=64,
            device=device,
        )
        threshold = tune_threshold(val[2], proba)
        m = classification_metrics(val[2], [int(x >= threshold) for x in proba])
        history.append({"epoch": epoch, "train_loss": running / n, "val_f1": m["f1"]})
        log(
            f"epoch {epoch}/{cfg.epochs} loss={running / n:.4f} val_f1={m['f1']:.4f} "
            f"thr={threshold:.2f} elapsed={time.perf_counter() - start:.0f}s"
        )
        if m["f1"] > best["f1"]:
            best = {
                "f1": m["f1"],
                "epoch": epoch,
                "threshold": threshold,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "val": m,
            }
    model.load_state_dict(best["state"])
    return {
        "config": asdict(cfg),
        "device": str(device),
        "cuda_used": device.type == "cuda",
        "train_seconds": round(time.perf_counter() - start, 1),
        "best_epoch": best["epoch"],
        "threshold": best["threshold"],
        "val": best["val"],
        "history": history,
    }


def save_student(model: nn.Module, tokenizer: Any, out_dir: Path, report: dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "max_length": report["config"]["max_length"],
        "threshold": report["threshold"],
        "base_model": report["config"]["base_model"],
        "train_report": report,
    }
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
