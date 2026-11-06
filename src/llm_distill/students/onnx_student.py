"""ONNX export of the S1 cross-encoder and an ONNX Runtime inference path (no torch needed)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from llm_distill.students.cross_encoder import META_NAME

INPUT_NAMES = ["input_ids", "attention_mask", "token_type_ids"]


class _LogitsOnly(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        ).logits


def export_onnx(model: nn.Module, tokenizer: Any, out_path: Path, *, max_length: int = 256) -> Path:
    """Export ``model`` with dynamic batch and sequence axes (legacy TorchScript exporter)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    enc = tokenizer(
        ["How long is the canary?", "What is the limit?"],
        ["The canary runs for ten minutes.", "Pull requests stay small."],
        truncation="only_second",
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    args = tuple(enc[name] for name in INPUT_NAMES)
    torch.onnx.export(
        _LogitsOnly(model),
        args,
        str(out_path),
        input_names=INPUT_NAMES,
        output_names=["logits"],
        dynamic_axes={
            **{name: {0: "batch", 1: "seq"} for name in INPUT_NAMES},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    return out_path


def _softmax_pos(logits: np.ndarray) -> np.ndarray:
    if logits.shape[-1] == 1:
        return (1.0 / (1.0 + np.exp(-logits[:, 0]))).astype(np.float64)
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e[:, 1] / e.sum(axis=-1)).astype(np.float64)


class OnnxStudent:
    name = "s1-minilm-onnx"

    def __init__(
        self,
        onnx_path: Path,
        tokenizer: Any,
        *,
        max_length: int = 256,
        batch_size: int = 32,
        intra_op_threads: int | None = None,
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if intra_op_threads:
            opts.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            str(onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def predict_proba(self, questions: Sequence[str], passages: Sequence[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in range(0, len(questions), self.batch_size):
            enc = self.tokenizer(
                list(questions[i : i + self.batch_size]),
                list(passages[i : i + self.batch_size]),
                truncation="only_second",
                max_length=self.max_length,
                padding=True,
                return_tensors="np",
            )
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            logits = self.session.run(["logits"], feed)[0]
            out.append(_softmax_pos(logits))
        return np.concatenate(out) if out else np.zeros(0)

    @classmethod
    def from_dir(cls, student_dir: Path, onnx_path: Path | None = None, **kw: Any):
        """Load tokenizer, threshold and ONNX graph from a directory written by scripts/train.py."""
        from transformers import AutoTokenizer

        student_dir = Path(student_dir)
        meta = json.loads((student_dir / META_NAME).read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(student_dir)
        student = cls(
            onnx_path or student_dir / "model.onnx",
            tokenizer,
            max_length=int(meta["max_length"]),
            **kw,
        )
        return student, float(meta["threshold"])
