"""Export the trained S1 cross-encoder to ONNX and check parity with PyTorch.

uv run python scripts/export_onnx.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from llm_distill.dataset import read_pairs
from llm_distill.settings import get_settings
from llm_distill.students.cross_encoder import CrossEncoderStudent
from llm_distill.students.onnx_student import OnnxStudent, export_onnx


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student-dir", type=Path, default=s.student_dir)
    ap.add_argument("--out", type=Path, default=s.onnx_path)
    ap.add_argument("--pairs", type=Path, default=s.pairs_path)
    ap.add_argument("--parity-samples", type=int, default=64)
    args = ap.parse_args()

    torch_student, threshold = CrossEncoderStudent.from_dir(args.student_dir)
    export_onnx(
        torch_student.model, torch_student.tokenizer, args.out, max_length=torch_student.max_length
    )
    onnx_student = OnnxStudent(
        args.out, torch_student.tokenizer, max_length=torch_student.max_length
    )

    test = read_pairs(args.pairs, "test")[: args.parity_samples]
    qs, ps = [p.question for p in test], [p.passage for p in test]
    # Compare on CPU so that a CUDA-trained model is checked on the same hardware as ORT.
    torch_student.model.cpu()
    torch_student.device = torch_student.device.__class__("cpu")
    a = torch_student.predict_proba(qs, ps)
    b = onnx_student.predict_proba(qs, ps)
    report = {
        "onnx_path": str(args.out),
        "size_mb": round(args.out.stat().st_size / 1e6, 2),
        "parity_samples": len(qs),
        "max_abs_prob_diff": float(np.abs(a - b).max()),
        "decision_flips": int(((a >= threshold) != (b >= threshold)).sum()),
    }
    print(json.dumps(report, indent=2))
    if report["max_abs_prob_diff"] > 1e-3:
        raise SystemExit("ONNX output differs from PyTorch by more than 1e-3")


if __name__ == "__main__":
    main()
