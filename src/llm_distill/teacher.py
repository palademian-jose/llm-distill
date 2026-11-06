"""Teacher labelling: ask an LLMClient whether each passage answers its question."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from llm_distill.dataset import Pair
from llm_distill.llm.base import LLMClient
from llm_distill.prompts import build_messages, parse_verdict

FAKE_LABEL = "fake-provider"


class TeacherLabel(BaseModel):
    pair_id: str
    split: str
    kind: str
    truth: int  # derived ground truth
    teacher: int | None  # 1 yes, 0 no, None when the reply was neither
    raw: str
    provider: str
    provider_label: str  # "fake-provider" for the heuristic, else the provider name
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float | None = None  # provider-reported cost when available


def provider_label(provider: str) -> str:
    return FAKE_LABEL if provider == "fake" else provider


def label_one(client: LLMClient, pair: Pair, *, model: str | None = None, retries: int = 3):
    messages = build_messages(pair.question, pair.passage)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.complete(messages, model=model, temperature=0.0, max_tokens=4)
            break
        except Exception as exc:  # network and rate-limit errors differ per backend
            last = exc
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"teacher failed on {pair.pair_id}") from last
    verdict = parse_verdict(resp.text)
    return TeacherLabel(
        pair_id=pair.pair_id,
        split=pair.split,
        kind=pair.kind,
        truth=pair.label,
        teacher=None if verdict is None else int(verdict),
        raw=resp.text[:40],
        provider=client.provider,
        provider_label=provider_label(client.provider),
        model=resp.model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        latency_ms=resp.latency_ms,
        cost_usd=resp.cost_usd,
    )


def label_pairs(
    client: LLMClient,
    pairs: list[Pair],
    *,
    model: str | None = None,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[TeacherLabel]:
    workers = workers if workers is not None else (1 if client.provider == "fake" else 8)
    out: list[TeacherLabel] = []
    if workers <= 1:
        for i, p in enumerate(pairs, start=1):
            out.append(label_one(client, p, model=model))
            if progress:
                progress(i, len(pairs))
        return out
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, lab in enumerate(pool.map(lambda p: label_one(client, p, model=model), pairs), 1):
            out.append(lab)
            if progress:
                progress(i, len(pairs))
    return out


def write_labels(labels: list[TeacherLabel], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for lab in labels:
            fh.write(lab.model_dump_json() + "\n")


def read_labels(path: Path, split: str | None = None) -> list[TeacherLabel]:
    rows = [
        TeacherLabel.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in rows if split is None or r.split == split]
