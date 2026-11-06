"""Copy the docs-rag corpus and golden set into a committed, dated snapshot."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def refresh_snapshot(source_repo: Path, dest: Path) -> dict:
    """Replace ``dest`` with ``source_repo``'s ``data/corpus`` and ``evals/golden``."""
    source_repo = Path(source_repo)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source_repo / "data" / "corpus", dest / "corpus")
    shutil.copytree(source_repo / "evals" / "golden", dest / "golden")
    dirty = bool(_git(source_repo, "status", "--porcelain", "--", "data/corpus", "evals/golden"))
    meta = {
        "source_repo": "docs-rag",
        "source_commit": _git(source_repo, "rev-parse", "HEAD"),
        "source_tree_dirty": dirty,
        "snapshot_date": date.today().isoformat(),
        "corpus_files": sum(1 for p in (dest / "corpus").rglob("*") if p.is_file()),
    }
    (dest / "SNAPSHOT.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta
