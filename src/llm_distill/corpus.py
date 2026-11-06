"""Corpus loading and line-aligned chunking, copied from docs-rag so the dataset builder
reproduces exactly the chunks (and chunk ids) that docs-rag retrieves.

Chunks never split a line, so every chunk is addressable as ``doc_path#Lstart-Lend``.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_distill.textutil import _HEADING_RE, first_heading, is_heading, tokenize

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


@dataclass(frozen=True)
class Document:
    doc_path: str  # path relative to the corpus root, forward slashes
    title: str
    text: str

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()


def load_corpus(root: Path) -> list[Document]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(root).as_posix()
        title = first_heading(text) or path.stem.replace("-", " ").replace("_", " ").title()
        docs.append(Document(doc_path=rel, title=title, text=text))
    return docs


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_path: str
    doc_title: str
    heading: str | None
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str
    index: int  # position within the document

    @property
    def embed_text(self) -> str:
        """Text used for indexing: document title and section heading prepended for context."""
        context = self.doc_title if not self.heading else f"{self.doc_title} > {self.heading}"
        return f"{context}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def make_chunk_id(doc_path: str, start_line: int, end_line: int) -> str:
    return hashlib.sha1(f"{doc_path}:{start_line}-{end_line}".encode()).hexdigest()[:10]


def chunk_document(doc: Document, *, size: int, overlap: int) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")
    lines = doc.lines
    n = len(lines)
    line_tokens = [count_tokens(line) for line in lines]
    headings: list[str | None] = []
    current: str | None = None
    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            current = m.group(1).strip()
        headings.append(current)

    chunks: list[Chunk] = []
    start = 0
    while start < n:
        end = start
        total = 0
        while end < n and (end == start or total + line_tokens[end] <= size):
            # Soft boundary: once the chunk is at least half full, stop before a new heading so
            # sections stay together instead of straddling two chunks.
            if end > start and total >= size // 2 and is_heading(lines[end]):
                break
            total += line_tokens[end]
            end += 1
        # Trim trailing blank lines from the chunk without affecting the cursor.
        last = end
        while last > start + 1 and not lines[last - 1].strip():
            last -= 1
        text = "\n".join(lines[start:last])
        if text.strip():
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.doc_path, start + 1, last),
                    doc_path=doc.doc_path,
                    doc_title=doc.title,
                    heading=headings[start] if start < n else None,
                    start_line=start + 1,
                    end_line=last,
                    text=text,
                    index=len(chunks),
                )
            )
        if end >= n:
            break
        if overlap > 0:
            back = end
            acc = 0
            while back > start + 1 and acc < overlap:
                back -= 1
                acc += line_tokens[back]
            start = max(back, start + 1)
        else:
            start = end
    return chunks


def chunk_documents(docs: list[Document], *, size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, size=size, overlap=overlap))
    return out
