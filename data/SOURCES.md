# Data sources

| Item | Source | Licence | Retrieved |
|---|---|---|---|
| `docs-rag-snapshot/corpus/` | `docs-rag/data/corpus/`, a synthetic 40-page engineering handbook written for that project | MIT (the `docs-rag` repository licence) | see `docs-rag-snapshot/SNAPSHOT.json` |
| `docs-rag-snapshot/golden/` | `docs-rag/evals/golden/`, 199 template-generated and 23 hand-written questions | MIT | see `docs-rag-snapshot/SNAPSHOT.json` |
| `pairs.jsonl` | derived by `scripts/build_dataset.py` from the two rows above | MIT | regenerated on demand |
| `teacher_labels.jsonl` | derived by `scripts/label_teacher.py` (gitignored) | MIT | regenerated on demand |

`SNAPSHOT.json` records the `docs-rag` commit SHA and the date the snapshot was taken. Refresh it
with `uv run python scripts/build_dataset.py --refresh ../docs-rag`.

No personal data and no third-party text is involved. Model weights come from the Hugging Face
hub (`cross-encoder/ms-marco-MiniLM-L-6-v2`, Apache-2.0) and are downloaded to the default
Hugging Face cache, not to this repository.
