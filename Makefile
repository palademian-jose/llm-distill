.PHONY: help setup test lint format data label baseline train export compare gate demo clean

UV ?= uv

help:             ## list targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/\t/'

setup:            ## install Python 3.12 toolchain and dependencies (CPU torch)
	$(UV) python install 3.12
	$(UV) sync
	@test -f .env || cp .env.example .env

test:             ## run the test suite (offline, no model download)
	LLM_PROVIDER=fake $(UV) run pytest

lint:             ## ruff lint + format check
	$(UV) run ruff check src tests scripts
	$(UV) run ruff format --check src tests scripts

format:           ## auto-format
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

data:             ## build data/pairs.jsonl from the committed docs-rag snapshot
	$(UV) run python scripts/build_dataset.py

label:            ## label every pair with the teacher (fake provider unless a key is set)
	LLM_PROVIDER=$${LLM_PROVIDER:-fake} $(UV) run python scripts/label_teacher.py

baseline:         ## fit the S2 TF-IDF baseline
	$(UV) run python scripts/train_baseline.py

train:            ## fine-tune S1 (downloads the base model once; CUDA used if torch has it)
	$(UV) run python scripts/train.py
	$(UV) run python scripts/export_onnx.py

export:           ## export S1 to ONNX and check parity
	$(UV) run python scripts/export_onnx.py

compare:          ## teacher vs students on the test split -> reports/distillation.md
	LLM_PROVIDER=$${LLM_PROVIDER:-fake} $(UV) run python scripts/compare.py --readme README.md

gate:             ## regression gate against reports/baseline.json (evalkit gate)
	$(UV) run evalkit gate --baseline reports/baseline.json --candidate reports/eval-latest.json --config evals.yaml --md reports/gate.md

demo:             ## offline demo: teacher vs student verdicts on BM25 results
	LLM_PROVIDER=fake $(UV) run python scripts/build_dataset.py > /dev/null
	LLM_PROVIDER=fake $(UV) run python scripts/label_teacher.py > /dev/null
	$(UV) run python scripts/train_baseline.py > /dev/null
	$(UV) run python scripts/demo.py

clean:
	rm -rf .pytest_cache .ruff_cache models data/teacher_labels.jsonl
