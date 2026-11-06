"""Single source of configuration, read from the environment and an optional .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["fake", "openrouter", "anthropic", "openai", "student"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Providers. ``llm_provider`` left unset means "auto": openrouter > anthropic > openai > fake.
    llm_provider: LLMProvider | None = None
    llm_model: str = "anthropic/claude-sonnet-5.5"
    judge_model: str = "openai/gpt-5.4-mini"
    cheap_model: str = "anthropic/claude-haiku-5.5"
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_app_name: str = "llm-distill"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Data
    snapshot_dir: Path = Path("data/docs-rag-snapshot")
    pairs_path: Path = Path("data/pairs.jsonl")
    teacher_labels_path: Path = Path("data/teacher_labels.jsonl")
    chunk_size: int = Field(default=128, ge=32)
    chunk_overlap: int = Field(default=16, ge=0)

    # Student
    student_base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    student_dir: Path = Path("models/s1-minilm")
    onnx_path: Path = Path("models/s1-minilm/model.onnx")
    tfidf_path: Path = Path("models/s2-tfidf.joblib")
    # "torch" | "onnx" | "tfidf"; used by StudentClient.
    student_backend: Literal["torch", "onnx", "tfidf"] = "onnx"
    student_threshold: float = 0.5

    # Cost model for the students: CPU-seconds are priced at this hourly rate (USD). 0.10 is
    # a round figure for a small 2 vCPU cloud instance; change it to your own rate.
    cpu_usd_per_hour: float = 0.10
    pricing_path: Path = Path("pricing.json")

    def has_key(self, provider: str) -> bool:
        return {
            "openrouter": bool(self.openrouter_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "openai": bool(self.openai_api_key),
        }.get(provider, False)

    def resolved_llm_provider(self) -> LLMProvider:
        """Explicit LLM_PROVIDER wins; otherwise pick the first backend that has a key."""
        if self.llm_provider is not None:
            return self.llm_provider
        for candidate in ("openrouter", "anthropic", "openai"):
            if self.has_key(candidate):
                return candidate  # type: ignore[return-value]
        return "fake"


@lru_cache
def get_settings() -> Settings:
    return Settings()
