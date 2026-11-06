# llm-evalops is a path dependency outside this build context, so pass it as a named context:
#   docker build --build-context llm-evalops=../llm-evalops -t llm-distill:latest .
# The image serves the distilled student through `llm_distill.client.StudentClient`; run
# `make train` first so that models/ exists. It is not built in CI.

# ---- build stage ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY --from=llm-evalops pyproject.toml README.md /llm-evalops/
COPY --from=llm-evalops src /llm-evalops/src

COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---- runtime stage -------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app models ./models
COPY --chown=app:app pricing.json ./

ENV PATH="/app/.venv/bin:$PATH" \
    LLM_PROVIDER=student \
    STUDENT_BACKEND=onnx \
    HF_HUB_OFFLINE=1

USER app
CMD ["python", "-c", "from llm_distill.llm import get_client; print(get_client().provider)"]
