# syntax=docker/dockerfile:1
#
# IMPORTANT — read before building:
#
#   1. Ollama runs on the HOST, not inside this container.
#      Set OLLAMA_HOST=http://host.docker.internal:11434 so the container
#      can reach it. The docker-compose.yml wires this up automatically.
#
#   2. The ChromaDB index lives on disk (embedded, not a separate service).
#      Mount ./data as a volume so the container can read the pre-built index:
#        -v $(pwd)/data:/app/data
#      Run scripts/02_build_index.py on the host BEFORE starting the container.
#
#   3. OLLAMA_HOST must include the http:// scheme, e.g.
#        OLLAMA_HOST=http://host.docker.internal:11434   (Docker Desktop / Linux)
#        OLLAMA_HOST=http://172.17.0.1:11434             (fallback for Linux)

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv so the runtime stage stays clean.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
# Install project dependencies into the venv (no editable install for Docker).
RUN pip install --upgrade pip \
    && pip install --no-cache-dir .

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the venv from builder — no build tools needed at runtime.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only what the application needs.
COPY src/       ./src/
COPY configs/   ./configs/
COPY .env.example ./.env.example

# Non-root user for security.
RUN useradd --create-home --no-log-init --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
