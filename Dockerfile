# ── J-Wash Docker image ──────────────────────────────────────────────────────
# Requires an NVIDIA GPU with CUDA 12.4+ drivers on the host.
#   docker build -t j-wash .
#   docker run --gpus all -p 8381:8381 -v ./data:/app/data -v ./hf_cache:/app/hf_cache j-wash

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /app

COPY requirements.txt .
COPY requirements/jacobian-lens.commit requirements/jacobian-lens.commit
COPY scripts/bootstrap_jlens.py scripts/bootstrap_jlens.py

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

RUN python scripts/bootstrap_jlens.py

RUN pip install --no-cache-dir -r requirements.txt

COPY ui/package.json ui/package-lock.json ui/
RUN cd ui && npm ci

COPY . .

RUN cd ui && npm run build

ENV HF_HOME=/app/hf_cache
ENV JWASH_DATA_DIR=/app/data

EXPOSE 8381

VOLUME ["/app/data", "/app/hf_cache", "/app/lenses"]

ENTRYPOINT ["python3.11", "-X", "utf8", "run.py", "--host", "0.0.0.0"]
