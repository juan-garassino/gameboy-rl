# gameboy-rl training image (archetype B). Build it in CI (no workflow in this repo yet) — never on the laptop.
# No ROM ever goes into the image: the pod pulls it from private GCS and deletes it there (entrypoint.sh).
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg tini python3 \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 -s /bin/bash researcher \
    && mkdir -p /app /workspace && chown -R researcher:researcher /app /workspace
USER researcher
ENV PATH="/home/researcher/.local/bin:/app/.venv/bin:$PATH" UV_PYTHON_INSTALL_DIR=/home/researcher/.python
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY --chown=researcher:researcher pyproject.toml uv.lock ./
# pyproject pins CPU torch 2.5 for laptops/CI; the pod gets the CUDA 12.4 wheel of the same line
# (torch 2.5 has no cu126 build; cu124 runs on any recent 12.x driver).
RUN uv python install 3.12 \
    && uv sync --frozen --no-install-project --no-install-package torch \
    && uv pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.5,<2.6"

COPY --chown=researcher:researcher src ./src
COPY --chown=researcher:researcher runpod ./runpod
# PWhiddy's env + start states (RAM snapshots, no ROM data). v2/runs/ (the pretrained zip) is dockerignored:
# the pod pulls init/poke_26214400.zip from GCS instead.
COPY --chown=researcher:researcher v2 ./v2
COPY --chown=researcher:researcher *.state ./
RUN uv sync --frozen --inexact --no-install-package torch \
    && chmod +x runpod/entrypoint.sh runpod/scripts/*.sh

ENV WORKSPACE_DIR=/workspace APP_DIR=/app MODE=train
ENTRYPOINT ["/usr/bin/tini", "--", "/app/runpod/entrypoint.sh"]
