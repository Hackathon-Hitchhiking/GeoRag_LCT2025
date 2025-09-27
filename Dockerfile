FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

ENV UV_CACHE_DIR=/tmp/uvcache
ENV HF_HOME=/models \
    HF_HUB_CACHE=/models

WORKDIR /app

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        zlib1g-dev \
        curl \
        ca-certificates \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        ffmpeg; \
    rm -rf /var/lib/apt/lists/*; \
    useradd -m -u 10001 appuser; \
    mkdir -p "${UV_CACHE_DIR}" /models; \
    chown -R appuser:appuser /app /home/appuser "${UV_CACHE_DIR}" /models

USER 10001:10001

RUN python -m pip install --no-cache-dir --user uv

COPY --chown=10001:10001 pyproject.toml uv.lock /app/
RUN uv sync --no-dev --frozen

COPY --chown=10001:10001 ./app /app/app
COPY --chown=10001:10001 ./entrypoint.sh /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
