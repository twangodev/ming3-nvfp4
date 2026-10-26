# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.11.2@sha256:c4f5de312ee66d46810635ffc5df34a1973ba753e7241ce3a08ef979ddd7bea5 AS uv

FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2 AS evaluation
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_PYTHON_PREFERENCE=only-managed UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface HOME=/tmp
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && uv python install 3.12.13
WORKDIR /opt/ming
COPY LICENSE ./
COPY evaluation/pyproject.toml evaluation/uv.lock evaluation/
COPY docker/upstream.json docker/fetch_upstream.py docker/
RUN uv sync --project evaluation --python 3.12.13 --locked --group dev --no-cache \
    && evaluation/.venv/bin/python docker/fetch_upstream.py
COPY evaluation/ evaluation/
COPY docker/Dockerfile docker/Dockerfile
ENTRYPOINT ["/opt/ming/evaluation/.venv/bin/python"]
CMD ["evaluation/prepare.py", "--help"]

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 AS quantize
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=evaluation /opt/python /opt/python
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_PYTHON_PREFERENCE=only-managed UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface HOME=/tmp \
    XDG_CACHE_HOME=/work/.cache TRITON_CACHE_DIR=/work/.cache/triton
# CUDA is supplied by the pinned base; only Ubuntu utilities are added here.
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/cuda*.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ffmpeg libsndfile1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/ming
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --extra data --python 3.12.13 --no-dev --no-cache --no-install-project
COPY LICENSE README.md .python-version ./
COPY src/ src/
RUN uv pip install --python .venv/bin/python --no-deps --no-cache -e .
COPY recipes/ recipes/
COPY evaluation/diagnostics/hf_reference.py evaluation/diagnostics/hf_reference.py
COPY docker/Dockerfile docker/Dockerfile
ENTRYPOINT ["/opt/ming/.venv/bin/ming-quant"]
CMD ["--help"]

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS serving
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=evaluation /opt/python /opt/python
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_PYTHON_PREFERENCE=only-managed UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface HOME=/tmp \
    XDG_CACHE_HOME=/work/.cache TRITON_CACHE_DIR=/work/.cache/triton \
    VLLM_PLUGINS=ming_nvfp4,vllm_omni_register_models \
    MING_NVFP4_FUSED_INPUT=0 MING_NVFP4_FUSED_OUTPUT=0
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/cuda*.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ffmpeg libsndfile1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/ming
COPY serving/pyproject.toml serving/uv.lock serving/measured-runtime.txt serving/
RUN uv sync --project serving --locked --python 3.12.13 --no-dev --no-cache --no-install-project \
    && uv pip install --python serving/.venv/bin/python --no-deps --no-cache -r serving/measured-runtime.txt
COPY serving/ serving/
RUN uv pip install --python serving/.venv/bin/python --no-deps --no-cache -e serving
COPY --from=evaluation /opt/ming/evaluation /opt/ming/evaluation
COPY --from=evaluation /etc/gitconfig /etc/gitconfig
COPY LICENSE ./
COPY docker/Dockerfile docker/Dockerfile
ENV PATH=/opt/ming/serving/.venv/bin:$PATH
ENTRYPOINT ["/opt/ming/serving/.venv/bin/python", "-m", "vllm.entrypoints.cli.main", "serve"]
CMD ["--help"]
