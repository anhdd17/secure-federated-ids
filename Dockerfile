FROM python:3.11-slim

# torch is installed separately from requirements.txt because the wheel depends on
# the CUDA version. torch 2.15.0 on Linux ships only cu132 / cu134 / cpu.
# If the GPU server's driver is older (check `nvidia-smi`), lower both args together, e.g.
#   docker build --build-arg TORCH_VERSION=<older> --build-arg TORCH_CUDA=<cuXXX> .
ARG TORCH_VERSION=2.15.0
ARG TORCH_CUDA=cu132

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "scripts/gpu_sanity.py"]
