# Astra backend image — serves both the API (uvicorn) and the worker (arq);
# they differ only by the compose `command`.
#
# Build context is this repo (Astra-Back). The compression engine (`astra/`) is
# vendored into the repo, so the image is fully self-contained — no sibling
# checkout or manual copy step is required. `.dockerignore` keeps caches,
# local databases and storage out of the build context.
FROM python:3.13-slim

# libgomp1: required by onnxruntime/torch native kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only torch wheels — avoids pulling multi-GB CUDA builds. Pre-installed
# before the app so the `.[prod,engine]` step below reuses this wheel.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4.0"

# Format readers/converters so UPLOADED Keras/.tflite/.pb/.safetensors/.mlmodel/
# .gguf models take the REAL pipeline instead of failing at import time. On Linux
# plain `tensorflow` is the CPU build (CUDA wheels come only from the
# `tensorflow[and-cuda]` extra) and pulls keras 3 + h5py transitively. Pinned to
# the versions validated on the dev host. Its own cached layer (independent of
# app code) so editing the backend doesn't trigger a TensorFlow reinstall.
RUN pip install \
        "tensorflow==2.21.0" \
        "tf2onnx==1.17.0" \
        "h5py==3.14.0" \
        "safetensors==0.7.0" \
        "coremltools==9.0" \
        "gguf==0.19.0"

# Install the backend + vendored engine + production extras. `app/` and the
# vendored `astra/` package are both copied; `.[prod,engine]` installs the
# production infra plus the engine's third-party deps (onnx/onnxruntime/optuna/
# sklearn). CPU torch is already present from the layer above.
COPY . /app/
RUN pip install ".[prod,engine]"

# Run as a NON-ROOT user. The worker deserializes uploaded models
# (torch/pickle/joblib in the vendored engine), so a deserialization exploit
# must not land as uid 0. All pip installs above run as root; drop privileges
# here, after the app is installed. The worker's scratch dir must be writable.
RUN useradd --create-home --uid 10001 astra \
    && mkdir -p /tmp/astra-work \
    && chown -R astra:astra /app /tmp/astra-work
USER astra

EXPOSE 8000

# Default to the API; the worker service overrides this in compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
