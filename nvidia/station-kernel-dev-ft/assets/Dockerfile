# Development container for profiler-driven kernel optimization.
# Based on NVIDIA's PyTorch NGC image, which includes PyTorch, CUDA, and Triton.
FROM nvcr.io/nvidia/pytorch:26.01-py3

# Install Python dependencies for model loading, profiling, and benchmarking.
# Triton is already included in the NGC PyTorch image — no separate install needed.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
