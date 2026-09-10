# Accelerate JAX Performance

> GPU-compiled array math with automatic differentiation

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

JAX lets you write **NumPy-style Python code** and run it fast on GPUs without writing CUDA. It does this by:

- **NumPy on accelerators**: Use `jax.numpy` just like NumPy, but arrays live on the GPU.
- **Function transformations**:
  - `jit` → Compiles your function into fast GPU code
  - `grad` → Gives you automatic differentiation
  - `vmap` → Vectorizes your function across batches
  - `pmap` → Runs across multiple GPUs in parallel
- **XLA backend**: JAX hands your code to XLA (Accelerated Linear Algebra compiler), which fuses operations and generates optimized GPU kernels.

## What you'll accomplish

You'll set up a JAX development environment on your **hardware platform** that enables high-performance machine learning prototyping with familiar NumPy-like abstractions, GPU acceleration, and performance optimization.

## What to know before starting

**Required:**

- Comfortable with Python and NumPy programming
- Experience working in a terminal
- Experience using and building containers

**Optional:**

- General understanding of machine learning workflows and techniques
- Familiarity with CUDA versions
- Basic understanding of linear algebra (high-school level math is sufficient)

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Local Docker image `jax-playbook`; marimo on port 8080 | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- GPU access verified with `nvidia-smi`
- Port 8080 available for marimo notebook access (or map a different host port)

**Software requirements**

- Docker or a compatible container runtime: `docker --version`
- NVIDIA Container Toolkit configured (`nvidia-smi` works inside a GPU container)
- Network access to pull the base CUDA image and build the playbook container

## Ancillary files

All required assets can be found [in the playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-jax/).

- `assets/jax-intro.py` — JAX introduction notebook covering programming model differences from NumPy and performance evaluation
- `assets/numpy-som.py` — NumPy reference implementation of a self-organized map (SOM) training algorithm
- `assets/som-jax.py` — Iteratively refined JAX SOM implementations for performance comparison
- `assets/00-toc.py` — Marimo table-of-contents entry point for the tutorial
- `assets/Dockerfile` — Package dependencies and container setup
- `assets/batch-som.mp4` — Sample SOM visualization used by the notebooks

## Time & risk

- **Estimated time:** 2 HOURS (including setup, tutorial completion, and validation)
- **Risk level:** Medium
  - Package dependency conflicts can appear if the container environment drifts from the playbook Dockerfile
  - Performance validation may need tuning for your workload size and available memory
- **Rollback:** Container environments provide isolation; stop and remove containers, then rebuild from the playbook Dockerfile to reset state
- **Last Updated:** 08/03/2026
  - Set up a containerized JAX environment with marimo notebooks comparing NumPy and JAX SOM performance

## Instructions

## Step 1. Verify system prerequisites

Confirm GPU access and Docker GPU support on your hardware platform.

```bash
## Verify GPU access
nvidia-smi

## Check Docker GPU support
docker run --gpus all --rm nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04 nvidia-smi
```

Expected output should show GPU information from both `nvidia-smi` and the container run.

If you see a permission-denied error (for example, permission denied while trying to connect to the Docker daemon socket), add your user to the docker group so you do not need `sudo` for Docker commands:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Clone the playbook repository

Clone the playbook assets and open the assets directory:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd client-hardware-playbooks/nvidia/playbook-jax/assets
```

## Step 3. Build the Docker image

> [!WARNING]
> This command downloads a base image and builds a container locally for this environment.

```bash
docker build -t jax-playbook .
```

## Step 4. Launch the Docker container

Run the JAX development environment in a Docker container with GPU support and port forwarding for marimo:

```bash
docker run --gpus all --rm -it \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8080:8080 \
    jax-playbook
```

The container starts a marimo notebook server on port 8080.

## Step 5. Access the marimo interface

Open a browser to `http://localhost:8080`. From another device, use `http://<HARDWARE_IP>:8080` where `<HARDWARE_IP>` is your hardware platform's reachable address.

The interface loads a table-of-contents display and a brief introduction to marimo.

## Step 6. Complete the JAX introduction tutorial

Work through the introductory material to understand how the JAX programming model differs from NumPy.

Navigate to and complete the JAX introduction notebook, which covers:

- JAX programming model fundamentals
- Key differences from NumPy
- Performance evaluation techniques

## Step 7. Implement the NumPy baseline

Complete the NumPy-based self-organized map (SOM) implementation to establish a performance baseline.

Work through the NumPy SOM notebook to:

- Understand the SOM training algorithm
- Implement the algorithm using familiar NumPy operations
- Record performance metrics for comparison

## Step 8. Optimize with JAX implementations

Progress through the iteratively refined JAX implementations to see performance improvements.

Complete the JAX SOM notebook sections:

- Basic JAX port of the NumPy implementation
- Performance-optimized JAX version
- GPU-accelerated parallel JAX implementation
- Compare performance across all versions

## Step 9. Validate performance gains

The notebooks show how to measure performance for each SOM training implementation. JAX implementations should improve on the NumPy baseline (some substantially).

Visually inspect the SOM training output on random color data to confirm algorithm correctness.

## Step 10. Cleanup (optional)

When you are finished, stop the container (Ctrl+C in the terminal where it is running). Because the container was started with `--rm`, it is removed automatically on exit.

Optionally remove the local image:

```bash
docker rmi jax-playbook
```

## Step 11. Next steps

Apply JAX optimization techniques to your own NumPy-based machine learning code.

```bash
## Example: Profile your existing NumPy code
python -m cProfile your_numpy_script.py

## Then adapt to JAX and compare performance
```

Try adapting familiar NumPy algorithms to JAX and measure performance improvements on your hardware platform's GPU.

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every platform listed in this playbook.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| `nvidia-smi` not found | All hardware platforms | Missing NVIDIA drivers | Install NVIDIA drivers for your hardware platform |
| Container fails to access GPU | All hardware platforms | Missing or misconfigured NVIDIA Container Toolkit | Install and configure `nvidia-container-toolkit`, then restart Docker |
| "permission denied" when running docker | All hardware platforms | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| JAX only uses CPU | All hardware platforms | CUDA/JAX version mismatch in the environment | Rebuild from the playbook `Dockerfile` so JAX is installed with CUDA support |
| Port 8080 unavailable | All hardware platforms | Port already in use | Use `-p 8081:8080` (or another free host port) or stop the process using 8080 |
| Package conflicts in Docker build | All hardware platforms | Stale or modified environment pins | Rebuild from the playbook `Dockerfile` without local edits to dependency pins |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
