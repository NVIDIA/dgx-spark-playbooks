# Run ML Interatomic Potentials and GPU Molecular Dynamics on DGX Spark

> MACE and GROMACS on GB10 — and the two environment fixes the stock aarch64 CUDA stack needs

*This playbook distills the GPU/quantum compute core of the [NovoMCP](https://novomcp.com) engine's DGX Spark port — NovoMCP is the open computational‑chemistry engine for drug discovery and materials science. Full write‑up: [What actually breaks when you port a chemistry stack to DGX Spark](https://www.novomcp.com/news/what-actually-breaks-when-you-port-a-chemistry-stack-to-dgx-spark).*

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Computational‑chemistry GPU software splits into two stacks that port to Grace very differently, and knowing which is which saves a lot of time:

- **PyTorch‑based ML interatomic potentials (MACE, ANI‑2x).** Install from **stock CUDA‑13 wheels** — the wheels ship `sm_120` kernels, and those **forward‑compat to GB10's `sm_121`**, so no source build is needed. The catch is two runtime issues in the *bundled* aarch64 libraries that crash the stack before your code runs, until you set two environment variables.
- **Compiled‑CUDA molecular dynamics (GROMACS).** Build from source with the `sm_121` arch flag and ARM SIMD, using the **2026.x** line (the 2025.x line fails to link its PME‑GPU kernels against CUDA 13 — a version issue, not an architecture one).

This playbook gets both running on a single DGX Spark, validates each with a known‑good number, and puts the two Grace‑specific fixes up front so you don't lose a day rediscovering them.

## What you'll accomplish

- Run a **MACE‑MPA‑0** single‑point and relaxation on the GPU (ethanol, validated against a reference energy).
- Build **GROMACS 2026.x** for GB10 and run an **SPC/E water MD scaling benchmark** from ~100 K to multi‑million atoms — with an optional ~28 M‑atom run whose *host‑side* setup needs ~69 GB RAM, comfortably absorbed by GB10's 128 GB unified pool. (The GPU‑resident footprint of this MD is modest — ~11 GB — so it is not a VRAM‑capacity demonstration; the benefit here is the large unified host memory.)

## What to know before starting

**Required:**

- Comfortable in a terminal and with Python
- Experience building and running containers

**Optional:**

- Familiarity with molecular dynamics or ML interatomic potentials
- Awareness of CUDA versions and GPU compute capabilities (`sm_*`)

## Supported hardware platforms

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Two local Docker images (`mace-spark`, `gromacs-spark`); `OMP_NUM_THREADS` + jemalloc set in the MACE image | — |

## Prerequisites

**Hardware requirements**

- DGX Spark (GB10) — GPU access verified with `nvidia-smi`
- Disk for two container images (the GROMACS build image is several GB)

**Software requirements**

- Docker or a compatible container runtime: `docker --version`
- NVIDIA Container Toolkit configured (`nvidia-smi` works inside a GPU container)
- Network access to pull base images and Python/CUDA packages

## Ancillary files

All assets are in this playbook's [`assets/`](assets/) directory:

- `assets/Dockerfile.mace` — MACE / ML‑potential environment: stock CUDA‑13 PyTorch wheels + the two Grace fixes + the Triton C‑compiler/headers (`gcc`, `libc6-dev`) baked in
- `assets/validate_mace.py` — MACE‑MPA‑0 single‑point and relaxation on ethanol, with the expected energy
- `assets/Dockerfile.gromacs` — GROMACS 2026.x built for `sm_121` + ARM SIMD
- `assets/water_benchmark.sh` — SPC/E water MD scaling benchmark (energy‑minimize → production MD, per size)
- `assets/topol_template.top` — SPC/E water topology used by the benchmark

## Time & risk

- **Estimated time:** ~1.5 HOURS (the GROMACS source build is the long part, ~30–45 min; everything else is minutes)
- **Risk level:** Low–Medium
  - The two MACE fixes are deterministic; once set, the stack is stable
  - A source build can drift if base‑image versions move — pin them if you reuse this
  - The largest benchmark size is host‑memory‑heavy at setup by design; scale it to your run
- **Rollback:** Everything is containerized. Stop/remove containers and delete the two images to reset.
- **Provenance:** These recipes were verified end‑to‑end on **two** independent GB10 boxes (CUDA 13.0; drivers 580.14x and 580.173.02; DGX OS), with the MACE and benchmark numbers below reproduced on the second. A retail Spark's DGX‑OS/driver may differ — pin your own versions beside any number you reuse.
- **Last Updated:** 10/2026

## Instructions

## Step 1. Verify GPU access

```bash
## Host GPU
nvidia-smi

## GPU visible inside a container
docker run --gpus all --rm nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04 nvidia-smi
```

Both should print GPU information. If you hit a Docker permission error, add your user to the `docker` group:

```bash
sudo usermod -aG docker $USER && newgrp docker
```

## Step 2. Understand the two Grace fixes (read before building)

The ML‑potential stack installs cleanly from stock CUDA‑13 wheels, but **two runtime failures** in the bundled aarch64 libraries will stop it before your code runs. Both are handled by the `Dockerfile.mace` in this playbook; they're called out here because they're non‑obvious and apply to any PyTorch/e3nn workload on Grace:

1. **`import torch` heap‑corrupts unless `OMP_NUM_THREADS` is bounded.** PyTorch's bundled `libnvpl_lapack_lp64_gomp` and `libarm_compute` spawn unbounded OpenMP threads at static‑init on the 20‑core chip and corrupt the allocator (`malloc(): corrupted top size`) before any user code runs. Any bounded value fixes it:

   ```bash
   export OMP_NUM_THREADS=8
   ```

2. **e3nn's GPU forward pass trips a glibc‑allocator corruption** (`free(): corrupted unsorted chunks`), reliably on a multi‑step relaxation. Preloading jemalloc clears it:

   ```bash
   export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
   ```

Neither is a CUDA or kernel fault — both are allocator/threading interactions in the stock aarch64 stack.

## Step 3. Build the MACE (ML‑potential) image

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-novomcp-spark/assets
docker build -f Dockerfile.mace -t mace-spark .
```

The image installs stock CUDA‑13 PyTorch wheels (no source build) plus `mace-torch`, and bakes in `OMP_NUM_THREADS` and the jemalloc preload.

## Step 4. Validate MACE on the GPU

```bash
docker run --gpus all --rm mace-spark python validate_mace.py
```

Expected output: an ethanol single‑point energy of about **−46.21 eV** and a relaxed energy near **−46.28 eV** (MACE‑MPA‑0, float32 default). A single‑point within a few meV confirms the kernels run correctly on `sm_121`. Running the same single‑point in **float64** lands on **−46.2554 eV**, matching the published reference exactly (0.00 meV) — the float64 path is the one to use for geometry work.

## Step 5. Build GROMACS 2026.x for GB10

> [!WARNING]
> This compiles GROMACS from source (CUDA + ARM SIMD) and takes ~30–45 minutes.

```bash
docker build -f Dockerfile.gromacs -t gromacs-spark .
docker run --gpus all --rm gromacs-spark gmx --version | grep -iE "GROMACS version|GPU support"
```

The build sets `GMX_CUDA_TARGET_SM=121` and `GMX_SIMD=ARM_NEON_ASIMD`, and uses the **2026.x** line. (The 2025.x line fails to link its PME‑GPU kernels against CUDA 13 — see Troubleshooting.)

## Step 6. Run the water MD scaling benchmark

```bash
docker run --gpus all --rm -v "$PWD:/work" -w /work gromacs-spark bash water_benchmark.sh
```

The script builds SPC/E water boxes at several sizes, energy‑minimizes each, and runs production MD with nonbonded + PME + update all on the GPU. GB10 numbers measured over 3 repeats per size (mean ± sd), with GPU power sampled on the host during each production run:

| Atoms | ns/day (mean ± sd) | Power (mean / peak) |
| :---- | :---- | :---- |
| ~98,000 | 220.3 ± 3.1 | 64 W / 71 W |
| ~975,000 | 19.21 ± 0.02 | 66 W / 71 W |
| ~3,875,000 | 4.244 ± 0.008 | 56 W / 63 W |

Reproducibility is tight (sd well under 2%). Note the draw *drops* at the largest size — the run becomes PME/bandwidth‑bound, so the GPU's compute units idle more and pull less power.

To exercise the large unified host memory, uncomment the largest size in `water_benchmark.sh` (box 66 nm, ~28.3 M atoms). Its **host‑side** setup (solvate + grompp) drives system memory to ~69 GB, which GB10's 128 GB unified pool absorbs without a separate‑host staging step. This is a *host*‑memory point, not a VRAM one: the GPU‑resident footprint of this MD stays modest (~11 GB, measured on a 48 GB L40S running the same system), so it is **not** a case of exceeding a datacenter card's VRAM. Watch both with `nvidia-smi` and `free -g` in another terminal.

## Step 7. Cleanup (optional)

```bash
docker rmi mace-spark gromacs-spark
```

## Step 8. Next steps

The same pattern extends to the rest of a chemistry stack on Grace:

- **xTB / CREST / AmberTools** — install from conda‑forge's `linux-aarch64` channel (the PyPI builds are x86‑only).
- **ANI‑2x** — add `torchani` to the MACE image (install `--no-deps`; it runs on the pure‑torch path).
- **AutoDock‑GPU** — build from source with `make DEVICE=CUDA TARGETS=121`.
- **sTDA excited states** — build from source with meson + openblas (see Troubleshooting for the three packaging gotchas).

And beyond the individual tools: these are the GPU/quantum compute primitives behind the [NovoMCP engine](https://novomcp.com) — the open computational‑chemistry engine — which orchestrates MACE, GROMACS, AutoDock‑GPU and quantum tools as an autonomous discovery funnel on a single box.

## Troubleshooting

## Common issues

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| `import torch` aborts with `malloc(): corrupted top size` | DGX Spark | PyTorch's bundled NVPL / ARM‑Compute spawn unbounded OpenMP threads at init and corrupt the allocator | Set `OMP_NUM_THREADS` to any bounded value (baked into `Dockerfile.mace`) |
| `free(): corrupted unsorted chunks` during a MACE relaxation | DGX Spark | glibc‑allocator interaction in e3nn's GPU forward pass on aarch64 | `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2` (baked into `Dockerfile.mace`) |
| MACE aborts on first GPU kernel: Triton `Failed to find C compiler` or `gcc … cuda_utils … returned non‑zero` | DGX Spark (any slim base) | the cu130 torch wheel pulls in Triton, which JIT‑compiles a CUDA launcher stub on first kernel launch and needs `gcc` **plus** the standard C headers; `python:3.12-slim` ships neither, and `--no-install-recommends` drops `libc6-dev` even when you add `gcc` | Install `gcc` **and** `libc6-dev` explicitly (both baked into `Dockerfile.mace`). This is a packaging gap, not a Grace/arch issue. |
| GROMACS link fails: `undefined reference to pme_solve_kernel` | DGX Spark | GROMACS 2025.x does not link its PME‑GPU kernels against CUDA 13 | Build the **2026.x** line |
| MACE / torch runs on CPU only | DGX Spark | wrong wheel index | Install torch from the CUDA‑13 wheel index; do not rebuild from source |
| `pip install xtb` (or crest) fails or installs an x86 binary | DGX Spark | PyPI builds are x86‑only | Install from conda‑forge `linux-aarch64` |
| sTDA `stda` not found after a successful build | DGX Spark | the build produces a binary named `std2`; its `libcint.so` is off the path; `XTB4STDAHOME` is unset | Install `std2` as `stda`; copy `libcint.so` onto the library path + `ldconfig`; set `XTB4STDAHOME` to the parameter directory |
| Benchmark NaNs at step 0 | DGX Spark | freshly tiled water box has close contacts | Energy‑minimize first (the benchmark script does this with a flexible‑water steepest‑descent step) |
| `nvidia-smi` not found / container has no GPU | All hardware platforms | Missing drivers or NVIDIA Container Toolkit | Install NVIDIA drivers; install/configure `nvidia-container-toolkit` and restart Docker |

---

*Contributed by [NovoMCP](https://novomcp.com), the open computational‑chemistry engine.*
