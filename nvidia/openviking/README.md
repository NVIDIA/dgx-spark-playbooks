# OpenViking with Ollama and NVIDIA cuVS

> Run an all-local agent context database with GPU-accelerated dense search on DGX Spark.

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Operating and tuning the deployment](#operating-and-tuning-the-deployment)
- [Cleanup and rollback](#cleanup-and-rollback)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

[OpenViking](https://github.com/volcengine/OpenViking) is a context database for AI agents. It
organizes agent memories, knowledge, and skills behind a filesystem-like interface and semantic
retrieval APIs.

This playbook keeps the complete workload on one DGX Spark:

- [Ollama](https://ollama.com/) serves a local embedding model and a local vision-language model.
- OpenViking stores canonical records, metadata, scalar and path indexes, sparse data, and recovery
  state in its native local backend.
- [NVIDIA cuVS](https://docs.nvidia.com/cuvs/) provides an optional GPU snapshot for dense top-k
  search. It is a vector index library, not a replacement for the complete OpenViking database.

The configuration deliberately uses `backend: "local"` with `cuvs.auto_enable: true`. OpenViking
uses cuVS only when the GPU is available and its memory admission check passes. Otherwise, the
same query stays on the native index. This is safer on a unified-memory system where Ollama and
vector search share the 128 GB pool.

## What you'll accomplish

You will have:

- OpenViking 0.4.17 running as a user service on `127.0.0.1:1933`.
- Ollama 0.33.2 serving `qwen3-embedding:0.6b` and the vision-capable `qwen3.8:27b` locally.
- Exact cuVS brute-force search using float32 vectors, with native fallback and background snapshot
  rebuilds.
- A reproducible Python 3.12 / Linux aarch64 environment installed from a fully pinned,
  hash-checked lock.
- An end-to-end test that makes real embedding and VLM calls, writes a known OpenViking record,
  retrieves it, and confirms through request telemetry that the dense query used cuVS.

## When cuVS helps

cuVS is most useful for a large, dense, read-heavy corpus where GPU parallelism can amortize index
build and launch overhead. Keep the native route for small collections, frequent writes, selective
path/scalar filters, or sparse/hybrid-heavy retrieval.

| Workload | Starting point |
| --- | --- |
| Small corpus or frequent mutations | Native local search |
| Mixed workload sharing memory with a local VLM | This playbook: local + auto-cuVS |
| Large dense corpus, exact recall required | cuVS brute-force after an A/B benchmark |
| Large dense corpus where approximate recall is acceptable | Evaluate CAGRA with Recall@K |

Do not infer a performance win from the smoke test. For a small vector count, native search can be
faster because GPU startup and snapshot construction dominate. Native OpenViking uses an int8
representation; this playbook's cuVS snapshot uses float32, so compare Recall@K as well as latency
and memory when benchmarking.

## Prerequisites

- DGX Spark with current DGX OS (Ubuntu 24.04, Linux aarch64, glibc 2.39 or newer).
- Current NVIDIA driver with CUDA 13 support and a working `nvidia-smi`.
- Python 3.12 with the `venv` module.
- `curl`, `git`, `sha256sum`, and a working `systemd --user` session.
- At least 35 GB of free disk space for Ollama, the models, and the Python environment.
- Internet access to GitHub, PyPI, NVIDIA's Python index, and the Ollama model registry during setup.

## Time and risk

- **Duration**: About 30-60 minutes. Model download speed is the largest variable.
- **Risk level**: Medium. The setup installs Ollama system-wide, creates a user service, downloads
  large model files, and creates `~/.openviking/ov.conf` plus `~/.openviking/data`.
- **Memory**: The VLM, embedding model, CUDA runtime, and cuVS snapshot share DGX Spark unified
  memory. The provided configuration reserves 8 GiB outside auto-cuVS admission.
- **Rollback**: Stop and remove the OpenViking user service. Data and models are preserved unless
  you explicitly delete them in the cleanup steps.
- **Last Updated**: 2026-08-30

## Pinned versions

| Component | Version |
| --- | --- |
| OpenViking | [0.4.17](https://github.com/volcengine/OpenViking/releases/tag/v0.4.17) |
| Ollama | [0.33.2](https://github.com/ollama/ollama/releases/tag/v0.33.2) |
| cuVS / RAFT / RMM | 26.6.0 |
| CuPy | 14.1.1 |
| CUDA Python toolkit meta-package | 13.0.3.0 |
| Python | 3.12 |
| Embedding | [`qwen3-embedding:0.6b`](https://ollama.com/library/qwen3-embedding:0.6b) (1024 dimensions) |
| VLM | [`qwen3.8:27b`](https://ollama.com/library/qwen3.8:27b) |

The RAPIDS 26.06 stack is kept on the CUDA 13.0 dependency line. In particular,
`nvidia-nvjitlink` is pinned to 13.0.88 so dependency resolution cannot silently move the
environment to a newer CUDA minor release.

## Instructions

## Step 1. Verify the DGX Spark environment

Run these commands on the Spark:

```bash
uname -m
ldd --version | head -n 1
python3 --version
nvidia-smi
```

Expected results:

- `uname -m` prints `aarch64`.
- glibc is 2.39 or newer and Python is 3.12.
- `nvidia-smi` lists the NVIDIA GPU and reports a driver that supports CUDA 13.

Also confirm that no unexpected process is consuming most of unified memory:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
df -h "$HOME"
```

## Step 2. Clone this repository

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks.git
cd dgx-spark-playbooks/nvidia/openviking/assets
```

## Step 3. Install the pinned Ollama release

If Ollama 0.33.2 is already installed and running, skip the installer and continue with the model
pulls below.

Download the versioned upstream installer and verify it before execution:

```bash
curl --fail --location \
  https://github.com/ollama/ollama/releases/download/v0.33.2/install.sh \
  --output /tmp/ollama-install-0.33.2.sh
echo "25f64b810b947145095956533e1bdf56eacea2673c55a7e586be4515fc882c9f  /tmp/ollama-install-0.33.2.sh" \
  | sha256sum --check
OLLAMA_VERSION=0.33.2 sh /tmp/ollama-install-0.33.2.sh
```

The upstream installer uses `sudo` to install Ollama and create its system service. Verify the
service and version:

```bash
systemctl status ollama --no-pager
ollama --version
curl --fail http://127.0.0.1:11434/api/tags
```

Pull the local embedding model and VLM:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.8:27b
ollama list
```

The two exact model names must appear in `ollama list` before continuing.

## Step 4. Install and start OpenViking

Review the supplied configuration before installing it:

```bash
python3 -m json.tool ov.conf >/dev/null
sed -n '1,240p' ov.conf
```

Then run the setup script:

```bash
./setup.sh
```

The script:

1. Refuses non-aarch64 Linux, Python versions other than 3.12, glibc older than 2.39, a missing
   GPU/user systemd session, or an Ollama client/server version other than 0.33.2.
2. Installs only prebuilt wheels from the hash-checked dependency lock under
   `~/.local/share/openviking-cuvs/.venv`.
3. Writes `~/.openviking/ov.conf` only when it is absent or already identical. It never silently
   overwrites an existing configuration.
4. Runs `openviking-server doctor` and `pip check`.
5. Installs and starts `openviking-cuvs.service` in the user systemd manager.

The lock uses the official PyPI and NVIDIA indexes by default. If those endpoints are inaccessible
from a restricted network, optional mirrors can be supplied explicitly; the recorded SHA256 hashes
still have to match:

```bash
OPENVIKING_PYPI_INDEX_URL=https://your-pypi-mirror.example/simple \
OPENVIKING_NVIDIA_PYPI_INDEX_URL=https://your-nvidia-mirror.example/simple \
./setup.sh
```

Do not substitute an unverified mirror or remove `--require-hashes` from the setup script.

## Step 5. Run the full end-to-end check

```bash
~/.local/share/openviking-cuvs/.venv/bin/python ./verify.py
```

The check is intentionally stronger than `GET /health`. It verifies:

- CUDA, CuPy, cuVS, and the NVIDIA GPU are usable from the pinned environment.
- Ollama returns one 1024-dimensional embedding and identifies a generated red image through the
  VLM route.
- OpenViking reports healthy model state.
- A known record can be written with vector processing and retrieved by semantic search.
- Search telemetry reports `vector.cuvs.routes.cuvs >= 1` and a non-zero cuVS index size.

A successful run ends with:

```text
OPENVIKING_CUVS_E2E_OK
```

Syntax, lock, and schema checks can be run on another machine, but they do not establish GPU
compatibility. Only this successful E2E run on a DGX Spark proves that the CUDA 13 wheels load,
both Ollama models answer, OpenViking writes and retrieves a vector, and the request takes the cuVS
route.

Auto mode may initially route a query to native search while the background GPU snapshot is being
built. The verifier retries for up to three minutes and only succeeds after a real cuVS route. It
fails instead of treating a native fallback or an empty result as proof of GPU search.

## Step 6. Connect from another computer

OpenViking and Ollama remain bound to loopback. Forward only the OpenViking port over SSH:

```bash
ssh -N -L 1933:127.0.0.1:1933 <username>@<spark-address>
```

From the client computer, verify the tunnel:

```bash
curl --fail http://127.0.0.1:1933/health
```

Do not expose port 1933 or Ollama port 11434 directly to an untrusted LAN or the public internet.
Use OpenViking authentication in addition to a protected tunnel when multiple users share a Spark.

## Operating and tuning the deployment

Inspect service and data status:

```bash
systemctl --user status openviking-cuvs --no-pager
journalctl --user -u openviking-cuvs -n 100 --no-pager
curl --fail http://127.0.0.1:1933/api/v1/observer/vikingdb
curl --fail http://127.0.0.1:1933/api/v1/observer/models
ollama ps
nvidia-smi
```

The provided `brute_force` / float32 configuration is the exact-search baseline. Evaluate CAGRA
only on representative data and only after recording native and brute-force Recall@K, p50/p95/p99
latency, throughput, build time, peak memory, and behavior during mutations. To try it, stop the
service, back up `~/.openviking/ov.conf`, change `algorithm` to `cagra`, then restart and rerun the
E2E check:

```bash
systemctl --user restart openviking-cuvs
~/.local/share/openviking-cuvs/.venv/bin/python ./verify.py
```

For a small corpus or a write-heavy deployment, disabling `auto_enable` and staying on the native
local route is usually the simpler choice.

## Cleanup and rollback

Stop and remove only the OpenViking user service and Python environment:

```bash
systemctl --user disable --now openviking-cuvs
rm "$HOME/.config/systemd/user/openviking-cuvs.service"
systemctl --user daemon-reload
rm -rf "$HOME/.local/share/openviking-cuvs"
```

The commands above preserve OpenViking data and configuration. To remove them too, first inspect
the exact paths, then delete them explicitly:

```bash
du -sh "$HOME/.openviking/data" "$HOME/.openviking/ov.conf"
rm -rf "$HOME/.openviking/data"
rm "$HOME/.openviking/ov.conf"
```

> [!WARNING]
> Removing `~/.openviking/data` permanently deletes the local OpenViking database.

Ollama and its models are shared with other applications. Remove them only if nothing else uses
them:

```bash
ollama rm qwen3-embedding:0.6b
ollama rm qwen3.8:27b
```

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `setup.sh` rejects the platform | The lock targets DGX OS Ubuntu 24.04, Linux aarch64, Python 3.12, and glibc 2.39 | Update DGX OS or create and review a new lock for the actual platform; do not bypass the check |
| NVIDIA packages resolve to CUDA 13.3 | The constraints or lock were bypassed | Use `requirements-cu13.lock` with `--require-hashes`; confirm `nvidia-nvjitlink==13.0.88` |
| `systemctl --user` cannot connect to the bus | No user systemd session is available | Log in through a normal local or SSH session. For an always-on service after logout, an administrator can enable linger with `sudo loginctl enable-linger "$USER"` |
| OpenViking is healthy but E2E reports no cuVS route | Auto admission rejected GPU use, the snapshot is still building, or cuVS failed | Check the verifier's last `routes`, service logs, `ollama ps`, and `nvidia-smi`; stop competing workloads or reduce the corpus. Do not count native fallback as cuVS success |
| E2E returns the marker but route is `native_filter_threshold` | A path/scalar filter selected the native route | Run the unfiltered verifier as supplied. Native routing for selective filters is expected and can be faster |
| Search returns an empty list after a GPU error | Runtime failures can look like zero recall at a higher layer | Inspect OpenViking logs and telemetry, then rerun the known-result check; never use `/health` alone as the cutover gate |
| First query after a write is slower | The immutable GPU snapshot is rebuilding | Keep background rebuild enabled; dirty queries use native search until the new snapshot is committed |
| Ollama VLM times out or cuVS admission falls back | The VLM and vector index are competing for unified memory | Check `ollama ps` and `nvidia-smi`; use a smaller VLM, increase headroom, or keep dense search native |
| `pip` reports a hash mismatch from a mirror | Mirror content differs from the locked artifact | Use the official indexes or a mirror that serves the exact locked wheel; do not disable hash checking |

For cuVS memory formulas, numerical semantics, and current auto-mode behavior, see the
[OpenViking cuVS guide](https://github.com/volcengine/OpenViking/blob/v0.4.17/docs/en/guides/16-cuvs.md).
