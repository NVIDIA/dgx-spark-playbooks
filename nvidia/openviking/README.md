# OpenViking with Ollama and NVIDIA cuVS

> Run an all-local agent context database on DGX Spark

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Remote access](#remote-access)
- [Cleanup](#cleanup)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

[OpenViking](https://github.com/volcengine/OpenViking) stores agent memory, knowledge, and skills
in one context database. In this setup, OpenViking uses:

- [Ollama](../ollama/) for a local embedding model and a local vision-language model (VLM).
- Its native local store for canonical data, scalar and sparse retrieval, and recovery.
- [NVIDIA cuVS](https://docs.nvidia.com/cuvs/) for an optional GPU snapshot of dense vectors.

The supplied configuration keeps `backend` set to `local` and enables cuVS auto mode. A dense
query uses cuVS when the snapshot is ready and the unified-memory admission check passes. It falls
back to the native index when GPU memory is tight or the query is better handled on the CPU. cuVS
is an acceleration layer here, not a replacement for the rest of OpenViking's storage engine.

## What you'll accomplish

- Run OpenViking 0.4.17.1 on DGX Spark with a CUDA 13 cuVS environment.
- Serve `qwen3-embedding:0.6b` and `qwen3.8:27b` locally through Ollama.
- Generate semantic metadata, write vectors, and retrieve a known result through cuVS.
- Keep both APIs on loopback and connect through NVIDIA Sync or an SSH tunnel.

The final smoke test proves the integration works. It is not a performance benchmark. For a small
corpus, native search can be faster because GPU startup and snapshot construction dominate.

## Prerequisites

- A DGX Spark with current DGX OS and working NVIDIA drivers.
- Python 3.12 with the `venv` module.
- `git` and `curl`.
- At least 35 GB of free disk space for the models and Python environment.
- Ollama installed by following Steps 1 and 2 in the [Ollama playbook](../ollama/README.md).

The [CUDA-X Data Science playbook](../cuda-x-data-science/README.md#step-1-verify-system-requirements)
has the shared CUDA 13 environment check. This playbook only adds the packages and configuration
specific to OpenViking and cuVS.

## Time and risk

- **Duration**: About 30-60 minutes. Model download speed is the largest variable.
- **Risk level**: Medium. The setup downloads a 27B model and creates local OpenViking data.
- **Memory**: Ollama and cuVS share the Spark's unified memory. The configuration reserves 8 GiB
  outside cuVS admission.
- **Rollback**: Stop the foreground server, then remove the dedicated environment and data paths.
- **Last Updated**: 09/02/2026

## Tested versions

| Component | Version |
| --- | --- |
| OpenViking | [0.4.17.1](https://github.com/volcengine/OpenViking/releases/tag/v0.4.17.1) |
| cuVS / RAFT / RMM | 26.6.0 |
| CuPy | 14.1.1 |
| CUDA Python packages | 13.0 |
| Python | 3.12 |
| Embedding model | [`qwen3-embedding:0.6b`](https://ollama.com/library/qwen3-embedding:0.6b) |
| VLM | [`qwen3.8:27b`](https://ollama.com/library/qwen3.8:27b) |

The short requirements file pins the CUDA-facing packages to the CUDA 13.0 line. OpenViking is
licensed under [AGPL-3.0](https://github.com/volcengine/OpenViking/blob/v0.4.17.1/LICENSE); review
the license before modifying or providing the service to others.

## Instructions

## Step 1. Verify the Spark and install Ollama

Run the shared CUDA checks from the
[CUDA-X Data Science playbook](../cuda-x-data-science/README.md#step-1-verify-system-requirements):

```bash
nvidia-smi
nvcc --version
python3 --version
```

If Ollama is not already installed, complete Steps 1 and 2 in the
[Ollama playbook](../ollama/README.md#step-1-verify-ollama-installation-status). That playbook owns
the Ollama install, service, API test, tunnel, and removal flow. Return here instead of pulling its
example chat model.

Pull the models used by OpenViking:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.8:27b
ollama list
curl --fail http://127.0.0.1:11434/api/tags
```

Both model names should appear before continuing.

## Step 2. Install OpenViking and cuVS

Clone this repository and enter the OpenViking assets directory:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks.git
cd dgx-spark-playbooks/nvidia/openviking/assets
```

Create a dedicated environment and install the tested CUDA 13 package set:

```bash
python3 -m venv "$HOME/.venvs/openviking"
source "$HOME/.venvs/openviking/bin/activate"
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: \
  --extra-index-url https://pypi.nvidia.com \
  -r requirements-cu13.txt
python -m pip check
```

Confirm that Python can execute a CUDA operation and import cuVS:

```bash
python - <<'PY'
import cupy
import cuvs

value = cupy.asarray([1.0], dtype=cupy.float32).sum().item()
print(f"cuVS: {cuvs.__file__}")
print(f"CUDA device: {cupy.cuda.runtime.getDeviceProperties(0)['name']}")
print(f"CUDA smoke result: {value}")
PY
```

For why the `[ctk]` CuPy extra is needed and how OpenViking routes dense queries, see the
[versioned OpenViking cuVS guide](https://github.com/volcengine/OpenViking/blob/v0.4.17.1/docs/en/guides/16-cuvs.md).

## Step 3. Configure and start OpenViking

Inspect the configuration before installing it. Back up an existing configuration rather than
silently replacing it:

```bash
python -m json.tool ov.conf >/dev/null
mkdir -p "$HOME/.openviking"
if [ -f "$HOME/.openviking/ov.conf" ]; then
  cp "$HOME/.openviking/ov.conf" "$HOME/.openviking/ov.conf.backup"
fi
install -m 600 ov.conf "$HOME/.openviking/ov.conf"
```

The file binds OpenViking to `127.0.0.1:1933`, points both model providers at local Ollama, and
uses exact float32 brute-force search for the cuVS snapshot. Validate the model and storage setup:

```bash
openviking-server doctor
```

Start OpenViking in the foreground so its logs remain visible:

```bash
openviking-server --config "$HOME/.openviking/ov.conf"
```

For a persistent service or another process manager, follow the upstream
[OpenViking deployment guide](https://github.com/volcengine/OpenViking/blob/v0.4.17.1/docs/en/guides/03-deployment.md)
after the foreground flow succeeds.

## Step 4. Run the OpenViking/cuVS smoke test

In another terminal, return to the assets directory and run:

```bash
source "$HOME/.venvs/openviking/bin/activate"
python verify.py
```

The test creates a UUID-scoped resource, waits for semantic and vector processing, checks the
generated semantic sidecars, then searches for a unique marker. It succeeds only when telemetry
reports a real cuVS route and the expected result is present. The temporary resource is deleted at
the end.

A successful run ends with:

```text
OPENVIKING_CUVS_E2E_OK
```

Auto mode may serve the first query from the native index while it builds a GPU snapshot. The
test retries for up to three minutes. A native fallback is valid runtime behavior, but it does not
count as proof that cuVS worked.

## Remote access

OpenViking uses development authentication in this local example, so loopback plus the Spark OS
account is the security boundary. Do not expose ports 1933 or 11434 directly to an untrusted LAN.

To use NVIDIA Sync, follow the custom-app and tunnel steps in the
[Ollama playbook](../ollama/README.md#step-4-access-nvidia-sync-settings), but create an
`OpenViking` entry for port `1933` instead of the Ollama entry for port `11434`.

Alternatively, forward the port with SSH:

```bash
ssh -N -L 1933:127.0.0.1:1933 <username>@<spark-address>
curl --fail http://127.0.0.1:1933/health
```

For a shared deployment, configure authentication before changing the listener. See the
[OpenViking authentication guide](https://github.com/volcengine/OpenViking/blob/v0.4.17.1/docs/en/guides/04-authentication.md).

## Cleanup

Stop the foreground OpenViking process with `Ctrl+C`. The following removes the dedicated Python
environment but preserves the database and configuration:

```bash
rm -rf "$HOME/.venvs/openviking"
```

To delete the OpenViking data as well, inspect the exact paths first:

```bash
du -sh "$HOME/.openviking/data" "$HOME/.openviking/ov.conf"
rm -rf "$HOME/.openviking/data"
rm "$HOME/.openviking/ov.conf"
```

> [!WARNING]
> Removing `~/.openviking/data` permanently deletes the local context database.

Ollama is shared with other applications. Use the cleanup section of the
[Ollama playbook](../ollama/README.md#step-9-cleanup-and-rollback) only when nothing else depends
on it. Remove just this playbook's models with:

```bash
ollama rm qwen3-embedding:0.6b
ollama rm qwen3.8:27b
```

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `openviking-server doctor` cannot reach a model | Ollama is stopped or a model is missing | Run `ollama list`, then repeat Step 1 and the Ollama API check |
| `import cuvs` or the CUDA smoke fails | The wrong environment is active, or the NVIDIA wheels were not installed | Activate `~/.venvs/openviking`, reinstall from `requirements-cu13.txt`, and run `python -m pip check` |
| The smoke test keeps reporting native routes | cuVS is still building or unified-memory admission rejected it | Check `ollama ps` and `nvidia-smi`, stop unrelated GPU workloads, then retry |
| `/health` succeeds but search fails | Health only proves the server process is up | Read the foreground server log and rerun `openviking-server doctor` plus `verify.py` |
| Search is slower than native on a small corpus | GPU setup and snapshot costs dominate | Keep native search for small/write-heavy data; benchmark cuVS only on representative data |

For cuVS memory sizing, CAGRA tuning, numerical differences, and fallback behavior, use the
[OpenViking cuVS guide](https://github.com/volcengine/OpenViking/blob/v0.4.17.1/docs/en/guides/16-cuvs.md)
as the source of truth.
