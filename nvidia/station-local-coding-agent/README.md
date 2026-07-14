# Local Coding Agent

> Run local CLI coding agents with Claude Code and Ollama on DGX Station (NVIDIA GB300) using qwen3.6:27b


## Table of Contents

- [Overview](#overview)
- [Claude Code](#claude-code)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Use Ollama on **DGX Station (NVIDIA GB300)** to run a local coding model and connect a CLI coding agent. This
playbook uses **Claude Code** with `ollama launch` so you can work without external cloud APIs.

The DGX Station GPU (reported as **NVIDIA GB300** in `nvidia-smi`) provides ample memory to run **qwen3.6:27b** with Ollama for local coding-agent workflows.

## CLI agent

This playbook uses **Claude Code** as the CLI agent, connected to a local Ollama model for inference.

## What you'll accomplish

You will run **qwen3.6:27b** on your **DGX Station (NVIDIA GB300)** with Ollama, connect Claude Code to it, and complete a small coding task end-to-end.

## What to know before starting

- Comfort with Linux command line basics
- Experience running terminal-based tools and editors
- Familiarity with Python for the short coding task

## Prerequisites

- **DGX Station** with **NVIDIA GB300** (Grace Blackwell) and NVIDIA driver; `nvidia-smi` typically shows "NVIDIA GB300"
- Internet access to download model weights
- **Ollama 0.15.0 or newer**
- **GPU memory** on GB300 supports the recommended `qwen3.6:27b` model
- **Disk space** for the `qwen3.6:27b` model download

## Time & risk

* **Duration**: ~20–30 minutes (includes model download)
* **Risk level**: Low
  * Large model downloads can fail if network connectivity is unstable
  * Older Ollama versions will not load newer models
* **Rollback**: Stop Ollama and delete the downloaded model from `~/.ollama/models`
* **Last Updated:** 06/12/2026
  * Model path set to qwen3.6:27b with `ollama launch`; Python task now uses a virtual environment

## Claude Code

## Step 1. Confirm your environment

**Description**: Verify the GPU is visible before installing anything.

```bash
nvidia-smi
```

**Expected output** (example): A table showing driver version and GPU(s). On DGX Station, the GPU name may appear as **NVIDIA GB300** (without "Ultra"):

```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 5xx.xx    Driver Version: 5xx.xx    CUDA Version: 12.x          |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GB300        On   | 00000000:06:00.0 Off |                    0 |
...
```

## Step 2. Install or update Ollama

**Description**: Install Ollama or ensure it is recent enough for modern coding models.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

To install a specific version if needed:

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.15.0 sh
```

If Ollama is already present, simply run:

```bash
ollama --version
```

**Expected output** (example):

```text
ollama version is 0.15.0
```

## Step 3. Pull a coding model

**Description**: Download the model weights to your DGX Station.

This playbook uses **qwen3.6:27b** with Claude Code through Ollama:

```bash
ollama pull qwen3.6:27b
```

**Expected output** (example): Progress lines followed by "success" and the model in `ollama list`:

```bash
ollama list
```

```text
NAME                                ID              SIZE    MODIFIED
qwen3.6:27b                         abc123...       ...     1 minute ago
```

## Step 4. Test local inference

**Description**: Run a quick prompt to confirm the model loads.

```bash
ollama run qwen3.6:27b
```

Try a prompt like:

```text
Write a short README checklist for a Python project.
```

**Expected output**: The model replies with a short README checklist.

**Exit the Ollama REPL** when done: type `/bye` or press **Ctrl+D**.

## Step 5. Install Claude Code

**Description**: Install the CLI tool that will drive the local model.

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Verify the installation**:

```bash
claude --version
```

**Expected output** (example): A version string such as `claude 0.x.x` or similar. If you see `claude: command not found`, ensure the install script added the CLI to your PATH (e.g. restart the terminal or source your shell profile); see [Troubleshooting](troubleshooting.md).

## Step 6. Increase context length (optional)

**Description**: Ollama defaults to a 4096 token context length. For coding agents and
larger codebases, set it to 64K tokens. This increases memory usage.
For more details on configuring context length and other parameters, see the Ollama documentation (context window and runtime options).

Set the context length per session in the Ollama REPL:

```bash
ollama run qwen3.6:27b
```

Then, in the Ollama prompt:

```text
/set parameter num_ctx 64000

```

**Exit when done**: type `/bye` or press **Ctrl+D**.

Optional method (set globally when serving Ollama):

```bash
sudo systemctl stop ollama
OLLAMA_CONTEXT_LENGTH=64000 ollama serve 
```

Keep this terminal open and run the next step in a new terminal.

## Step 7. Connect Claude Code to Ollama

**Description**: Launch Claude Code through Ollama with the model you pulled.

```bash
ollama launch claude --model qwen3.6:27b
```

**Expected output**: Claude Code starts and uses the local Ollama model.

**Exit Claude Code** when done: type `/exit` or press **Ctrl+C**.

## Step 8. Complete a small coding task

**Description**: Create a tiny repo and let Claude Code implement a function and tests.

```bash
mkdir -p ~/cli-agent-demo
cd ~/cli-agent-demo
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pytest

printf 'def add(a, b):\n    """Return the sum of a and b."""\n    pass\n' > math_utils.py
printf 'import math_utils\n\n\ndef test_add():\n    assert math_utils.add(1, 2) == 3\n' > test_math_utils.py
```

If Claude Code is not already running, launch it:

```bash
ollama launch claude --model qwen3.6:27b
```

In Claude Code, enter:

```text
Please implement add() in math_utils.py and make sure the test passes.
```

**Exit Claude Code** when finished: type `/exit` or press **Ctrl+C**, then run the test:

```bash
python3 -m pytest -q
deactivate
```

Expected output should show the test passing.

## Step 9. Cleanup and rollback

**Description**: Remove the model and stop the Ollama service if you no longer need them. **Remove the model first** (while the Ollama server is running), then stop the service.

> [!WARNING]
> The following removes the downloaded model files from disk.

**1. Remove the model** (Ollama must be running). Use the same name you pulled:

```bash
ollama rm qwen3.6:27b
```

**2. Stop the Ollama service**:

```bash
sudo systemctl stop ollama
```

## Step 10. Next steps

- Use larger context (e.g. 64K–198K) for big codebases.
- Use Claude Code on multi-file refactors or test-generation tasks.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ollama: command not found` | Ollama not installed or PATH not updated | Rerun `curl -fsSL https://ollama.com/install.sh \| sh` and open a new shell |
| Model load fails with version error | Ollama is older than the model requires | Update Ollama to a current stable release. Do not pin to older versions. |
| `model not found` in Claude Code | Model was not pulled | Run `ollama pull qwen3.6:27b` and retry with `ollama launch claude --model qwen3.6:27b`. |
| `connection refused` to localhost:11434 | Ollama service not running | Start with `ollama serve` or `sudo systemctl start ollama` |
| Sharded GGUF model pull fails with HTTP 400 | Ollama does not support pulling sharded GGUF models from Hugging Face | Use the documented `qwen3.6:27b` model instead: `ollama pull qwen3.6:27b`. |
| `CUDA error: context is destroyed` on a dual-GPU Station | Ollama may fail when both the GB300 and RTX PRO 6000 GPUs are visible | Run Ollama with one visible GPU. For example, set `CUDA_VISIBLE_DEVICES=1` in the Ollama service environment, restart Ollama, and rerun the playbook. |
| Claude Code edit task fails through the direct Ollama endpoint | Direct endpoint wiring can fail with some Ollama/model combinations | Launch Claude Code through Ollama instead: `ollama launch claude --model qwen3.6:27b`. |
| `externally-managed-environment` or Python package install fails | System Python blocks direct package installs | Create and activate a virtual environment, then install pytest inside it: `python3 -m venv .venv`, `source .venv/bin/activate`, `python3 -m pip install -U pytest`. |
| Slow responses or OOM | Insufficient GPU memory or fragmentation | On DGX Station (NVIDIA GB300), ensure no other heavy GPU workloads. If OOM persists, unload other models or set `OLLAMA_MAX_LOADED_MODELS=1`. |
| `claude: command not found` after install | CLI not on PATH or install script did not complete | Restart the terminal or run `source ~/.bashrc` (or your shell profile). Check the install script output for the install path and add it to PATH. |
| Claude Code install fails (Node.js / network) | Node.js missing or install script cannot download | Ensure Node.js is installed (`node --version`). Run the installer with Bash: `curl -fsSL https://claude.ai/install.sh | bash`. If the install script fails with a network error, retry from a stable connection or download the Claude Code CLI from the official site. See [Claude Code documentation](https://docs.claude.com/en/docs/claude-code/overview) for alternatives. |

> [!NOTE]
> DGX Station with **NVIDIA GB300** provides ample GPU memory for the documented `qwen3.6:27b` workflow. Use `OLLAMA_MAX_LOADED_MODELS=1` if you hit memory limits with multiple models.
