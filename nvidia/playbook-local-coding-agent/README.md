# Set Up Claude Code with Local Inference

> A coding agent connected to a local Ollama model

## Table of Contents

- [Overview](#overview)
- [Claude Code](#claude-code)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Use [Ollama](https://ollama.com) on your **hardware platform** to run a local coding model and connect a CLI coding agent. This playbook uses **[Claude Code](https://docs.claude.com/en/docs/claude-code)** with Ollama's built-in [launch method](https://ollama.com/blog/launch) (`ollama launch claude`), so you can work without environment variables, provider config files, or external cloud APIs.

## CLI agent

This playbook uses **Claude Code** as the CLI agent, connected to a local Ollama model for inference.

## What you'll accomplish

You'll run a local coding model ([Qwen3.6](https://ollama.com/library/qwen3.6) `qwen3.6:27b`) on your **hardware platform** with Ollama, launch Claude Code against it with a single command, and complete a small coding task end-to-end.

## What to know before starting

**Required:**

- Comfort with Linux command line basics
- Experience running terminal-based tools and editors
- Familiarity with Python for the short coding task

**Optional:**

- Experience browsing models on the [Ollama library](https://ollama.com/library) (for example, [Qwen3.6](https://ollama.com/library/qwen3.6))

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Station** | DGX OS (Linux) | Large HBM + Grace DRAM | Ollama + `qwen3.6:27b`; Claude Code via `ollama launch` | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Hardware platform powered on, networked, and reachable for local or SSH terminal access
- Sufficient GPU memory for `qwen3.6:27b`
- Enough free storage for the model download

**Software requirements**

- A [current Ollama release](https://ollama.com/download) (required for [`ollama launch`](https://ollama.com/blog/launch) and the recommended model): `ollama --version`
- Internet access to download model weights
- Python 3 with `venv` support for the coding-task verification

## Find model recipes

Browse models you can pull and run with Ollama in the [Ollama library](https://ollama.com/library). Use the tags and sizes that fit your hardware platform’s memory and storage.

| Hardware platform | More recipes |
| ----------------- | ------------ |
| **DGX Station** | [Ollama library](https://ollama.com/library) · [Qwen3.6](https://ollama.com/library/qwen3.6) |

Use the **Claude Code** tab for the base workflow.

## Ancillary files

No local ancillary files are required. All steps use Ollama and Claude Code on your hardware platform.

## Time & risk

- **Estimated time:** 30 MIN (mostly model download time)
- **Risk level:** Low
  - Large model downloads can fail if network connectivity is unstable
  - Ollama must support [`ollama launch`](https://ollama.com/blog/launch) and the recommended model tag — verify with `ollama --version` and install from [ollama.com/download](https://ollama.com/download) if needed
- **Rollback:** Stop Ollama and delete the downloaded model from `~/.ollama/models` (see Cleanup in the Claude Code tab). Cleanup is optional and removes downloaded model files.
- **Last Updated:** 08/03/2026
  - Set up Claude Code against a local Qwen3.6 model with `ollama launch` on supported hardware platforms

## Claude Code

## Step 1. Confirm your environment

Verify the GPU is visible before installing anything.

```bash
nvidia-smi
```

Expected output should show a detected GPU and driver version for your hardware platform.

## Step 2. Install or update Ollama

Install [Ollama](https://ollama.com/download) or confirm your install supports [`ollama launch`](https://ollama.com/blog/launch).

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

If Ollama is already installed, just verify the version:

```bash
ollama --version
```

Expected output should show a current Ollama release.

## Step 3. Pull a coding model

Download the [Qwen3.6](https://ollama.com/library/qwen3.6) model weights to your hardware platform.

This playbook uses **qwen3.6:27b** with Claude Code through Ollama:

```bash
ollama pull qwen3.6:27b
```

Expected output should show progress lines followed by success. Confirm the model appears in the list:

```bash
ollama list
```

```text
NAME                                ID              SIZE    MODIFIED
qwen3.6:27b                         abc123...       ...     1 minute ago
```

## Step 4. Test local inference

Run a quick prompt to confirm the model loads.

```bash
ollama run qwen3.6:27b
```

Try a prompt like:

```text
Write a short README checklist for a Python project.
```

Expected output should show the model responding in the terminal. When you are done, type `/bye` or press **Ctrl+D** to exit the interactive session before continuing.

## Step 5. Install Claude Code

Install [Claude Code](https://docs.claude.com/en/docs/claude-code), the CLI tool that will drive the local model.

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Verify the installation:

```bash
claude --version
```

Expected output should show a version string such as `claude 0.x.x`. If you see `claude: command not found`, ensure the install script added the CLI to your PATH (for example, restart the terminal or source your shell profile); see [Troubleshooting](troubleshooting.md).

## Step 6. Increase context length (optional)

Ollama defaults to a 4096 token context length. For coding agents and larger codebases, set it to 64K tokens. This increases memory usage. For more details on configuring context length and other parameters, see the [Ollama documentation](https://ollama.com/docs).

Set the context length per session in the Ollama REPL:

```bash
ollama run qwen3.6:27b
```

Then, in the Ollama prompt:

```text
/set parameter num_ctx 64000
```

Exit when done: type `/bye` or press **Ctrl+D**.

Optional method (set globally when serving Ollama):

```bash
sudo systemctl stop ollama
OLLAMA_CONTEXT_LENGTH=64000 ollama serve
```

Keep this terminal open and run the next step in a new terminal.

## Step 7. Connect Claude Code to Ollama

Launch Claude Code through Ollama with the model you pulled. No environment variables or config files are required.

```bash
ollama launch claude --model qwen3.6:27b
```

Expected output should show Claude Code starting and using the local Ollama model.

Exit Claude Code when done: type `/exit` or press **Ctrl+C**.

## Step 8. Complete a small coding task

Create a tiny repo and let Claude Code implement a function and tests.

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

Exit Claude Code when finished: type `/exit` or press **Ctrl+C**, then run the test:

```bash
python3 -m pytest -q
deactivate
```

Expected output should show the test passing.

## Step 9. Cleanup

Remove the model and stop the Ollama service if you no longer need them. Cleanup is optional. **Remove the model first** (while the Ollama server is running), then stop the service.

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

- Use larger context (for example, 64K–198K) for big codebases
- Use Claude Code on multi-file refactors or test-generation tasks
- Browse additional models in the [Ollama library](https://ollama.com/library)

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ollama: command not found` | Ollama not installed or PATH not updated | Rerun `curl -fsSL https://ollama.com/install.sh \| sh` and open a new shell |
| `ollama launch` reports unknown command | `ollama launch` not available in this install | Install or reinstall Ollama from [ollama.com/download](https://ollama.com/download): `curl -fsSL https://ollama.com/install.sh \| sh` |
| Model load fails with version error | Installed Ollama does not accept the recommended model tag | Install or reinstall Ollama from [ollama.com/download](https://ollama.com/download): `curl -fsSL https://ollama.com/install.sh \| sh` |
| `model not found` in Claude Code | Model was not pulled | Run `ollama pull qwen3.6:27b` and retry with `ollama launch claude --model qwen3.6:27b` |
| `connection refused` to localhost:11434 | Ollama service not running | Start with `ollama serve` or `sudo systemctl start ollama` |
| Sharded GGUF model pull fails with HTTP 400 | The selected model tag is not supported by Ollama | Use the documented `qwen3.6:27b` model instead: `ollama pull qwen3.6:27b` |
| `CUDA error: context is destroyed` on a multi-GPU system | Ollama is initializing across a mixed-GPU topology | Pin Ollama to one GPU. For a foreground test, run `CUDA_VISIBLE_DEVICES=0 ollama serve`; for a system service, add `Environment="CUDA_VISIBLE_DEVICES=0"` (or another GPU index) to an Ollama systemd drop-in and restart Ollama |
| Claude Code edit task fails through a direct Ollama endpoint | Direct endpoint wiring can fail with some Ollama/model combinations | Launch Claude Code through Ollama instead: `ollama launch claude --model qwen3.6:27b` |
| `externally-managed-environment` or Python package install fails | System Python blocks direct package installs | Create and activate a virtual environment, then install pytest inside it: `python3 -m venv .venv`, `source .venv/bin/activate`, `python3 -m pip install -U pytest` |
| Slow responses or OOM | Insufficient GPU memory or fragmentation | Ensure no other heavy GPU workloads. If OOM persists, unload other models or set `OLLAMA_MAX_LOADED_MODELS=1` |
| `claude: command not found` after install | CLI not on PATH or install script did not complete | Restart the terminal or run `source ~/.bashrc` (or your shell profile). Check the install script output for the install path and add it to PATH |
| Claude Code install fails (Node.js / network) | Node.js missing or install script cannot download | Ensure Node.js is installed (`node --version`). Run the installer with Bash: `curl -fsSL https://claude.ai/install.sh \| bash`. If the install script fails with a network error, retry from a stable connection. See [Claude Code documentation](https://docs.claude.com/en/docs/claude-code/overview) for alternatives |

> [!NOTE]
> The recommended `qwen3.6:27b` workflow fits the Supported hardware platforms matrix defaults. Use `OLLAMA_MAX_LOADED_MODELS=1` if you hit memory limits with multiple models.

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
