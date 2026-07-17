# CLI Coding Agent

> Build local CLI coding agents with Ollama

## Table of Contents

- [Overview](#overview)
- [Claude Code](#claude-code)
- [OpenCode](#opencode)
- [Codex CLI](#codex-cli)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Use [Ollama](https://ollama.com) on [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) to run a local coding model and connect a CLI coding agent. This
playbook supports three options: **[Claude Code](https://docs.claude.com/en/docs/claude-code)**, **[OpenCode](https://opencode.ai)**, and **[Codex CLI](https://github.com/openai/codex)**. Each
agent is wired up with Ollama's built-in [launch method](https://ollama.com/blog/launch) (`ollama launch <agent>`), so you
can work without environment variables, provider config files, or external cloud APIs.

## Choose your CLI agent

Pick the tab that matches the CLI agent you want to use:

- **Claude Code**: Fastest path to a working CLI agent with a local Ollama model.
- **OpenCode**: Open-source CLI launched directly from Ollama.
- **Codex CLI**: OpenAI Codex CLI launched directly from Ollama against the local model.

## What you'll accomplish

You will run a local coding model ([Qwen3.6](https://ollama.com/library/qwen3.6)) on your DGX Spark with Ollama, launch your
chosen CLI agent against it with a single command, and complete a small coding task end-to-end.

## What to know before starting

- Comfort with Linux command line basics
- Experience running terminal-based tools and editors
- Familiarity with Python for the short coding task

## Prerequisites

- DGX Spark access with NVIDIA DGX OS 7.3.1 (Ubuntu 24.04.3 LTS base)
- Internet access to download model weights
- [Ollama](https://ollama.com/download) v0.15 or newer (required for [`ollama launch`](https://ollama.com/blog/launch))
- GPU memory depends on the Qwen3.6 variant you choose:
  - `qwen3.6:latest` (35B-a3b, MoE) — ~24GB, 256K context
  - `qwen3.6:35b-a3b-nvfp4` — ~22GB, NVIDIA FP4 build tuned for Blackwell (DGX Spark)
  - `qwen3.6:35b-a3b-q8_0` — ~39GB, higher-quality quant
  - `qwen3.6:35b-a3b-bf16` — ~71GB, full precision (fits Spark's unified memory)

## Time & risk

* **Duration**: ~15-25 minutes (mostly model download time)
* **Risk level**: Low
  * Large model downloads can fail if network connectivity is unstable
  * Ollama versions older than 0.15 do not support `ollama launch`
* **Rollback**: Stop Ollama and delete the downloaded model from `~/.ollama/models`
* **Last Updated:** 04/16/2026
  * Switched to `ollama launch` method and upgraded the default model to Qwen3.6

## Claude Code

## Step 1. Confirm your environment

**Description**: Verify the OS version and GPU are visible before installing anything.

```bash
cat /etc/os-release | head -n 2
nvidia-smi
```

Expected output should show Ubuntu 24.04.3 LTS (DGX OS 7.3.1 base) and a detected GPU.

## Step 2. Install or update Ollama

**Description**: Install [Ollama](https://ollama.com/download) or ensure it is recent enough to support [`ollama launch`](https://ollama.com/blog/launch).

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

If Ollama is already installed, just verify the version:

```bash
ollama --version
```

Expected output should show Ollama v0.15 or newer.

## Step 3. Pull Qwen3.6

**Description**: Download the [Qwen3.6](https://ollama.com/library/qwen3.6) model weights to your Spark node.

```bash
ollama pull qwen3.6
```

Optional variants if you want different memory footprints or precision:

```bash
ollama pull qwen3.6:35b-a3b-nvfp4   # NVIDIA FP4 build tuned for Blackwell (~22GB)
ollama pull qwen3.6:35b-a3b-q8_0    # Higher-quality 8-bit quant (~39GB)
ollama pull qwen3.6:35b-a3b-bf16    # Full precision (~71GB)
```

Expected output should show `qwen3.6` (and any optional variants) in `ollama list`.

## Step 4. Test local inference (optional)

**Description**: Run a quick prompt to confirm the model loads.

```bash
ollama run qwen3.6
```

Try a prompt like:

```text
Write a short README checklist for a Python project.
```

Expected output should show the model responding in the terminal. When you are done, type `/bye` or press `Ctrl+D` to exit the interactive session before continuing.

## Step 5. Install and launch Claude Code with Ollama

**Description**: Install [Claude Code](https://docs.claude.com/en/docs/claude-code), then use Ollama's built-in [launch method](https://ollama.com/blog/launch) to start Claude Code against your local model. No environment variables or config files are required.

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude --version
```

If Claude Code is already installed, just verify the version:

```bash
claude --version
```

```bash
ollama launch claude --model qwen3.6
```

Expected output should show Claude Code starting and using the local Qwen3.6 model. Qwen3.6 ships with a 256K context window by default; adjust context length through Ollama's settings if you need to tune it further.

## Step 6. Complete a small coding task

**Description**: Create a tiny repo and let Claude Code implement a function and tests.

```bash
mkdir -p ~/cli-agent-demo
cd ~/cli-agent-demo

printf 'def add(a, b):\n    """Return the sum of a and b."""\n    pass\n' > math_utils.py
printf 'import math_utils\n\n\ndef test_add():\n    assert math_utils.add(1, 2) == 3\n' > test_math_utils.py
```

If you do not already have pytest installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pytest
```

In Claude Code:

```text
Please implement add() in math_utils.py and make sure the test passes.
```

Run the test:

```bash
python3 -m pytest -q
```

Expected output should show the test passing. When you are done, run `deactivate` to exit the virtual environment.

## Step 7. Cleanup and rollback

**Description**: Remove the model and stop services if you no longer need them.

To stop the service:

```bash
sudo systemctl stop ollama
```

> [!WARNING]
> This will delete the downloaded model files.

```bash
ollama rm qwen3.6
```

## Step 8. Next steps

- Try the `qwen3.6:35b-a3b-nvfp4` or `bf16` variants for different quality/VRAM tradeoffs
- Use Claude Code on multi-file refactors or test-generation tasks
- Explore the full 256K context window on larger codebases

## OpenCode

## Step 1. Confirm your environment

**Description**: Verify the OS version and GPU are visible before installing anything.

```bash
cat /etc/os-release | head -n 2
nvidia-smi
```

Expected output should show Ubuntu 24.04.3 LTS (DGX OS 7.3.1 base) and a detected GPU.

## Step 2. Install or update Ollama

**Description**: Install [Ollama](https://ollama.com/download) or ensure it is recent enough to support [`ollama launch`](https://ollama.com/blog/launch).

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

If Ollama is already installed, just verify the version:

```bash
ollama --version
```

Expected output should show Ollama v0.15 or newer.

## Step 3. Pull Qwen3.6

**Description**: Download the [Qwen3.6](https://ollama.com/library/qwen3.6) model weights to your Spark node.

```bash
ollama pull qwen3.6
```

Optional variants if you want different memory footprints or precision:

```bash
ollama pull qwen3.6:35b-a3b-nvfp4   # NVIDIA FP4 build tuned for Blackwell (~22GB)
ollama pull qwen3.6:35b-a3b-q8_0    # Higher-quality 8-bit quant (~39GB)
ollama pull qwen3.6:35b-a3b-bf16    # Full precision (~71GB)
```

Expected output should show `qwen3.6` in `ollama list`.

## Step 4. Test local inference (optional)

**Description**: Run a quick prompt to confirm the model loads.

```bash
ollama run qwen3.6
```

Try a prompt like:

```text
Write a short README checklist for a Python project.
```

Expected output should show the model responding. When you are done, type `/bye` or press `Ctrl+D` to exit before continuing.

## Step 5. Launch OpenCode with Ollama

**Description**: Use Ollama's built-in [launch method](https://ollama.com/blog/launch) to start [OpenCode](https://opencode.ai) against your local model. No [`opencode.json`](https://opencode.ai/docs/config/) provider configuration is required.

```bash
ollama launch opencode --model qwen3.6
```

If you want to pre-configure OpenCode without launching immediately:

```bash
ollama launch opencode --config
```

Expected output should show OpenCode starting with Ollama preselected as the provider and Qwen3.6 as the model. Qwen3.6 ships with a 256K context window by default.

## Step 6. Complete a small coding task

**Description**: Create a tiny repo and let OpenCode implement a function and tests.

```bash
mkdir -p ~/cli-agent-demo
cd ~/cli-agent-demo

printf 'def add(a, b):\n    """Return the sum of a and b."""\n    pass\n' > math_utils.py
printf 'import math_utils\n\n\ndef test_add():\n    assert math_utils.add(1, 2) == 3\n' > test_math_utils.py
```

If you do not already have pytest installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pytest
```

In OpenCode:

```text
Please implement add() in math_utils.py and make sure the test passes.
```

Run the test:

```bash
python3 -m pytest -q
```

Expected output should show the test passing. When you are done, run `deactivate` to exit the virtual environment.

## Step 7. Cleanup and rollback

**Description**: Remove the model and stop services if you no longer need them.

To stop the service:

```bash
sudo systemctl stop ollama
```

> [!WARNING]
> This will delete the downloaded model files.

```bash
ollama rm qwen3.6
```

## Step 8. Next steps

- Try the `qwen3.6:35b-a3b-nvfp4` or `bf16` variants for different quality/VRAM tradeoffs
- Use OpenCode on multi-file changes or test-generation tasks
- Explore the full 256K context window on larger codebases

## Codex CLI

## Step 1. Confirm your environment

**Description**: Verify the OS version and GPU are visible before installing anything.

```bash
cat /etc/os-release | head -n 2
nvidia-smi
```

Expected output should show Ubuntu 24.04.3 LTS (DGX OS 7.3.1 base) and a detected GPU.

## Step 2. Install or update Ollama

**Description**: Install [Ollama](https://ollama.com/download) or ensure it is recent enough to support [`ollama launch`](https://ollama.com/blog/launch).

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

If Ollama is already installed, just verify the version:

```bash
ollama --version
```

Expected output should show Ollama v0.15 or newer.

## Step 3. Pull Qwen3.6

**Description**: Download the [Qwen3.6](https://ollama.com/library/qwen3.6) model weights to your Spark node.

```bash
ollama pull qwen3.6
```

Optional variants if you want different memory footprints or precision:

```bash
ollama pull qwen3.6:35b-a3b-nvfp4   # NVIDIA FP4 build tuned for Blackwell (~22GB)
ollama pull qwen3.6:35b-a3b-q8_0    # Higher-quality 8-bit quant (~39GB)
ollama pull qwen3.6:35b-a3b-bf16    # Full precision (~71GB)
```

Expected output should show `qwen3.6` in `ollama list`.

## Step 4. Test local inference (optional)

**Description**: Run a quick prompt to confirm the model loads.

```bash
ollama run qwen3.6
```

Try a prompt like:

```text
Write a short README checklist for a Python project.
```

Expected output should show the model responding. When you are done, type `/bye` or press `Ctrl+D` to exit before continuing.

## Step 5. Install and launch Codex CLI with Ollama

**Description**: Install [Codex CLI](https://github.com/openai/codex), then use Ollama's built-in [launch method](https://ollama.com/blog/launch) to start it against your local model. Ollama configures the local-model integration, but the Codex CLI binary must be installed first. No `~/.codex/config.toml` is required.

```bash
npm install -g @openai/codex
codex --version
ollama launch codex --model qwen3.6
```

Expected output should show Codex CLI starting with Ollama as the provider and Qwen3.6 as the model. Qwen3.6 ships with a 256K context window by default, which is well suited to Codex's agentic workflows.

## Step 6. Complete a small coding task

**Description**: Create a tiny repo and let Codex implement a function and tests.

```bash
mkdir -p ~/cli-agent-demo
cd ~/cli-agent-demo

printf 'def add(a, b):\n    """Return the sum of a and b."""\n    pass\n' > math_utils.py
printf 'import math_utils\n\n\ndef test_add():\n    assert math_utils.add(1, 2) == 3\n' > test_math_utils.py
```

If you do not already have pytest installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pytest
```

In Codex:

```text
Please implement add() in math_utils.py and make sure the test passes.
```

Run the test:

```bash
python3 -m pytest -q
```

Expected output should show the test passing. When you are done, run `deactivate` to exit the virtual environment.

## Step 7. Cleanup and rollback

**Description**: Remove the model and stop services if you no longer need them.

To stop the service:

```bash
sudo systemctl stop ollama
```

> [!WARNING]
> This will delete the downloaded model files.

```bash
ollama rm qwen3.6
```

## Step 8. Next steps

- Try the `qwen3.6:35b-a3b-nvfp4` or `bf16` variants for different quality/VRAM tradeoffs
- Use Codex CLI on multi-file changes or test-generation tasks
- Explore the full 256K context window on larger codebases

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ollama: command not found` | Ollama not installed or PATH not updated | Rerun `curl -fsSL https://ollama.com/install.sh \| sh` and open a new shell |
| `ollama launch` reports unknown command | Ollama is older than v0.15 | Update Ollama: `curl -fsSL https://ollama.com/install.sh \| sh` |
| Model load fails with version error or HTTP 412 | Ollama version is too old for the model | Update Ollama: `curl -fsSL https://ollama.com/install.sh \| sh` |
| `model not found` when launching an agent | Model was not pulled | Run `ollama pull qwen3.6` and retry |
| `connection refused` to localhost:11434 | Ollama service not running | Start with `ollama serve` or `sudo systemctl start ollama` |
| `ollama launch <agent>` exits immediately | Agent integration failed to initialize | Re-run `ollama launch <agent>`; if it persists, check `journalctl -u ollama` |
| Slow responses or OOM errors | Model variant too large for GPU memory | Switch to `qwen3.6:35b-a3b-nvfp4` or close other GPU workloads |
| `python3 -m pip install -U pytest` reports `externally-managed-environment` | Ubuntu 24.04 protects the system Python environment | Create and activate a virtual environment first: `python3 -m venv .venv && source .venv/bin/activate` |
| `ollama pull` reports that a model tag is a sharded GGUF | The selected model tag is not supported by Ollama | Use the Qwen3.6 commands in Step 3 instead of sharded GGUF tags |
| `ollama run` fails with `CUDA error: context is destroyed` on a multi-GPU system | Ollama is initializing across a mixed-GPU topology | Pin Ollama to one GPU. For a foreground test, run `CUDA_VISIBLE_DEVICES=0 ollama serve`; for a system service, add `Environment="CUDA_VISIBLE_DEVICES=0"` to an Ollama systemd drop-in and restart Ollama |
| A direct Claude Code setup using an Anthropic-compatible Ollama endpoint produces prose but does not edit files | Some model/server combinations do not emit tool calls reliably | Use `ollama launch claude` with Qwen3.6 as shown in this playbook |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing
> between the GPU and CPU. If you see memory pressure, flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```
