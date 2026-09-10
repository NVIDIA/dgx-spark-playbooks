# Set Up CLI Coding Agents with Local Inference

> Your chosen agent connected to a local model in one command

## Table of Contents

- [Overview](#overview)
- [Claude Code](#claude-code)
- [OpenCode](#opencode)
- [Codex CLI](#codex-cli)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Use [Ollama](https://ollama.com) on your **hardware platform** to run a local coding model and connect a CLI coding agent. This playbook supports three options: **[Claude Code](https://docs.claude.com/en/docs/claude-code)**, **[OpenCode](https://opencode.ai)**, and **[Codex CLI](https://github.com/openai/codex)**. Each agent is wired up with Ollama's built-in [launch method](https://ollama.com/blog/launch) (`ollama launch <agent>`), so you can work without environment variables, provider config files, or external cloud APIs.

## Choose your CLI agent

Pick the tab that matches the CLI agent you want to use:

- **Claude Code**: Fastest path to a working CLI agent with a local Ollama model.
- **OpenCode**: Open-source CLI installed locally, then configured and launched through Ollama.
- **Codex CLI**: OpenAI Codex CLI launched directly from Ollama against the local model.

## What you'll accomplish

You'll run a local coding model ([Qwen3.6](https://ollama.com/library/qwen3.6)) on your **hardware platform** with Ollama, launch your chosen CLI agent against it with a single command, and complete a small coding task end-to-end.

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
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Ollama + `qwen3.6:35b-a3b-mtp-q4_K_M` (~23 GB); Claude Code / OpenCode / Codex via `ollama launch` | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Hardware platform powered on, networked, and reachable for local or SSH terminal access
- Sufficient memory for your chosen Qwen3.6 variant (about 23 GB for the default MTP Q4_K_M; about 39 GB for `q8_0`; about 71 GB for `bf16`)
- Enough free storage for model downloads

**Software requirements**

- A [current Ollama release](https://ollama.com/download) (required for [`ollama launch`](https://ollama.com/blog/launch) and the default MTP Q4_K_M model): `ollama --version`
- Internet access to download model weights
- For Codex CLI: Node.js / npm available to install `@openai/codex`
- Python 3 with `venv` support for the optional coding-task verification

## Find model recipes

Browse models you can pull and run with Ollama in the [Ollama library](https://ollama.com/library). Use the tags and sizes that fit your hardware platform’s memory and storage.

| Hardware platform | More recipes |
| ----------------- | ------------ |
| **DGX Spark** | [Ollama library](https://ollama.com/library) · [Qwen3.6](https://ollama.com/library/qwen3.6) |

Use the **Claude Code**, **OpenCode**, or **Codex CLI** tab for the base workflow.

## Time & risk

- **Estimated time:** 20 MIN (mostly model download time)
- **Risk level:** Low
  - Large model downloads can fail if network connectivity is unstable
  - Ollama must support [`ollama launch`](https://ollama.com/blog/launch) and the default MTP Q4_K_M model tag — verify with `ollama --version` and install from [ollama.com/download](https://ollama.com/download) if needed
- **Rollback:** Stop Ollama and delete the downloaded model from `~/.ollama/models` (see Cleanup in each agent tab). Cleanup is optional and removes downloaded model files.
- **Last Updated:** 08/03/2026
  - Set up Claude Code, OpenCode, or Codex CLI against a local Qwen3.6 model with `ollama launch` on supported hardware platforms

## Claude Code

## Step 1. Confirm your environment

Verify the OS version and GPU are visible before installing anything.

```bash
cat /etc/os-release | head -n 2
nvidia-smi
```

Expected output should show a supported Linux OS for your hardware platform and a detected GPU.

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

## Step 3. Pull Qwen3.6

Download the [Qwen3.6](https://ollama.com/library/qwen3.6) model weights to your hardware platform.

```bash
ollama pull qwen3.6:35b-a3b-mtp-q4_K_M
```

The default is the MTP Q4_K_M variant recommended in the Supported hardware platforms matrix. Optional higher-memory variants:

```bash
ollama pull qwen3.6:35b-a3b-q8_0    # Higher-quality 8-bit quant (~39GB)
ollama pull qwen3.6:35b-a3b-bf16    # Full precision (~71GB)
```

Expected output should show `qwen3.6:35b-a3b-mtp-q4_K_M` (and any optional variants) in `ollama list`.

## Step 4. Test local inference (optional)

Run a quick prompt to confirm the model loads.

```bash
ollama run qwen3.6:35b-a3b-mtp-q4_K_M
```

Try a prompt like:

```text
Write a short README checklist for a Python project.
```

Expected output should show the model responding in the terminal. When you are done, type `/bye` or press `Ctrl+D` to exit the interactive session before continuing.

## Step 5. Install and launch Claude Code with Ollama

Install [Claude Code](https://docs.claude.com/en/docs/claude-code), then use Ollama's built-in [launch method](https://ollama.com/blog/launch) to start Claude Code against your local model. No environment variables or config files are required.

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude --version
```

If Claude Code is already installed, just verify the version:

```bash
claude --version
```

```bash
ollama launch claude --model qwen3.6:35b-a3b-mtp-q4_K_M
```

Expected output should show Claude Code starting and using the local Qwen3.6 model. Qwen3.6 ships with a 256K context window by default; adjust context length through Ollama's settings if you need to tune it further.

## Step 6. Complete a small coding task

Create a tiny repo and let Claude Code implement a function and tests.

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

## Step 7. Cleanup

Remove the model and stop services if you no longer need them. Cleanup is optional.

To stop the service:

```bash
sudo systemctl stop ollama
```

> [!WARNING]
> This will delete the downloaded model files.

```bash
ollama rm qwen3.6:35b-a3b-mtp-q4_K_M
```

## Step 8. Next steps

- Try the `q8_0` or `bf16` variants for higher-quality, higher-memory tradeoffs if your hardware platform has enough memory
- Use Claude Code on multi-file refactors or test-generation tasks
- Explore the full 256K context window on larger codebases
- Browse additional models in the [Ollama library](https://ollama.com/library)

## OpenCode

## Step 1. Confirm your environment

Verify the OS version and GPU are visible before installing anything.

```bash
cat /etc/os-release | head -n 2
nvidia-smi
```

Expected output should show a supported Linux OS for your hardware platform and a detected GPU.

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

## Step 3. Pull Qwen3.6

Download the [Qwen3.6](https://ollama.com/library/qwen3.6) model weights to your hardware platform.

```bash
ollama pull qwen3.6:35b-a3b-mtp-q4_K_M
```

The default is the MTP Q4_K_M variant recommended in the Supported hardware platforms matrix. Optional higher-memory variants:

```bash
ollama pull qwen3.6:35b-a3b-q8_0    # Higher-quality 8-bit quant (~39GB)
ollama pull qwen3.6:35b-a3b-bf16    # Full precision (~71GB)
```

Expected output should show `qwen3.6:35b-a3b-mtp-q4_K_M` in `ollama list`.

## Step 4. Test local inference (optional)

Run a quick prompt to confirm the model loads.

```bash
ollama run qwen3.6:35b-a3b-mtp-q4_K_M
```

Try a prompt like:

```text
Write a short README checklist for a Python project.
```

Expected output should show the model responding. When you are done, type `/bye` or press `Ctrl+D` to exit before continuing.

## Step 5. Install and launch OpenCode with Ollama

Install [OpenCode](https://opencode.ai) first, then use Ollama's built-in [launch method](https://ollama.com/blog/launch) to configure it against your local model. Ollama supplies the provider configuration, so no [`opencode.json`](https://opencode.ai/docs/config/) file is required.

```bash
curl -fsSL https://opencode.ai/install | bash
export PATH="$HOME/.opencode/bin:$PATH"
opencode --version
ollama launch opencode --model qwen3.6:35b-a3b-mtp-q4_K_M
```

The installer adds OpenCode under `~/.opencode/bin`. The `export` command makes it available in the current shell immediately.

If you want to pre-configure OpenCode without launching immediately:

```bash
ollama launch opencode --config
```

Expected output should show OpenCode starting with Ollama preselected as the provider and Qwen3.6 as the model. Qwen3.6 ships with a 256K context window by default.

## Step 6. Complete a small coding task

Create a tiny repo and let OpenCode implement a function and tests.

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

## Step 7. Cleanup

Remove the model and stop services if you no longer need them. Cleanup is optional.

To stop the service:

```bash
sudo systemctl stop ollama
```

> [!WARNING]
> This will delete the downloaded model files.

```bash
ollama rm qwen3.6:35b-a3b-mtp-q4_K_M
```

## Step 8. Next steps

- Try the `q8_0` or `bf16` variants for higher-quality, higher-memory tradeoffs if your hardware platform has enough memory
- Use OpenCode on multi-file changes or test-generation tasks
- Explore the full 256K context window on larger codebases
- Browse additional models in the [Ollama library](https://ollama.com/library)

## Codex CLI

## Step 1. Confirm your environment

Verify the OS version and GPU are visible before installing anything.

```bash
cat /etc/os-release | head -n 2
nvidia-smi
```

Expected output should show a supported Linux OS for your hardware platform and a detected GPU.

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

## Step 3. Pull Qwen3.6

Download the [Qwen3.6](https://ollama.com/library/qwen3.6) model weights to your hardware platform.

```bash
ollama pull qwen3.6:35b-a3b-mtp-q4_K_M
```

The default is the MTP Q4_K_M variant recommended in the Supported hardware platforms matrix. Optional higher-memory variants:

```bash
ollama pull qwen3.6:35b-a3b-q8_0    # Higher-quality 8-bit quant (~39GB)
ollama pull qwen3.6:35b-a3b-bf16    # Full precision (~71GB)
```

Expected output should show `qwen3.6:35b-a3b-mtp-q4_K_M` in `ollama list`.

## Step 4. Test local inference (optional)

Run a quick prompt to confirm the model loads.

```bash
ollama run qwen3.6:35b-a3b-mtp-q4_K_M
```

Try a prompt like:

```text
Write a short README checklist for a Python project.
```

Expected output should show the model responding. When you are done, type `/bye` or press `Ctrl+D` to exit before continuing.

## Step 5. Install and launch Codex CLI with Ollama

Install [Codex CLI](https://github.com/openai/codex), then use Ollama's built-in [launch method](https://ollama.com/blog/launch) to start it against your local model. Ollama configures the local-model integration, but the Codex CLI binary must be installed first. No `~/.codex/config.toml` is required.

```bash
npm install -g @openai/codex
codex --version
ollama launch codex --model qwen3.6:35b-a3b-mtp-q4_K_M
```

Expected output should show Codex CLI starting with Ollama as the provider and Qwen3.6 as the model. Qwen3.6 ships with a 256K context window by default, which is well suited to Codex's agentic workflows.

## Step 6. Complete a small coding task

Create a tiny repo and let Codex implement a function and tests.

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

## Step 7. Cleanup

Remove the model and stop services if you no longer need them. Cleanup is optional.

To stop the service:

```bash
sudo systemctl stop ollama
```

> [!WARNING]
> This will delete the downloaded model files.

```bash
ollama rm qwen3.6:35b-a3b-mtp-q4_K_M
```

## Step 8. Next steps

- Try the `q8_0` or `bf16` variants for higher-quality, higher-memory tradeoffs if your hardware platform has enough memory
- Use Codex CLI on multi-file changes or test-generation tasks
- Explore the full 256K context window on larger codebases
- Browse additional models in the [Ollama library](https://ollama.com/library)

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ollama: command not found` | Ollama not installed or PATH not updated | Rerun `curl -fsSL https://ollama.com/install.sh \| sh` and open a new shell |
| `ollama launch` reports unknown command | `ollama launch` not available in this install | Install or reinstall Ollama from [ollama.com/download](https://ollama.com/download): `curl -fsSL https://ollama.com/install.sh \| sh` |
| Model load fails with version error or HTTP 412 | Installed Ollama does not accept the default MTP Q4_K_M tag | Install or reinstall Ollama from [ollama.com/download](https://ollama.com/download): `curl -fsSL https://ollama.com/install.sh \| sh` |
| `model not found` when launching an agent | Model was not pulled | Run `ollama pull qwen3.6:35b-a3b-mtp-q4_K_M` and retry |
| `connection refused` to localhost:11434 | Ollama service not running | Start with `ollama serve` or `sudo systemctl start ollama` |
| `ollama launch <agent>` exits immediately | Agent integration failed to initialize | Re-run `ollama launch <agent>`; if it persists, check `journalctl -u ollama` |
| Slow responses or OOM errors | Model variant too large for available memory | Use the default `qwen3.6:35b-a3b-mtp-q4_K_M` variant and close other GPU workloads |
| `python3 -m pip install -U pytest` reports `externally-managed-environment` | System Python environment is protected | Create and activate a virtual environment first: `python3 -m venv .venv && source .venv/bin/activate` |
| `ollama pull` reports that a model tag is a sharded GGUF | The selected model tag is not supported by Ollama | Use the Qwen3.6 commands in Step 3 instead of sharded GGUF tags |
| `ollama run` fails with `CUDA error: context is destroyed` on a multi-GPU system | Ollama is initializing across a mixed-GPU topology | Pin Ollama to one GPU. For a foreground test, run `CUDA_VISIBLE_DEVICES=0 ollama serve`; for a system service, add `Environment="CUDA_VISIBLE_DEVICES=0"` to an Ollama systemd drop-in and restart Ollama |
| A direct Claude Code setup using an Anthropic-compatible Ollama endpoint produces prose but does not edit files | Some model/server combinations do not emit tool calls reliably | Use `ollama launch claude` with Qwen3.6 as shown in this playbook |
| OpenCode launch reports `opencode is not installed` or fails while fetching its version | OpenCode is not installed, or Ollama cannot complete its installer flow | Install OpenCode directly: `curl -fsSL https://opencode.ai/install \| bash`, then run `export PATH="$HOME/.opencode/bin:$PATH"` and retry `ollama launch opencode --model qwen3.6:35b-a3b-mtp-q4_K_M` |

> [!NOTE]
> Some hardware platforms use Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. With many applications still updating to take advantage of UMA, you may encounter memory pressure even when within rated capacity. If that happens, manually flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
