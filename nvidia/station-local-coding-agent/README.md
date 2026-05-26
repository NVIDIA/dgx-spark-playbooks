# Local Coding Agent

> Run local CLI coding agents with Ollama on DGX Station (NVIDIA GB300) using glm-4.7-flash (fast) or unsloth/GLM-4.7-GGUF:Q8_0 (best quality)


## Table of Contents

- [Overview](#overview)
- [Claude Code](#claude-code)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Use Ollama on **DGX Station (NVIDIA GB300)** to run local coding models and connect a CLI coding agent. This
playbook uses **Claude Code** to talk to Ollama for local inference, so you can work without external cloud APIs.

The DGX Station GPU (reported as **NVIDIA GB300** in `nvidia-smi`) provides ample memory to run **glm-4.7-flash** (fast loading and testing) and larger models such as **unsloth/GLM-4.7-GGUF:Q8_0** (best quality), both supported on Ollama.

## CLI agent

This playbook uses **Claude Code** as the CLI agent, connected to a local Ollama model for inference.

## What you'll accomplish

You will run a local coding model on your **DGX Station (NVIDIA GB300)** with Ollama, connect Claude Code to it, and complete a small coding task end-to-end. Use **glm-4.7-flash** (including high-quality variants) or **unsloth/GLM-4.7-GGUF:Q8_0** for best quality.

## What to know before starting

- Comfort with Linux command line basics
- Experience running terminal-based tools and editors
- Familiarity with Python for the short coding task

## Prerequisites

- **DGX Station** with **NVIDIA GB300** (Grace Blackwell) and NVIDIA driver; `nvidia-smi` typically shows "NVIDIA GB300"
- Internet access to download model weights
- **Ollama 0.15.0 or newer** (required for GLM-4.7-Flash; do not pin to 0.14.3)
- **GPU memory** on GB300 supports both recommended models:
  - **glm-4.7-flash**: ~19 GB (`latest`) to ~60 GB (bf16) — **recommended for fast loading and testing**
  - **unsloth/GLM-4.7-GGUF:Q8_0** (Hugging Face on Ollama): larger model — **recommended for best quality**
  - Other variants (e.g. `glm-4.7-flash:bf16`, `glm-4.7-flash:q8_0`) fit on GB300
- **Disk space** for model downloads: plan for ~19 GB for `glm-4.7-flash:latest`, plus additional space for the Q8_0 or bf16 variants if you use them

## Time & risk

* **Duration**: ~20–30 minutes (includes model download)
* **Risk level**: Low
  * Large model downloads can fail if network connectivity is unstable
  * Older Ollama versions will not load newer models
* **Rollback**: Stop Ollama and delete the downloaded model from `~/.ollama/models`
* **Last Updated:** 03/06/2026
  * Model set to glm-4.7-flash; Ollama 0.15.0+; cleanup order and docs refresh

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

To install a specific version (e.g. 0.15.0 or newer, required for GLM-4.7-Flash):

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.15.0 sh
```

If Ollama is already present and the version is 0.15.0 or newer, simply run:

```bash
ollama --version
```

**Expected output** (example):

```text
ollama version is 0.15.0
```

## Step 3. Pull a coding model

**Description**: Download the model weights to your DGX Station. This playbook supports two model options on Ollama; choose one (or both) depending on whether you want **fast loading and testing** or **best quality**.

**For fast loading and testing** — **glm-4.7-flash** (~19 GB for `latest`; loads quickly; ensure Ollama 0.15.0+):

```bash
ollama pull glm-4.7-flash
```

**For best quality** — **unsloth/GLM-4.7-GGUF:Q8_0** from Hugging Face (larger, higher quality; supported on Ollama):

```bash
ollama pull hf.co/unsloth/GLM-4.7-GGUF:Q8_0
```

**Other glm-4.7-flash variants** on GB300 (more GPU memory; bf16 is ~60 GB):

```bash
ollama pull glm-4.7-flash:q8_0
ollama pull glm-4.7-flash:bf16
```

**Expected output** (example): Progress lines followed by "success" and the model in `ollama list`:

```bash
ollama list
```

```text
NAME                                ID              SIZE    MODIFIED
glm-4.7-flash:latest                abc123...       19 GB   1 minute ago
unsloth/GLM-4.7-GGUF:Q8_0           def456...       ...    ...
```

## Step 4. Test local inference

**Description**: Run a quick prompt to confirm the model loads. Use the same model name you pulled (e.g. `glm-4.7-flash` for fast testing, or `hf.co/unsloth/GLM-4.7-GGUF:Q8_0` for best quality).

```bash
ollama run glm-4.7-flash
```

Or, if you pulled the larger model:

```bash
ollama run hf.co/unsloth/GLM-4.7-GGUF:Q8_0
```

Try a prompt like:

```text
Write a short README checklist for a Python project.
```

**Expected output**: GLM-4.7-Flash may show **Thinking...** and reasoning text before the final answer, then the model's response. This is normal; wait for the reply to complete.

**Exit the Ollama REPL** when done: type `/bye` or press **Ctrl+D**.

## Step 5. Install Claude Code

**Description**: Install the CLI tool that will drive the local model.

```bash
curl -fsSL https://claude.ai/install.sh | sh
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

Set the context length per session in the Ollama REPL (use the same model name you pulled, e.g. `glm-4.7-flash` or `hf.co/unsloth/GLM-4.7-GGUF:Q8_0`):

```bash
ollama run glm-4.7-flash
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

**Description**: Point Claude Code to the local Ollama server and launch it. Use the model you pulled: `glm-4.7-flash` (fast) or `hf.co/unsloth/GLM-4.7-GGUF:Q8_0` (best quality).

```bash
export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_BASE_URL=http://localhost:11434

claude --model glm-4.7-flash
```

If you are using the larger model:

```bash
claude --model hf.co/unsloth/GLM-4.7-GGUF:Q8_0
```

- **`ANTHROPIC_AUTH_TOKEN=ollama`**: Claude Code treats the literal value `ollama` as a special token that means "use the local Ollama backend" instead of Anthropic's cloud API. No real API key is needed when using Ollama.
- **`ANTHROPIC_BASE_URL`**: Tells Claude Code to send requests to your local Ollama server at port 11434.

**Persist these variables** (optional) so you don't have to re-export every terminal session. Add to `~/.bashrc` or your shell profile (e.g. `~/.zshrc`):

```bash
echo 'export ANTHROPIC_AUTH_TOKEN=ollama' >> ~/.bashrc
echo 'export ANTHROPIC_BASE_URL=http://localhost:11434' >> ~/.bashrc
source ~/.bashrc
```

**Expected output**: Claude Code starts and uses the local model.

**Exit Claude Code** when done: type `/exit` or press **Ctrl+C**.

## Step 8. Complete a small coding task

**Description**: Create a tiny repo and let Claude Code implement a function and tests.

```bash
mkdir -p ~/cli-agent-demo
cd ~/cli-agent-demo

printf 'def add(a, b):\n    """Return the sum of a and b."""\n    pass\n' > math_utils.py
printf 'import math_utils\n\n\ndef test_add():\n    assert math_utils.add(1, 2) == 3\n' > test_math_utils.py
```

If you do not already have pytest installed:

```bash
python -m pip install -U pytest
```

In Claude Code, enter:

```text
Please implement add() in math_utils.py and make sure the test passes.
```

**Exit Claude Code** when finished: type `/exit` or press **Ctrl+C**, then run the test:

```bash
python -m pytest -q
```

Expected output should show the test passing.

## Step 9. Cleanup and rollback

**Description**: Remove the model and stop the Ollama service if you no longer need them. **Remove the model first** (while the Ollama server is running), then stop the service.

> [!WARNING]
> The following removes the downloaded model files from disk.

**1. Remove the model** (Ollama must be running). Use the same name you pulled:

```bash
ollama rm glm-4.7-flash
```

Or, for the Hugging Face model:

```bash
ollama rm hf.co/unsloth/GLM-4.7-GGUF:Q8_0
```

Use the exact tag you pulled (e.g. `glm-4.7-flash:bf16` if you used that variant).

**2. Stop the Ollama service**:

```bash
sudo systemctl stop ollama
```

## Step 10. Next steps

- **Fast loading and testing:** use **glm-4.7-flash** for quick iteration and smaller downloads.
- **Best quality:** use **unsloth/GLM-4.7-GGUF:Q8_0** (Hugging Face on Ollama) or **glm-4.7-flash** high-quality variants (`glm-4.7-flash:bf16`, `glm-4.7-flash:q8_0`) on DGX Station (NVIDIA GB300).
- Use larger context (e.g. 64K–198K) for big codebases.
- Use Claude Code on multi-file refactors or test-generation tasks.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ollama: command not found` | Ollama not installed or PATH not updated | Rerun `curl -fsSL https://ollama.com/install.sh | sh` and open a new shell |
| Model load fails with version error | Ollama is older than 0.15.0 | Update Ollama to 0.15.0 or newer (required for GLM-4.7-Flash). Do not pin to 0.14.3. |
| `model not found` in Claude Code | Model was not pulled | Run `ollama pull glm-4.7-flash` or `ollama pull hf.co/unsloth/GLM-4.7-GGUF:Q8_0` and retry. Use the same model name in `claude --model ...`. |
| `connection refused` to localhost:11434 | Ollama service not running | Start with `ollama serve` or `sudo systemctl start ollama` |
| Slow responses or OOM | Insufficient GPU memory or fragmentation | On DGX Station (NVIDIA GB300), ensure no other heavy GPU workloads. If OOM persists, use a smaller variant (e.g. `glm-4.7-flash:q8_0` or `glm-4.7-flash:q4_K_M`) or `OLLAMA_MAX_LOADED_MODELS=1`. |
| `claude: command not found` after install | CLI not on PATH or install script did not complete | Restart the terminal or run `source ~/.bashrc` (or your shell profile). Check the install script output for the install path and add it to PATH. |
| Claude Code install fails (Node.js / network) | Node.js missing or install script cannot download | Ensure Node.js is installed (`node --version`). If the install script fails with a network error, retry from a stable connection or download the Claude Code CLI from the official site. See [Claude Code documentation](https://claude.ai/docs) for alternatives. |

> [!NOTE]
> DGX Station with **NVIDIA GB300** provides ample GPU memory for **glm-4.7-flash** (fast testing) and **unsloth/GLM-4.7-GGUF:Q8_0** (best quality), plus variants (e.g. `glm-4.7-flash:bf16`). Use `OLLAMA_MAX_LOADED_MODELS=1` if you hit memory limits with multiple models.
