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

Use Ollama on DGX Spark to run local coding models and connect a CLI coding agent. This
playbook supports three options: **Claude Code**, **OpenCode**, and **Codex CLI**. Each
agent talks to Ollama for local inference, so you can work without external cloud APIs.

## Choose your CLI agent

Pick the tab that matches the CLI agent you want to use:

- **Claude Code**: Fastest path to a working CLI agent with a local Ollama model.
- **OpenCode**: Open-source CLI with provider configuration; this guide targets Ollama.
- **Codex CLI**: OpenAI Codex CLI configured to run against Ollama locally.

## What you'll accomplish

You will run a local coding model on your DGX Spark with Ollama, connect it to your
chosen CLI agent, and complete a small coding task end-to-end.

## What to know before starting

- Comfort with Linux command line basics
- Experience running terminal-based tools and editors
- Familiarity with Python for the short coding task

## Prerequisites

- DGX Spark access with NVIDIA DGX OS 7.3.1 (Ubuntu 24.04.3 LTS base)
- Internet access to download model weights
- Ollama 0.14.3 or newer
- GPU memory depends on the model you choose. Example requirements for GLM-4.7-Flash:
  - 19GB+ for `glm-4.7-flash:latest`
  - 32GB+ for `glm-4.7-flash:q8_0`
  - 60GB+ for `glm-4.7-flash:bf16`

## Time & risk

* **Duration**: ~20-30 minutes (includes model download time)
* **Risk level**: Low
  * Large model downloads can fail if network connectivity is unstable
  * Older Ollama versions will not load the model
* **Rollback**: Stop Ollama and delete the downloaded model from `~/.ollama/models`
* **Last Updated:** 01/21/2026
  * First publication

## Claude Code

## Step 1. Confirm your environment

**Description**: Verify the OS version and GPU are visible before installing anything.

```bash
cat /etc/os-release | head -n 2
nvidia-smi
```

Expected output should show Ubuntu 24.04.3 LTS (DGX OS 7.3.1 base) and a detected GPU.

## Step 2. Install or update Ollama

**Description**: Install Ollama or ensure it is recent enough for modern coding models.

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.14.3 sh
ollama --version
```

If the ollama is already present and the version is 0.14.3 or newer, simply run:

```bash
ollama --version
```

Expected output should show `ollama --version` as 0.14.3 or newer.

## Step 3. Pull GLM-4.7-Flash

**Description**: Download the model weights to your Spark node.

```bash
ollama pull glm-4.7-flash
```

Optional variants if you need different memory footprints:

```bash
ollama pull glm-4.7-flash:q4_K_M
ollama pull glm-4.7-flash:q8_0
ollama pull glm-4.7-flash:bf16
```

Expected output should show `glm-4.7-flash` (and any optional variants you pulled) in `ollama list`.

## Step 4. Test local inference

**Description**: Run a quick prompt to confirm the model loads.

```bash
ollama run glm-4.7-flash
```

Try a prompt like:

```text
Write a short README checklist for a Python project.
```

Expected output should show the model responding in the terminal.

## Step 5. Install Claude Code

**Description**: Install the CLI tool that will drive the local model.

```bash
curl -fsSL https://claude.ai/install.sh | sh
```

## Step 6. Increase context length (optional)

**Description**: Ollama defaults to a 4096 token context length. For coding agents and
larger codebases, set it to 64K tokens. This increases memory usage.
For more details on configuring context length, see the [Ollama documentation](https://ollama.com/docs/faq#how-can-i-increase-the-context-length).

Set the context length per session in the Ollama REPL:

```bash
ollama run glm-4.7-flash
```

Then, in the Ollama prompt:

```text
/set parameter num_ctx 64000

```

Optional method (set globally when serving Ollama):

```bash
sudo systemctl stop ollama
OLLAMA_CONTEXT_LENGTH=64000 ollama serve 
```

Keep this terminal open and run the next step in a new terminal.

## Step 7. Connect Claude Code to Ollama

**Description**: Point Claude Code to the local Ollama server and launch it.

```bash
export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_BASE_URL=http://localhost:11434

claude --model glm-4.7-flash
```

Expected output should show Claude Code starting and using the local model.

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

In Claude Code:

```text
Please implement add() in math_utils.py and make sure the test passes.
```

Run the test:

```bash
python -m pytest -q
```

Expected output should show the test passing.

## Step 9. Cleanup and rollback

**Description**: Remove the model and stop services if you no longer need them.

To stop the service:

```bash
sudo systemctl stop ollama
```

> [!WARNING]
> This will delete the downloaded model files.

```bash
ollama rm glm-4.7-flash
```

## Step 10. Next steps

- Try larger code tasks with the 198K context window
- Experiment with `glm-4.7-flash:q8_0` or `glm-4.7-flash:bf16` for higher quality
- Use Claude Code on multi-file refactors or test-generation tasks

## OpenCode

## Step 1. Confirm your environment

**Description**: Verify the OS version and GPU are visible before installing anything.

```bash
cat /etc/os-release | head -n 2
nvidia-smi
```

Expected output should show Ubuntu 24.04.3 LTS (DGX OS 7.3.1 base) and a detected GPU.

## Step 2. Install or update Ollama

**Description**: Install Ollama or ensure it is recent enough for modern coding models.

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.14.3 sh
ollama --version
```

If Ollama is already installed and the version is 0.14.3 or newer, simply run:

```bash
ollama --version
```

Expected output should show `ollama --version` as 0.14.3 or newer.

## Step 3. Pull a coding model

**Description**: Download a local coding model to your Spark node.

```bash
ollama pull glm-4.7-flash
```

Optional variants if you need different memory footprints:

```bash
ollama pull glm-4.7-flash:q4_K_M
ollama pull glm-4.7-flash:q8_0
ollama pull glm-4.7-flash:bf16
```

Expected output should show your model in `ollama list`.

## Step 4. Install OpenCode

**Description**: Install the OpenCode CLI using the official Linux instructions.

Follow the install guide at https://opencode.ai/docs, then verify:

```bash
opencode --version
```

## Step 5. Configure OpenCode to use Ollama

**Description**: Point OpenCode to your local Ollama server with an `opencode.json`.

Create `opencode.json` in your project directory (or the location you prefer for OpenCode config):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "glm-4.7-flash": {
          "name": "glm-4.7-flash"
        }
      }
    }
  }
}
```

Replace `glm-4.7-flash` with the model you pulled. If Ollama is running on another host,
update the `baseURL` accordingly.

## Step 6. Increase context length (optional)

**Description**: Ollama defaults to a 4096 token context length. For coding agents and
larger codebases, set it to 64K tokens. This increases memory usage.
For more details, see the [Ollama documentation](https://ollama.com/docs/faq#how-can-i-increase-the-context-length).

Set the context length per session in the Ollama REPL:

```bash
ollama run glm-4.7-flash
```

Then, in the Ollama prompt:

```text
/set parameter num_ctx 64000

```

Optional method (set globally when serving Ollama):

```bash
sudo systemctl stop ollama
OLLAMA_CONTEXT_LENGTH=64000 ollama serve 
```

Keep this terminal open and run the next step in a new terminal.

## Step 7. Launch OpenCode

**Description**: Start the OpenCode CLI and select the Ollama provider and model.

```bash
opencode
```

If prompted, select the Ollama provider and the model you configured.

## Step 8. Complete a small coding task

**Description**: Create a tiny repo and let OpenCode implement a function and tests.

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

In OpenCode:

```text
Please implement add() in math_utils.py and make sure the test passes.
```

Run the test:

```bash
python -m pytest -q
```

Expected output should show the test passing.

## Step 9. Cleanup and rollback

**Description**: Remove the model and stop services if you no longer need them.

To stop the service:

```bash
sudo systemctl stop ollama
```

> [!WARNING]
> This will delete the downloaded model files.

```bash
ollama rm glm-4.7-flash
```

## Step 10. Next steps

- Try other coding models available in Ollama
- Experiment with higher context lengths for larger refactors
- Use OpenCode on multi-file changes or test-generation tasks

## Codex CLI

## Step 1. Confirm your environment

**Description**: Verify the OS version and GPU are visible before installing anything.

```bash
cat /etc/os-release | head -n 2
nvidia-smi
```

Expected output should show Ubuntu 24.04.3 LTS (DGX OS 7.3.1 base) and a detected GPU.

## Step 2. Install or update Ollama

**Description**: Install Ollama or ensure it is recent enough for modern coding models.

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.14.3 sh
ollama --version
```

If Ollama is already installed and the version is 0.14.3 or newer, simply run:

```bash
ollama --version
```

Expected output should show `ollama --version` as 0.14.3 or newer.

## Step 3. Install Codex CLI

**Description**: Install the Codex CLI.

```bash
npm install -g @openai/codex
codex --version
```

## Step 4. Start Codex with Ollama

**Description**: Launch Codex with the OSS flag to use Ollama.

```bash
codex --oss
```

By default, Codex uses the local `gpt-oss:20b` model.

## Step 5. Optional settings

**Description**: Adjust the model or context length if needed.

To use GLM-4.7-Flash with Codex, pull the model and start Codex with `-m`:

```bash
ollama pull glm-4.7-flash
codex --oss -m glm-4.7-flash
```

To switch to other models, use the `-m` flag:

```bash
codex --oss -m gpt-oss:120b
```

To use a cloud model:

```bash
codex --oss -m gpt-oss:120b-cloud
```

Codex works best with a large context window. We recommend 64K tokens.
For more details, see the [Ollama documentation](https://ollama.com/docs/faq#how-can-i-increase-the-context-length).

Set the context length per session in the Ollama REPL:

```bash
ollama run glm-4.7-flash
```

Then, in the Ollama prompt:

```text
/set parameter num_ctx 64000

```

Optional method (set globally when serving Ollama):

```bash
sudo systemctl stop ollama
OLLAMA_CONTEXT_LENGTH=64000 ollama serve 
```

Replace `glm-4.7-flash` with the model you are using (for example, `gpt-oss:20b`).

Keep this terminal open and run the next step in a new terminal.

## Step 6. Advanced configuration (optional)

**Description**: Set defaults or point Codex at a remote Ollama server.

Create or edit `~/.codex/config.toml`:

```toml
model = "glm-4.7-flash"
model_provider = "ollama"

[model_providers.ollama]
base_url = "http://localhost:11434/v1"
```

If Ollama is running on another host, update the `base_url` accordingly. You can set
`model` to any Ollama model you want Codex to use.

## Step 7. Complete a small coding task

**Description**: Create a tiny repo and let Codex implement a function and tests.

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

In Codex:

```text
Please implement add() in math_utils.py and make sure the test passes.
```

Run the test:

```bash
python -m pytest -q
```

Expected output should show the test passing.

## Step 8. Cleanup and rollback

**Description**: Remove the model and stop services if you no longer need them.

To stop the service:

```bash
sudo systemctl stop ollama
```

> [!WARNING]
> This will delete the downloaded model files.

```bash
ollama rm gpt-oss:20b
```

Replace `gpt-oss:20b` with the model you used.

## Step 9. Next steps

- Try other Ollama coding models with Codex CLI
- Experiment with higher context lengths for larger refactors
- Use Codex CLI on multi-file changes or test-generation tasks

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ollama: command not found` | Ollama not installed or PATH not updated | Rerun `curl -fsSL https://ollama.com/install.sh | sh` and open a new shell |
| Model load fails with version error | Ollama is older than 0.14.3 | Update Ollama to 0.14.3 or newer |
| `model not found` in Claude Code | Model was not pulled | Run `ollama pull glm-4.7-flash` and retry |
| `opencode: command not found` | OpenCode not installed or PATH not updated | Install OpenCode and open a new shell |
| OpenCode cannot reach Ollama | `baseURL` misconfigured or Ollama not running | Set `baseURL` to `http://localhost:11434/v1` and start Ollama |
| `codex: command not found` | Codex CLI not installed or PATH not updated | Install Codex CLI and open a new shell |
| Codex CLI uses the wrong model/provider | `~/.codex/config.toml` not pointing to Ollama | Set `model_provider = "ollama"` and `base_url = "http://localhost:11434/v1"` |
| `connection refused` to localhost:11434 | Ollama service not running | Start with `ollama serve` or `systemctl start ollama` |
| Slow responses or OOM errors | Model variant too large for GPU memory | Use `glm-4.7-flash:q4_K_M` or close other GPU workloads |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing
> between the GPU and CPU. If you see memory pressure, flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```
