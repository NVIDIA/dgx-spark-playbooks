# Run OpenClaw with a Local LLM

> Install a local-first AI agent and connect it to a private OpenAI-compatible model endpoint


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Option A — vLLM (recommended on large-memory Linux hardware platforms)](#option-a-vllm-recommended-on-large-memory-linux-hardware-platforms)
  - [Option B — LM Studio (simple path on discrete-GPU hardware platforms)](#option-b-lm-studio-simple-path-on-discrete-gpu-hardware-platforms)
  - [Option C — Ollama](#option-c-ollama)
- [Agent-ready Models](#agent-ready-models)
  - [Recommendations by hardware platform](#recommendations-by-hardware-platform)
  - [Context window guidance](#context-window-guidance)
  - [Large-memory Linux hardware platforms — vLLM agent-ready path](#large-memory-linux-hardware-platforms-vllm-agent-ready-path)
  - [Discrete-GPU hardware platforms — LM Studio / Ollama / vLLM](#discrete-gpu-hardware-platforms-lm-studio-ollama-vllm)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

OpenClaw is a **local-first** AI agent that runs on your machine. It remembers conversations, adapts to your usage, runs continuously, uses context from your files and apps, and can be extended with community **skills**.

Running OpenClaw with a **local LLM** keeps your data private and avoids ongoing cloud API costs. Your hardware platform provides the GPU acceleration that agent workflows need for responsive tool calling and multi-turn sessions.

## What you'll accomplish

You will install OpenClaw on your hardware platform, connect it to a local OpenAI-compatible model endpoint, and verify the agent in the OpenClaw web UI. Optionally, you can add communication channels and skills. The agent and models run on your hardware—no data leaves your machine unless you add cloud or external integrations.

## Popular use cases

- **Personal secretary**: With access to your inbox, calendar, and files, OpenClaw can help manage your schedule, draft replies, send reminders, and find meeting slots.
- **Proactive project management**: Check project status over email or messaging, send status updates, and follow up or send reminders.
- **Research agent**: Combine web search and your local files to produce reports with personalized context.
- **Install helper**: Search for apps/libraries, run installations, and debug errors using terminal access (larger models recommended).

## What to know before starting

**Required:**

- Basic use of the terminal and a text editor
- Awareness of the security considerations below

**Optional:**

- Familiarity with a local inference backend (vLLM, Ollama, or LM Studio) if you plan to use a local model

## Important: security and risks

AI agents can introduce real risks. Read OpenClaw’s guidance: [OpenClaw Gateway Security](https://docs.openclaw.ai/gateway/security).

Main risks:

1. **Data exposure**: Personal information or files may be leaked or stolen.
2. **Malicious code**: The agent or connected tools may expose you to malware or attacks.

You cannot eliminate all risk; proceed at your own risk. **Critical security measures:**

- **STRONGLY RECOMMENDED:** Run OpenClaw on a dedicated or isolated system (for example, a clean hardware platform or VM) and only copy in the data the agent needs. Do not run this on your primary workstation with sensitive data.
- Use **dedicated accounts** for the agent instead of your main accounts; grant only the minimum access it needs.
- Enable only **skills you trust**, preferably those vetted by the community. Skills that provide terminal or file system access increase risk significantly.
- **CRITICAL:** Ensure the OpenClaw web UI and any messaging channels are **never exposed** to the public internet without strong authentication. Use SSH tunneling or VPN if accessing remotely.
- Where possible, **limit internet access** for the agent using firewall rules or network isolation.
- **Monitor activity**: Regularly review logs and commands executed by the agent.

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, OS, memory, and whether multi-node applies.

| Hardware platform | OS | Memory  | Multi-node capable hardware |
| :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | — |
| **RTX** | Linux, Windows, WSL | Dedicated VRAM (size varies) | — |


> [!NOTE]
> Only platforms listed in the Supported hardware platforms table above are covered by this playbook.

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Enough memory for your chosen model (see **Agent-ready Models** for recommendations)

**Software requirements**

- Terminal access to the hardware platform (local or SSH)
- A local OpenAI-compatible inference backend, or willingness to install one during the Instructions tab
- Network access to download OpenClaw and your chosen model checkpoint

## Find model guidance

This playbook is an **agent** workflow, not a general inference-serving guide. For recommended local models by hardware platform, use the **Agent-ready Models** tab. For full vLLM container recipes and launch settings, see [Serve LLMs with vLLM](https://build.nvidia.com/playbooks/vllm). For Ollama or LM Studio setup, see those playbooks in **Resources**.

## Time & risk

- **Estimated time:** 30 MIN for install and first-time model setup; model download time depends on size and network
- **Risk level:** **Medium to High**—the agent has access to whatever files, tools, and channels you configure. Risk increases significantly if you enable terminal/command execution skills or connect external accounts. Without proper isolation, this setup could expose sensitive data or allow code execution. **Always follow the security measures above.**
- **Rollback:** Stop the OpenClaw gateway and uninstall via the same install script or by removing its directory; stop the local inference server separately if desired
- **Last Updated:** 07/27/2026
  - Added supported hardware matrix and Agent-ready Models tab for DGX Spark and RTX

## Instructions

> [!CAUTION]
> **Before proceeding, review the security risks in the Overview tab.** OpenClaw is an AI agent that can access your files, execute commands, and connect to external services. Data exposure and malicious code execution are real risks. **Strongly recommended:** Run OpenClaw on an isolated system or VM, use dedicated accounts (not your main accounts), and never expose the dashboard to the public internet without authentication.

## Step 1. Prepare your environment (Windows / WSL only)

> [!NOTE]
> Skip this step on Linux hardware platforms.

If you are on Windows, you can install OpenClaw on native Windows or in Windows Subsystem for Linux (WSL). WSL provides a Linux-style environment if your skills need it; native Windows is often simpler and can connect more easily to Windows apps.

If you choose WSL and it is not already installed:

1. Open **PowerShell as Administrator**.
2. Install WSL:

```powershell
wsl --install
```

3. Verify:

```powershell
wsl --version
```

4. Start WSL:

```powershell
wsl
```

## Step 2. Install OpenClaw

On your hardware platform, open a terminal and run the official install script:

```bash
curl -fsSL https://openclaw.ai/install.sh | bash
```

On native Windows, follow the install options documented at [openclaw.ai](https://openclaw.ai) if the curl install path does not apply.

After dependencies are downloaded, OpenClaw will show a **security warning**. Read the risks; if you accept them, use the arrow keys to select **Yes** and press Enter.

## Step 3. Complete the OpenClaw onboarding

Work through the prompts as follows.

1. **Quickstart vs Manual**: Choose **Quickstart**.

2. **Model provider**:
   - **Recommended for a local model:** If your backend is not listed yet (or you will configure it after the server is up), go to the bottom of the list and select **Skip for now**—you’ll configure the model in a later step.
   - If your inference backend appears in the list (for example Ollama, LM Studio, or vLLM), you can select it now and fill in API key (leave blank or use a dummy value), Base URL (default is usually fine), and Model ID (must match the exact handle served by the backend).

3. **Filtering models by provider**: If prompted, select **All Providers**. On the next prompt for the default model, choose **Keep Current** unless you already selected a provider.

4. **Communication channel**: You can connect a channel (for example messaging) to use the bot when away from the machine, or select **Skip for Now** and configure it later.

5. **Skills**: We recommend selecting **No** for now. You can add skills later from the web UI or Clawhub after you’ve tested the basics.

6. **Homebrew**: If you are prompted to install Homebrew, select **No**—Homebrew is for macOS only and is not needed on Linux or Windows for this playbook.

7. **Hooks**: We recommend selecting all three for a better experience. Note that this may log data locally; enable only if you’re comfortable with that.

8. **Dashboard URL**: The terminal will print a URL for the OpenClaw dashboard. **Save this URL** (and any access token shown)—you’ll need it to open the web UI.

9. **Finish**: Select **Yes** on the final prompt to complete installation.

You can now open the OpenClaw dashboard in a browser using the URL and token from the installer.

## Step 4. Serve a local model

OpenClaw connects to a local, OpenAI-compatible endpoint. Pick a backend that fits your hardware platform, then use the matching subsection. Model recommendations are in the **Agent-ready Models** tab.

### Option A — vLLM (recommended on large-memory Linux hardware platforms)

Use when you want an OpenAI-compatible HTTP server and validated agent-ready recipes.

1. In a **separate terminal**, launch the recommended model for your hardware platform from the **Agent-ready Models** tab (or follow [Serve LLMs with vLLM](https://build.nvidia.com/playbooks/vllm) for container setup).
2. Wait until the server reports startup complete, then verify:

```bash
curl http://localhost:8000/v1/models
```

You should see your model handle in the returned list. The default OpenClaw `baseUrl` for this path is `http://localhost:8000/v1`.

### Option B — LM Studio (simple path on discrete-GPU hardware platforms)

Install LM Studio, download a model that fits your VRAM (see the **Agent-ready Models** table — the 27B example below is for 24GB+ only), and serve it with a large context window (32K minimum; 64K+ recommended when VRAM allows):

```bash
curl -fsSL https://lmstudio.ai/install.sh | bash
lms get qwen/qwen3.6-27b
lms load qwen/qwen3.6-27b --context-length 65536
lms server start
```

Use the Model ID that matches what LM Studio serves. Prefer this path when you want a GUI-oriented, llama.cpp-backed workflow.

### Option C — Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.6:27b
ollama run qwen3.6:27b
```

Inside the Ollama session (or via your usual Ollama config), set context high enough for agent use—for example `/set parameter num_ctx 65536`. Use the exact Ollama model tag as the OpenClaw Model ID.

> [!NOTE]
> Free as much VRAM as possible before loading the model (close other GPU workloads; enable only the skills you need). Smaller GPUs should use smaller models—see **Agent-ready Models**.

## Step 5. Configure OpenClaw to use the local server

If you already selected a provider during onboarding and chat works, you can skip to **Step 6**.

Otherwise, open the OpenClaw config file:

```bash
~/.openclaw/openclaw.json
```

Example with nano:

```bash
nano ~/.openclaw/openclaw.json
```

Add or update the `models` section so it includes your provider. Example for a vLLM server (no API key required—any non-empty placeholder works):

```json
"models": {
  "mode": "merge",
  "providers": {
    "vllm": {
      "baseUrl": "http://localhost:8000/v1",
      "apiKey": "vllm",
      "api": "openai-responses",
      "models": [
        {
          "id": "nvidia/Qwen3.6-35B-A3B-NVFP4",
          "name": "nvidia/Qwen3.6-35B-A3B-NVFP4",
          "reasoning": true,
          "input": ["text"],
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "contextWindow": 262144,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

Replace `id`, `name`, `baseUrl`, and `contextWindow` to match the model and backend you launched. The `id` and `name` must match the handle served by the backend.

> [!NOTE]
> If OpenClaw reports an unsupported-endpoint error against the Responses API, change `"api": "openai-responses"` to the OpenAI chat-completions variant for your OpenClaw version — vLLM always exposes `/v1/chat/completions`.
>
> If OpenClaw runs in WSL and cannot reach a server on the Windows host, replace `127.0.0.1` / `localhost` with the Windows host IP as seen from WSL.

If the OpenClaw gateway is already running, restart it so it reloads `~/.openclaw/openclaw.json`.

## Step 6. Verify the setup

1. In a browser, open the **OpenClaw dashboard URL** (and use the access token if required).
2. Start a **new** conversation and send a short message.
3. If you get a reply from the agent, the setup is working.

You can also ask OpenClaw which model it’s using. In the gateway chat UI you can switch models by typing: **`/model MODEL_NAME`**.

## Step 7. Optional: add skills and learn more

- **Skills** add capabilities but also risk; only enable skills you trust (for example, community-vetted ones). To add a skill:
  - Ask OpenClaw to configure a skill, or
  - Use the sidebar in the web UI to enable skills, or
  - Browse [Clawhub](https://docs.openclaw.ai/tools/clawhub) for community skills.

- For more usage and configuration details, see the [OpenClaw documentation](https://docs.openclaw.ai).

## Agent-ready Models

## Agent-ready models

Agent-ready models are tuned for **agentic workloads** — tool calling, reasoning traces, and long multi-turn sessions. Use this tab to pick a recommended local model for your hardware platform, then return to the **Instructions** tab to connect OpenClaw to the OpenAI-compatible endpoint.

### Recommendations by hardware platform

| Hardware platform | Recommended model | Example handle / tag | Inference path |
| ----------------- | ----------------- | -------------------- | -------------- |
| **DGX Spark** | Agent-ready Qwen3.6-35B-A3B (NVFP4) | `nvidia/Qwen3.6-35B-A3B-NVFP4` | vLLM — see [Serve LLMs with vLLM → Agent-ready Models](https://build.nvidia.com/playbooks/vllm/agent-ready-models) |
| **RTX** (24GB+ VRAM) | Qwen3.6 27B | `qwen/qwen3.6-27b` (LM Studio) or `qwen3.6:27b` (Ollama) | LM Studio, Ollama, or vLLM |
| **RTX** (12–16GB VRAM) | Qwen 3.5 9B / Gemma 4 12B | backend-specific tags | LM Studio or Ollama |
| **RTX** (6–8GB VRAM) | Qwen 3.5 4B | backend-specific tags | LM Studio or Ollama |

Only platforms listed in the Supported hardware platforms table (Overview) are listed above.

### Context window guidance

For OpenClaw, set the local server context window to **at least 32K tokens**. If your hardware platform has additional memory headroom, **64K or higher** is recommended so multi-turn agent sessions and skills have room to work.

### Large-memory Linux hardware platforms — vLLM agent-ready path

On large-memory Linux hardware platforms, use the agent-ready `nvidia/Qwen3.6-35B-A3B-NVFP4` recipe from the Serve LLMs with vLLM playbook’s [Agent-ready Models](https://build.nvidia.com/playbooks/vllm/agent-ready-models) tab. That path serves an OpenAI-compatible API (typically at `http://localhost:8000/v1`) suitable for OpenClaw.

1. Follow the Serve LLMs with vLLM playbook’s [Agent-ready Models](https://build.nvidia.com/playbooks/vllm/agent-ready-models) tab for the launch command for your hardware platform.
2. Verify:

```bash
curl http://localhost:8000/v1/models
```

3. In OpenClaw, set provider `baseUrl` to `http://localhost:8000/v1` and set `id` / `name` to `nvidia/Qwen3.6-35B-A3B-NVFP4`. Match `contextWindow` to the server `--max-model-len` (the agent-ready recipe uses a large context suitable for multi-turn agent sessions).

### Discrete-GPU hardware platforms — LM Studio / Ollama / vLLM

On discrete-GPU hardware platforms, choose a model that fits available VRAM (table above), then serve it with your preferred backend:

- **LM Studio** — easy GUI path; MTP may be enabled by default on supported models.
- **Ollama** — simple CLI path.
- **vLLM** — maximum configurability on Linux; use when you want OpenAI-compatible HTTP serving and custom flags.

Example LM Studio download + serve (24GB+ recommendation):

```bash
lms get qwen/qwen3.6-27b
lms load qwen/qwen3.6-27b --context-length 65536
lms server start
```

Example Ollama pull + context setting:

```bash
ollama pull qwen3.6:27b
ollama run qwen3.6:27b
## then: /set parameter num_ctx 65536
```

Use the **exact** model handle or tag as the OpenClaw Model ID.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| OpenClaw dashboard URL not loading | Gateway not running or wrong host/port | **Restart the OpenClaw gateway** so it reloads `~/.openclaw/openclaw.json`. **Verify:** Check that the gateway process is running with `pgrep -f openclaw` or `ps aux \| grep openclaw`. **Find URL/token:** Check the original installer output (scroll up in your terminal) or look in gateway logs (typically `~/.openclaw/logs/`) for the dashboard URL and access token |
| "Connection refused" to model (for example localhost:8000) | Local inference server not running, still loading, or wrong port | Confirm the backend is up (`curl http://localhost:8000/v1/models` for vLLM, or your LM Studio / Ollama status checks) and that `baseUrl` in `openclaw.json` matches the server |
| OpenClaw says no model available | Provider not configured or model handle mismatch | Add the provider to `~/.openclaw/openclaw.json` and ensure `id`/`name` exactly match the served handle |
| Out-of-memory or very slow inference | Model too large for available GPU memory or other GPU workloads | Use a smaller model from **Agent-ready Models**, lower context length / GPU memory utilization for vLLM, free GPU memory (close other apps), or check usage with `nvidia-smi` |
| WSL cannot reach model server on Windows host | `localhost` inside WSL is not the Windows host | Replace `127.0.0.1` / `localhost` in OpenClaw `baseUrl` with the Windows host IP as seen from WSL |
| Install script fails or dependencies missing | Missing system packages | Install curl and any required build tools; see [OpenClaw documentation](https://docs.openclaw.ai) for current requirements |
| Config changes not applied | Gateway not reloaded | Restart the OpenClaw gateway so it reloads `~/.openclaw/openclaw.json` |
