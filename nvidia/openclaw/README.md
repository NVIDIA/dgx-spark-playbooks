# OpenClaw 🦞

> Run OpenClaw locally on DGX Spark with LM Studio or Ollama

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

OpenClaw (formerly Clawdbot & Moltbot) is a **local-first** AI agent that runs on your machine. It combines multiple capabilities into a single assistant: it remembers conversations, adapts to your usage, runs continuously, uses context from your files and apps, and can be extended with community **skills**.

Running OpenClaw and its LLMs **fully on your DGX Spark** keeps your data private and avoids ongoing cloud API costs. DGX Spark is well suited for this: it runs Linux, is designed to stay on, and has **128GB memory**, so you can run large local models for better accuracy and more capable behavior.

## What you'll accomplish

You will have OpenClaw installed on your DGX Spark and connected to a local LLM (via LM Studio or Ollama). You can use the OpenClaw web UI to chat with your agent, and optionally connect communication channels and skills. The agent and models run entirely on your Spark—no data leaves your machine unless you add cloud or external integrations.

## Popular use cases

- **Personal secretary**: With access to your inbox, calendar, and files, OpenClaw can help manage your schedule, draft replies, send reminders, and find meeting slots.
- **Proactive project management**: Check project status over email or messaging, send status updates, and follow up or send reminders.
- **Research agent**: Combine web search and your local files to produce reports with personalized context.
- **Install helper**: Search for apps/libraries, run installations, and debug errors using terminal access (larger models recommended).

## What to know before starting

- Basic use of the Linux terminal and a text editor
- Optional: familiarity with Ollama or LM Studio if you plan to use a local model
- Awareness of the security considerations below

## Important: security and risks

AI agents can introduce real risks. Read OpenClaw’s guidance: [OpenClaw Gateway Security](https://docs.openclaw.ai/gateway/security).

Main risks:

1. **Data exposure**: Personal information or files may be leaked or stolen.
2. **Malicious code**: The agent or connected tools may expose you to malware or attacks.

You cannot eliminate all risk; proceed at your own risk. **Critical security measures:**

- **STRONGLY RECOMMENDED:** Run OpenClaw on a dedicated or isolated system (e.g., a clean DGX Spark or VM) and only copy in the data the agent needs. Do not run this on your primary workstation with sensitive data.
- Use **dedicated accounts** for the agent instead of your main accounts; grant only the minimum access it needs.
- Enable only **skills you trust**, preferably those vetted by the community. Skills that provide terminal or file system access increase risk significantly.
- **CRITICAL:** Ensure the OpenClaw web UI and any messaging channels are **never exposed** to the public internet without strong authentication. Use SSH tunneling or VPN if accessing remotely.
- Where possible, **limit internet access** for the agent using firewall rules or network isolation.
- **Monitor activity**: Regularly review logs and commands executed by the agent.

## Prerequisites

- DGX Spark running Linux, connected to your network
- Terminal (SSH or local) access to the Spark
- For local LLMs: enough GPU memory for your chosen model (see Instructions for size guidance; DGX Spark’s 128GB supports large models)

## Time and risk

- **Duration**: About 30 minutes for install and first-time model setup; model download time depends on size and network (gpt-oss-120b is ~65GB and may take longer on slower connections).
- **Risk level**: **Medium to High**—the agent has access to whatever files, tools, and channels you configure. Risk increases significantly if you enable terminal/command execution skills or connect external accounts. Without proper isolation, this setup could expose sensitive data or allow code execution. **Always follow the security measures above.**
- **Rollback**: You can stop the OpenClaw gateway and uninstall via the same install script or by removing its directory; uninstall Ollama or LM Studio separately if desired.
- **Last Updated**: 03/11/2026
  - First Publication

## Instructions

> [!CAUTION]
> **Before proceeding, review the security risks in the Overview tab.** OpenClaw is an AI agent that can access your files, execute commands, and connect to external services. Data exposure and malicious code execution are real risks. **Strongly recommended:** Run OpenClaw on an isolated system or VM, use dedicated accounts (not your main accounts), and never expose the dashboard to the public internet without authentication.

## Step 1. Install OpenClaw on your DGX Spark

On your DGX Spark, open a terminal and run the official install script. This installs OpenClaw and its dependencies on your Linux system.

```bash
curl -fsSL https://openclaw.ai/install.sh | bash
```

After dependencies are downloaded, OpenClaw will show a **security warning**. Read the risks; if you accept them, use the arrow keys to select **Yes** and press Enter.

## Step 2. Complete the OpenClaw onboarding

Work through the prompts as follows.

1. **Quickstart vs Manual**: Choose **Quickstart**.

2. **Model provider**: To use a **local model** (recommended for DGX Spark), go to the bottom of the list and select **Skip for now**—you’ll configure the model later. To use a cloud model instead, pick a provider and follow its instructions.

3. **Filtering models by provider**: Select **All Providers**. On the next prompt for the default model, choose **Keep Current**.

4. **Communication channel**: You can connect a channel (e.g., messaging) to use the bot when away from the machine, or select **Skip for Now** and configure it later.

5. **Skills**: We recommend selecting **No** for now. You can add skills later from the web UI or Clawhub after you’ve tested the basics.

6. **Homebrew**: If you are prompted to install Homebrew, select **No**—Homebrew is for macOS only and is not needed on Linux.

7. **Hooks**: We recommend selecting all three for a better experience. Note that this may log data locally; enable only if you’re comfortable with that.

8. **Dashboard URL**: The terminal will print a URL for the OpenClaw dashboard. **Save this URL** (and any access token shown)—you’ll need it to open the web UI.

9. **Finish**: Select **Yes** on the final prompt to complete installation.

You can now open the OpenClaw dashboard in a browser using the URL and token from the installer.

## Step 3. Choose and install a local LLM backend

OpenClaw can use a local LLM via **LM Studio** (best raw performance, uses Llama.cpp) or **Ollama** (simpler and good for deployment). Use a **separate terminal** on your DGX Spark for the backend so the gateway and the model server can run side by side.

**Install one of the following:**

**Option A – LM Studio**

```bash
curl -fsSL https://lmstudio.ai/install.sh | bash
```

**Option B – Ollama**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Step 4. Select and download a model

Model quality and capability scale with size. Free as much GPU memory as possible (avoid other GPU workloads, enable only the skills you need). DGX Spark has **128GB unified memory**, so you can run large models with room to spare.

**Suggested models by GPU memory:**

| GPU memory   | Suggested model                    | Model size | Notes |
|-------------|-------------------------------------|-----------|-------|
| 8–12 GB     | qwen3-4B-Thinking-2507             | ~5GB      | —     |
| 16 GB       | gpt-oss-20b                        | ~12GB     | Lower latency, good for interactive use |
| 24–48 GB    | Nemotron-3-Nano-30B-A3B            | ~20GB     | —     |
| 128 GB      | gpt-oss-120b                       | ~65GB     | **Best quality on DGX Spark** (quantized); leaves ~63GB for context window and other processes; use 20B/30B if you prefer faster responses |

**Quality vs. latency:** The 120B model gives the best accuracy and capability but has higher per-token latency. If you prefer snappier replies, use **gpt-oss-20b** (or a 30B model) instead; both run comfortably on DGX Spark with plenty of memory headroom.

**Download the model:**

**LM Studio**

```bash
lms get openai/gpt-oss-120b
```

**Ollama**

```bash
ollama pull gpt-oss:120b
```

(Use the model name that matches your choice from the table; adjust the `lms get` or `ollama pull` command accordingly.)

## Step 5. Run the model with a large context window

OpenClaw works best with a context window of **32K tokens or more**.

**LM Studio**

```bash
lms load openai/gpt-oss-120b --context-length 32768
```

**Ollama**

```bash
ollama run gpt-oss:120b
```

Once the interactive prompt appears, set the context window (type the following at the Ollama prompt; do not include any `>>>` prefix):

```
/set parameter num_ctx 32768
```

Keep this terminal (or process) running so the model stays loaded. You can now chat with the model or press Ctrl+D to exit the interactive mode while keeping the model server running.

> [!TIP]
> **If you see out-of-memory (OOM) errors:** Try a smaller context (e.g. `16384`) or switch to a smaller model (e.g. gpt-oss-20b). Monitor memory with `nvidia-smi` while the model is loaded.

## Step 6. Configure OpenClaw to use your local model

**If you use LM Studio:**

1. Open the OpenClaw config file in your preferred editor (e.g. `nano`, `vim`, or a graphical editor). The config path is:
   ```bash
   ~/.openclaw/openclaw.json
   ```
   Example with nano:
   ```bash
   nano ~/.openclaw/openclaw.json
   ```

2. Add or update the `models` section so it includes the LM Studio provider. Example for **gpt-oss-120b** (DGX Spark):

```json
"models": {
  "mode": "merge",
  "providers": {
    "lmstudio": {
      "baseUrl": "http://localhost:1234/v1",
      "apiKey": "lmstudio",
      "api": "openai-responses",
      "models": [
        {
          "id": "openai/gpt-oss-120b",
          "name": "openai/gpt-oss-120b",
          "reasoning": false,
          "input": ["text"],
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "contextWindow": 32768,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```

For **gpt-oss-20b** or another model, use the same structure but set `id` and `name` to match the model you loaded (e.g. `openai/gpt-oss-20b`). Adjust `contextWindow` and `maxTokens` if needed.

**If you use Ollama:**

> [!NOTE]
> `ollama launch openclaw` requires **Ollama v0.15 or later**. If you see an "unknown command" error, upgrade Ollama (`ollama --version`) and retry.

Run:

```bash
ollama launch openclaw
```

If the OpenClaw gateway is already running, it should pick up the new configuration automatically. You can add `--config` to configure without launching the gateway yet.

## Step 7. Verify the setup

1. In a browser, open the **OpenClaw dashboard URL** (and use the access token if required).
2. Start a **new** conversation and send a short message.
3. If you get a reply from the agent, the setup is working.

You can also ask OpenClaw which model it’s using. In the gateway chat UI you can switch models by typing: **`/model MODEL_NAME`**.

## Step 8. Optional: add skills and learn more

- **Skills** add capabilities but also risk; only enable skills you trust (e.g., community-vetted ones). To add a skill:
  - Ask OpenClaw to configure a skill, or
  - Use the sidebar in the web UI to enable skills, or
  - Browse [Clawhub](https://docs.openclaw.ai/tools/clawhub) for community skills.

- For more usage and configuration details, see the [OpenClaw documentation](https://docs.openclaw.ai).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| OpenClaw dashboard URL not loading | Gateway not running or wrong host/port | **Restart the OpenClaw gateway:** For Ollama, run `ollama launch openclaw` to restart an already-configured gateway. For LM Studio, restart the OpenClaw gateway via the LM Studio UI or restart the OpenClaw service/container. **Verify:** Check that the gateway process is running with `pgrep -f openclaw` or `ps aux \| grep openclaw`. **Find URL/token:** Check the original installer output (scroll up in your terminal) or look in gateway logs (typically `~/.openclaw/logs/`) for the dashboard URL and access token |
| "Connection refused" to model (e.g. localhost:1234 or Ollama port) | LM Studio or Ollama not running, or wrong port | Start the model in a separate terminal (`lms load ...` or `ollama run ...`) and ensure the port in `openclaw.json` matches (1234 for LM Studio, 11434 for Ollama) |
| OpenClaw says no model available | Model provider not configured or model not loaded | Add the `models` section to `~/.openclaw/openclaw.json` for LM Studio, or run `ollama launch openclaw` for Ollama; ensure the model is loaded/running |
| Out-of-memory or very slow inference on DGX Spark | Model too large for available GPU memory or other GPU workloads | Free GPU memory (close other apps), choose a smaller model, or check usage with `nvidia-smi` |
| Install script fails or dependencies missing | Missing system packages on Linux | Install curl and any required build tools; see [OpenClaw documentation](https://docs.openclaw.ai) for current requirements |
| Config changes not applied | Gateway not reloaded | Restart the OpenClaw gateway so it reloads `~/.openclaw/openclaw.json` |
