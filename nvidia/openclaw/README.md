# OpenClaw 🦞

> Run OpenClaw locally on DGX Spark with a vLLM-served local model

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

You will have OpenClaw installed on your DGX Spark and connected to a local LLM served by **vLLM** (the agent-ready `nvidia/Qwen3.6-35B-A3B-NVFP4` recipe). You can use the OpenClaw web UI to chat with your agent, and optionally connect communication channels and skills. The agent and models run entirely on your Spark—no data leaves your machine unless you add cloud or external integrations.

## Popular use cases

- **Personal secretary**: With access to your inbox, calendar, and files, OpenClaw can help manage your schedule, draft replies, send reminders, and find meeting slots.
- **Proactive project management**: Check project status over email or messaging, send status updates, and follow up or send reminders.
- **Research agent**: Combine web search and your local files to produce reports with personalized context.
- **Install helper**: Search for apps/libraries, run installations, and debug errors using terminal access (larger models recommended).

## What to know before starting

- Basic use of the Linux terminal and a text editor
- Optional: familiarity with Docker and vLLM if you plan to use a local model
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

- **Duration**: About 30 minutes for install and first-time model setup; model download time depends on size and network (the NVFP4 checkpoint is downloaded once and cached for later launches).
- **Risk level**: **Medium to High**—the agent has access to whatever files, tools, and channels you configure. Risk increases significantly if you enable terminal/command execution skills or connect external accounts. Without proper isolation, this setup could expose sensitive data or allow code execution. **Always follow the security measures above.**
- **Rollback**: You can stop the OpenClaw gateway and uninstall via the same install script or by removing its directory; stop the vLLM container separately (`docker rm`/`docker rmi`) if desired.
- **Last Updated**: 06/12/2026
  - Switch local inference backend to vLLM (agent-ready Qwen3.6 35B recipe)
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

## Step 3. Serve the model with vLLM on your DGX Spark

OpenClaw will connect to a local, OpenAI-compatible endpoint served by **vLLM**. This playbook uses the agent-ready `nvidia/Qwen3.6-35B-A3B-NVFP4` recipe — the same one documented in the vLLM playbook's [Run Agent Ready Qwen3.6 35B Model with vLLM](https://build.nvidia.com/spark/vllm/agent-ready-qwen35b) tab. The NVFP4 quantization and speculative decoding give strong tool-calling and reasoning quality while leaving headroom on DGX Spark's 128GB unified memory.

In a **separate terminal** on your DGX Spark, follow the vLLM playbook's [Run Agent Ready Qwen3.6 35B Model with vLLM](https://build.nvidia.com/spark/vllm/agent-ready-qwen35b) tab to launch the server. Run it on its own terminal so the gateway and the model server can run side by side. That tab serves `nvidia/Qwen3.6-35B-A3B-NVFP4` on an OpenAI-compatible API at `http://localhost:8000/v1`.

Once the server reports `Application startup complete`, verify it from another terminal before continuing:

```bash
curl http://localhost:8000/v1/models
```

You should see `nvidia/Qwen3.6-35B-A3B-NVFP4` in the returned list.

## Step 4. Configure OpenClaw to use the vLLM server

1. Open the OpenClaw config file in your preferred editor (e.g. `nano`, `vim`, or a graphical editor). The config path is:
   ```bash
   ~/.openclaw/openclaw.json
   ```
   Example with nano:
   ```bash
   nano ~/.openclaw/openclaw.json
   ```

2. Add or update the `models` section so it includes the vLLM provider pointing at the endpoint from Step 3. vLLM does not require an API key, so any non-empty placeholder works:

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

The `id` and `name` must match the model handle served by vLLM (`nvidia/Qwen3.6-35B-A3B-NVFP4`). `contextWindow` matches the `--max-model-len` from Step 3.

> [!NOTE]
> If OpenClaw reports an unsupported-endpoint error against the Responses API, change `"api": "openai-responses"` to the OpenAI chat-completions variant for your OpenClaw version — vLLM always exposes `/v1/chat/completions`.

3. If the OpenClaw gateway is already running, restart it so it reloads `~/.openclaw/openclaw.json` and picks up the new provider.

## Step 5. Verify the setup

1. In a browser, open the **OpenClaw dashboard URL** (and use the access token if required).
2. Start a **new** conversation and send a short message.
3. If you get a reply from the agent, the setup is working.

You can also ask OpenClaw which model it’s using. In the gateway chat UI you can switch models by typing: **`/model MODEL_NAME`** (e.g. `/model nvidia/Qwen3.6-35B-A3B-NVFP4`).

## Step 6. Optional: add skills and learn more

- **Skills** add capabilities but also risk; only enable skills you trust (e.g., community-vetted ones). To add a skill:
  - Ask OpenClaw to configure a skill, or
  - Use the sidebar in the web UI to enable skills, or
  - Browse [Clawhub](https://docs.openclaw.ai/tools/clawhub) for community skills.

- For more usage and configuration details, see the [OpenClaw documentation](https://docs.openclaw.ai).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| OpenClaw dashboard URL not loading | Gateway not running or wrong host/port | **Restart the OpenClaw gateway** so it reloads `~/.openclaw/openclaw.json`. **Verify:** Check that the gateway process is running with `pgrep -f openclaw` or `ps aux \| grep openclaw`. **Find URL/token:** Check the original installer output (scroll up in your terminal) or look in gateway logs (typically `~/.openclaw/logs/`) for the dashboard URL and access token |
| "Connection refused" to model (e.g. localhost:8000) | vLLM server not running, still loading, or wrong port | Confirm the vLLM container is up and finished loading (`curl http://localhost:8000/v1/models` lists the model) and that `baseUrl` in `openclaw.json` is `http://localhost:8000/v1` |
| OpenClaw says no model available | Provider not configured or model handle mismatch | Add the `vllm` provider to `~/.openclaw/openclaw.json` and ensure `id`/`name` exactly match the served handle (`nvidia/Qwen3.6-35B-A3B-NVFP4`) |
| Out-of-memory or very slow inference on DGX Spark | Model too large for available GPU memory or other GPU workloads | Lower `--gpu-memory-utilization` or `--max-model-len` when launching vLLM, free GPU memory (close other apps), or check usage with `nvidia-smi` |
| Install script fails or dependencies missing | Missing system packages on Linux | Install curl and any required build tools; see [OpenClaw documentation](https://docs.openclaw.ai) for current requirements |
| Config changes not applied | Gateway not reloaded | Restart the OpenClaw gateway so it reloads `~/.openclaw/openclaw.json` |
