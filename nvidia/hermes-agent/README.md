# Hermes-agent with Local Models

> Install and run the Hermes self-improving AI agent on DGX Spark.

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

[Hermes Agent](https://github.com/NousResearch/hermes-agent) is a **self-improving** AI agent built by [Nous Research](https://nousresearch.com). It runs as a terminal TUI on your machine and, through a built-in gateway, can also be reached from messaging platforms like Telegram, Discord, and Slack. It creates skills from experience, improves them during use, persists memory across sessions, and can run scheduled tasks via its built-in cron.

Running Hermes and its LLM **fully on your DGX Spark** keeps your conversations and data private and avoids ongoing cloud API costs. DGX Spark is well suited for this: it runs Linux, is designed to stay on, and has **128GB memory**, so you can serve large local models for better reasoning quality and connect to the agent from your phone over Telegram while the heavy work runs locally.

## What you'll accomplish

You will have Hermes installed on your DGX Spark and connected to a local LLM served by Ollama. You can chat with the agent from the DGX Spark terminal and from Telegram on your phone or laptop. The gateway runs as a system service, so the agent stays reachable across reboots without anyone logging in.

- Install Ollama and pull a local model
- Install Hermes and configure it against the local Ollama endpoint
- Set up a Telegram bot so you can message Hermes from any Telegram client
- Resume past sessions, switch models, update, and uninstall using the `hermes` CLI

## Popular use cases

- **Personal assistant from your phone**: Chat with Hermes via Telegram while the model runs on your Spark — manage email drafts, summarize docs, or answer questions on the go.
- **Multi-step task automation**: Ask the agent to walk you through configurations (e.g., setting up email); on non-trivial tasks Hermes can autonomously persist a reusable skill for next time.
- **Scheduled checks**: Use the built-in cron to watch a product price online or run a daily check, and have results delivered to your Telegram home channel.
- **Reasoning-visible problem solving**: Use `/reasoning show` in the TUI to follow the agent's intermediate reasoning on complex problems.

## What to know before starting

- Basic use of the Linux terminal and a text editor
- Familiarity with Ollama or willingness to follow the [Ollama playbook](../ollama/) first
- A Telegram account if you want to use the messaging gateway
- Awareness of the security considerations below

## Important: security and risks

AI agents that can execute commands and reach external services introduce real risks. Read the upstream guidance: [Hermes documentation](https://hermes-agent.nousresearch.com/docs/).

Main risks:

1. **Data exposure**: Personal information or files on your DGX Spark may be leaked through agent actions or messaging channels.
2. **Unauthorized access**: A Telegram bot left open to anyone who finds it can be misused; a model endpoint exposed beyond `localhost` can be abused.

You cannot eliminate all risk; proceed at your own risk. **Recommended security measures:**

- **Restrict the Telegram bot** by entering one or more numeric Telegram user IDs at the *"Allowed user IDs"* prompt during install. Leaving this blank allows anyone who finds the bot to use it.
- Keep the Ollama endpoint bound to **`localhost` only**; do not expose `http://<spark-ip>:11434` to your LAN or the public internet without strong authentication.
- Run Hermes on a Spark dedicated to this purpose where possible, and only place files on it that the agent is allowed to access.
- **Monitor activity**: Periodically review the gateway service logs (`journalctl -u <hermes-gateway-service>`) and the Hermes session history.

## Prerequisites

- DGX Spark running Linux, connected to your network
- Terminal (SSH or local) access to the Spark
- `curl` and `git` installed (verified in Step 1 of the instructions)
- Enough disk and GPU memory for the Ollama model you plan to serve (the playbook uses `qwen3.6:27b` as the example; pick a smaller model if you want a faster first install)
- A Telegram account and the ability to create a bot via [@BotFather](https://t.me/BotFather) if you plan to use the messaging gateway

## Time and risk

- **Duration**: About 30 minutes for install and first-time setup; model download time depends on size and network speed.
- **Risk level**: **Medium** — the agent can execute commands, persist skills, and is reachable from Telegram. Risk increases if you skip the allowed-user-IDs restriction or expose the local model endpoint beyond `localhost`. Always follow the security measures above.
- **Rollback**: Run `hermes uninstall` to remove Hermes, the gateway service, and the shell-profile entry. Uninstall Ollama separately if desired.
- **Last Updated**: 2026-04-26
  - First Publication

## Instructions

## Step 1. Verify your environment

Before installing Hermes, confirm that your DGX Spark is running DGX OS, has network access, and exposes the basic command-line tools used during install.

```bash
uname -a
curl --version
git --version
```

Expected output should show DGX OS and working `curl` / `git` binaries.

## Step 2. Install Ollama and pull a model

Hermes will be configured against a local Ollama endpoint, so Ollama must be installed and serving at least one model before you run the Hermes installer. If you have already completed the [Ollama playbook](../ollama/), you can skip this step.

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify the Ollama service is running and reachable on the default port:

```bash
curl http://localhost:11434/api/tags
```

Pull the model you intend to use with Hermes (this playbook uses `qwen3.6:27b` as the example):

```bash
ollama pull qwen3.6:27b
```

## Step 3. Install Hermes

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

The installer will walk you through an interactive setup. Respond to each prompt in the order they appear:

1. **"Install ripgrep for faster file search ffmpeg for TTS voice messages? [Y/n]"** — Press **Enter** to accept the default and install both helpers.

2. **"How would you like to set up Hermes?"** — Choose **Quick setup** to proceed with the recommended defaults.

3. **"Select Provider"** — Choose **Custom endpoint (enter URL manually)** so Hermes can be pointed at the model endpoint running on your DGX Spark.

4. **"API base URL [e.g. https://api.example.com/v1]:"** — Enter the URL of your local model server. For a local Ollama endpoint, use `http://localhost:11434/v1`.

5. **"API key [optional]"** — Leave blank and press **Enter**; no key is required for a local model.

6. **Model selection** — The installer lists the models available from your local Ollama instance. Select one to use with Hermes (for example, `qwen3.6:27b`).

7. **"Context length in tokens [leave blank for auto-detect]:"** — Press **Enter** to let Hermes auto-detect the context length from the selected model.

8. **"Display name [Local (localhost:11434)]"** — Press **Enter** to accept the suggested label, or type a custom name to identify this endpoint in the Hermes UI.

9. **"Connect a messaging platform? (Telegram, Discord, etc.)"** — Choose **Set up messaging now (recommended)** to configure a gateway during installation.

10. **"Select platforms to configure:"** — Choose **Telegram**. The remaining steps in this playbook use Telegram as the example; the same flow applies to the other supported gateways.

11. **"Telegram bot token:"** — Open Telegram and start a chat with [@BotFather](https://t.me/BotFather), follow its guided flow to create a new bot, then paste the token BotFather returns into this prompt. The terminal will not echo any characters as the token is pasted — this is expected. Press **Enter** to submit; the installer should respond with `Telegram token saved`.

12. **"Allowed user IDs (comma-separated, leave empty for open access):"** — To restrict the bot to specific Telegram accounts, follow the on-screen instructions to look up your numeric Telegram user ID, then enter one or more IDs separated by commas. Leaving this field blank allows anyone who can reach the bot to use it, which is generally not recommended.

13. **"Use your user ID (\<your-id\>) as the home channel? [Y/n]:"** — Press **Enter** to accept. This designates your own Telegram account as the default channel Hermes will use for proactive messages and scheduled deliveries.

14. **"Install the gateway as a systemd service? (runs in background, starts on boot) [Y/n]:"** — Press **Enter** to accept. The gateway will run as a background service and start automatically whenever your DGX Spark boots.

15. **"Choose how the gateway should run in the background:"** — Choose **System service**. The DGX Spark is typically an always-on machine, and a system service starts on boot without requiring an interactive login or the `linger` workaround that user services need. The service will still run under your user account so it can read your Hermes configuration; only installation requires `sudo`.

16. **"Launch hermes chat now? [Y/n]:"** — Press **Enter** to launch the Hermes TUI immediately and verify the installation end-to-end. Once the TUI is open, type `hello` and press **Enter**; the agent should respond, confirming that the model endpoint and Hermes are wired up correctly. When you're done, type `/exit` to leave the chat and return to your shell. On exit, Hermes prints the exact command needed to resume this conversation later — `hermes --resume <sessionId>`. Save it if you want to pick up where you left off.

17. **"Would you like to install the gateway as a background service? [Y/n]:"** — Press **Enter** to accept. This finalizes the gateway as a background service so it stays available for messaging-platform traffic outside of an interactive Hermes session.

18. **Reload your shell** to make the `hermes` command available:

    ```bash
    source ~/.bashrc
    ```

## Step 4. Switch to a different Ollama model (optional)

You configured an initial model during the Hermes install. To switch to a different one later, pull the new model with Ollama and then re-point Hermes at the same local endpoint.

1. Pull the new model with Ollama (replace `<model-name>` with the model you want):

    ```bash
    ollama pull <model-name>
    ```

2. Launch the Hermes model picker:

    ```bash
    hermes model
    ```

3. At the **"Select Provider"** prompt, choose **Custom endpoint (enter URL manually)**.

4. At the **"API base URL"** prompt, enter the same local Ollama endpoint as before:

    ```
    http://localhost:11434/v1
    ```

5. When the installer lists the models served by Ollama, choose the one you just pulled. Hermes will use it for subsequent sessions.

## Step 5. Resume a previous Hermes session

To pick up a past conversation, launch Hermes with the `--resume` flag and the session ID printed when you exited that chat:

```bash
hermes --resume <sessionId>
```

The TUI will reopen with the prior conversation history restored, ready for follow-up prompts.

## Step 6. Talk to Hermes from Telegram

The Telegram gateway you configured during install is already running as a background service, so you can reach Hermes from any Telegram client without a terminal session.

1. Open Telegram (mobile or desktop) and search for your bot by the username you assigned through @BotFather.

2. Open the chat with the bot and tap **Start** (or send `/start`) on first contact.

3. Send the message `hello`. Hermes will reply through the bot, confirming the gateway is wired to your DGX Spark and the underlying model.

From here you can send any prompt you would normally type in the TUI — Hermes will run on your DGX Spark and stream the response back to Telegram.

## Step 7. Update Hermes

To upgrade an existing Hermes installation to the latest release, run:

```bash
hermes update
```

The command pulls the latest Hermes version, applies any required dependency changes, and restarts the gateway service so the new version takes effect.

## Step 8. Cleanup

> [!WARNING]
> This removes the Hermes installation and the gateway service. By default, `~/.hermes/` (configuration, conversation history, and skills) is preserved unless you opt into a full uninstall at the on-screen prompt.

Because the gateway was installed as a **System service** in Step 15, run the uninstall with `sudo` so it has permission to remove the system-scope systemd unit:

```bash
sudo hermes uninstall
```

Follow the on-screen prompts to confirm removal. `sudo hermes uninstall` automatically:

- Stops and removes the systemd gateway service.
- Removes the `hermes` wrapper script and the PATH entries added to your shell profile.
- Deletes the Hermes installation directory.

## Step 9. Next steps

1. **Inspect the agent's reasoning.** Inside the TUI, run `/reasoning show` to surface the model's intermediate reasoning alongside its responses. This is especially useful for following the agent's progress on multi-step or complex problems and for debugging unexpected answers.
2. **Try a multi-step task to trigger skill creation.** For example, ask the agent how to set up email — Hermes will walk through the configuration with you and, on completing a non-trivial task like this, may autonomously persist a reusable skill so the next email-related request is faster.
3. **Configure scheduled automations via the built-in cron.** For example, ask Hermes to check the price of a product online once a day and notify you on Telegram when it drops below a threshold. Hermes will schedule the task with its built-in cron and deliver each result through the messaging gateway you set up.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `hermes: command not found` after install | Shell profile not reloaded in the current session | Run `source ~/.bashrc` (or `source ~/.zshrc`) and retry. Open a new terminal if the issue persists. |
| Hermes installer can't list any models at the model-selection prompt | Ollama is not running or has no models pulled | Sanity-check Ollama in another terminal: list installed models with `ollama list`, hit the API with `curl http://localhost:11434/api/tags`, and confirm a model can actually serve requests by running `ollama run <model-name>` (e.g. `ollama run qwen3.6:27b`) and sending a test prompt. If the list is empty or the API is unreachable, start Ollama and pull a model with `ollama pull <model-name>`, then re-run the Hermes installer. |
| `Connection refused` to `http://localhost:11434/v1` from Hermes | Ollama service not running on the default port | Start the Ollama service and confirm it is listening on `11434`. On systemd hosts: `systemctl status ollama` and `systemctl start ollama`. |
| Pasting the Telegram bot token shows nothing on the screen | Expected — the installer hides token characters as a security measure | Paste the token, then press **Enter**. The installer should respond with `Telegram token saved`. |
| Telegram bot does not reply when you send `hello` | Gateway service not running, or your account is not in the allowed user IDs list | Check the gateway service status with `systemctl status` (look for the Hermes gateway unit installed in Step 14). If your Telegram user ID was not added during install, re-run `hermes` setup or update the gateway config to include it. |
| Out-of-memory or very slow inference | Selected Ollama model is too large for available GPU memory, or other GPU workloads are competing | Check usage with `nvidia-smi`, free GPU memory by closing other workloads, or pull a smaller model with `ollama pull <smaller-model>` and switch to it via `hermes model`. |
| `hermes update` fails or the gateway does not restart | Gateway service still bound to the previous version, or insufficient permissions on a system-service install | Re-run `hermes update` with `sudo` if the gateway was installed as a **System service**. If the service is stuck, restart it manually: `sudo systemctl restart <hermes-gateway-unit>`. |
| Cannot resume a previous session | The `<sessionId>` value is missing or wrong | Launch `hermes` without `--resume` to start fresh; past session IDs are printed to the terminal each time you `/exit` a chat. |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU.
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

For latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
