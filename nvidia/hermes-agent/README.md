# Run Hermes Agent with Local Models

> Install and run the Hermes self-improving AI agent on DGX Spark.

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Verify outbound HTTPS to Telegram (gateway requirement)](#verify-outbound-https-to-telegram-gateway-requirement)
  - [3a. Installer basics](#3a-installer-basics)
  - [3b. Point Hermes at your local model](#3b-point-hermes-at-your-local-model)
  - [3c. Choose the terminal backend](#3c-choose-the-terminal-backend)
  - [3d. Finish the wizard and verify](#3d-finish-the-wizard-and-verify)
  - [3e. Configure Telegram](#3e-configure-telegram)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

[Hermes Agent](https://github.com/NousResearch/hermes-agent) is a **self-improving** AI agent built by [Nous Research](https://nousresearch.com). It runs as a terminal TUI on your machine and, through a built-in gateway, can also be reached from messaging platforms like Telegram, Discord, and Slack. It creates skills from experience, improves them during use, persists memory across sessions, and can run scheduled tasks via its built-in cron.

Running Hermes and its LLM **fully on your DGX Spark** keeps your conversations and data private and avoids ongoing cloud API costs. DGX Spark is well suited for this: it runs Linux, is designed to stay on, and has **128GB memory**, so you can serve large local models for better reasoning quality and connect to the agent from your phone over Telegram while the heavy work runs locally.

## What you'll accomplish

You will have Hermes installed on your DGX Spark and connected to a local LLM served by **vLLM** (the agent-ready `nvidia/Qwen3.6-35B-A3B-NVFP4` recipe). You can chat with the agent from the DGX Spark terminal and from Telegram on your phone or laptop. The gateway runs as a system service, so the agent stays reachable across reboots without anyone logging in.

- Serve a local model with vLLM
- Install Hermes and configure it against the local vLLM endpoint
- Set up a Telegram bot so you can message Hermes from any Telegram client
- Resume past sessions, switch models, update, and uninstall using the `hermes` CLI

## Popular use cases

- **Personal assistant from your phone**: Chat with Hermes via Telegram while the model runs on your Spark — manage email drafts, summarize docs, or answer questions on the go.
- **Multi-step task automation**: Ask the agent to walk you through configurations (e.g., setting up email); on non-trivial tasks Hermes can autonomously persist a reusable skill for next time.
- **Scheduled checks**: Use the built-in cron to watch a product price online or run a daily check, and have results delivered to your Telegram home channel.
- **Reasoning-visible problem solving**: Use `/reasoning show` in the TUI to follow the agent's intermediate reasoning on complex problems.

## What to know before starting

- Basic use of the Linux terminal and a text editor
- Familiarity with Docker and vLLM, or willingness to follow the [vLLM for Inference playbook](https://build.nvidia.com/spark/vllm) first
- A Telegram account if you want to use the messaging gateway
- Awareness of the security considerations below

## Important: security and risks

AI agents that can execute commands and reach external services introduce real risks. Read the upstream guidance, especially the dedicated security topics: [Hermes Agent — Security](https://hermes-agent.nousresearch.com/docs/user-guide/security).

Main risks:

1. **Data exposure**: Personal information or files on your DGX Spark may be leaked through agent actions or messaging channels.
2. **Unauthorized access**: A Telegram bot left open to anyone who finds it can be misused; a model endpoint exposed beyond `localhost` can be abused.

You cannot eliminate all risk; proceed at your own risk. **Recommended security measures:**

- **Restrict the Telegram bot** by entering one or more numeric Telegram user IDs at the *"Allowed user IDs"* prompt during install. Leaving this blank allows anyone who finds the bot to use it.
- Keep the vLLM endpoint bound to the Spark; do not forward `http://<spark-ip>:8000` to your LAN or the public internet without strong authentication.
- Run Hermes on a Spark dedicated to this purpose where possible, and only place files on it that the agent is allowed to access.
- **Monitor activity**: Periodically review the gateway service logs (`sudo journalctl -u <hermes-gateway-unit> -e`) and the Hermes session history.

## Prerequisites

- DGX Spark running Linux, connected to your network
- Terminal (SSH or local) access to the Spark
- `curl` and `git` installed (verified in Step 1 of the instructions)
- Interactive terminal access for the setup wizard and any `sudo` password prompts. Non-interactive SSH is supported with the config-command fallback in the Instructions tab.
- Docker with the NVIDIA Container Toolkit, plus a HuggingFace token to download the model (the playbook serves `nvidia/Qwen3.6-35B-A3B-NVFP4` with vLLM)
- A Telegram account and the ability to create a bot via [@BotFather](https://t.me/BotFather) if you plan to use the messaging gateway

## Time and risk

- **Duration**: About 30 minutes for install and first-time setup; model download time depends on size and network speed.
- **Risk level**: **Medium** — the agent can execute commands, persist skills, and is reachable from Telegram. Risk increases if you skip the allowed-user-IDs restriction or expose the local model endpoint beyond `localhost`. Always follow the security measures above.
- **Rollback**: Run `hermes uninstall` (with `sudo` if you installed the gateway as a system service) to remove Hermes, the gateway service, and the shell-profile entry. The data directory `~/.hermes` may still be present afterward; remove it manually if you want a full reset (see the Cleanup and Troubleshooting tabs). Stop the vLLM container separately (`docker rm`/`docker rmi`) if desired.
- **Last Updated**: 2026-06-12
  - Switch local inference backend to vLLM (agent-ready Qwen3.6 35B recipe)
  - First Publication

## Instructions

## Step 1. Verify your environment

Before installing Hermes, confirm that your DGX Spark is running DGX OS, has network access, and exposes the basic command-line tools used during install.

```bash
uname -a
curl --version
git --version
```

**What to look for:** DGX Spark ships with **DGX OS**, which is a specialized Ubuntu-based Linux image. The `uname -a` line will not always contain the literal string “DGX OS”. A healthy Spark typically shows **Linux**, **Ubuntu**, and **nvidia** (kernel or platform identifiers) in that output. Confirm that `curl --version` and `git --version` print version lines without errors.

### Verify outbound HTTPS to Telegram (gateway requirement)

The Hermes **Telegram gateway** talks to Telegram’s cloud API over **HTTPS**. On some corporate or lab networks, **outbound HTTPS to `api.telegram.org` is blocked**, which produces a working local install but a **bot that never responds**. Before you invest time in gateway setup, run this quick check from the same network you will use for the Spark:

```bash
curl -sS --connect-timeout 10 -o /dev/null -w "HTTP %{http_code}\n" https://api.telegram.org/
```

You should see an **HTTP status line** such as **`HTTP 404`**, **`HTTP 200`**, or **`HTTP 302`** (Telegram’s edge often answers bare `GET` requests with a short JSON or redirect). The important part is that the request **completes over TLS** without hanging. **Timeouts**, **“Could not resolve host”**, or **connection refused** mean the gateway will not reach Telegram from this network—try a path that allows that traffic (for example a personal hotspot) or ask your network administrator to allow **HTTPS to `api.telegram.org`**.

## Step 2. Serve a model with vLLM

Hermes will be configured against a local, OpenAI-compatible endpoint, so a model server must be running before you launch the Hermes installer. This playbook uses **vLLM** with the agent-ready `nvidia/Qwen3.6-35B-A3B-NVFP4` recipe — the same one documented in the vLLM playbook's [Run Agent Ready Qwen3.6 35B Model with vLLM](https://build.nvidia.com/spark/vllm/agent-ready-qwen35b) tab.

Follow that tab to launch the server in a **separate terminal** on the Spark so it can run alongside Hermes. It serves `nvidia/Qwen3.6-35B-A3B-NVFP4` on an OpenAI-compatible API at `http://localhost:8000/v1`.

Once the server reports `Application startup complete`, verify the API on **8000** in another terminal. A healthy server returns **JSON** with a top-level **`"data"`** array listing the served model:

```bash
curl -sS http://localhost:8000/v1/models
```

You should see `nvidia/Qwen3.6-35B-A3B-NVFP4` in the returned list.

> [!NOTE]
> Keep the vLLM endpoint bound to the Spark only. The container publishes port `8000`; do not forward `http://<spark-ip>:8000` to your LAN or the public internet without strong authentication.

## Step 3. Install Hermes

The steps below were verified against **Hermes Agent v0.18.0 (2026.7.1) · upstream `676236bb`**. Newer installer versions may word prompts differently or reorder them; follow the closest matching prompt.

Run the installer from an **interactive terminal** on the Spark. If you are connected over SSH, use a normal SSH session where you can answer prompts and enter your `sudo` password when requested. If you run the installer from a non-interactive automation shell, Hermes can install but the setup wizard and optional system-package prompts may be skipped; use the **Non-interactive SSH fallback** below in that case.

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

The installer walks you through an interactive setup wizard. This playbook uses the **Blank Slate** path, which keeps the wizard short — basics, model endpoint, and terminal backend — and then configures Telegram afterwards with a single `hermes gateway setup` command (Step 3e). The subsections below follow the wizard's prompt order.

> [!IMPORTANT]
> **OpenClaw on the same machine (out of scope for this playbook):** If another tool such as **OpenClaw** was installed previously, the Hermes installer may ask whether you want to **import** or **migrate** from it. For the steps in *this* playbook, answer **`n`** (no) so Hermes does not pull in OpenClaw configuration. Mixing migrations can leave Telegram or gateway state inconsistent; if you already migrated by mistake, prefer a clean reinstall (see **Start over from scratch** in the Troubleshooting tab) before continuing.

### 3a. Installer basics

1. **"Install ripgrep for faster file search ffmpeg for TTS voice messages? [Y/n]"** — Press **Enter** to accept the default and install both helpers. If `sudo` asks for your password, enter your Linux user password. If you skip this step or run without a terminal, Hermes still works, but file search falls back to slower tools and TTS voice-message support is limited. You can install the helpers later with `sudo apt install -y ripgrep ffmpeg`.

2. **"How would you like to set up Hermes?"** — Choose **Blank Slate**. It starts with everything off except the bare minimum, still offers the provider prompts used in the rest of this step, and lets you opt in to individual capabilities later. For context on the other options: **Quick Setup** does not offer provider selection — it signs in through **https://portal.nousresearch.com** (registration or login required) instead, so it won't point Hermes at the local model endpoint on your DGX Spark — and **Full setup** walks through every provider, tool, and plugin option — this playbook uses Blank Slate to get users started quickly.

### 3b. Point Hermes at your local model

1. **"Select provider"** — The wizard lists many hosted providers; scroll past them and choose **Custom endpoint (enter URL manually)** so Hermes can be pointed at the model endpoint running on your DGX Spark. (If the machine already has saved endpoints, entries like **Local (localhost:8000)** appear near the bottom of the list — selecting the matching saved entry works too.)

2. **"API base URL [e.g. https://api.example.com/v1]:"** — *If this prompt appears*, enter the URL of your local model server. For the local vLLM endpoint from Step 2, use `http://localhost:8000/v1`. (Depending on installer version or prior config, this question is sometimes skipped when the endpoint is already inferred—continue with the prompts you do see.)

3. **"API key [optional]"** — Leave blank and press **Enter**; vLLM does not require a key for a local model.

4. **Model selection** — The installer lists the models served by your local endpoint (vLLM reports these via `/v1/models`). Select `nvidia/Qwen3.6-35B-A3B-NVFP4`.

5. **"Context length in tokens [leave blank for auto-detect]:"** — Press **Enter** to let Hermes auto-detect the context length from the served model (the recipe serves `--max-model-len 262144`).

6. **"Display name [Local (localhost:8000)]"** — Press **Enter** to accept the suggested label, or type a custom name to identify this endpoint in the Hermes UI.

**Non-interactive SSH fallback:** If the installer prints **"Setup wizard skipped (no terminal available)"**, or if you are validating the playbook through non-interactive SSH, configure the local vLLM endpoint with Hermes' config command instead of the prompts above:

```bash
export PATH="$HOME/.local/bin:$PATH"
hermes config set model.provider custom
hermes config set model.base_url http://localhost:8000/v1
hermes config set model.default nvidia/Qwen3.6-35B-A3B-NVFP4
hermes -z "Reply exactly HERMES_OK"
```

The last command should return `HERMES_OK`, confirming that Hermes can call the local vLLM model without opening the TUI.

### 3c. Choose the terminal backend

1. **"Select terminal backend:"** — Choose **Keep current (local)** (or **Local** if no current option is shown). This determines where Hermes executes shell commands when the agent uses its terminal tool; the local backend runs them directly on the DGX Spark, which is what this playbook assumes. The other options (Docker, Modal, SSH, Daytona, Singularity/Apptainer) run commands in containers or off-box environments and are out of scope here.

### 3d. Finish the wizard and verify

1. **"Your minimal agent is ready. What next?"** — Choose **Start with everything disabled — finish now (most minimal)**. You can enable capabilities later at any time.

2. **Reload your shell** to make the `hermes` command available, then verify the command resolves:

    ```bash
    source ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
    which hermes
    ```

3. **Verify the model wiring.** Run `hermes` to open the TUI, type `hello`, and press **Enter**; the agent should respond, confirming that the model endpoint and Hermes are wired up correctly. When you're done, type `/exit` to leave the chat and return to your shell. On exit, Hermes prints the exact command needed to resume this conversation later — `hermes --resume <sessionId>`. Save it if you want to pick up where you left off.

### 3e. Configure Telegram

The Blank Slate path skips messaging setup, so configure the Telegram gateway now with:

```bash
hermes gateway setup
```

1. **"Select platforms to configure:"** — The command shows a multi-select list of all supported messaging platforms (Telegram, Discord, Slack, WhatsApp, and many more). Navigate to **Telegram**, press **SPACE** to select it, then press **ENTER** to confirm. The remaining steps in this playbook use Telegram as the example; the same flow applies to the other supported gateways.

    > [!TIP]
    > **If Telegram questions are skipped:** Some users see **“Setup complete”** or **“Messaging Platforms (Gateway) configuration complete!”** immediately after choosing Telegram, without token or user-ID prompts. That usually means a prior partial Telegram state exists. Re-run **`hermes gateway setup`** and select Telegram again to supply the bot token and allowed user IDs. (If the CLI suggests `hermes setup gateway` but that flow still skips prompts, use **`hermes gateway setup`**—that is the command most users report as working for a full Telegram reconfiguration; if it keeps skipping, see **Start over from scratch** in the Troubleshooting tab.) Follow the printed **`sudo`** lines to register the gateway service (if a printed `sudo hermes …` command fails with `command not found`, see the Troubleshooting tab).

2. **"Telegram bot token:"** — Open Telegram and start a chat with [@BotFather](https://t.me/BotFather), follow its guided flow to create a new bot, then paste the token BotFather returns into this prompt. **Tip:** Installing [Telegram Desktop](https://desktop.telegram.org/) on the same machine as your SSH session lets you **copy the token from Telegram and paste into the terminal** without retyping it from your phone. The terminal will not echo any characters as the token is pasted — this is expected. Press **Enter** to submit; the installer should respond with `Telegram token saved`.

3. **"Allowed user IDs (comma-separated, leave empty for open access):"** — To restrict the bot to specific Telegram accounts, follow the on-screen instructions to look up your numeric Telegram user ID, then enter one or more IDs separated by commas. Leaving this field blank allows anyone who can reach the bot to use it, which is generally not recommended.

4. **"Use your user ID (\<your-id\>) as the home channel? [Y/n]:"** — Press **Enter** to accept. This designates your own Telegram account as the default channel Hermes will use for proactive messages and scheduled deliveries.

5. **"Install the gateway as a systemd service? (runs in background, starts on boot) [Y/n]:"** — Press **Enter** to accept. The gateway will run as a background service.

6. **"Choose how the gateway should run in the background:"** — Choose **System service** if you want Hermes to start at boot without requiring an interactive login. The service will still run under your user account so it can read your Hermes configuration; only installation requires `sudo`. If this prompt doesn't appear or you need to (re)install the service manually, run `sudo "$(which hermes)" gateway install --system --run-as-user "$USER"`.

7. **Verify the gateway.** After configuration, confirm the gateway unit is active and recent logs look healthy (replace `<hermes-gateway-unit>` with the **exact** `*.service` name the installer printed—often something containing `hermes` and `gateway`):

    ```bash
    systemctl list-units --type=service --all | grep -i hermes
    systemctl --user list-units --type=service --all | grep -i hermes
    sudo systemctl status <hermes-gateway-unit>
    sudo journalctl -u <hermes-gateway-unit> -e --no-pager -n 50
    ```

    If `systemctl status` or `systemctl --user status` shows **active (running)** and logs are not repeating connection errors to Telegram, the service side is in good shape. If logs show TLS timeouts or “connection refused” to Telegram hosts, re-run the **outbound HTTPS** check at the top of this page.

8. **Talk to Hermes from Telegram.** The gateway is now running as a background service, so you can reach Hermes from any Telegram client without a terminal session:

    - Open Telegram (mobile or desktop) and search for your bot by the username you assigned through @BotFather.
    - Open the chat with the bot and tap **Start** (or send `/start`) on first contact.
    - Send the message **`hello`**. Hermes will reply through the bot, confirming the gateway is wired to your DGX Spark and the underlying model.

    > [!NOTE]
    > After **`/start`**, Telegram may show a generic **“Unknown command”**-style message from the bot. That can be normal for bots that only implement free-form chat. **Ignore that message and send `hello` anyway**—Hermes should respond to normal text once the gateway and model are healthy.

    From here you can send any prompt you would normally type in the TUI — Hermes will run on your DGX Spark and stream the response back to Telegram.

## Step 4. Switch to a different model

You configured an initial model during the Hermes install. To switch to a different one later, restart vLLM serving the new model handle, then re-point Hermes at the same local endpoint.

1. Stop the current vLLM container (Ctrl+C in its terminal) and relaunch it with the new model handle in place of `nvidia/Qwen3.6-35B-A3B-NVFP4`. Use the same `docker run` invocation from the vLLM playbook's [Run Agent Ready Qwen3.6 35B Model with vLLM](https://build.nvidia.com/spark/vllm/agent-ready-qwen35b) tab, swapping the model handle (and any flags appropriate for the new model).

2. Launch the Hermes model picker:

    ```bash
    hermes model
    ```

3. At the **"Select Provider"** prompt, choose **Custom endpoint (enter URL manually)**.

4. **If you see the “API base URL” prompt**, enter the same local vLLM endpoint as before:

    ```
    http://localhost:8000/v1
    ```

5. When Hermes lists the models served by the endpoint, choose the one you just started serving. Hermes will use it for subsequent sessions.

If you are in a non-interactive SSH session, switch models with config commands instead:

```bash
hermes config set model.provider custom
hermes config set model.base_url http://localhost:8000/v1
hermes config set model.default <new-model-handle>
hermes -z "Reply exactly MODEL_OK"
```

## Step 5. Configure tools

The Blank Slate install from Step 3 starts with the agent's tools disabled. To enable more tools or modify the existing tool configuration, run:

```bash
hermes tools
```

This opens the same multi-select tool list as the setup wizard (web search, browser automation, terminal, file operations, code execution, and more). Toggle entries with **SPACE** and press **ENTER** to confirm.

## Step 6. Resume a previous Hermes session

To pick up a past conversation, launch Hermes with the `--resume` flag and the session ID printed when you exited that chat:

```bash
hermes --resume <sessionId>
```

The TUI will reopen with the prior conversation history restored, ready for follow-up prompts.

## Step 7. Update Hermes

To upgrade an existing Hermes installation to the latest release, run:

```bash
hermes update
```

The command pulls the latest Hermes version, applies any required dependency changes, and restarts the gateway service so the new version takes effect.

## Step 8. Cleanup

> [!WARNING]
> This removes the Hermes installation and the gateway service. By default, `~/.hermes/` (configuration, conversation history, and skills) is preserved unless you opt into a full uninstall at the on-screen prompt.

Run cleanup from an **interactive terminal**. The uninstaller may refuse non-interactive subprocesses and still asks you to choose whether to keep data or perform a full uninstall. For a full wipe, choose **Full uninstall** and type **`yes`** at the confirmation prompt.

Because the gateway was installed as a **System service** in Step 3e, run the uninstall with `sudo` so it has permission to remove the system-scope systemd unit. If `sudo hermes uninstall` fails with **command not found**, it is because `sudo` does not inherit your user `PATH`; invoke the binary by its full path instead:

```bash
export PATH="$HOME/.local/bin:$PATH"
HERMES_BIN="$(command -v hermes || printf '%s\n' "$HOME/.local/bin/hermes")"
sudo "$HERMES_BIN" uninstall
```

Follow the on-screen prompts to confirm removal. The uninstaller typically:

- Stops and removes the systemd gateway service.
- Removes the `hermes` wrapper script and the PATH entries added to your shell profile.
- Deletes the Hermes application directory.

**Data directory:** The **`~/.hermes`** directory (configuration, sessions, skills) is **not always removed** by `uninstall`, depending on the options you choose at prompts. After uninstall, check whether it still exists:

```bash
ls -la ~/.hermes
```

If you intend a **full** removal, delete it manually (this is irreversible):

```bash
rm -rf ~/.hermes
```

## Step 9. Next steps

1. **Inspect the agent's reasoning.** Inside the TUI, run `/reasoning show` to surface the model's intermediate reasoning alongside its responses. This is especially useful for following the agent's progress on multi-step or complex problems and for debugging unexpected answers.
2. **Try a multi-step task to trigger skill creation.** For example, ask the agent how to set up email — Hermes will walk through the configuration with you and, on completing a non-trivial task like this, may autonomously persist a reusable skill so the next email-related request is faster.
3. **Configure scheduled automations via the built-in cron.** For example, ask Hermes to check the price of a product online once a day and notify you on Telegram when it drops below a threshold. Hermes will schedule the task with its built-in cron and deliver each result through the messaging gateway you set up.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `hermes: command not found` after install | Shell profile not reloaded in the current session | Run `source ~/.bashrc` (or `source ~/.zshrc`) and retry. Open a new terminal if the issue persists. |
| `source ~/.bashrc` works in an interactive terminal, but `hermes` is still missing from a scripted SSH command | Many Ubuntu `.bashrc` files return early for non-interactive shells before the installer-added PATH lines run | In automation, run `export PATH="$HOME/.local/bin:$PATH"` before `hermes`, or call `~/.local/bin/hermes` directly. |
| `sudo: hermes: command not found` during gateway install, uninstall, or printed `sudo hermes …` steps | `sudo` resets `PATH` and does not see the user-level `hermes` shim | Run `which hermes` as your normal user, then invoke that path with sudo, e.g. `sudo "$(which hermes)" uninstall` or `sudo /full/path/from/which/hermes gateway …`. |
| Installer prints **"Setup wizard skipped (no terminal available)"** | The installer was launched from a non-interactive shell, CI job, or SSH command without a usable TTY | Either re-run `hermes setup` in an interactive terminal, or configure the endpoint directly: `hermes config set model.provider custom`, `hermes config set model.base_url http://localhost:8000/v1`, and `hermes config set model.default nvidia/Qwen3.6-35B-A3B-NVFP4`. |
| Installer cannot install `ripgrep` / `ffmpeg`, or prints `Non-interactive mode and no terminal available` | Optional helper install needs `sudo`, but the current shell cannot prompt for a password | Install manually in an interactive terminal with `sudo apt install -y ripgrep ffmpeg`. Hermes still runs without them, but file search is slower and TTS voice-message support is limited. |
| Browser tools show `system dependency not met`, or Playwright Chromium install fails | Playwright needs Linux shared libraries installed through `sudo`, and the installer could not obtain sudo access | Core chat and Telegram can still work. To enable browser tools, run `cd ~/.hermes/hermes-agent && npx playwright install --with-deps chromium` in an interactive terminal and enter your sudo password. |
| You want the gateway to start at boot, but `hermes gateway install` creates a user service | Current Hermes installs a user service by default unless `--system` is supplied | Use `sudo "$(which hermes)" gateway install --system --run-as-user "$USER"` (or replace `$(which hermes)` with `~/.local/bin/hermes` if needed). |
| `hermes uninstall --yes` says it requires an interactive terminal, or still prompts for uninstall options | The uninstaller protects data deletion and expects a real TTY for confirmation | Run it directly in your terminal, or allocate a TTY over SSH (`ssh -t <spark> 'hermes uninstall'`). For a full wipe, select **Full uninstall** and type `yes` when prompted. |
| Telegram bot never answers; gateway logs show timeouts or TLS errors to `api.telegram.org` | Outbound **HTTPS to Telegram is blocked** on the current network (common on locked-down corporate LANs) | From the Spark, run `curl -sS --connect-timeout 10 -o /dev/null -w "HTTP %{http_code}\n" https://api.telegram.org/` (see Instructions). If this hangs or fails, move the Spark to a network that allows Telegram **or** ask IT to allow HTTPS to **`api.telegram.org`**. The rest of the playbook can succeed locally while the bot stays silent. |
| Installer asks about **OpenClaw import / migration** | Another agent framework was previously installed | For this playbook, answer **`n`**. OpenClaw migration is **out of scope** here and can leave gateway or Telegram state confusing. If you already migrated by mistake, use **Start over from scratch** below. |
| Choosing **Telegram** during install immediately shows “setup complete” without token / user ID prompts | Stale or partial Hermes gateway config; installer short-circuit | After `source ~/.bashrc`, run **`hermes gateway setup`**, select Telegram, and complete token and allowed-user steps. Install or restart the systemd service using the printed commands (with `sudo "$(which hermes)"` if needed). |
| `/start` shows “Unknown command” (or similar) in Telegram | Bot does not define a custom `/start` handler | Send a normal text message such as **`hello`** after `/start`. Hermes responds to conversational text, not necessarily slash commands. |
| `~/.hermes` still exists after `uninstall` | Uninstaller preserves data unless you explicitly remove it | This is expected in some flows. Remove manually only if you want a full wipe: `rm -rf ~/.hermes` (see **Start over from scratch**). |
| Hermes installer can't list any models at the model-selection prompt | vLLM is not running yet or is still loading the checkpoint | Sanity-check the endpoint in another terminal: `curl http://localhost:8000/v1/models` should return a `"data"` array containing `nvidia/Qwen3.6-35B-A3B-NVFP4`. If it is empty or unreachable, confirm the vLLM container is up and has finished loading (watch its terminal for `Application startup complete`), then re-run the Hermes installer. |
| `Connection refused` to `http://localhost:8000/v1` from Hermes | vLLM server not running, still loading, or wrong port | Confirm the vLLM container is up and listening on `8000` (`docker ps`, then `curl http://localhost:8000/v1/models`). If it exited, relaunch it (see Instructions — Step 2). |
| Pasting the Telegram bot token shows nothing on the screen | Expected — the installer hides token characters as a security measure | Paste the token, then press **Enter**. The installer should respond with `Telegram token saved`. |
| Telegram bot does not reply when you send `hello` | Gateway service not running, your account is not in the allowed user IDs list, **or outbound HTTPS to Telegram is blocked** | (1) Confirm Telegram HTTPS from the Spark (Instructions — network check). (2) List Hermes units with `systemctl list-units --type=service --all`, locate the gateway unit by name, then `sudo systemctl status <hermes-gateway-unit>` and `sudo journalctl -u <hermes-gateway-unit> -e --no-pager -n 80`. (3) If logs show reachability to Telegram but messages are ignored, verify your numeric user ID is in the allowed list via `hermes gateway setup` or the [Hermes messaging gateway docs](https://hermes-agent.nousresearch.com/docs/user-guide/messaging). |
| Out-of-memory or very slow inference | Served model is too large for available GPU memory, or other GPU workloads are competing | Check usage with `nvidia-smi`, free GPU memory by closing other workloads, or relaunch vLLM with a lower `--gpu-memory-utilization` / `--max-model-len` (or a smaller model handle) and re-point Hermes via `hermes model`. |
| `hermes update` fails or the gateway does not restart | Gateway service still bound to the previous version, or insufficient permissions on a system-service install | Re-run `sudo "$(which hermes)" update` if the gateway was installed as a **System service** and plain `hermes update` cannot restart it. If the service is stuck, restart it manually: `sudo systemctl restart <hermes-gateway-unit>`. |
| Cannot resume a previous session | The `<sessionId>` value is missing or wrong | Use `hermes --resume <sessionId>` with the exact ID Hermes printed when you `/exit` that chat. If the ID is lost, start a new session with `hermes` (omit `--resume`). |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU.
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

For latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
