# Run Hermes Agent with Local Models

> Install and run the Hermes self-improving AI agent on DGX Spark.

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Verify outbound HTTPS to Telegram (gateway requirement)](#verify-outbound-https-to-telegram-gateway-requirement)
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
- Familiarity with Ollama or willingness to follow the [Ollama on Spark playbook](https://build.nvidia.com/spark/ollama) first
- A Telegram account if you want to use the messaging gateway
- Awareness of the security considerations below

## Important: security and risks

AI agents that can execute commands and reach external services introduce real risks. Read the upstream guidance, especially the dedicated security topics: [Hermes Agent — Security](https://hermes-agent.nousresearch.com/docs/user-guide/security).

Main risks:

1. **Data exposure**: Personal information or files on your DGX Spark may be leaked through agent actions or messaging channels.
2. **Unauthorized access**: A Telegram bot left open to anyone who finds it can be misused; a model endpoint exposed beyond `localhost` can be abused.

You cannot eliminate all risk; proceed at your own risk. **Recommended security measures:**

- **Restrict the Telegram bot** by entering one or more numeric Telegram user IDs at the *"Allowed user IDs"* prompt during install. Leaving this blank allows anyone who finds the bot to use it.
- Keep the Ollama endpoint bound to **`localhost` only**; do not expose `http://<spark-ip>:11434` to your LAN or the public internet without strong authentication.
- Run Hermes on a Spark dedicated to this purpose where possible, and only place files on it that the agent is allowed to access.
- **Monitor activity**: Periodically review the gateway service logs (`sudo journalctl -u <hermes-gateway-unit> -e`) and the Hermes session history.

## Prerequisites

- DGX Spark running Linux, connected to your network
- Terminal (SSH or local) access to the Spark
- `curl` and `git` installed (verified in Step 1 of the instructions)
- Interactive terminal access for the setup wizard and any `sudo` password prompts. Non-interactive SSH is supported with the config-command fallback in the Instructions tab.
- Enough disk and GPU memory for the Ollama model you plan to serve (the playbook uses `qwen3.6:27b` as the example; pick a smaller model if you want a faster first install)
- A Telegram account and the ability to create a bot via [@BotFather](https://t.me/BotFather) if you plan to use the messaging gateway

## Time and risk

- **Duration**: About 30 minutes for install and first-time setup; model download time depends on size and network speed.
- **Risk level**: **Medium** — the agent can execute commands, persist skills, and is reachable from Telegram. Risk increases if you skip the allowed-user-IDs restriction or expose the local model endpoint beyond `localhost`. Always follow the security measures above.
- **Rollback**: Run `hermes uninstall` (with `sudo` if you installed the gateway as a system service) to remove Hermes, the gateway service, and the shell-profile entry. The data directory `~/.hermes` may still be present afterward; remove it manually if you want a full reset (see the Cleanup and Troubleshooting tabs). Uninstall Ollama separately if desired.
- **Last Updated**: 2026-05-08
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

## Step 2. Install Ollama and pull a model

Hermes will be configured against a local Ollama endpoint, so Ollama must be installed and serving at least one model before you run the Hermes installer. If you have already completed the [Ollama on Spark playbook](https://build.nvidia.com/spark/ollama), you can skip this step.

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

> [!NOTE]
> During `install.sh` you might see a message that **systemd is not running** or that a service could not be enabled. On a normal DGX Spark appliance with systemd this is uncommon. If you are on a minimal container, chroot, or unusual environment, Ollama may still run via the `ollama` CLI once the binary is installed; on a standard Spark, prefer fixing the service (`systemctl status ollama`) if the installer warns. If Ollama otherwise starts and answers on port **11434**, you can treat a one-off installer warning as informational.

Verify the Ollama daemon is running and the HTTP API on **11434** responds. The command below asks Ollama for the **list of pulled models** (`GET /api/tags`). A healthy daemon returns **JSON** with a top-level **`"models"`** array (it may be empty until you pull a model):

```bash
curl -sS http://localhost:11434/api/tags
```

Optional: confirm the daemon build string:

```bash
curl -sS http://localhost:11434/api/version
```

Pull the model you intend to use with Hermes (this playbook uses `qwen3.6:27b` as the example):

```bash
ollama pull qwen3.6:27b
```

## Step 3. Install Hermes

Run the installer from an **interactive terminal** on the Spark. If you are connected over SSH, use a normal SSH session where you can answer prompts and enter your `sudo` password when requested. If you run the installer from a non-interactive automation shell, Hermes can install but the setup wizard and optional system-package prompts may be skipped; use the **Non-interactive SSH fallback** below in that case.

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

The installer will walk you through an interactive setup. Respond to each prompt in the order they appear:

> [!IMPORTANT]
> **OpenClaw on the same machine (out of scope for this playbook):** If another tool such as **OpenClaw** was installed previously, the Hermes installer may ask whether you want to **import** or **migrate** from it. For the steps in *this* playbook, answer **`n`** (no) so Hermes does not pull in OpenClaw configuration. Mixing migrations can leave Telegram or gateway state inconsistent; if you already migrated by mistake, prefer a clean reinstall (see **Start over from scratch** in the Troubleshooting tab) before continuing.

1. **"Install ripgrep for faster file search ffmpeg for TTS voice messages? [Y/n]"** — Press **Enter** to accept the default and install both helpers. If `sudo` asks for your password, enter your Linux user password. If you skip this step or run without a terminal, Hermes still works, but file search falls back to slower tools and TTS voice-message support is limited. You can install the helpers later with `sudo apt install -y ripgrep ffmpeg`.

2. **"How would you like to set up Hermes?"** — Choose **Quick setup** to proceed with the recommended defaults.

3. **"Select Provider"** — Choose **Custom endpoint (enter URL manually)** so Hermes can be pointed at the model endpoint running on your DGX Spark.

4. **"API base URL [e.g. https://api.example.com/v1]:"** — *If this prompt appears*, enter the URL of your local model server. For a local Ollama endpoint, use `http://localhost:11434/v1`. (Depending on installer version or prior config, this question is sometimes skipped when the endpoint is already inferred—continue with the prompts you do see.)

5. **"API key [optional]"** — Leave blank and press **Enter**; no key is required for a local model.

6. **Model selection** — The installer lists the models available from your local Ollama instance. Select one to use with Hermes (for example, `qwen3.6:27b`).

7. **"Context length in tokens [leave blank for auto-detect]:"** — Press **Enter** to let Hermes auto-detect the context length from the selected model.

8. **"Display name [Local (localhost:11434)]"** — Press **Enter** to accept the suggested label, or type a custom name to identify this endpoint in the Hermes UI.

9. **"Connect a messaging platform? (Telegram, Discord, etc.)"** — Choose **Set up messaging now (recommended)** to configure a gateway during installation.

10. **"Select platforms to configure:"** — Choose **Telegram**. The remaining steps in this playbook use Telegram as the example; the same flow applies to the other supported gateways.

    > [!TIP]
    > **If Telegram questions are skipped:** Some users see **“Setup complete”** or **“Messaging Platforms (Gateway) configuration complete!”** immediately after choosing Telegram, without token or user-ID prompts. That usually means the installer thinks Telegram is already configured, or a prior partial state exists. Exit any TUI, reload your shell (`source ~/.bashrc`), then run **`hermes gateway setup`** and select Telegram there to supply the bot token and allowed user IDs. (If the CLI suggests `hermes setup gateway` but that flow still skips prompts, use **`hermes gateway setup`**—that is the command most users report as working for a full Telegram reconfiguration.) Follow the printed **`sudo`** lines to register the gateway service (see **Sudo and `hermes` PATH** below).

11. **"Telegram bot token:"** — Open Telegram and start a chat with [@BotFather](https://t.me/BotFather), follow its guided flow to create a new bot, then paste the token BotFather returns into this prompt. **Tip:** Installing [Telegram Desktop](https://desktop.telegram.org/) on the same machine as your SSH session lets you **copy the token from Telegram and paste into the terminal** without retyping it from your phone. The terminal will not echo any characters as the token is pasted — this is expected. Press **Enter** to submit; the installer should respond with `Telegram token saved`.

12. **"Allowed user IDs (comma-separated, leave empty for open access):"** — To restrict the bot to specific Telegram accounts, follow the on-screen instructions to look up your numeric Telegram user ID, then enter one or more IDs separated by commas. Leaving this field blank allows anyone who can reach the bot to use it, which is generally not recommended.

13. **"Use your user ID (\<your-id\>) as the home channel? [Y/n]:"** — Press **Enter** to accept. This designates your own Telegram account as the default channel Hermes will use for proactive messages and scheduled deliveries.

14. **"Install the gateway as a systemd service? (runs in background, starts on boot) [Y/n]:"** — Press **Enter** to accept. The gateway will run as a background service.

15. **"Choose how the gateway should run in the background:"** — Choose **System service** if you want Hermes to start at boot without requiring an interactive login. The service will still run under your user account so it can read your Hermes configuration; only installation requires `sudo`. If you install the gateway after setup instead of through the wizard, use the system-service form shown in **Sudo and `hermes` PATH** below.

16. **"Launch hermes chat now? [Y/n]:"** — Press **Enter** to launch the Hermes TUI immediately and verify the installation end-to-end. Once the TUI is open, type `hello` and press **Enter**; the agent should respond, confirming that the model endpoint and Hermes are wired up correctly. When you're done, type `/exit` to leave the chat and return to your shell. On exit, Hermes prints the exact command needed to resume this conversation later — `hermes --resume <sessionId>`. Save it if you want to pick up where you left off.

17. **"Would you like to install the gateway as a background service? [Y/n]:"** — Press **Enter** to accept. This finalizes the gateway as a background service so it stays available for messaging-platform traffic outside of an interactive Hermes session.

18. **Reload your shell** to make the `hermes` command available, then verify the command resolves:

    ```bash
    source ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
    which hermes
    ```

#### Non-interactive SSH fallback

If the installer prints **"Setup wizard skipped (no terminal available)"**, or if you are validating the playbook through non-interactive SSH, configure the local Ollama endpoint with Hermes' config command:

```bash
export PATH="$HOME/.local/bin:$PATH"
hermes config set model.provider custom
hermes config set model.base_url http://localhost:11434/v1
hermes config set model.default qwen3.6:27b
hermes -z "Reply exactly HERMES_OK"
```

The last command should return `HERMES_OK`, confirming that Hermes can call the local Ollama model without opening the TUI.

#### Sudo and `hermes` PATH

`sudo` runs with a minimal environment and often **does not inherit your user `PATH`**, so `sudo hermes …` can fail with **`hermes: command not found`** even though `hermes` works without `sudo`. Use the real binary path, for example:

```bash
export PATH="$HOME/.local/bin:$PATH"
HERMES_BIN="$(command -v hermes || printf '%s\n' "$HOME/.local/bin/hermes")"
sudo "$HERMES_BIN" uninstall
```

Or paste the absolute path printed by `which hermes` in place of `hermes` in any `sudo` command the installer prints. For a boot-time Linux system service, the current Hermes CLI supports:

```bash
sudo "$HERMES_BIN" gateway install --system --run-as-user "$USER"
```

#### Verify the Telegram gateway (after Step 3)

After configuration, confirm the gateway unit is active and recent logs look healthy (replace `<hermes-gateway-unit>` with the **exact** `*.service` name the installer printed—often something containing `hermes` and `gateway`):

```bash
systemctl list-units --type=service --all | grep -i hermes
systemctl --user list-units --type=service --all | grep -i hermes
sudo systemctl status <hermes-gateway-unit>
sudo journalctl -u <hermes-gateway-unit> -e --no-pager -n 50
```

If `systemctl status` or `systemctl --user status` shows **active (running)** and logs are not repeating connection errors to Telegram, the service side is in good shape. If logs show TLS timeouts or “connection refused” to Telegram hosts, re-run the **outbound HTTPS** check at the top of this page.

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

4. **If you see the “API base URL” prompt**, enter the same local Ollama endpoint as before:

    ```
    http://localhost:11434/v1
    ```

5. When the installer lists the models served by Ollama, choose the one you just pulled. Hermes will use it for subsequent sessions.

If you are in a non-interactive SSH session, switch models with config commands instead:

```bash
hermes config set model.provider custom
hermes config set model.base_url http://localhost:11434/v1
hermes config set model.default <model-name>
hermes -z "Reply exactly MODEL_OK"
```

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

3. Send the message **`hello`**. Hermes will reply through the bot, confirming the gateway is wired to your DGX Spark and the underlying model.

    > [!NOTE]
    > After **`/start`**, Telegram may show a generic **“Unknown command”**-style message from the bot. That can be normal for bots that only implement free-form chat. **Ignore that message and send `hello` anyway**—Hermes should respond to normal text once the gateway and model are healthy.

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

Run cleanup from an **interactive terminal**. The uninstaller may refuse non-interactive subprocesses and still asks you to choose whether to keep data or perform a full uninstall. For a full wipe, choose **Full uninstall** and type **`yes`** at the confirmation prompt.

Because the gateway was installed as a **System service** in Step 15, run the uninstall with `sudo` so it has permission to remove the system-scope systemd unit. If `sudo hermes uninstall` fails with **command not found**, use the same **full-path** pattern as in **Sudo and `hermes` PATH** above:

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
| Installer prints **"Setup wizard skipped (no terminal available)"** | The installer was launched from a non-interactive shell, CI job, or SSH command without a usable TTY | Either re-run `hermes setup` in an interactive terminal, or configure Ollama directly: `hermes config set model.provider custom`, `hermes config set model.base_url http://localhost:11434/v1`, and `hermes config set model.default qwen3.6:27b`. |
| Installer cannot install `ripgrep` / `ffmpeg`, or prints `Non-interactive mode and no terminal available` | Optional helper install needs `sudo`, but the current shell cannot prompt for a password | Install manually in an interactive terminal with `sudo apt install -y ripgrep ffmpeg`. Hermes still runs without them, but file search is slower and TTS voice-message support is limited. |
| Browser tools show `system dependency not met`, or Playwright Chromium install fails | Playwright needs Linux shared libraries installed through `sudo`, and the installer could not obtain sudo access | Core chat and Telegram can still work. To enable browser tools, run `cd ~/.hermes/hermes-agent && npx playwright install --with-deps chromium` in an interactive terminal and enter your sudo password. |
| You want the gateway to start at boot, but `hermes gateway install` creates a user service | Current Hermes installs a user service by default unless `--system` is supplied | Use `sudo "$(which hermes)" gateway install --system --run-as-user "$USER"` (or replace `$(which hermes)` with `~/.local/bin/hermes` if needed). |
| `hermes uninstall --yes` says it requires an interactive terminal, or still prompts for uninstall options | The uninstaller protects data deletion and expects a real TTY for confirmation | Run it directly in your terminal, or allocate a TTY over SSH (`ssh -t <spark> 'hermes uninstall'`). For a full wipe, select **Full uninstall** and type `yes` when prompted. |
| Telegram bot never answers; gateway logs show timeouts or TLS errors to `api.telegram.org` | Outbound **HTTPS to Telegram is blocked** on the current network (common on locked-down corporate LANs) | From the Spark, run `curl -sS --connect-timeout 10 -o /dev/null -w "HTTP %{http_code}\n" https://api.telegram.org/` (see Instructions). If this hangs or fails, move the Spark to a network that allows Telegram **or** ask IT to allow HTTPS to **`api.telegram.org`**. The rest of the playbook can succeed locally while the bot stays silent. |
| Installer asks about **OpenClaw import / migration** | Another agent framework was previously installed | For this playbook, answer **`n`**. OpenClaw migration is **out of scope** here and can leave gateway or Telegram state confusing. If you already migrated by mistake, use **Start over from scratch** below. |
| Choosing **Telegram** during install immediately shows “setup complete” without token / user ID prompts | Stale or partial Hermes gateway config; installer short-circuit | After `source ~/.bashrc`, run **`hermes gateway setup`**, select Telegram, and complete token and allowed-user steps. Install or restart the systemd service using the printed commands (with `sudo "$(which hermes)"` if needed). |
| `/start` shows “Unknown command” (or similar) in Telegram | Bot does not define a custom `/start` handler | Send a normal text message such as **`hello`** after `/start`. Hermes responds to conversational text, not necessarily slash commands. |
| `~/.hermes` still exists after `uninstall` | Uninstaller preserves data unless you explicitly remove it | This is expected in some flows. Remove manually only if you want a full wipe: `rm -rf ~/.hermes` (see **Start over from scratch**). |
| Hermes installer can't list any models at the model-selection prompt | Ollama is not running or has no models pulled | Sanity-check Ollama in another terminal: list installed models with `ollama list`, hit the API with `curl http://localhost:11434/api/tags`, and confirm a model can actually serve requests by running `ollama run <model-name>` (e.g. `ollama run qwen3.6:27b`) and sending a test prompt. If the list is empty or the API is unreachable, start Ollama and pull a model with `ollama pull <model-name>`, then re-run the Hermes installer. |
| `Connection refused` to `http://localhost:11434/v1` from Hermes | Ollama service not running on the default port | Start the Ollama service and confirm it is listening on `11434`. On systemd hosts: `systemctl status ollama` and `systemctl start ollama`. |
| Pasting the Telegram bot token shows nothing on the screen | Expected — the installer hides token characters as a security measure | Paste the token, then press **Enter**. The installer should respond with `Telegram token saved`. |
| Telegram bot does not reply when you send `hello` | Gateway service not running, your account is not in the allowed user IDs list, **or outbound HTTPS to Telegram is blocked** | (1) Confirm Telegram HTTPS from the Spark (Instructions — network check). (2) List Hermes units with `systemctl list-units --type=service --all`, locate the gateway unit by name, then `sudo systemctl status <hermes-gateway-unit>` and `sudo journalctl -u <hermes-gateway-unit> -e --no-pager -n 80`. (3) If logs show reachability to Telegram but messages are ignored, verify your numeric user ID is in the allowed list via `hermes gateway setup` or the [Hermes messaging gateway docs](https://hermes-agent.nousresearch.com/docs/user-guide/messaging). |
| Out-of-memory or very slow inference | Selected Ollama model is too large for available GPU memory, or other GPU workloads are competing | Check usage with `nvidia-smi`, free GPU memory by closing other workloads, or pull a smaller model with `ollama pull <smaller-model>` and switch to it via `hermes model`. |
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
