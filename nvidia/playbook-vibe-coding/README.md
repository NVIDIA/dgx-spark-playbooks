# Set Up Vibe Coding in VS Code

> Code completion, chat, and edits from a local model

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook walks you through setting up your hardware platform as a **Vibe Coding assistant** — locally or as a remote coding companion for VS Code with Continue.dev.

It uses **Ollama** with **GPT-OSS 120B** so you can run code completion, chat, and edits from a local model in VS Code. Optional steps also show how to expose Ollama on your local network so a separate workstation can use the same assistant. The guide assumes a fresh OS install; if you hit issues on an existing system, see the **Troubleshooting** tab.

## What you'll accomplish

You'll configure your **hardware platform** to:

- Run local code assistance through Ollama
- Serve models remotely for Continue and VS Code integration
- Host large LLMs such as GPT-OSS 120B for coding workflows

## What to know before starting

**Required:**

- Familiarity with opening a Linux terminal and copying commands
- A user account with sudo privileges on the hardware platform
- Basic familiarity with VS Code extensions

**Optional:**

- Firewall control if you plan to enable remote access from another machine on your network

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Ollama + `gpt-oss:120b`; Continue in VS Code; API on port `11434` | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Hardware platform powered on, networked, and reachable for local or SSH terminal access
- Sufficient memory for your chosen model (128 GB unified memory recommended for `gpt-oss:120b`)

**Software requirements**

- **Ollama** and an LLM of your choice (for example, `gpt-oss:120b`)
- **VS Code**
- **Continue** VS Code extension
- Internet access for model downloads
- Optional: firewall control for remote access configuration

## Time & risk

- **Estimated time:** 30 MIN (longer on first run due to model download)
- **Risk level:** Low
  - Large model downloads may be slow or fail due to network issues
- **Rollback:** No permanent system changes during normal usage; stop Ollama and remove the Continue configuration if you no longer need the assistant
- **Last Updated:** 07/31/2026
  - Set up Ollama with Continue in VS Code for local and remote vibe coding on supported hardware platforms

## Instructions

## Step 1. Install Ollama

Install the latest version of Ollama on your hardware platform:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Once the service is running, pull the desired model:

```bash
ollama pull gpt-oss:120b
```


## Step 2. (Optional) Enable remote access

To allow remote connections (for example, from a workstation using VS Code and Continue), modify the Ollama systemd service on the hardware platform:

```bash
sudo systemctl edit ollama
```

Add the following lines beneath the commented section:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"
```

Reload and restart the service:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

If using a firewall, open port 11434:

```bash
sudo ufw allow 11434/tcp
```

Verify that the workstation can connect to the Ollama server on your hardware platform:

```bash
curl -v http://YOUR_HARDWARE_IP:11434/api/version
```

Replace **YOUR_HARDWARE_IP** with your hardware platform's reachable IP address. If the connection fails, see the **Troubleshooting** tab.

## Step 3. Install VS Code

Download and install VS Code for the architecture you will use for editing:

1. Go to [https://code.visualstudio.com/download](https://code.visualstudio.com/download)
2. Download the Linux package that matches your system architecture (for example, ARM64 on ARM-based hardware platforms, or the matching package on a remote workstation)
3. After the download completes, note the package name and install it:

```bash
sudo dpkg -i DOWNLOADED_PACKAGE_NAME
```

If you use a remote workstation, install VS Code appropriate for that system's architecture.

## Step 4. Install the Continue extension

Open VS Code and install **Continue** from the Marketplace:

1. Open the **Extensions** view in VS Code
2. Search for **Continue** published by [Continue.dev](https://www.continue.dev/) and install the extension
3. After installation, click the Continue icon in the activity bar

## Step 5. Local inference setup

On the same machine where Ollama is running:

1. Click **Or, configure your own models**
2. Click **Click here to view more providers**
3. Choose **Ollama** as the provider
4. For Model, select **Autodetect**
5. Test inference by sending a test prompt

Your downloaded model (for example, `gpt-oss:120b`) becomes the default for inference.

## Step 6. Connect a workstation to the remote Ollama server

To connect a workstation running VS Code to Ollama on a remote hardware platform, complete the following on that workstation:

1. Install Continue as instructed in Step 4
2. Click the **Continue** icon
3. Click **Or, configure your own models**
4. Click **Click here to view more providers**
5. Select **Ollama** as the provider
6. Select **Autodetect** as the model

Continue will fail to detect the model because it is attempting to connect to a locally hosted Ollama server. Point it at your hardware platform instead:

1. Click the gear icon in the upper right of the Continue window
2. On the left pane, click **Models**
3. Next to the first dropdown under **Chat**, click the gear icon
4. Continue's `config.yaml` opens — replace the configuration with the following, substituting **YOUR_HARDWARE_IP** with your hardware platform's IP address:

```yaml
name: Config
version: 1.0.0
schema: v1

assistants:
  - name: default
    model: OllamaRemote

models:
  - name: OllamaRemote
    provider: ollama
    model: gpt-oss:120b
    apiBase: http://YOUR_HARDWARE_IP:11434
    title: gpt-oss:120b
    roles:
      - chat
      - edit
      - autocomplete
```

Add additional model entries for any other Ollama models you want to host remotely.

## Step 7. Cleanup

When you no longer need the assistant:

- Stop the Ollama service if desired: `sudo systemctl stop ollama`
- Remove or revert remote-access overrides under `/etc/systemd/system/ollama.service.d/` if you enabled Step 2
- Remove or edit the Continue `config.yaml` entries that point at the remote server

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Ollama not starting | GPU drivers may not be installed correctly | Run `nvidia-smi` in the terminal. If the command fails, check system updates for your hardware platform (for example, via the platform dashboard or package updates). |
| Continue can't connect over the network | Port 11434 may not be open or accessible | Run `ss -tuln \| grep 11434`. If the output does not show a listen on `*:11434`, return to Step 2 and run the `ufw` allow command. |
| Continue can't detect a locally running Ollama model | Configuration not properly set or detected | Check `OLLAMA_HOST` and `OLLAMA_ORIGINS` in `/etc/systemd/system/ollama.service.d/override.conf`. If those values are correct, add these lines to `~/.bashrc`: `export OLLAMA_HOST=0.0.0.0:11434` and `export OLLAMA_ORIGINS=*`. Then run `source ~/.bashrc` (or reopen the terminal). |
| High memory usage | Model size too large for available memory | Confirm no other large models or containers are running with `nvidia-smi`. Use a smaller model such as `gpt-oss:20b` for lightweight usage. |

> [!NOTE]
> Some hardware platforms use Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. With many applications still updating to take advantage of UMA, you may encounter memory pressure even when within rated capacity. If that happens, manually flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
