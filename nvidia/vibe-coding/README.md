# Vibe Coding in VS Code

> Use DGX Spark as a local or remote Vibe Coding assistant with Ollama and Continue


## Table of Contents

- [Overview](#overview)
  - [What You'll Accomplish](#what-youll-accomplish)
  - [Prerequisites](#prerequisites)
  - [Time & risk](#time-risk)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook walks you through setting up DGX Spark as a **Vibe Coding assistant** — locally or as a remote coding companion for VSCode with Continue.dev.  
This guide uses **Ollama** with **GPT-OSS 120B** to provide easy deployment of a coding assistant to VSCode. Included is advanced instructions to allow DGX Spark and Ollama to provide the coding assistant to be available over your local network. This guide is also written on a **fresh installation** of the OS. If your OS is not freshly installed and you have issues, see the troubleshooting tab.

### What You'll Accomplish

You'll have a fully configured DGX Spark system capable of:
- Running local code assistance through Ollama.
- Serving models remotely for Continue and VSCode integration.
- Hosting large LLMs like GPT-OSS 120B using unified memory.

### Prerequisites

- DGX Spark (128GB unified memory recommended)
- **Ollama** and an LLM of your choice (e.g., `gpt-oss:120b`)
- **VSCode**
- **Continue** VSCode extension
- Internet access for model downloads
- Basic familiarity with opening the Linux terminal, copying and pasting commands.
- Having sudo access.
- Optional: firewall control for remote access configuration

### Time & risk
* **Duration:** About 30 minutes 
* **Risks:** Data download slowness or failure due to network issues
* **Rollback:** No permanent system changes made during normal usage.
* **Last Updated:** 10/21/2025
  * First publication

## Instructions

## Step 1. Install Ollama

Install the latest version of Ollama using the following command:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
Once the service is running, pull the desired model:

```bash
ollama pull gpt-oss:120b
```

## Step 2. (Optional) Enable Remote Access

To allow remote connections (e.g., from a workstation using VSCode and Continue), modify the Ollama systemd service:

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

Verify that the workstation can connect to your DGX Spark's Ollama server:

  ```bash
  curl -v http://YOUR_SPARK_IP:11434/api/version
  ```
 Replace **YOUR_SPARK_IP** with your DGX Spark's IP address.
 If the connection fails please see the Troubleshooting tab.

## Step 3. Install VSCode

For DGX Spark (ARM-based), download and install VSCode:
  Navigate to https://code.visualstudio.com/download and download the Linux ARM64 version of VSCode. After
  the download completes note the downloaded package name. Use it in the next command in place of DOWNLOADED_PACKAGE_NAME.
```bash
sudo dpkg -i DOWNLOADED_PACKAGE_NAME
```

If using a remote workstation, **install VSCode appropriate for your system architecture**.

## Step 4. Install Continue.dev Extension

Open VSCode and install **Continue.dev** from the Marketplace:
- Go to **Extensions view** in VSCode
- Search for **Continue** published by [Continue.dev](https://www.continue.dev/) and install the extension.
After installation, click the Continue icon on the right-hand bar.

## Step 5. Local Inference Setup
- Click `Or, configure your own models`
- Click `Click here to view more providers`
- Choose `Ollama` as the Provider
- For Model, select `Autodetect`
- Test inference by sending a test prompt.

Your downloaded model will now be the default (e.g., `gpt-oss:120b`) for inference.

## Step 6. Setting up a Workstation to Connect to the DGX Spark' Ollama Server

To connect a workstation running VSCode to a remote DGX Spark instance the following must be completed on that workstation:
  - Install Continue as instructed in Step 4
  - Click on the `Continue` icon on the left pane
  - Click `Or, configure your own models`
  - Click `Click here to view more providers`
  - Select `Ollama` as the Provider
  - Select `Autodetect` as the Model.

Continue **will** fail to detect the model as it is attempting to connect to a locally hosted Ollama server.
  - Find the `gear` icon in the upper right corner of the Continue window and click on it.
  - On the left pane, click **Models**
  - Next to the first dropdown menu under **Chat** click the gear icon.
  - Continue's `config.yaml` will open. Take note of your DGX Spark's IP address.
  - Replace the configuration with the following. **YOUR_SPARK_IP** should be replaced with your DGX Spark's IP.


```yaml
name: Config
version: 1.0.0
schema: v1

assistants:
  - name: default
    model: OllamaSpark

models:
  - name: OllamaSpark
    provider: ollama
    model: gpt-oss:120b
    apiBase: http://YOUR_SPARK_IP:11434
    title: gpt-oss:120b
    roles:
      - chat
      - edit
      - autocomplete
```

Replace `YOUR_SPARK_IP` with the IP address of your DGX Spark.  
Add additional model entries for any other Ollama models you wish to host remotely.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| **WiFi connection drops or becomes unreachable** (especially in headless mode) | Aggressive WiFi power-saving settings in NetworkManager | Edit `/etc/NetworkManager/conf.d/default-wifi-powersave-on.conf`, set `wifi.powersave = 2`, and run `sudo systemctl restart NetworkManager`. |
| **Random reboots and "00" error code on the display** | Watchdog timer module (`sbsa_gwdt`) not loaded | Add `sbsa_gwdt` to `/etc/modules-load.d/watchdog.conf` and reboot to ensure the hardware watchdog is correctly managed by the kernel. |
| Ollama not starting | GPU drivers may not be installed correctly | Run `nvidia-smi` in the terminal. If the command fails check DGX Dashboard for updates to your DGX Spark. |
| Continue can't connect over the network | Port 11434 may not be open or accessible | Run command `ss -tuln \| grep 11434`. If the output does not reflect `tcp LISTEN 0 4096 *:11434 *:*`, go back to step 2 and run the ufw command. |
| Continue can't detect a locally running Ollama model | Configuration not properly set or detected | Check `OLLAMA_HOST` and `OLLAMA_ORIGINS` in `/etc/systemd/system/ollama.service.d/override.conf` file. If `OLLAMA_HOST` and `OLLAMA_ORIGINS` are set correctly, add these lines to your `~/.bashrc` file. |
| High memory usage | Model size too big | Confirm no other large models or containers are running with `nvidia-smi`. Use smaller models such as `gpt-oss:20b` for lightweight usage. |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU.
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
