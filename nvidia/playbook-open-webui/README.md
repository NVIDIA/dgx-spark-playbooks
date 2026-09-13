# Chat with LLMs Using Open WebUI and Ollama

> A self-hosted browser interface with models running locally on your GPU

## Table of Contents

- [Overview](#overview)
- [Open WebUI Remotely](#open-webui-remotely)
- [Open WebUI on Desktop](#open-webui-on-desktop)
- [Agent-ready Models](#agent-ready-models)
  - [Recommendations by hardware platform](#recommendations-by-hardware-platform)
  - [DGX Spark: Agent-ready Qwen3.6-35B-A3B (Ollama)](#dgx-spark-agent-ready-qwen36-35b-a3b-ollama)
  - [Context window guidance](#context-window-guidance)
  - [Next steps](#next-steps)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Open WebUI is a self-hosted chat application that you can run entirely on your hardware platform. This gives you privacy and security because everything stays on your system. The Open WebUI application operates offline, and your queries go to a model running locally.

Open WebUI is browser based, so you can use it to chat with a model from your laptop, or directly on the hardware platform as a desktop.

This playbook shows you two paths to run Open WebUI:

* **Remotely with NVIDIA Sync:** Run Open WebUI on a remote hardware platform from your laptop.
* **Manually on a desktop:** Run Open WebUI on a local desktop session, or if you want to get under the hood.

Both paths lead to the same outcome: chatting with a model running on your hardware platform through Open WebUI.

## What you'll accomplish

You'll download an Open WebUI container image with Ollama onto your hardware platform, run the container, then use the Open WebUI browser interface to download and run a model. Then you'll chat with it.

The container setup includes integrated Ollama for model management, persistent data storage, and GPU acceleration for model inference.

## What to know before starting

**Required:**

- Basic familiarity with terminal commands
- For the remote path: cutting and pasting terminal commands, and how to use NVIDIA Sync ([documentation](https://docs.nvidia.com/sync/latest/direct-connections.html#nvidia-sync-direct-connections))
- For the desktop path: familiarity with Docker commands

**Optional:**

- Experience browsing models on the [Ollama models page](https://ollama.com/search) (for example, [qwen3.6](https://ollama.com/library/qwen3.6) and [gpt-oss](https://ollama.com/library/gpt-oss))

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Open WebUI + Ollama container (`ghcr.io/open-webui/open-webui:ollama`); desktop on port `8080`, NVIDIA Sync custom app on port `12000` | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Hardware platform powered on, networked, and reachable for local desktop or NVIDIA Sync access
- Enough disk space for the container image and models (about 7 GB for the container image; about 15 GB for `gpt-oss:20b` or about 25 GB for `qwen3.6:latest`)

**Software requirements**

- Docker available on the hardware platform (`docker ps`)
- Web browser access to Open WebUI (port `8080` locally, or port `12000` via NVIDIA Sync)
- Network access from the hardware platform to download the container image and models
- Remotely with NVIDIA Sync path: NVIDIA Sync installed on your laptop and connected to your hardware platform

## Find model recipes

Browse models you can pull and run with the integrated Ollama stack in the [Ollama library](https://ollama.com/library). Use the tags and sizes that fit your hardware platform’s memory and storage.

| Hardware platform | More recipes |
| ----------------- | ------------ |
| **DGX Spark** | [Ollama library](https://ollama.com/library) |

Use the **Open WebUI Remotely** or **Open WebUI on Desktop** tab for the base workflow. For agentic workloads, see the **Agent-ready Models** tab.

## Time & risk

- **Estimated time:** 15–20 MIN for setup, including the Open WebUI container download and model download (time varies with your internet speed)
- **Risk level:** Low
  - Docker permission issues may require user group changes and a session restart
  - Large model downloads may take significant time depending on network speed
- **Rollback:** Stop and remove the container, image, and volumes (see Cleanup in each instructions tab). Cleanup is optional and destructive to chat history and downloaded models.
- **Last Updated:** 07/31/2026
  - Run Open WebUI with integrated Ollama on your hardware platform, chat from a browser, and pull models locally

## Open WebUI Remotely

> [!TIP]
> These steps assume NVIDIA Sync is installed and connected to your hardware platform. If you still need to set that up, complete NVIDIA Sync remote-access setup for your hardware platform first. See also the [NVIDIA Sync documentation](https://docs.nvidia.com/sync/latest/direct-connections.html#nvidia-sync-direct-connections).

## Step 1. Use NVIDIA Sync to connect and open a terminal

From your laptop:

- Open NVIDIA Sync with the desktop icon or from the system tray or taskbar.
- Select your hardware platform from the device dropdown.
- Select **Connect**.
- After the connection is established, select Terminal to open a terminal on the hardware platform.

## Step 2. Configure Docker permissions

You must first make sure that your user account can run Docker commands on the hardware platform without sudo.

To test that, in the terminal run:

```bash
docker ps > /dev/null
```

**Success:** If the command returns a blank, then skip ahead to Step 3.

Otherwise, you will see a permission denied error, which means you still need to remove the sudo requirement.
To do that, add your user to the docker group with the commands below.

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Then verify the change is set by testing Docker access again with the command:

```bash
docker ps > /dev/null
```

## Step 3. Download the Open WebUI container image

Pull the container image onto your hardware platform with the command:

```bash
docker pull ghcr.io/open-webui/open-webui:ollama
```

Wait for the image to download, then go to Step 4.

## Step 4. Add Open WebUI as a custom application through NVIDIA Sync

A custom application lets NVIDIA Sync start Open WebUI and automatically forward its port.

In the NVIDIA Sync device window:

- Select **Add New** in the **Custom** section.
- Fill out the form with these values:
  - **Name:** Open WebUI
  - **Port:** 12000
  - **Auto open in browser at the following path:** Check this checkbox

- Then, copy and paste the entire script below into the **Launch Script** field

```bash
#!/usr/bin/env bash
set -euo pipefail

NAME="open-webui"
IMAGE="ghcr.io/open-webui/open-webui:ollama"

cleanup() {
  echo "Signal received; stopping ${NAME}..."
  docker stop "${NAME}" >/dev/null 2>&1 || true
  exit 0
}
trap cleanup INT TERM HUP QUIT EXIT

## Ensure Docker CLI and daemon are available
if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon not reachable." >&2
  exit 1
fi

## Already running?
if [ -n "$(docker ps -q --filter "name=^${NAME}$" --filter "status=running")" ]; then
  echo "Container ${NAME} is already running."
else
#  # Exists but stopped? Start it.
  if [ -n "$(docker ps -aq --filter "name=^${NAME}$")" ]; then
    echo "Starting existing container ${NAME}..."
    docker start "${NAME}" >/dev/null
  else
#    # Not present: create and start it.
    echo "Creating and starting ${NAME}..."
    docker run -d -p 12000:8080 --gpus=all \
      -v open-webui:/app/backend/data \
      -v open-webui-ollama:/root/.ollama \
      --name "${NAME}" "${IMAGE}" >/dev/null
  fi
fi

echo "Running. Press Ctrl+C to stop ${NAME}."
## Keep the script alive until a signal arrives
while :; do sleep 86400; done
```

- Finally, click the **Add** button to finish the configuration.

## Step 5. Launch Open WebUI and create an administrator account

Once the app is configured, you can launch it from NVIDIA Sync and connect to it with your browser.

In the NVIDIA Sync application window for your hardware platform, select **Open WebUI** in the **Custom** section.

The application should open in your web browser at the URL `http://localhost:12000`.

If it does not, open your web browser and go to `http://localhost:12000`.

Open WebUI uses a local administrator account to control access. The account credentials are stored locally on your hardware platform.

When the app opens in your browser, create your admin account as follows:

- Select **Get Started** at the bottom of the screen.
- Complete the admin account creation with easily remembered details.
- Select **Create Admin Account** to complete.

## Step 6. Select a model to download

> [!TIP]
> The Open WebUI container doesn't come with a model, so you must download one before chat will work.
> Open WebUI downloads selected models from the Ollama [registry](https://ollama.com/search).

Do the following in the Open WebUI application:

- Click **Select a model** in the top left corner of the Open WebUI interface.
- Type `gpt-oss:20b` in the search field.
- Click the **Pull "gpt-oss:20b" from Ollama.com** button that appears.
- Wait for the model to fully download. You can monitor progress in the interface.

Alternatively, you can enter `qwen3.6:latest` instead of `gpt-oss:20b`.

After the download completes, the model appears in the **Select a model** menu.

## Step 7. Load the model and submit a query

> [!TIP]
> Selecting an available model loads it onto the GPU, which can take up to 30 seconds, depending on the model size.
> This can delay server response to your initial query.

- Select the model from the **Select a model** menu in the top-left corner.
- In the chat box, enter a prompt such as `Write me a haiku about GPUs` and press Enter.

## Step 8. Stop Open WebUI with NVIDIA Sync

When you finish your session, you can stop the Open WebUI container from the NVIDIA Sync application window.

- Click on the NVIDIA Sync icon in your system tray or taskbar to open the main application window.
- Under the **Custom** section, click the `x` icon on the right of the **Open WebUI** entry.
- This closes the tunnel and stops the Open WebUI Docker container.

## Step 9. Next steps

You can follow up with other playbooks or use different models.

- For agentic workloads, see the **Agent-ready Models** tab for the recommended model on your hardware platform.
- [Find and compare models from the Ollama model registry](https://ollama.com/library).
- Monitor GPU and system usage during inference (for example, via tools available through NVIDIA Sync).

## Step 10. Cleanup

Use these steps when you want to remove Open WebUI from your hardware platform. Cleanup is optional rollback—not required to finish the playbook.

> [!WARNING]
> These commands permanently delete all Open WebUI data and downloaded models on the hardware platform.

1. Stop the Open WebUI application in the NVIDIA Sync device window (this will also stop the container).

2. Open a terminal on the hardware platform using the Terminal App in the NVIDIA Sync device window.

3. Remove the container with the command:

```bash
docker rm open-webui
```

4. Remove the downloaded image with the command:

```bash
docker rmi ghcr.io/open-webui/open-webui:ollama
```

5. Remove the persistent data volumes with the command:

```bash
docker volume rm open-webui open-webui-ollama
```

6. Remove the custom application from NVIDIA Sync by opening the device window and deleting the **Open WebUI** entry from the **Custom** section.

## Open WebUI on Desktop

> [!TIP]
> Use this tab when you are working on a local desktop session on the hardware platform, or when you want to run Docker commands directly. For remote laptop access through NVIDIA Sync, use the **Open WebUI Remotely** tab instead.

## Step 1. Configure Docker permissions

You should first make sure you can run Docker commands without entering your sudo password.

To test that, open a terminal on the hardware platform and run:

```bash
docker ps > /dev/null
```

**Success:** If the command returns a blank, then skip ahead to Step 2.

Otherwise, you will see a permission denied error, which means you still need to remove the sudo requirement.
To do that, add your user to the docker group with the commands below.

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Then verify the change is set by testing Docker access again with the command:

```bash
docker ps > /dev/null
```

## Step 2. Download the Open WebUI container image

Pull the container image onto your hardware platform with the command:

```bash
docker pull ghcr.io/open-webui/open-webui:ollama
```

Wait for the image to download, then go to Step 3.

## Step 3. Start the Open WebUI container

Start the Open WebUI container by running:

```bash
docker run -d -p 8080:8080 --gpus=all \
  -v open-webui:/app/backend/data \
  -v open-webui-ollama:/root/.ollama \
  --name open-webui ghcr.io/open-webui/open-webui:ollama
```

This will start the Open WebUI container and make it accessible at `http://localhost:8080`. You can access the Open WebUI interface from your local web browser.

> [!NOTE]
> Application data will be stored in the `open-webui` volume and model data will be stored in the `open-webui-ollama` volume.

## Step 4. Create administrator account

Set up the initial administrator account for Open WebUI. This is a local account that you will use to access the Open WebUI interface.

- In the Open WebUI interface, click the **Get Started** button at the bottom of the screen.
- Fill out the administrator account creation form with your preferred credentials.
- Click the registration button to create your account and access the main interface.

## Step 5. Download and configure a model

You'll then download a language model through Ollama and configure it for use in Open WebUI. This download happens on your hardware platform and may take several minutes.

- Click on the **Select a model** dropdown in the top left corner of the Open WebUI interface.
- Type `gpt-oss:20b` in the search field.
- Click the **Pull 'gpt-oss:20b' from Ollama.com** button that appears.
- Wait for the model download to complete. You can monitor progress in the interface.
- Once complete, select **gpt-oss:20b** from the model dropdown.

## Step 6. Test the model

You can verify that the setup is working properly by testing model inference through the web interface.

- In the chat text area at the bottom of the Open WebUI interface, enter: **Write me a haiku about GPUs**.
- Press Enter to send the message and wait for the model's response.

## Step 7. Next steps

Try downloading different models from the Ollama library at https://ollama.com/library.

For agentic workloads, see the **Agent-ready Models** tab for the recommended model on your hardware platform.

You can also use the **Open WebUI Remotely** tab so that you can reach the same setup from your laptop through NVIDIA Sync and monitor GPU and memory usage as you try different models.

If Open WebUI reports an update is available, you can update the container image by running:

```bash
docker pull ghcr.io/open-webui/open-webui:ollama
```

## Step 8. Cleanup

Use these steps when you want to completely remove the Open WebUI installation and free up resources. Cleanup is optional rollback—not required to finish the playbook.

> [!WARNING]
> These commands permanently delete all Open WebUI data and downloaded models.

Stop and remove the Open WebUI container:

```bash
docker stop open-webui
docker rm open-webui
```

Remove the downloaded images:

```bash
docker rmi ghcr.io/open-webui/open-webui:ollama
```

Remove persistent data volumes:

```bash
docker volume rm open-webui open-webui-ollama
```

## Agent-ready Models

## Agent-ready models

Agent-ready models are tuned for **agentic workloads** — tool calling, reasoning traces, and long multi-turn sessions. Use this tab to pick a recommended model for your hardware platform, then pull it through Open WebUI’s integrated Ollama stack using either instructions tab.

### Recommendations by hardware platform

| Hardware platform | Recommended agent-ready model | Ollama library tag |
| ----------------- | ----------------------------- | ------------------ |
| **DGX Spark** | Agent-ready Qwen3.6-35B-A3B (Ollama) | [`qwen3.6:35b-a3b`](https://ollama.com/library/qwen3.6:35b-a3b) |

### DGX Spark: Agent-ready Qwen3.6-35B-A3B (Ollama)

Complete the setup through Step 5 in the **Open WebUI Remotely** tab (through creating the administrator account), or through Step 4 in the **Open WebUI on Desktop** tab, so Open WebUI is running and you are signed in.

Then pull and select the recommended model in the Open WebUI interface:

1. Click **Select a model** in the top left corner.
2. Type `qwen3.6:35b-a3b` in the search field.
3. Click the **Pull "qwen3.6:35b-a3b" from Ollama.com** button that appears.
4. Wait for the download to finish, then select **qwen3.6:35b-a3b** from the model dropdown.

#### Verify in chat

In the chat text area, enter a prompt such as **Write me a haiku about GPUs and AI**, then press Enter and wait for the response.

### Context window guidance

For agentic use, prefer a context window of **at least 32K tokens**. If your hardware platform has additional memory headroom, **64K or higher** is recommended so multi-turn sessions and tools have room to work. Adjust context settings in Open WebUI or the model controls for your session when available.

### Next steps

- **Remote workflow:** see the **Open WebUI Remotely** tab
- **Desktop workflow:** see the **Open WebUI on Desktop** tab
- **Browse other models:** [Ollama library](https://ollama.com/library)

## Troubleshooting

## Common issues with setting up via NVIDIA Sync

| Symptom | Cause | Fix |
|---------|-------|-----|
| Permission denied on docker ps | User not in docker group | Run Docker permissions setup completely, including terminal restart |
| Browser doesn't open automatically | Auto-open setting disabled | Manually navigate to `http://localhost:12000` |
| Model download fails | Network connectivity issues | Check internet connection, retry download |
| GPU not detected in container | Missing `--gpus=all` flag | Recreate the container with the correct start script |
| Port 12000 already in use | Another application using port | Change port in Custom App settings or stop the conflicting service |

## Common issues with manual setup

| Symptom | Cause | Fix |
|---------|-------|-----|
| Permission denied on docker ps | User not in docker group | Run Docker permissions setup completely, including logging out and logging back in, or use sudo |
| Model download fails | Network connectivity issues | Check internet connection, retry download |
| GPU not detected in container | Missing `--gpus=all` flag | Recreate the container with the correct command |
| Port 8080 already in use | Another application using port | Change port in the Docker command or stop the conflicting service |

> [!NOTE]
> Some hardware platforms use Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. If you hit memory pressure even when within rated capacity, manually flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
