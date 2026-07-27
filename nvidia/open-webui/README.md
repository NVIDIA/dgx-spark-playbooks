# Open WebUI with Ollama

> Install Open WebUI and use Ollama to chat with models on your Spark

## Table of Contents

- [Overview](#overview)
- [Open WebUI on Remote Spark](#open-webui-on-remote-spark)
- [Open WebUI on Desktop Spark](#open-webui-on-desktop-spark)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Open WebUI is a self-hosted chat application that you can run entirely on your DGX Spark.
This gives you privacy and security because everything stays on your system.
The Open WebUI application operates offline, and your queries go to a model running on your Spark.

Open WebUI is browser based, so you can use it to chat with a model on a Spark from your laptop, or directly on the Spark itself as a desktop.

This playbook shows you two different paths to run Open WebUI. 

* **Remotely with NVIDIA Sync**: Use this path to run Open WebUI on a remote Spark from your laptop. 
* **Manually on a Desktop**: Use this path to run Open WebUI on a desktop Spark or if you want to get under the hood.

Both paths lead to the same outcome: chatting with a model running on your DGX Spark through Open WebUI.

## What you'll accomplish

You will download an Open WebUI container image with Ollama onto your Spark, run the container and then use the Open WebUI browser interface to download and run a model.
Then, you will chat with it. 

The container setup includes integrated Ollama for model management, persistent data storage, and GPU acceleration for model inference.

## What to know before starting

* The [Ollama models page](https://ollama.com/search) to get information on models, e.g. [qwen3.6](https://ollama.com/library/qwen3.6) and [gpt-oss](https://ollama.com/library/gpt-oss)
* Remote with NVIDIA Sync: Cut and paste terminal commands; How to use NVIDIA Sync. [See documentation here](https://docs.nvidia.com/sync/latest/direct-connections.html#nvidia-sync-direct-connections) and [installation here](https://build.nvidia.com/spark/connect-to-your-spark/sync).
* Manually on a Desktop: Terminal experience; Familiarity with Docker commands


## Prerequisites

-  DGX Spark [device is set up](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html) and accessible
-  Enough disk space for the container image and models (7GB for container image, 25GB for qwen3.6:latest or 15GB for gpt-oss:latest)
-  Remotely with NVIDIA Sync path: [NVIDIA Sync installed on your laptop and connected to the Spark](https://build.nvidia.com/spark/connect-to-your-spark/sync) to your DGX Spark

## Time & risk

* **Duration**: 15-20 minutes for setup, includes Open WebUI container download as well as model download (time varies per your internet speed)
* **Risks**:
  * Docker permission issues may require user group changes and session restart
  * Large model downloads may take significant time depending on network speed
* **Last Updated**: 07/25/2026 (minor copy edits)

## Open WebUI on Remote Spark

## Step 1. Use NVIDIA Sync to connect to the Spark and open a terminal

> [!TIP]
> If you haven't already installed NVIDIA Sync, [learn how here.](/spark/connect-to-your-spark/sync)

From your laptop:

- Open NVIDIA Sync with the desktop icon or from the system tray or taskbar.
- Select your Spark from the device dropdown.
- Select **Connect**.
- After the connection is established, select Terminal to open a terminal on the Spark

## Step 2. Configure Docker permissions

You must first make sure that your user account can run Docker commands on the Spark without sudo.

To test that, in the terminal run:

```bash
docker ps > /dev/null
```

**Success**: If the command returns a blank, then skip ahead to Step 3.

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

Pull the container image onto your Spark with the command:

```bash
docker pull ghcr.io/open-webui/open-webui:ollama
```

Wait for the image to download, then go to Step 4.


## Step 4. Add Open WebUI as a custom application through NVIDIA Sync

A custom application lets NVIDIA Sync start Open WebUI and automatically forward its port.

In the NVIDIA Sync device window:

- Select **Add New** in the **Custom** section.
- Fill out the form with these values:
  - **Name**: Open WebUI
  - **Port**: 12000
  - **Auto open in browser at the following path**: Check this checkbox

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

- Finally, click the "Add" button to finish the configuration.

## Step 5. Launch Open WebUI and create an administrator account

Once the app is configured, you can launch it from NVIDIA Sync and connect to it with your browser.

In the NVIDIA Sync application window for the Spark, select "Open WebUI" in the "Custom" section.

The application should open in your web browser at the URL `http://localhost:12000`.

If it does not, open your web browser and go to `http://localhost:12000`.

Open WebUI uses a local administrator account to control access. The account credentials are stored locally on your Spark.

When the app opens in your browser, create your admin account as follows:
- Select "Get Started" at the bottom of the screen.
- Complete the admin account creation with easily remembered details.
- Select "Create Admin Account" to complete. 

## Step 6. Select a model to download

> [!TIP]
> The Open WebUI container doesn't come with a model so you must download one before chat will work. 
> Open WebUI downloads selected models from the Ollama [registry here](https://ollama.com/search).

Do the following in the Open WebUI application:

- Click "Select a model" in the top left corner of the Open WebUI interface.
- Type `gpt-oss:latest` in the search field.
- Click the `Pull "gpt-oss:latest" from Ollama.com` button that appears.
- Wait for the model to fully download. You can monitor progress in the interface.

Alternatively, you can enter `qwen3.6:latest` instead of `gpt-oss:latest`.

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
- Under the "Custom" section, click the `x` icon on the right of the "Open WebUI" entry.
- This closes the tunnel and stops the Open WebUI Docker container.

## Step 9. Next steps

You can follow up with other playbooks or use different models.

- [Use DGX Dashboard to monitor GPU and memory utilization while working with the model](https://build.nvidia.com/spark/dgx-dashboard/instructions)
- [Find and compare models from the Ollama model registry](https://ollama.com/library).

## Step 10. Cleanup and rollback

Steps to remove the Open WebUI from your Spark.

> [!WARNING]
> These commands will permanently delete all Open WebUI data and downloaded models on the Spark.

1. Stop the Open WebUI application in the NVIDIA Sync device window (this will also stop the container)

2. Open a terminal on the Spark using the Terminal App in the NVIDIA Sync device window

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

## Open WebUI on Desktop Spark

## Step 1. Configure Docker permissions

You should first make sure you can run Docker commands without entering your sudo password. 

To test that, open a terminal on the Spark and run:

```bash
docker ps > /dev/null
```

**Success**: If the command returns a blank, then skip ahead to Step 2.

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

Pull the container image onto your Spark with the command:

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

- In the Open WebUI interface, click the "Get Started" button at the bottom of the screen.
- Fill out the administrator account creation form with your preferred credentials.
- Click the registration button to create your account and access the main interface.

## Step 5. Download and configure a model

You'll then download a language model through Ollama and configure it for use in
Open WebUI. This download happens on your DGX Spark device and may take several minutes.

- Click on the "Select a model" dropdown in the top left corner of the Open WebUI interface.
- Type `gpt-oss:20b` in the search field.
- Click the "Pull 'gpt-oss:20b' from Ollama.com" button that appears.
- Wait for the model download to complete. You can monitor progress in the interface.
- Once complete, select "gpt-oss:20b" from the model dropdown.

## Step 6. Test the model

You can verify that the setup is working properly by testing model
inference through the web interface.

- In the chat text area at the bottom of the Open WebUI interface, enter: **Write me a haiku about GPUs**.
- Press Enter to send the message and wait for the model's response.

## Step 7. Next steps

Try downloading different models from the Ollama library at https://ollama.com/library.

You can try this [set up with NVIDIA Sync](/spark/open-webui/sync) so that you can monitor GPU and memory usage through the DGX Dashboard as you try different models.

If Open WebUI reports an update is available, you can update the container image by running:

```bash
docker pull ghcr.io/open-webui/open-webui:ollama
```

## Step 8. Cleanup and rollback

Steps to completely remove the Open WebUI installation and free up resources.

> [!WARNING]
> These commands will permanently delete all Open WebUI data and downloaded models.

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

## Troubleshooting

## Common issues with setting up via NVIDIA Sync

| Symptom | Cause | Fix |
|---------|-------|-----|
| Permission denied on docker ps | User not in docker group | Run Step 1 completely, including terminal restart |
| Browser doesn't open automatically | Auto-open setting disabled | Manually navigate to localhost:12000 |
| Model download fails | Network connectivity issues | Check internet connection, retry download |
| GPU not detected in container | Missing `--gpus=all flag` | Recreate container with correct start script |
| Port 12000 already in use | Another application using port | Change port in Custom App settings or stop conflicting service |

## Common issues with manual setup

| Symptom | Cause | Fix |
|---------|-------|-----|
| Permission denied on docker ps | User not in docker group | Run Step 1 completely, including logging out and logging back in or use sudo|
| Model download fails | Network connectivity issues | Check internet connection, retry download |
| GPU not detected in container | Missing `--gpus=all flag` | Recreate container with correct command |
| Port 8080 already in use | Another application using port | Change port in docker command or stop conflicting service |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
