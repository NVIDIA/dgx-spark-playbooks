# Serve LLMs with vLLM on One Device

> A locally accessible model API from one DGX Spark or Station

## Table of Contents

- [Overview](#overview)
- [Beginner](#beginner)
- [Intermediate](#intermediate)
  - [Step 1. Connect with NVIDIA Sync](#step-1-connect-with-nvidia-sync)
  - [Step 2. Check the Docker group and the Hugging Face CLI](#step-2-check-the-docker-group-and-the-hugging-face-cli)
  - [Step 3. Download the container and model for the device recommended recipe](#step-3-download-the-container-and-model-for-the-device-recommended-recipe)
  - [Step 4. Add the launch script to NVIDIA Sync as a custom app](#step-4-add-the-launch-script-to-nvidia-sync-as-a-custom-app)
  - [Step 5. Launch the app and monitor vLLM startup](#step-5-launch-the-app-and-monitor-vllm-startup)
  - [Step 6. Verify the server through the locally available endpoint](#step-6-verify-the-server-through-the-locally-available-endpoint)
  - [Next steps](#next-steps)
- [Advanced Qwen on Spark](#advanced-qwen-on-spark)
  - [Step 1. Select the recipe and save the Docker command](#step-1-select-the-recipe-and-save-the-docker-command)
  - [Step 2. Get the container and model on the device](#step-2-get-the-container-and-model-on-the-device)
  - [Step 3. Use the copied Docker command to run the vLLM container](#step-3-use-the-copied-docker-command-to-run-the-vllm-container)
  - [Step 4. Test the vLLM server and tool calling](#step-4-test-the-vllm-server-and-tool-calling)
  - [Step 5. (optional) Save the Docker run command and tests as a Bash script](#step-5-optional-save-the-docker-run-command-and-tests-as-a-bash-script)
  - [Next steps](#next-steps)
- [Advanced DeepSeek on Station](#advanced-deepseek-on-station)
  - [Step 1. Select the recipe and save the Docker command](#step-1-select-the-recipe-and-save-the-docker-command)
  - [Step 2. Get the container and model on the device](#step-2-get-the-container-and-model-on-the-device)
  - [Step 3. Run the vLLM container](#step-3-run-the-vllm-container)
  - [Step 4. Test the server and tool calling](#step-4-test-the-server-and-tool-calling)
  - [Step 5. (optional) Save the Docker run command and tests as a Bash script](#step-5-optional-save-the-docker-run-command-and-tests-as-a-bash-script)
  - [Step 6. (conditional) Configure the endpoint for Hermes or OpenShell](#step-6-conditional-configure-the-endpoint-for-hermes-or-openshell)
  - [Next steps](#next-steps)
- [Troubleshooting](#troubleshooting)
  - [Intermediate: NVIDIA Sync Custom App](#intermediate-nvidia-sync-custom-app)
  - [Advanced: Docker port mapping and SSH tunnel](#advanced-docker-port-mapping-and-ssh-tunnel)

---

## Overview

## Basic idea

[vLLM](https://vllm.ai/) is a high-performance inference engine for serving text and multimodal language models at datacenter scale.
It offers considerable performance and flexibility, but configuring it for a particular model and use case takes skill and experience, especially on a local system.

This playbook provides three paths of increasing complexity to set up vLLM to run a text model on a DGX Spark or DGX Station ([for a cluster, go here](https://build.nvidia.com/playbooks/vllm-cluster)). 

The paths go from a fully software-assisted model launch to manually configuring a recipe yourself.

At the end you will be able to use the remote endpoint securely at `localhost` on your laptop, or you will have a harness running on the remote consuming the endpoint directly on that device. 

## What you'll accomplish

- **[Beginner](beginner.md):** Use NVIDIA Sync's model launch feature to run a model on a remote Spark or Station.
- **[Intermediate](intermediate.md):** Manually prepare a recommended model on the remote device, add its launch script as a Sync Custom App, and then verify the API from your laptop.
- **Advanced ([Qwen on 128GB Spark](advanced-qwen-on-spark.md) or [DeepSeek V4.1 Flash on DGX Station](advanced-deepseek-on-station.md)):** Do the entire process manually to set up the server, primarily aimed at [OpenShell](https://build.nvidia.com/playbooks/openshell) or [Hermes](https://build.nvidia.com/playbooks/hermes-agent) playbook support.

## What to know before starting

- Using [NVIDIA Sync](https://docs.nvidia.com/sync/latest/index.html) for device connection and launching an application. **Beginner · Intermediate** (optional for Advanced)
- Reading Docker and Hugging Face CLI commands. **Intermediate · Advanced**
- Choosing a model recipe and understanding model size and precision. **Advanced**
- Reading or editing a Bash launch script. **Advanced**
- Configuring `docker run` and `vllm serve` arguments. **Advanced**

## Supported hardware platforms

Check the table below to confirm which path this playbook supports for your hardware.

| Hardware platform | OS | Memory | One device |
| :---- | :---- | :---- | :----: |
| **DGX Spark** | DGX OS (Linux) | 64 GB or 128 GB Unified Memory | ✅ |
| **DGX Station** | Ubuntu (Linux) | Large HBM + Grace DRAM | ✅ |

## Prerequisites

**Hardware requirements**

- One DGX Spark or DGX Station. 
- For Beginner and Intermediate using NVIDIA Sync, the device must be reachable from your laptop.

**Software and access requirements**

- [NVIDIA Sync installed](https://docs.nvidia.com/sync/latest/getting-started.html#installation-and-onboarding) on your laptop, with the device [added and connected](https://docs.nvidia.com/sync/latest/direct-connections.html#adding-a-device-for-a-direct-connection). **Beginner · Intermediate** (optional for Advanced)
- Docker installed on the DGX device and [available to your user](https://docs.docker.com/engine/install/linux-postinstall/#add-your-user-to-the-docker-group). **Beginner · Intermediate · Advanced**
- A Hugging Face token if the selected model or launcher asks for one. **Beginner**
- A Hugging Face account and an [installed](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#standalone-installer-recommended), [authenticated](https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication) Hugging Face CLI on the device. **Intermediate · Advanced**
- Configured SSH access to the remote device. **Advanced**

## Time & risk

- **Estimated time:** Beginner and Intermediate times vary between 10 and 25 minutes depending on your network speed. Advanced takes about 30 minutes more to account for manual steps.
- **Risk level:** Low
- **Stop or rollback:** You can stop configuration flows at any time, and you can remove the container or model from the device manually.
- **Last Updated:** 10/07/2026

## Beginner

## Use the NVIDIA Sync application to set everything up for you

You will use NVIDIA Sync to access the DGX device in appliance mode.

Then you will use the [model launcher](https://docs.nvidia.com/sync/latest/index.html) feature to run the model.

Finally, you will start OpenCode on the remote to query the model.

## Step 1. Install NVIDIA Sync on your laptop and add a DGX Spark or Station (one time)

**First, download and install NVIDIA Sync on your laptop.**

::spark-download

**For Windows:** Double-click the `.exe` installer and follow the instructions.

**For macOS:** Open `nvidia-sync.dmg`, drag and drop it into the Applications folder, then launch it from Applications.

**For Ubuntu:** Follow the Ubuntu installation instructions [here](https://docs.nvidia.com/sync/latest/getting-started.html#installation).

**Then, add your DGX Spark or Station device to NVIDIA Sync.**

NVIDIA Sync scans for mDNS broadcasting devices and checks your local SSH config file, but if your remote device doesn't appear then select **Add a device manually**, enter the information below, then select **Add** to proceed.

1. `Name`: a display name  
2. `Hostname or IP`: the IP address
3. `Username`: your username for the device (should have `sudo` privileges)
4. `Password`: password for that username 

**Success**: A **Get started** modal will appear. 

## Step 2. Configure the model (one time)

**First, go to Settings and configure the container, model and OpenCode.**

1. Go to **Settings > Models tab** in NVIDIA Sync
2. Select the device from the drop down and read the configuration guidance
3. Select **Configure** and follow the container and model download  
4. When the downloads complete, follow the OpenCode install on the remote

> [!NOTE]
> Downloading the container and model can take a long time.
> The DGX Spark container and model are roughly 50 GBs in size.
> The DGX Station container and model are roughly 300 GB in size.

**Leave the endpoint access mode at "Private".**

The default mode securely tunnels the endpoint to `localhost:18765/v1` on your laptop. 

> [!NOTE]
> The **Shared** mode opens the endpoint at the device's IP address on your network and adds a password to it.
> This will break the OpenCode launch, so leave it at **Private** until you are comfortable with how things work ([see NVIDIA Sync docs](https://docs.nvidia.com/sync/latest/index.html)). 
 
**Success**: The model card shows details for the device.

## Step 3. Launch the container and follow the vLLM server until it's ready (repeat use)

**First, launch the model from the model card in the Models tab in Settings.**

Select "Start model" from the option dots on the right hand side of the card.
This runs the container on the remote with the proper recipe. 

> [!NOTE]
> It can take up to 10 minutes for the server to load the model and be ready for inference.

**Then, open the resource monitor to watch for the model loading on the GPU.**

1. Open the NVIDIA Sync task bar utility
2. Select the [Resource Monitor](https://docs.nvidia.com/sync/latest/resource-monitor.html)
3. Select the `GPU UTILIZATION` tab

Both `GPU UTILIZATION` and `UNIFIED MEMORY` go up as the server loads the model. 

**Finally, watch the Docker logs to monitor the server status for ready state.**

1. Open a terminal from the task bar utility
2. Run `docker ps` to see the container name, `<container-name>`
3. Then run `watch docker logs <container-name>` to see the server status

**Success**: The model section in NVIDIA Sync transitions to `Running`. The Docker logs show `Application startup complete`.

## Step 4. Launch OpenCode. (repeat use)

**Once the model is available, launch OpenCode from the task bar utility.**

1. Click the OpenCode button in the NVIDIA Sync utility
2. Wait for the browser to open to the application, and then create a user and a password
3. Then chat with the agent as it's configured to use the endpoint on the remote device.

**Success**: OpenCode returns a response from the model.

## Step 5. Next Steps

1. Learn about **Shared** mode in the [NVIDIA Sync documentation](https://docs.nvidia.com/sync/latest/index.html).
2. Follow the [playbook to cluster two or more DGX Spark devices](https://build.nvidia.com/playbooks/vllm-cluster).
3. Use [Troubleshooting](troubleshooting.md) if Sync setup, model launch, or OpenCode does not work as expected.

## Intermediate

## Set up and serve a recommended vLLM recipe as an NVIDIA Sync custom application

You will set up the remote system to run a device recommended model in a container.
It's a one time setup.

Then you will add a container launch script to Sync to create a custom app.
It's also a one time setup.

After that, you can launch the containerized model directly from NVIDIA Sync.

### Step 1. Connect with NVIDIA Sync

**Connect to your DGX Spark or Station with NVIDIA Sync and open a remote terminal.**

See the [Beginner tab](beginner.md) for onboarding instructions, or follow this [demo video](https://www.youtube.com/watch?v=MehBUQtb9qM&t).

### Step 2. Check the Docker group and the Hugging Face CLI

The launch scripts assume that Docker and the Hugging Face CLI work appropriately.

**First, verify that your user can run `docker` commands without requiring a password every time.**

You need to be able to run `docker` commands without entering a password every time.

Check that with:

```bash
docker ps > /dev/null
```

If the command reports permission denied, then add your user to the `docker` group.

1. Add the user with `sudo usermod -aG docker $USER`
2. Make those new permissions available in the terminal with `newgrp docker`
3. Then verify it worked with `docker ps >/dev/null`

> [!NOTE]
> Adding your user to the `docker` group comes with risk of granting root access without a password.
> See the [Docker user guide](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user) for more information.

**Next, make sure the Hugging Face CLI is installed and authenticated.**

You need the Hugging Face CLI installed.
It should be authenticated to access private models and avoid download rate limits.

Check that with:

```bash
hf auth whoami
```

1. If it returns `command not found`, install the CLI [per Hugging Face instructions](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#standalone-installer-recommended) and then refresh the terminal with `source $HOME/.bashrc`
2. If it returns `Not logged in`, then authenticate the CLI [per Hugging Face instructions](https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication)

**Success**: `docker ps` does not return a permission error, and `hf auth whoami` prints your username and organizations.

### Step 3. Download the container and model for the device recommended recipe

**Use the commands in the table to get the proper container and model on disk.**

1. Run the Docker pull command to download and extract the container
2. Run the Hugging Face CLI command to download the model

| Device | Memory | Recipe | Docker pull command | Model download command |
| --- | --- | --- | --- | --- |
| DGX Spark | 64GB unified memory | [Qwen/Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B?hardware=dgx_spark_gb10) | `docker pull vllm/vllm-openai:qwen38` | `hf download nvidia/Qwen3.8-27B-NVFP4` |
| DGX Spark | 128GB unified memory | [Qwen3.6-35B-A3B (NVFP4)](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B?hardware=dgx_spark_gb10&variant=nvfp4) | `docker pull vllm/vllm-openai:v0.28.0` | `hf download nvidia/Qwen3.6-35B-A3B-NVFP4` |
| DGX Station | 748GB coherent memory | [Nemotron 3 Ultra (NVFP4)](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Ultra/StationDeploymentGuide) | `docker pull vllm/vllm-openai:v0.22.0` | `hf download nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` |

> [!NOTE]
> Container and checkpoint sizes vary by recipe, and downloads can take a long time.
> The DGX Station checkpoint is especially large, so allow additional download time and confirm you have sufficient storage.
> The Nemotron checkpoint may require Hugging Face access. Accept the model terms and authenticate before downloading if prompted.

### Step 4. Add the launch script to NVIDIA Sync as a custom app

These launch scripts adapt the device-specific recipes from [vLLM recipes](https://recipes.vllm.ai/) and the [Nemotron 3 Ultra Station deployment guide](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Ultra/StationDeploymentGuide) to the NVIDIA Sync [custom app feature](https://docs.nvidia.com/sync/latest/applications.html#custom-scripts-and-applications).

**First, go to the pre-written launch script for your device and recipe.**

| Device | Memory | App name | Launch script |
| --- | --- | --- | --- |
| DGX Spark | 64GB | `vLLM-Qwen3.8-27B` | [64GB Spark launch script](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm/assets/sync-vllm-single-spark-64gb.sh) |
| DGX Spark | 128GB | `vLLM-Qwen3.6-35B-A3B` | [128GB Spark launch script](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm/assets/sync-vllm-single-spark-128gb.sh) |
| DGX Station | 748GB | `vLLM-Nemotron-3-Ultra` | [Station launch script](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm/assets/sync-vllm-single-station.sh) |

**Then, create the custom app for the device.**

1. Open **NVIDIA Sync** and select the device from the drop down
2. Select **Custom** > **+ Add New** to open the custom application form
3. Then, enter the **Name** per the table and enter `8000` for **Port** (leave "Auto open" and "URL Path" blank)
4. Next, paste the script into **Launch Script** (leave"Launch in Terminal" blank)
5. Finally, select **Add**

> [!NOTE]
> For either DGX Spark script, if port `8000` is already in use, choose another Sync app port and change the `PORT` variable in the script to match. The script maps that DGX host port to container port `8000`.
> The Station recipe uses Docker host networking, so there is no Docker port mapping; leave the Sync app port and `vllm serve --port 8000` unchanged.
> Its script binds the endpoint to the DGX device's loopback interface, matching the local-only access pattern used by this guide.

**Success**: The application addition completes and a new entry appears in the **Custom** section.

### Step 5. Launch the app and monitor vLLM startup

Getting the vLLM endpoint available can take a while because various runtime files need to be generated and it can take a while for the model weights to load into memory. 

**First, launch the app to run the container and start the vLLM server.**

Just click the app name in NVIDIA Sync to launch the container.

**Next, open the resource monitor to watch the server create the runtime files and load the model weights.**

The server start up will actively use the system resources.

1. Click the resource monitor in NVIDIA Sync
2. When the monitor window opens, select the different resources to see the activity. 

**Then, monitor server state through the `docker` logs in a terminal.**

The logs show startup progress and can help diagnose errors. Use the health check and functional request in Step 6 to verify API readiness.

1. Open a terminal through NVIDIA Sync
2. For the 64GB DGX Spark, run `docker logs --tail 50 --follow vllm-qwen38`
3. For the 128GB DGX Spark, run `docker logs --tail 50 --follow vllm-qwen36-35b-a3b`
4. For DGX Station, run `docker logs --tail 50 --follow nemotron-ultra-vllm`


**Success**: The container is running. Continue to Step 6 to verify the API and test inference.

### Step 6. Verify the server through the locally available endpoint

The vLLM server has a variety of endpoints that you can use to check status.

**First, open a local terminal and run commands from the table below to check on the inference server.**

Replace port `8000` in the commands below if you configured another port for the app.

1. Open a **local** terminal application on your laptop
2. Use the health check command to see if the server is up
3. Get the server's model id for the model being used   

| Laptop terminal | Health check | List served models |
| --- | --- | --- |
| Windows PowerShell | `curl.exe --fail-with-body -i http://localhost:8000/health` | `curl.exe --fail-with-body -sS http://localhost:8000/v1/models` |
| macOS or Linux | `curl --fail-with-body -i http://localhost:8000/health` | `curl --fail-with-body -sS http://localhost:8000/v1/models` |

**Then, use the model ID, `<model-id>` returned by `/v1/models` to test inference with a chat request.**

1. Copy the query command to a file or paste it into a terminal
2. Copy the model ID, `<model-id>` from the `/v1/models` endpoint and put it in the appropriate place in the query command
3. Send the query

| Laptop terminal | Chat request |
| --- | --- |
| Windows PowerShell | `(Invoke-RestMethod -Uri http://localhost:8000/v1/chat/completions -Method Post -ContentType "application/json" -Body '{"model":"<model-id>","messages":[{"role":"user","content":"Write a haiku about a GPU."}],"max_tokens":4096}').choices` |
| macOS or Linux | `curl --fail-with-body -sS http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"<model-id>","messages":[{"role":"user","content":"Write a haiku about a GPU."}],"max_tokens":4096}'` |

**Success:** The response contains a `choices` array with the model's answer.

### Next steps

- Connect a local harness to the endpoint you verified above.
- Continue to [Advanced: Qwen on DGX Spark](advanced-qwen-on-spark.md) for the 128GB Spark or [Advanced: DeepSeek V4.1 Flash on DGX Station](advanced-deepseek-on-station.md) for Station to prepare the endpoint for Hermes or OpenShell.
- Use [Troubleshooting](troubleshooting.md) if the app does not start or the laptop cannot reach the API.
- Use [the vLLM cluster playbook](https://build.nvidia.com/playbooks/vllm-cluster) to serve across multiple DGX Sparks.

## Advanced Qwen on Spark

## Advanced: Serve Qwen on DGX Spark for use by Hermes or Openshell

You will set up an OpenAI-compatible vLLM endpoint for the Hermes or OpenShell playbooks using a 128GB DGX Spark recipe. 

You will download the container and model, run and test the server, then save the container launch as a Bash script.

After that you can follow the [Hermes](https://build.nvidia.com/playbooks/hermes-agent) or [OpenShell](https://build.nvidia.com/playbooks/openshell) playbooks.

These instructions are meant to provide guidance that you can generalize to other recipes.

### Step 1. Select the recipe and save the Docker command

**Go to the Qwen recipe page for the 128GB DGX Spark.**

| Device | Memory | Hugging Face repository | vLLM Recipe |
| --- | --- | --- | --- |
| DGX Spark | 128GB unified memory | `nvidia/Qwen3.6-35B-A3B-NVFP4` | [Qwen3.6-35B-A3B NVFP4 recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B?hardware=dgx_spark_gb10&variant=nvfp4) |

**Then, select Docker and copy the commands to a file for future reference.**

1. Open the `INSTALL` drop down and select `docker`
2. Copy the `docker pull` and `docker run` commands to a file
3. Copy the Hugging Face repository name from the table to the same file

> [!NOTE]
> The [vLLM recipe](https://recipes.vllm.ai/) site is essentially a user-friendly configuration engine that lets select options to narrow down to a model checkpoint and vLLM server/Docker command to serve the model on your device. Each model checkpoint page has further options you can select to modify the commands. 

### Step 2. Get the container and model on the device

If you can't access the device with a keyboard and monitor, SSH into it.

**First, open a terminal and pull the container using the copies `docker pull` command.**

You can follow the layer downloads and extractions in the terminal.

**Then, verify the Hugging Face CLI is installed and authenticated.** 

In the terminal, run

```bash
hf auth whoami
```

If the CLI isn't installed, install it with [the standalone installer](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#standalone-installer-recommended) and authenticate it ([see here]((https://build.nvidia.com/playbooks/openshell))).

**Finally, download the model to the device.**

In the terminal, run the following command and follow progress in the terminal.

```bash
hf download nvidia/Qwen3.6-35B-A3B-NVFP4
```

> [!NOTE]
> Some recipes require additional model files beyond the main checkpoint. For example, speculative-decoding modes may use separate draft models, such as DSpark or DFlash to create drafts for Nemotron 3.5 or Qwen3.8. Follow guidance for individual recipes to download those called for.

### Step 3. Use the copied Docker command to run the vLLM container

Run the complete `docker run` command you saved in the device terminal, then follow vLLM startup and model loading in the terminal.

**Success:** The server output includes:

```text
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

> [!NOTE]
> Startup time, readiness messages, and host ports can vary by recipe. Keep the complete recipe command, wait for that server's ready signal, and use its host port in later endpoint checks.  

### Step 4. Test the vLLM server and tool calling

You can manually test that things are working before moving on to the Hermes or OpenShell playbooks.

**First, check the endpoint health and the model ID endpoints.**

1. In a terminal on the device, run `curl --fail-with-body -i http://localhost:8000/health` to test health
2. Then, verify the model ID with `curl --fail-with-body -sS http://localhost:8000/v1/models`

**Then, use the model ID to send a chat request.**

1. Swap `<model-id>` in the command below for the actual ID returned by `/v1/models`
2. Run the command in the terminal on the device

```bash
curl --fail-with-body -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Reply with a short greeting."}],"max_tokens":64}'
```
**Success**: The response contains a `choices` array with the model's answer.

**Finally, verify tool calling with a request that declares a tool.** 

You can spoof the actual tool to check vLLM's tool-call formatting without installing Hermes or OpenShell. 
This test checks that vLLM returns a structured tool call; it does not execute the tool.

1. Swap `<model-id>` in the command below for the actual ID returned by `/v1/models`
2. Run the command in the terminal on the device

```bash
curl --fail-with-body -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"What is the weather in Boston? Use the available tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location.","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"tool_choice":"required"}'
```

**Success:** The response contains a `tool_calls` entry and `finish_reason` is `tool_calls`.  

> [!NOTE]
> A working chat endpoint does not necessarily mean tool calling is enabled. Some recipes require a tool-call parser or extra request fields, and reasoning models may need a larger `max_tokens` budget. Follow the selected recipe's settings. 

### Step 5. (optional) Save the Docker run command and tests as a Bash script

Once the Docker command works, you may want to put it in a bash script that you can run on
the DGX device without copy and paste every time.  

1. Give an AI agent the tested `docker run` command and ask it to put that command in a Bash script without changing its options. 
2. Save the script on the DGX and make it executable
3. Run the script when you want to start the container again

## Step 6. (conditional) Configure the endpoint for Hermes or OpenShell

If you are continuing with Hermes or OpenShell, follow the matching instructions below.

**Verify the endpoint is set up for Hermes running on the DGX device.** 

In the [Hermes playbook](https://build.nvidia.com/playbooks/hermes-agent), use `http://localhost:8000/v1` as the API base URL and `nvidia/Qwen3.6-35B-A3B-NVFP4` as the model ID.
Hermes runs directly on the DGX, so `localhost` reaches vLLM.

**Verify the endpoint is set up for the gatway container OpenShell runs on the DGX device.**

OpenClaw runs in a sandbox on the DGX, with its own network context.
Use the DGX's network IP (not `localhost`) for the OpenShell provider.

1. On the DGX, find its network IP:

   ```bash
   DGX_IP="$(hostname -I | awk '{print $1}')"
   printf 'DGX IP: %s\n' "$DGX_IP"
   ```

2. While vLLM is running, verify the endpoint at that IP:

   ```bash
   curl --fail-with-body -sS "http://${DGX_IP}:8000/v1/models"
   ```

3. In the [OpenShell playbook](https://build.nvidia.com/playbooks/openshell), set the provider URL to `http://<DGX-IP>:8000/v1`, replacing `<DGX-IP>` with the address printed in Step 1. Set the model ID to `nvidia/Qwen3.6-35B-A3B-NVFP4`.

> [!NOTE]
> For a different client or deployment, use an endpoint address reachable from where that client runs. `localhost` refers to the client's own network context, which may not be the DGX.

### Next steps

- [Do the Hermes Playbook](https://build.nvidia.com/playbooks/hermes-agent)
- [Do the OpenShell Playbook](https://build.nvidia.com/playbooks/openshell)
- [vLLM Qwen3.6 recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B?hardware=dgx_spark_gb10&variant=nvfp4)
- [Troubleshooting](troubleshooting.md)

## Advanced DeepSeek on Station

## Advanced: Serve DeepSeek on DGX Station

You will set up an OpenAI-compatible vLLM endpoint for the Hermes or OpenShell playbooks using a DGX Station recipe.

You will download the container and model, run and test the server, then save the container launch as a Bash script.

After that you can follow the [Hermes](https://build.nvidia.com/playbooks/hermes-agent) or [OpenShell](https://build.nvidia.com/playbooks/openshell) playbooks.

### Step 1. Select the recipe and save the Docker command

**Go to the DeepSeek V4.1 Flash recipe page for DGX Station.**

| Device | Memory | Hugging Face repository | vLLM Recipe |
| --- | --- | --- | --- |
| DGX Station | 748GB coherent memory | `deepseek-ai/DeepSeek-V4.1-Flash` | [DeepSeek V4.1 Flash, DGX Station](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash?hardware=dgx_station_gb300) |

**Then, select Docker, enable tool calling, and copy the commands to a file for future reference.**

1. Open the `INSTALL` drop down and select `docker`
2. Enable the recipe's tool-calling feature, then copy the `docker pull` and `docker run` commands to a file
3. Copy the Hugging Face repository name from the table to the same file

Use the generated command for DGX Station; do not substitute generic vLLM flags. The tool-calling configuration uses the `deepseek_v41` parser.

> [!NOTE]
> The [vLLM recipe](https://recipes.vllm.ai/) site lets you select a model checkpoint and device-specific options, then generates the corresponding server and Docker commands. Options such as tool calling can change the generated launch command.

### Step 2. Get the container and model on the device

If you can't access the device with a keyboard and monitor, SSH into it.

**First, open a terminal and pull the container using the copied `docker pull` command.**

You can follow the layer downloads and extractions in the terminal.

**Then, verify the Hugging Face CLI is installed.**

In the terminal, run:

```bash
hf auth whoami
```

If the CLI isn't installed, install it with [the standalone installer](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#standalone-installer-recommended). The model repository is publicly downloadable; logging in is optional but can help avoid download rate limits.

**Finally, download the model to the device.**

In the terminal, run the following command and follow progress in the terminal. The checkpoint is about 476 GiB; confirm you have enough free storage and allow substantial download time.

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash
```

> [!NOTE]
> Unlike the Intermediate Station recipe, this uses a different model repository and the vLLM nightly image. Expect to download the DeepSeek checkpoint and a new image; Docker may reuse layers shared with images already on the device.

> [!NOTE]
> Some recipes require additional model files beyond the main checkpoint. For example, optional speculative-decoding modes may use a separate draft model. Check the selected recipe to see which downloads its launch command needs, and confirm there is enough storage.

### Step 3. Run the vLLM container

Run the complete `docker run` command you saved in the device terminal, then follow vLLM startup and model loading in the terminal. This recipe uses the vLLM nightly image (vLLM 0.30.0 or later). The first startup with a new image or flag set can spend 75–90 minutes tuning kernels before the model is ready.

**Success:** The server output includes:

```text
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

> [!NOTE]
> Startup time, readiness messages, and host ports can vary by recipe. Keep the complete generated command, wait for that server's ready signal, and use its host port in later endpoint checks. This DeepSeek image starts the server itself, so do not add a second `vllm serve`. It publishes port `8000` for OpenShell to reach using the DGX IP; because the API is unauthenticated, keep the device on a trusted network.

### Step 4. Test the server and tool calling

You can manually test that things are working before moving on to the Hermes or OpenShell playbooks.

**First, check the endpoint health and model ID.**

1. In a terminal on the device, run `curl --fail-with-body -i http://localhost:8000/health` to test health
2. Then, verify the model ID with `curl --fail-with-body -sS http://localhost:8000/v1/models`

**Then, use the model ID to send a chat request.**

Swap `<model-id>` in the command below for the actual ID returned by `/v1/models`, then run it in the terminal on the device. DeepSeek's default reasoning can consume a short output-token budget, so this smoke test turns thinking off for a concise response:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Reply with a short greeting."}],"max_tokens":64,"chat_template_kwargs":{"thinking":false}}'
```

**Success:** The response contains a `choices` array with the model's answer.

**Finally, verify tool calling with a request that declares a tool.**

You can spoof the actual tool to check vLLM's tool-call formatting without installing Hermes or OpenShell. Swap `<model-id>` for the actual ID returned by `/v1/models`, then run the command in the terminal on the device.

```bash
curl --fail-with-body -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"What is the weather in Boston? Use the available tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location.","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"tool_choice":"required","chat_template_kwargs":{"thinking":false}}'
```

**Success:** The response contains a `tool_calls` entry and `finish_reason` is `tool_calls`.

> [!NOTE]
> A successful chat request does not show that tool calling is configured. Use the model's parser and request options in its recipe; reasoning models may also need extra options or a larger token budget. This test checks that vLLM returns a structured tool call; it does not execute the tool.

### Step 5. (optional) Save the Docker run command and tests as a Bash script

Once the Docker command works, you may want to put it in a bash script that you can run on the DGX device without copy and paste every time.

1. Give an AI agent the tested `docker run` command and ask it to put that command in a Bash script without changing its options.
2. Save the script on the DGX and make it executable.
3. Run the script when you want to start the container again.

### Step 6. (conditional) Configure the endpoint for Hermes or OpenShell

If you are continuing with Hermes or OpenShell, follow the matching instructions below.

**Verify the endpoint is set up for Hermes running on the DGX Station.**

In the [Hermes playbook](https://build.nvidia.com/playbooks/hermes-agent), use `http://localhost:8000/v1` as the API base URL and `deepseek-ai/DeepSeek-V4.1-Flash` as the model ID. Hermes runs directly on the DGX, so `localhost` reaches vLLM.

**Verify the endpoint is set up for OpenShell running on the DGX Station.**

OpenClaw runs in a sandbox on the DGX, with its own network context. Use the DGX's network IP (not `localhost`) for the OpenShell provider.

1. On the DGX, find its network IP:

   ```bash
   DGX_IP="$(hostname -I | awk '{print $1}')"
   printf 'DGX IP: %s\n' "$DGX_IP"
   ```

2. While vLLM is running, verify the endpoint at that IP:

   ```bash
   curl --fail-with-body -sS "http://${DGX_IP}:8000/v1/models"
   ```

3. In the [OpenShell playbook](https://build.nvidia.com/playbooks/openshell), set the provider URL to `http://<DGX-IP>:8000/v1`, replacing `<DGX-IP>` with the address printed in Step 1. Set the model ID to `deepseek-ai/DeepSeek-V4.1-Flash`.

> [!NOTE]
> For a different client or deployment, use an endpoint address reachable from where that client runs. `localhost` refers to the client's own network context, which may not be the DGX.

### Next steps

- [Do the OpenShell Playbook](https://build.nvidia.com/playbooks/openshell)
- [Do the Hermes Playbook](https://build.nvidia.com/playbooks/hermes-agent)
- [DeepSeek V4.1 Flash recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash?hardware=dgx_station_gb300)
- [Troubleshooting](troubleshooting.md)

## Troubleshooting

## Common issues

The **Applies to** column identifies the relevant path and hardware. "All paths · all hardware platforms" applies throughout this playbook.

| Symptom | Applies to | Cause | Fix |
|---------|------------|-------|-----|
| "permission denied" when running docker | All paths · all hardware platforms | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Container fails to start with GPU error | All paths · all hardware platforms | NVIDIA Container Toolkit not configured | Run `nvidia-ctk runtime configure --runtime=docker` and restart Docker |
| Hugging Face authentication failure, gated model access denied, or model download hangs/fails | All paths · all hardware platforms | Missing or invalid token, restricted model access, or network issue | Run `hf auth whoami` on the device if using the Hugging Face CLI. Authenticate with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication), request access if the model is gated, and retry the download. |
| CUDA out of memory | All paths · all hardware platforms | Context length too large / model too big | Reduce context length or concurrency settings in the selected recipe, or choose a smaller model or lower-precision checkpoint. |
| Port 8000 is in use | Intermediate · Advanced | Another app is using the DGX host port | Follow the instructions for your path under **Use another API port** below. |
| Laptop cannot reach the API, but `/health` returns HTTP `200` on the DGX device | Intermediate · Advanced | The selected forwarding route does not match the DGX host port, or the forwarding process is not active | Check the path-specific forwarding steps below. On Windows, run PowerShell commands in PowerShell rather than WSL unless you have configured forwarding there. |
| PowerShell prompts for `Uri` when you enter a `curl` command | Intermediate · Advanced | `curl` is a PowerShell alias for `Invoke-WebRequest` | Use `curl.exe` when following a curl command in PowerShell. The playbook's Windows chat-request examples use `Invoke-RestMethod`. |
| Chat request returns `The model 'unknown' does not exist` | Intermediate · Advanced | The request's `model` value does not match a served model | Get the model ID from `/v1/models` and use that exact ID in the chat request. |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |
| Chat completion returns `content: null` with `finish_reason: length` | Intermediate · Advanced | The request's `max_tokens` was exhausted during reasoning | Increase `max_tokens` in the request and retry. |

## Use another API port

Choose the instructions for the path you are using. Keep the laptop port, DGX host port, and container port distinct; the correct change depends on how the path forwards traffic.

### Intermediate: NVIDIA Sync Custom App

If DGX host port 8000 is already in use, choose 8001 for the custom app:

1. Set the NVIDIA Sync custom app port to **8001**.
2. In its launch script, change `PORT=8000` to `PORT=8001`. Leave the container port and `vllm serve --port 8000` unchanged.
3. Restart the custom app. In the Intermediate guide's API-check URLs, use `http://localhost:8001` instead of `http://localhost:8000`.

The script binds the DGX host's `127.0.0.1:8001` to port 8000 inside the container. NVIDIA Sync forwards laptop port 8001 to that DGX host port.

### Advanced: Docker port mapping and SSH tunnel

If the selected recipe maps DGX host port 8000 and it is already in use, change the host-side port in the recipe's `docker run -p` mapping (the value to the left of the colon) to an unused port, such as 8001. Keep the container-side port and `vllm serve --port` unchanged. Use the new DGX host port for the device-local API checks and as `<dgx-host-port>` in the SSH tunnel command. Use the selected `<laptop-port>` in requests sent through the tunnel.

For tunnel-only access, bind the DGX host-side port to `127.0.0.1`; publish it on other network interfaces only with deliberate access controls.

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

> [!NOTE]
> **Monitoring GPU memory with UMA.** Because of unified memory, `nvidia-smi --query-gpu` memory fields report `N/A`. Use plain `nvidia-smi` instead.
