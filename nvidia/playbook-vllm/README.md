# Serve LLMs with vLLM on One Device

> A locally accessible model API from one DGX Spark or Station

## Table of Contents

- [Overview](#overview)
- [Intermediate](#intermediate)
  - [Step 1. Install NVIDIA Sync on your laptop](#step-1-install-nvidia-sync-on-your-laptop)
  - [Step 2. Add your DGX Spark or Station to NVIDIA Sync](#step-2-add-your-dgx-spark-or-station-to-nvidia-sync)
  - [Step 3. Connect, open a terminal and check the tools](#step-3-connect-open-a-terminal-and-check-the-tools)
  - [Step 4. Choose a model and download its files](#step-4-choose-a-model-and-download-its-files)
  - [Step 5. Add the model to NVIDIA Sync](#step-5-add-the-model-to-nvidia-sync)
  - [Step 6. Launch the app and watch the server start](#step-6-launch-the-app-and-watch-the-server-start)
  - [Step 7. Test the API from your laptop](#step-7-test-the-api-from-your-laptop)
  - [Step 8. Stop the model and start it again](#step-8-stop-the-model-and-start-it-again)
  - [Next steps](#next-steps)
- [Advanced](#advanced)
  - [Step 1. Choose a model and recipe](#step-1-choose-a-model-and-recipe)
  - [Step 2. Read the settings that shape the service](#step-2-read-the-settings-that-shape-the-service)
  - [Step 3. Run the recipe](#step-3-run-the-recipe)
  - [Step 4. Test inference and tool calling](#step-4-test-inference-and-tool-calling)
  - [Step 5. (optional) Tune the service for your workload](#step-5-optional-tune-the-service-for-your-workload)
  - [Step 6. (optional) Save a repeatable launch script](#step-6-optional-save-a-repeatable-launch-script)
  - [Step 7. Connect Hermes, OpenShell, or a laptop client](#step-7-connect-hermes-openshell-or-a-laptop-client)
  - [Next steps](#next-steps)
- [Troubleshooting](#troubleshooting)
  - [Intermediate: NVIDIA Sync Custom App](#intermediate-nvidia-sync-custom-app)
  - [Advanced: Host networking, port mapping and SSH tunnel](#advanced-host-networking-port-mapping-and-ssh-tunnel)

---

## Overview

## Basic idea

[vLLM](https://vllm.ai/) is a high-performance inference engine for serving LLMs at datacenter scale.
However, getting the desired performance requires significant background knowledge and configuration experience, especially when setting up on a local system.

This playbook shows you two paths to set up vLLM and configure the endpoint on a remote DGX Spark or DGX Station ([for a Spark cluster, go here](https://build.nvidia.com/playbooks/vllm-cluster)). 


## What you'll accomplish

- **Intermediate:** Manually prepare a recommended model on the remote device and configure the vLLM container as an NVIDIA Sync Custom App.
- **Advanced:** Get under the hood and do everything manually to learn about vLLM and optimize model performance.

## What to know before starting

- How to use [NVIDIA Sync](https://docs.nvidia.com/sync/latest/index.html). **Intermediate**
- How to read Docker and Hugging Face CLI commands. **Intermediate · Advanced**
- How to read and edit a Bash launch script. **Intermediate · Advanced**
- How to choose a model recipe and understanding model size and precision. **Advanced**
- How to configure `docker run` and `vllm serve` arguments. **Advanced**

## Supported hardware platforms

Check the table below to confirm which path this playbook supports for your hardware.

| Hardware platform | OS | Memory | One device |
| :---- | :---- | :---- | :----: |
| **DGX Spark** | DGX OS (Linux) | 64 GB or 128 GB Unified Memory | ✅ |
| **DGX Station** | Ubuntu (Linux) | Large HBM + Grace DRAM | ✅ |

## Prerequisites

**Hardware requirements**

- One DGX Spark or DGX Station. 

**Software and access requirements**

- [NVIDIA Sync installed](https://docs.nvidia.com/sync/latest/getting-started.html#installation-and-onboarding) on your laptop, with the DGX device [connected](https://docs.nvidia.com/sync/latest/direct-connections.html#adding-a-device-for-a-direct-connection). **Intermediate** 
- Docker installed on the DGX device and [available to your user](https://docs.docker.com/engine/install/linux-postinstall/#add-your-user-to-the-docker-group). **Intermediate · Advanced**
- An [installed](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#standalone-installer-recommended) Hugging Face CLI on the remote device. **Intermediate · Advanced**
- SSH access to the remote device. **Advanced**

## Time & risk

- **Estimated time:** Intermediate varies between 15 and 30 minutes depending on network speed. Advanced takes about 30 minutes more to account for manual steps.
- **Risk level:** Low
- **Stop or rollback:** You can stop configuration flows at any time, and you can remove the container or model from the device manually.
- **Last Updated:** 10/19/2026
  - Intermediate offers two models per device and memory size, with matching NVIDIA Sync launch scripts
  - Advanced combines Spark and Station in one recipe workflow, with explanations of memory, scheduling, and model settings
  - Advanced includes API and tool-call checks, manual launch scripts, and client setup for Hermes, OpenShell, and laptop access

## Intermediate

## Set up a vLLM container as an NVIDIA Sync custom app

You will use NVIDIA Sync to connect to a DGX Spark or Station, then download a vLLM container and a model for your device.

Then, you will add a supplied launch script as a custom app in NVIDIA Sync.
Once the app is set up, you can use Sync to start and stop the model and reach its API from your laptop.

Sync sets up the SSH tunnel and port forwarding for you.

### Step 1. Install NVIDIA Sync on your laptop

**First, download NVIDIA Sync for your laptop.**

NVIDIA Sync is a desktop app that lets you connect to a remote Linux device and run apps on it.

::spark-download

**Then, install and open NVIDIA Sync.**

**For Windows:** Double-click the `.exe` installer and follow the steps.

**For macOS:** Open `nvidia-sync.dmg`, drag the app into Applications, then open it.

**For Ubuntu:** Follow the [Ubuntu install steps](https://docs.nvidia.com/sync/latest/getting-started.html#installation).

**Success:** NVIDIA Sync opens on your laptop.

### Step 2. Add your DGX Spark or Station to NVIDIA Sync

**First, choose how to add your device.**

Your laptop must be able to reach the device on the network.
You can add it in one of three ways:

1. **mDNS:** Select a device that Sync finds on the network, then enter your username and password
2. **Manual:** Select **Add a device manually**, enter the device's IP address, then enter your username and password
3. **Import:** Select a working SSH alias from your local SSH config, then add it

For mDNS or Manual, have these details ready:

- **Name:** a name to help you find the device in Sync
- **Hostname or IP:** the device's network name or IP address
- **Username:** your username on the device
- **Password:** the password for that user

> [!NOTE]
> If you use an SSH key to sign in, first set up an SSH alias, then import it into Sync. See [Add a device for a direct connection](https://docs.nvidia.com/sync/latest/direct-connections.html#adding-a-device-for-a-direct-connection).

**Then, allow NVIDIA Sync to set up the Docker group if prompted.**

If Sync asks to add your user to the `docker` group, click **Confirm**.
This lets the launch script run Docker as your user.
You will need `sudo` access for this step.

> [!NOTE]
> Access to the `docker` group grants root access without a password, which is a risk. Learn more in the [Docker user guide](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

**Success:** Your device appears in NVIDIA Sync.

### Step 3. Connect, open a terminal and check the tools

**First, connect to the device and open a terminal.**

1. Select your device from the NVIDIA Sync drop-down
2. Click **Connect**
3. Select the **Terminal** app

The terminal opens an SSH session on the DGX device.
You will run the download commands in this terminal.

**Then, check that the Hugging Face CLI is installed.**

Run:

```bash
hf --help
```

If it returns `command not found`, install the CLI with the [Hugging Face install steps](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#standalone-installer-recommended), then open a new terminal.

**Success:** `hf --help` shows the CLI help.

### Step 4. Choose a model and download its files

**First, choose one of the two rows for your device and memory size.**

Each row has a matching launch script in Step 5.
The recipe links give you more detail about the model settings used by these scripts.

| Device | Memory | Model | Docker pull command | Model download command |
| --- | --- | --- | --- | --- |
| DGX Spark | 64GB unified memory | [Qwen3.8 27B (NVFP4)](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark/index.html?model=qwen%2Fqwen3.8-27b&os=linux&backend=vllm&memory=64&precision=nvfp4&variant=&memoryConfig=64) | `docker pull vllm/vllm-openai:v0.28.0` | `hf download nvidia/Qwen3.8-27B-NVFP4` |
| DGX Spark | 64GB unified memory | [Nemotron 3.5 Bolt 74B A9B](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark/index.html?model=nvidia%2Fnemotron-3.5-bolt-74b-a9b&os=linux&backend=vllm&memory=64&precision=iq2&variant=&memoryConfig=64) | `docker pull eugr/spark-vllm:nightly-20261016` | `hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` |
| DGX Spark | 128GB unified memory | [Qwen3.8 Flash-Next](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark/index.html?model=qwen%2Fqwen3.8-flash-next&os=linux&backend=vllm&memory=128&precision=fp4-local&variant=&memoryConfig=128) | `docker pull eugr/spark-vllm-b12x:nightly-20261015` | `hf download nvidia/Qwen3.8-Flash-Next-FP4-Local` |
| DGX Spark | 128GB unified memory | [Qwen3.8 27B (NVFP4)](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark/index.html?model=qwen%2Fqwen3.8-27b&os=linux&backend=vllm&memory=128&precision=nvfp4&variant=&memoryConfig=128) | `docker pull vllm/vllm-openai:v0.28.0` | `hf download nvidia/Qwen3.8-27B-NVFP4` |
| DGX Station | 748GB coherent memory | [Qwen3.8 Flash-Next](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-station/index.html?model=qwen%2Fqwen3.8-flash-next&os=linux&backend=vllm&memory=&precision=nvfp4&variant=&memoryConfig=748) | `docker pull vllm/vllm-openai:qwen38-flash-next` | `hf download Inferact/Qwen3.8-Flash-Next-NVFP4` |
| DGX Station | 748GB coherent memory | [Nemotron 3 Super 120B A12B](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-station/index.html?model=nvidia%2Fnemotron-3-super-120b-a12b&os=linux&backend=vllm&memory=&precision=nvfp4&variant=mtp&memoryConfig=748) | `docker pull vllm/vllm-openai:v0.31.0` | `hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` |

**Next, download the container and model on the DGX device.**

1. Run the `docker pull` command from your row and wait for it to finish
2. Run the `hf download` command from the same row and wait for it to finish

**If you selected Qwen3.8 27B or Nemotron 3.5 Bolt, download the draft model needed for speculative decoding.**

The Qwen3.8 27B and Nemotron 3.5 Bolt scripts use a second, smaller model to help speed up replies through something called "[speculative decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)".

Run the command for your model below.

| Model | Draft model download command |
| --- | --- |
| Qwen3.8 27B, on either Spark memory size | `hf download z-lab/Qwen3.8-27B-DFlash2` |
| Nemotron 3.5 Bolt 74B A9B | `hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` |

> [!NOTE]
> The container and model files are large. Check that the device has enough free disk space and allow time for all downloads to finish.

**Success:** The container, model and any draft model have finished downloading.

### Step 5. Add the model to NVIDIA Sync

**First, open the launch script for the row you chose in Step 4.**

Each script starts the container and runs the model with the recipe's settings. It also stops the container when you stop the app in Sync.

| Device | Memory | App name | Launch script |
| --- | --- | --- | --- |
| DGX Spark | 64GB | `vLLM-Qwen3.8-27B-64GB` | [Qwen3.8 27B on 64GB Spark](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm/assets/sync-vllm-spark-64gb-qwen38-27b.sh) |
| DGX Spark | 64GB | `vLLM-Nemotron3.5-Bolt` | [Nemotron 3.5 Bolt on 64GB Spark](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm/assets/sync-vllm-spark-64gb-nemotron35-bolt.sh) |
| DGX Spark | 128GB | `vLLM-Qwen3.8-Flash-Next` | [Qwen3.8 Flash-Next on 128GB Spark](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm/assets/sync-vllm-spark-128gb-qwen38-flash-next.sh) |
| DGX Spark | 128GB | `vLLM-Qwen3.8-27B-128GB` | [Qwen3.8 27B on 128GB Spark](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm/assets/sync-vllm-spark-128gb-qwen38-27b.sh) |
| DGX Station | 748GB | `vLLM-Qwen3.8-Flash-Next` | [Qwen3.8 Flash-Next on Station](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm/assets/sync-vllm-station-qwen38-flash-next.sh) |
| DGX Station | 748GB | `vLLM-Nemotron3-Super` | [Nemotron 3 Super on Station](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm/assets/sync-vllm-station-nemotron3-super.sh) |

**Then, add the script as a custom app.**

1. In NVIDIA Sync, connect to your device
2. Select **Custom** > **+ Add New**
3. Enter the **App name** from your row in **Name**, and enter `8000` in **Port**
4. Paste the full script into **Launch Script**
5. Leave **Auto open** and **Launch in Terminal** off, and leave **URL Path** blank
6. Select **Add**

Learn more about [Sync custom apps](https://docs.nvidia.com/sync/latest/applications.html#custom-scripts-and-applications).

> [!NOTE]
> Start with one model app at a time. If port `8000` is in use, choose a free port, such as `8001`. Set both the Sync app's **Port** and `PORT=8000` near the top of its launch script to that number. Use the same port in the laptop commands in Step 7.

**Success:** The app appears in the **Custom** section for your device.

### Step 6. Launch the app and watch the server start

**First, click the app name in NVIDIA Sync.**

The script starts the container and loads the model. Sync forwards the model's API port to your laptop.

**Next, open the resource monitor to watch the device.**

1. Click the resource monitor in NVIDIA Sync
2. Select the views to see CPU, GPU and memory use as the server starts

The first start can take a while as vLLM builds runtime files and loads the model. Device activity shows that work is in progress; use Step 7 to check that the API is ready.

**Then, check the container logs in a terminal on the DGX device.**

Open **Terminal** through Sync and run the command for your app.

| App name | Log command |
| --- | --- |
| `vLLM-Qwen3.8-27B-64GB` | `docker logs --tail 50 --follow vllm-spark-64gb-qwen38-27b` |
| `vLLM-Nemotron3.5-Bolt` | `docker logs --tail 50 --follow vllm-spark-64gb-nemotron35-bolt` |
| `vLLM-Qwen3.8-Flash-Next` (128GB DGX Spark) | `docker logs --tail 50 --follow vllm-spark-128gb-qwen38-flash-next` |
| `vLLM-Qwen3.8-27B-128GB` | `docker logs --tail 50 --follow vllm-spark-128gb-qwen38-27b` |
| `vLLM-Qwen3.8-Flash-Next` (DGX Station) | `docker logs --tail 50 --follow vllm-station-qwen38-flash-next` |
| `vLLM-Nemotron3-Super` | `docker logs --tail 50 --follow vllm-station-nemotron3-super` |

Press `Ctrl+C` to leave the log view. The model app keeps running.

**Success:** The logs show the server is ready to accept requests. Continue to Step 7 to check the API and send a request.

### Step 7. Test the API from your laptop

**First, open a local terminal on your laptop and check the server.**

Keep the model app running and Sync connected.
Use the commands for your laptop below.
If you chose a port other than `8000`, use that port in each URL.

1. Run the health check and look for HTTP `200`
2. Run the model-list command and copy the model ID from the reply

| Laptop terminal | Health check | List served models |
| --- | --- | --- |
| Windows PowerShell | `curl.exe --fail-with-body -i http://localhost:8000/health` | `curl.exe --fail-with-body -sS http://localhost:8000/v1/models` |
| macOS or Linux | `curl --fail-with-body -i http://localhost:8000/health` | `curl --fail-with-body -sS http://localhost:8000/v1/models` |

If the health check cannot connect while the model loads, wait and try again. Check the container logs if startup fails.

**Then, use the model ID to send a chat request.**

1. Copy the chat command for your laptop
2. Replace `<model-id>` with the exact ID returned by `/v1/models`
3. Run the command and wait for the reply

| Laptop terminal | Chat request |
| --- | --- |
| Windows PowerShell | `(Invoke-RestMethod -Uri http://localhost:8000/v1/chat/completions -Method Post -ContentType "application/json" -Body '{"model":"<model-id>","messages":[{"role":"user","content":"Write a haiku about a GPU."}],"max_tokens":4096}').choices` |
| macOS or Linux | `curl --fail-with-body -sS http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"<model-id>","messages":[{"role":"user","content":"Write a haiku about a GPU."}],"max_tokens":4096}'` |

**Success:** The response has a `choices` array with the model's answer. Your laptop can now reach the model through Sync.

### Step 8. Stop the model and start it again

**First, stop the app through NVIDIA Sync.**

1. Open the NVIDIA Sync pop-up window
2. Click the **x** beside the running custom app

The supplied script stops and removes its container. The model files and cached runtime files stay on the DGX device for the next start.

**Then, start the model when you need it again.**

1. Connect to the device through Sync
2. Click the custom app name
3. Wait for the model to load, then run the health check from Step 7

**Success:** The health check returns HTTP `200` again. You can start and stop this model through Sync each time you need it.

### Next steps

- Connect an app on your laptop to `http://localhost:8000/v1`, using the model ID from Step 7. Use your chosen port if you changed it.
- Continue to the Advanced tab to learn how to choose and run a recipe yourself.
- Use the Troubleshooting Tab if the app fails to start or your laptop cannot reach the API.
- Use [the vLLM cluster playbook](https://build.nvidia.com/playbooks/vllm-cluster) to serve across more than one DGX Spark.

## Advanced

## Choose and run a vLLM recipe on DGX Spark or Station

You will choose a recipe for your device, run its commands, and test the model's API and tool calling.

Along the way, you will see how the recipe uses memory, batches requests, and selects the code that runs the model.

Once the endpoint works, you can tune it for your workload, save the launch command, and use it with a client such as Hermes or OpenShell.

### Step 1. Choose a model and recipe

**First, open the recipe page for your device and memory size.**

Use [NVIDIA Local AI Inference Recipes](https://nvidia.github.io/local-ai-inference-recipes/) to select a model, checkpoint precision, and vLLM configuration.

| Device | Memory | Recipes |
| --- | --- | --- |
| DGX Spark | 64GB unified memory | [64GB Spark recipes](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark/index.html?memoryConfig=64) |
| DGX Spark | 128GB unified memory | [128GB Spark recipes](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark/index.html?memoryConfig=128) |
| DGX Station | 748GB coherent memory | [Station recipes](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-station/index.html?memoryConfig=748) |

**Then, select a model and save the full recipe commands.**

1. Set the hardware and memory size to match your device
2. Select the model, **Linux**, and **vLLM**
3. Choose the checkpoint precision and any recipe variant
4. Read the prerequisites and limits, then save the setup and server commands and the recipe URL

The checkpoint, image, environment settings, and server flags work together.
For example, a low-precision checkpoint needs a backend that can load and run its format, and a speculative-decoding variant may need a second checkpoint.
Start with the full configuration from one recipe.

> [!NOTE]
> [DeepSeek V4.1 Flash on Station](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-station/index.html?model=deepseek%2Fdeepseek-v4.1-flash&os=linux&backend=vllm&precision=fp4-fp8-mixed&memoryConfig=748) has extra host-memory, swap, and disk requirements. Its setup command includes a pinned model download and file-cache preparation. Keep those steps with the launch command.

**Success:** You have saved a complete recipe for your device and chosen model.

### Step 2. Read the settings that shape the service

**First, check how the recipe divides memory between weights and requests.**

The model weights take up part of the device's memory.
Serving also needs space for runtime work and the KV cache, which holds attention keys and values from tokens already processed.
Longer requests and more requests in flight compete for that space.

| Setting | Why it is in the recipe |
| --- | --- |
| Checkpoint precision and `--quantization` | Set how model weights are stored and run. Lower precision can reduce weight memory use; the checkpoint and backend must support it. |
| `--gpu-memory-utilization` | Sets the fraction of GPU memory vLLM budgets for its executor. A higher value can leave more room for KV cache, with less headroom for other use. |
| `--max-model-len` | Caps the total input and output tokens in a request. A large limit needs enough cache to serve long requests. |
| `--max-num-seqs` | Caps the number of sequences processed in one iteration. More active sequences can raise throughput and memory use. |
| `--kv-cache-dtype` | Sets KV-cache precision separately from weight precision. A supported lower-precision cache can hold more tokens in the same memory. |
| `--cpu-offload-gb` | Places some model weights in host memory. This can make a large model fit, with performance tied to host memory and the CPU–GPU link. |

See the [vLLM serve options](https://docs.vllm.ai/en/latest/cli/serve/) and [quantized KV-cache guide](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) for details.

**Next, check how the recipe schedules work and reuses it.**

vLLM first processes the input prompt, called *prefill*, then generates output tokens, called *decode*.
The best settings depend on prompt length, output length, and the number of clients.

| Setting | Why it is in the recipe |
| --- | --- |
| `--max-num-batched-tokens` | Sets the token budget for one scheduler iteration. With chunked prefill, smaller values can help active replies keep moving; larger values can process more prompt tokens at once. |
| `--enable-chunked-prefill` | Splits long prompt processing into chunks that can share batches with decode work. |
| `--enable-prefix-caching` | Reuses cached work for shared prompt prefixes, such as repeated system prompts or document text. It saves prompt work when the prefix matches. |
| `--speculative-config` | Uses a draft model or native prediction heads to propose tokens for the main model to check. The gain depends on the model and workload. Preserve any draft-checkpoint settings. |

Read more about [scheduling and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/), [prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/), and [speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/).

**Then, keep the recipe's model-specific runtime settings together.**

Load formats, attention and MoE backends, compilation settings, and environment variables select code paths supported by that image and model.
Keep them for the first run so you have a working baseline.

For agent use, also keep the model's tool-call and reasoning parsers.
`--enable-auto-tool-choice` allows the model to choose a tool, while `--tool-call-parser` reads its tool-call format.
`--reasoning-parser` separates reasoning from the final answer; the model's chat template and request settings control whether it thinks.
See [tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/) and [reasoning outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/).

**Success:** You know which settings control capacity, scheduling, and the response format your client will receive.

### Step 3. Run the recipe

**First, check the setup command in a terminal on the DGX device.**

Use a local terminal on the device or an SSH session.
Set any host variables called for by the recipe, such as `HF_TOKEN`.

**For Station, check that the command selects the GB300.**

If the recipe uses `--gpus all` and your Station has more than one GPU, get the GB300's ID on the host:

```bash
GB300_UUID=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | \
  awk -F, 'tolower($2) ~ /gb300/ {gsub(/[[:space:]]/, "", $1); print $1; exit}')
: "${GB300_UUID:?No GB300 found}"
```

Before running the container, replace `--gpus all` with `--gpus "device=${GB300_UUID}"`.
Keep any GPU selection already supplied by the recipe.

**Then, run the full setup command on the DGX host.**

Docker pulls the image if it is missing.
The setup or server command downloads model files as needed; keep any explicit download, revision, and local-path options in the recipe.

The image supplies the vLLM build and libraries used by the configuration.
The Hugging Face cache mount keeps model files on the host.
Mounts for the vLLM, FlashInfer, and Triton caches also let later containers reuse compiled work.
First startup can include downloads, compilation, and kernel tuning.

**Next, run the server command inside the container.**

A setup command that ends with `--entrypoint /bin/bash` and an image name opens a shell in that image.
In that shell:

1. Apply any listed environment settings
2. Run the complete `vllm serve` command
3. Leave the terminal open and follow startup output

If the recipe already launches the server in its Docker command, follow that form directly.

**Finally, note the API address and port.**

With `--network host`, vLLM listens directly on the DGX host's port.
With a Docker `-p` mapping, use the published host port for checks outside the container.
With host networking, a loopback bind is enough for host-local clients and SSH forwarding.
With port mapping, keep vLLM bound to `0.0.0.0` inside the container and choose the host-side bind in the mapping.
The OpenShell handoff in Step 7 needs the service reachable from its sandbox.

The commands below use DGX host port `8000`.
Use your recipe's host port if it differs.

**Success:** The server reports that it is ready to accept requests. Confirm readiness with the API checks in Step 4.

### Step 4. Test inference and tool calling

**First, open a second terminal on the DGX device and check the API.**

Run:

```bash
curl --fail-with-body -i http://localhost:8000/health
curl --fail-with-body -sS http://localhost:8000/v1/models
```

The health check should return HTTP `200`, and the model list should have a non-empty `data` array.
Copy the served model's `id` for the requests below.
It can differ from the checkpoint name when the recipe sets `--served-model-name`.

**Then, send a chat request with that model ID.**

Replace `<model-id>` and run:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Reply with a short greeting."}],"max_tokens":4096}'
```

**Success:** `choices[0].message.content` contains an answer.
Reasoning can use the output-token budget before an answer appears; follow the recipe's request settings or raise the budget if the response ends with `finish_reason: length`.

> [!NOTE]
> For DeepSeek V4.1 Flash, add `"chat_template_kwargs":{"thinking":false}` to the JSON body of both tests here for a check with reasoning off. Reasoning controls differ by model; use the selected model's settings.

**Finally, test a tool call.**

Replace `<model-id>` and run:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Use get_weather to look up the weather in Boston."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location.","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"tool_choice":"auto","max_tokens":4096}'
```

**Success:** The response contains a `tool_calls` entry for `get_weather` with Boston in its arguments, and `finish_reason` is `tool_calls`.
This checks the model's automatic tool-call path and response format; the request declares the tool, and your client would execute it.
If the model replies in plain text, check the recipe's tool-call settings and the chosen model's tool support.

### Step 5. (optional) Tune the service for your workload

**First, record a baseline with representative requests.**

Use the prompt lengths, output lengths, and client count you expect to serve.
Record time to first token, output tokens per second, and memory use.
For multiple clients, distinguish each client's generation rate from total throughput.

**Then, change one setting and repeat the same workload.**

1. Set `--max-model-len` to cover your actual input and output needs
2. Try `--max-num-seqs` and `--max-num-batched-tokens` to balance concurrency and latency
3. Compare repeated-prefix workloads with prefix caching, or test the recipe's speculative-decoding variant
4. Keep the settings that improve your target measure, then rerun the chat and tool tests

Memory capacity, request length, and concurrency form a shared budget.
A context limit of 262144 tokens and `--max-num-seqs 8` do not guarantee that eight full-length requests fit at once.
Read the KV-cache capacity and concurrency estimates in startup output, then test the load you intend to run.

**Success:** You have measured settings that suit your workload and still pass the functional checks.

### Step 6. (optional) Save a repeatable launch script

**First, save the working host setup and server commands.**

Keep the image tag, checkpoint revision or path, GPU selection, cache mounts, environment settings, and tested vLLM options together.

**Then, combine the commands if you want one launch script.**

For a recipe that opens `/bin/bash` before you run `vllm serve`:

1. Move the server's environment settings into Docker `-e NAME=value` options
2. Replace `--entrypoint /bin/bash` with `--entrypoint vllm`
3. After the image name, append `serve`, the checkpoint or path, and the tested server arguments
4. Keep any host preparation before `docker run`, including recipe-specific downloads or cache setup
5. Add a container name, such as `--name vllm-local`, and save the result as `launch-vllm.sh`

Make it executable and run it on the DGX device:

```bash
chmod +x launch-vllm.sh
./launch-vllm.sh
```

For a background service, replace the interactive `-it` options with `-d`.
Follow its output with `docker logs --tail 50 --follow vllm-local` and stop it with `docker stop vllm-local`.
If the recipe uses `--rm`, run the saved script again to create the next container.

**Success:** The saved script starts the same service and passes the checks in Step 4.

### Step 7. Connect Hermes, OpenShell, or a laptop client

**First, use an API address reachable from your client.**

Keep the server running and use the exact model ID returned by `/v1/models`.

| Client | API base URL | Setup |
| --- | --- | --- |
| Hermes or another process on the DGX host | `http://localhost:8000/v1` | Follow the client setup with the served model ID. |
| A client on your laptop | `http://localhost:8000/v1` | Forward a local port to the DGX host port with SSH, as shown below. |
| The OpenShell sandbox in its playbook | `http://host.openshell.internal:8000/v1` | Keep vLLM reachable from the sandbox; the playbook uses `--host 0.0.0.0` and host port `8000`. |

The recipe's `0.0.0.0` bind can expose an unauthenticated API on the device's network interfaces.
Use a trusted network or suitable access controls for that route.

**For a laptop client, open an SSH tunnel from the laptop.**

Replace `user@dgx-host` with your SSH login and device address, then run:

```bash
ssh -N -L 8000:127.0.0.1:8000 user@dgx-host
```

Keep the tunnel open, then run the health and model-list checks from a second laptop terminal.
Use `curl.exe` for those checks in Windows PowerShell.
If laptop port `8000` is busy, change the leftmost port to `8001` and use `http://localhost:8001/v1` in the client.

**For Hermes or OpenShell, continue with the matching playbook.**

- In the [Hermes playbook](https://build.nvidia.com/playbooks/hermes-agent), use the host-local API URL and the served model ID. For DeepSeek V4.1 Flash, follow that playbook's model-specific reasoning settings.
- In the [OpenShell playbook](https://build.nvidia.com/playbooks/openshell), reuse this endpoint and continue at Step 6 after exporting `MODEL_HANDLE` as in its Step 5. Use the ID from `/v1/models`; the recipe may give the service an alias.

**Success:** Your client reaches the existing vLLM service and uses the model you tested.

### Next steps

- Use the Troubleshooting Tab for startup, memory, or connection errors.
- Review the [vLLM tuning guide](https://docs.vllm.ai/en/latest/configuration/optimization/) for further performance work.
- Use [the vLLM cluster playbook](https://build.nvidia.com/playbooks/vllm-cluster) for more than one DGX Spark.

## Troubleshooting

## Common issues

The **Applies to** column identifies the relevant path and hardware. "All paths · all hardware platforms" applies throughout this playbook.

| Symptom | Applies to | Cause | Fix |
|---------|------------|-------|-----|
| "permission denied" when running docker | Advanced · all hardware platforms | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Container fails to start with GPU error | All paths · all hardware platforms | NVIDIA Container Toolkit not configured | Run `nvidia-ctk runtime configure --runtime=docker` and restart Docker |
| CUDA out of memory | All paths · all hardware platforms | Other models are still running, or the model settings need more memory than is free | For Intermediate, stop other model apps in Sync and check that the launch script matches your device and memory size in Steps 4 and 5. For Advanced, reduce context length or concurrency in the selected recipe, or choose a smaller model or lower-precision checkpoint. |
| Model or draft-model files are missing | Intermediate · all hardware platforms | A download did not finish, the wrong launch script was chosen, or the files were downloaded under another user or cache path | Finish the downloads in Step 4, including the draft model for Qwen3.8 27B or Nemotron 3.5 Bolt. Use the matching script from Step 5 and the same DGX user for downloads and Sync. If you use `HF_HOME`, check that the script uses the same path as the downloads. |
| `Could not find a GB300 GPU with nvidia-smi.` | Intermediate · DGX Station | The script cannot find the Station's GB300 | Check that Sync is connected to the right Station. Run `nvidia-smi` on that device and confirm it lists a GB300 before trying again. |
| Port 8000 is in use | Intermediate · Advanced | Another app is using the port on the laptop or DGX device | Stop the other app, or follow the instructions for your path under **Use another API port** below. |
| Laptop cannot reach the API, but `/health` returns HTTP `200` on the DGX device | Intermediate · Advanced | Port forwarding is not active or uses a different port | For Intermediate, keep Sync connected and the custom app running. Check that the Sync app's **Port**, the script's `PORT`, and the laptop URL use the same port. Run the laptop checks in a local terminal as shown in Step 7. For Advanced, check the forwarding route for your client. On Windows, run PowerShell commands in PowerShell rather than WSL unless you have configured forwarding there. |
| PowerShell prompts for `Uri` when you enter a `curl` command | Intermediate · Advanced | `curl` is a PowerShell alias for `Invoke-WebRequest` | Use `curl.exe` when following a curl command in PowerShell. The playbook's Windows chat-request examples use `Invoke-RestMethod`. |
| Chat request returns `The model 'unknown' does not exist` | Intermediate · Advanced | The request's `model` value does not match a served model | Get the model ID from `/v1/models` and use that exact ID in the chat request. |
| `torch.OutOfMemoryError` on a GPU with about 95 GiB while loading DeepSeek | Advanced (DGX Station) | The recipe's `--gpus all` let vLLM pick the RTX PRO 6000 instead of the GB300 | Pin the GB300 as in the Advanced tab, Step 3: replace `--gpus all` with `--gpus "device=${GB300_UUID}"`. |
| `engine input registration timed out after 3600s` during FlashInfer autotune | Advanced (DGX Station) | First-run kernel tuning takes longer than the engine-ready timeout | Raise `VLLM_ENGINE_READY_TIMEOUT_S` (for example `7200`) and mount `~/.cache/vllm` and `~/.cache/flashinfer` so tuning results persist. |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |
| Chat completion returns `content: null` with `finish_reason: length` | Intermediate · Advanced | The request's `max_tokens` was exhausted during reasoning | Increase `max_tokens` in the request and retry. |

## Use another API port

Choose the instructions for the path you are using. Keep the laptop port, DGX host port, and container port distinct; the correct change depends on how the path forwards traffic.

### Intermediate: NVIDIA Sync Custom App

If port 8000 is already in use on your laptop or DGX device, choose a free port, such as 8001:

1. Stop the custom app in Sync by clicking the **x** beside it.
2. Edit the custom app and set **Port** to **8001**.
3. In **Launch Script**, change `PORT=8000` to `PORT=8001`.
4. Save the change and start the app again.
5. In the Intermediate guide's Step 7 URLs, use `http://localhost:8001` instead of `http://localhost:8000`.

All supplied Intermediate scripts use Docker host networking. The `PORT` value sets vLLM's port on the DGX device, and Sync forwards the same port to your laptop. With `PORT=8001`, vLLM listens at `127.0.0.1:8001` on the DGX device.

### Advanced: Host networking, port mapping and SSH tunnel

For a recipe with `--network host`, change the server's `--port 8000` to an unused port, such as `--port 8001`. Use that port for the DGX-local API checks and as the remote port in the SSH tunnel. If you use OpenShell, also update the port in its provider, network policy, and client settings.

If the selected recipe maps DGX host port 8000 and it is already in use, change the host-side port in the recipe's `docker run -p` mapping (the value to the left of the colon) to an unused port, such as 8001. Keep the container-side port and `vllm serve --port` unchanged. Use the new DGX host port for the device-local API checks and as `<dgx-host-port>` in the SSH tunnel command. Use the selected `<laptop-port>` in requests sent through the tunnel.

For tunnel-only access, bind the DGX host-side port to `127.0.0.1`; publish it on other network interfaces only with deliberate access controls.

In the Advanced tab's Step 7 tunnel command, the leftmost port is the laptop port and the rightmost port is the DGX host port. If only the laptop port is busy, change the leftmost port and use it in the laptop client's API URL.

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

> [!NOTE]
> **Monitoring GPU memory with UMA.** Because of unified memory, `nvidia-smi --query-gpu` memory fields report `N/A`. Use plain `nvidia-smi` instead.
