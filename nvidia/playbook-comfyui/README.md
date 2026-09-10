# Generate Images and Videos with ComfyUI

> Node-based diffusion workflows for images and videos with FLUX, Wan, HunyuanVideo, and Stable Diffusion


## Table of Contents

- [Overview](#overview)
- [Image Gen Quick Start](#image-gen-quick-start)
- [Video Gen Workflow](#video-gen-workflow)
  - [Text-to-video with Wan 2.1 (Tier 1)](#text-to-video-with-wan-21-tier-1)
  - [Intermediate workflows (Tier 2)](#intermediate-workflows-tier-2)
  - [Advanced workflows (Tier 3)](#advanced-workflows-tier-3)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

ComfyUI is an open-source, node-based web application for AI image and video generation with diffusion models. Instead of a single text box, you connect processing nodes — model loaders, text encoders, samplers, decoders — into a graph that gives full control over every generation step.

- **Node-based workflows** let you build, modify, and share generation pipelines visually. Workflows save as JSON for versioning, collaboration, and reproducibility.
- **Multi-model support** covers Stable Diffusion, FLUX for images, Wan 2.1 and HunyuanVideo for video, and NVIDIA Cosmos for world generation.
- **GPU-accelerated inference** runs on your hardware platform so generation stays local.

## What you'll accomplish

Install and run ComfyUI on your hardware platform, then generate images (and optionally video) from a browser UI on port 8188.

Start with the **Image Gen Quick Start** tab for a lighter host Python install with Stable Diffusion 1.5. Use the **Video Gen Workflow** tab for container-based FLUX / Wan / HunyuanVideo / Cosmos / ControlNet workflows.

## What to know before starting

**Required:**

- Basic command-line and terminal usage
- Familiarity with Python virtual environments *or* Docker containers (depending on install path)

**Optional:**

- Familiarity with generative AI concepts (prompts, diffusion models, checkpoints)
- Hugging Face account for gated model downloads (container / large-model path)

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, OS, memory, and whether multi-node applies.

| Hardware platform | OS | Memory  | Multi-node capable hardware |
| :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | — |
| **DGX Station** | DGX OS (Linux) | Large HBM + Grace DRAM | — |


## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Image Gen Quick Start (host Python): at least ~20 GB free disk; ~8 GB GPU memory for Stable Diffusion–class models
- Video Gen Workflow (container + tiered models): at least ~70 GB free disk for Tier 1 (~230 GB for all tiers); peak GPU memory about **~80 GB / ~100 GB / ~120 GB** for Tiers 1–3 (see that tab)

**Software requirements**

- Image Gen Quick Start: Python 3.8+, pip, Git, CUDA toolkit compatible with your GPU, network access to Hugging Face
- Video Gen Workflow: Docker, NVIDIA Container Toolkit, Hugging Face access token, network access to NGC / Hugging Face / GitHub
- Web browser access to port `8188` on the hardware platform

## Ancillary files

Playbook assets (Video Gen Workflow) live under `assets/` in this repository:

- `assets/Dockerfile` — Builds the ComfyUI container image from an NGC PyTorch base
- `assets/scripts/download-models.sh` — Downloads model weights from Hugging Face (`hf` CLI)
- `assets/workflows/*.json` — UI workflows for **Load** in the web UI
- `assets/workflow_api/*.api.json` — Same graphs in API format for `/prompt` and automation
- `assets/scripts/api_to_ui_workflow.py` — Regenerates UI JSON from API JSON if you edit a graph programmatically

For Image Gen Quick Start, clone [ComfyUI on GitHub](https://github.com/comfyanonymous/ComfyUI) directly (`requirements.txt`, `main.py`, and checkpoint directories).

## Time & risk

- **Estimated time:** 45 MIN (longer on first run when downloading large models)
- **Risk level:** Medium
  - Model downloads are large and may fail due to network or auth issues
  - Port 8188 must be reachable for the web UI
- **Rollback:** Remove the virtual environment and clone (Image Gen Quick Start), or stop/remove the container and optionally delete `models/` (Video Gen Workflow) — non-destructive to the host OS
- **Last Updated:** 07/27/2026
  - Image Gen Quick Start and Video Gen Workflow tabs; start with image gen, then scale to tiered video models

## Image Gen Quick Start

> [!NOTE]
> These instructions target **Linux**. This tab is a lightweight host Python install for Stable Diffusion–class image generation — a good first try on any supported hardware platform. For FLUX, Wan, HunyuanVideo, Cosmos, and playbook workflows, continue to the **Video Gen Workflow** tab.

## Quick start (optional)

If you prefer an automated setup, download and run the provided script to perform Steps 1–6 in one go (prerequisite check, virtual environment, PyTorch, ComfyUI, dependencies, and model download):

```bash
curl -fsSL "https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-comfyui/assets/setup.sh" | bash
```

When it finishes, launch the server from the **same directory** where you ran `setup.sh` (it expects `comfyui-env/` and `ComfyUI/` in the current directory):

```bash
curl -fsSL "https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-comfyui/assets/launch.sh" | bash
```

Then continue from [Step 8. Validate installation](#step-8-validate-installation).

To learn what each step does, follow the manual instructions below instead.

## Step 1. Verify system prerequisites

Check that your hardware platform meets the requirements before proceeding with installation.

```bash
python3 --version
pip3 --version
nvidia-smi
```

Expected output should show Python 3.8+, pip available, and GPU detection.

## Step 2. Create Python virtual environment

You will install ComfyUI on your host system, so you should create an isolated environment to avoid conflicts with system packages.

```bash
python3 -m venv comfyui-env
source comfyui-env/bin/activate
```

Verify the virtual environment is active by checking the command prompt shows `(comfyui-env)`.

## Step 3. Install PyTorch with CUDA support

Install PyTorch with CUDA 13.0 support.

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

This installation targets CUDA 13.0 compatibility with Blackwell architecture GPUs.

## Step 4. Clone ComfyUI repository

Download the ComfyUI source code from the official repository.

```bash
git clone --branch v0.33.2 https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI/
```

## Step 5. Install ComfyUI dependencies

Install the required Python packages for ComfyUI operation.

```bash
pip install -r requirements.txt
```

This installs all necessary dependencies including web interface components and model handling libraries.

## Step 6. Download model checkpoint that will be used in Step 9

```bash
cd models/checkpoints/
wget https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors
cd ../../
```

The download will be approximately 2GB and may take several minutes depending on network speed.

## Step 7. Launch ComfyUI server

Start the ComfyUI web server with network access enabled.

```bash
python main.py --listen 0.0.0.0
```

The server will bind to all network interfaces on port 8188, making it accessible from other devices.

## Step 8. Validate installation

The server from Step 7 keeps running in the foreground, so run the following in a **second terminal**.

Check that ComfyUI is running correctly and accessible via your web browser.

```bash
curl -I http://localhost:8188
```

Expected output should show HTTP 200 response indicating the web server is operational.

Open a web browser and navigate to `http://<HARDWARE_IP>:8188` where `<HARDWARE_IP>` is your device's IP address.

## Step 9. Run a template flow

Test the installation with a basic image generation workflow:

1. Access the web interface at `http://<HARDWARE_IP>:8188`

   > [!NOTE]
   > If the page fails to load, make sure the device you are browsing from is allowed to access your local network:
   > - **macOS:** Open **System Settings → Privacy & Security → Local Network** and enable access for your browser. macOS blocks local-network connections until an app is granted this permission. See [Control access to your local network on Mac](https://support.apple.com/guide/mac-help/control-access-to-your-local-network-on-mac-mchla4f49138/mac).
   > - **Windows:** Set your network profile to **Private** (not Public) so the device can reach others on the network. See [Make a Wi-Fi network public or private in Windows](https://support.microsoft.com/en-us/help/4043043/windows-10-make-network-public-private).
2. Load a starter workflow:
   1. Click **Templates** on the left side of the menu (skip this if the template window pops up automatically)
   2. Choose **Getting Started** on the left side of the template window
   3. Choose **1.1 Starter-Text to Image**
   4. Click the **Run** button at the top right
3. Monitor GPU usage with `nvidia-smi` in a separate terminal

The image generation should complete within 30 seconds.

## Step 10. Optional - Cleanup and rollback

If you need to remove the installation completely, follow these steps:

> [!WARNING]
> This will delete all installed packages and downloaded models.

```bash
deactivate
rm -rf comfyui-env/
rm -rf ComfyUI/
```

To rollback during installation, press `Ctrl+C` to stop the server and remove the virtual environment.

## Video Gen Workflow

> [!NOTE]
> These instructions target **Linux**. This tab needs substantially more memory than **Image Gen Quick Start**:
>
> | Tier | Disk (weights) | Peak GPU memory (approx.) |
> |------|----------------|---------------------------|
> | **1 — Getting Started** | ~70 GB | ~80 GB (Wan 720p) |
> | **2 — Intermediate** | ~180 GB | ~100 GB |
> | **3 — Advanced** | ~230 GB | ~120 GB (Hunyuan 1080p) |
>
> Match the tier to your hardware platform’s available memory. On unified-memory platforms, start with Tier 1 and reduce resolution or frame count if you hit memory pressure. For a lighter Stable Diffusion–only start (~8 GB class), use the **Image Gen Quick Start** tab.

## Step 1. Verify your environment

```bash
nvidia-smi
docker --version
df -h /
```

Expected: a detected NVIDIA GPU, Docker 24+, enough free disk for your model tier (Tier 1 ~70 GB; all tiers ~230 GB), and enough GPU memory for that tier’s peak (Tier 1 ~80 GB; see the note above).

If you have not already added your user to the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Set Hugging Face credentials

```bash
## Required for gated models. Run in the same shell as the download script.
## Get a token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_huggingface_token"
```

Accept model licenses when prompted on Hugging Face, for example:

- [FLUX.1 dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [HiDream-I1 Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full)

## Step 3. Clone this playbook and build the image

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-comfyui
docker build -t comfyui -f assets/Dockerfile .
```

The build clones ComfyUI, installs dependencies (preserving the NGC-optimized PyTorch), and pre-installs custom nodes for video, ControlNet, and IP-Adapter. Expect about 5–10 minutes.

## Step 4. Download models by tier

| Tier | Models | Disk space | Peak VRAM (approx.) | Workflows enabled |
|------|--------|------------|---------------------|-------------------|
| **1 — Getting Started** | FLUX.1 dev, Wan 2.1 T2V 14B | ~70 GB | ~80 GB (Wan 720p clip) | Text-to-image, text-to-video |
| **2 — Intermediate** | + HiDream-I1, Wan 2.1 I2V, Cosmos-Predict2 | ~180 GB | ~100 GB (FLUX→Wan two-model graph) | + HiDream, image-to-video, FLUX→Wan, Cosmos Video2World |
| **3 — Advanced** | + HunyuanVideo, FLUX ControlNet (Canny) | ~230 GB | ~120 GB (Hunyuan 1080p / long clips) | + 1080p video, ControlNet-guided generation |

Peak VRAM depends on resolution, frame count, and precision. Match the tier to your hardware platform's available memory. On unified-memory platforms, large video models may need reduced resolution or frames.

Install the Hugging Face Hub CLI if needed:

```bash
pip3 install --break-system-packages huggingface-hub
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
hf --version
```

```bash
## Tier 1 only:
bash assets/scripts/download-models.sh 1

## All tiers:
bash assets/scripts/download-models.sh
```

After Tier 1 completes:

```bash
ls -la ./models/diffusion_models/
ls -la ./models/text_encoders/ | head
```

## Step 5. Launch the container

Identify the target GPU index when more than one GPU is present:

```bash
nvidia-smi --query-gpu=index,name --format=csv,noheader
```

Use `--gpus '"device=N"'` with that index (default `0` on single-GPU systems). `--gpus all` is fine when only one GPU is available; on multi-GPU systems, pin the device you intend to use.

```bash
docker run -d \
  --name comfyui \
  --gpus '"device=<GPU_ID>"' \
  --ipc host \
  --ulimit memlock=-1 \
  -p 8188:8188 \
  -v "$(pwd)/models:/opt/ComfyUI/models" \
  -v "$(pwd)/output:/opt/ComfyUI/output" \
  -v "$(pwd)/input:/opt/ComfyUI/input" \
  -v "$(pwd)/assets/workflows:/opt/ComfyUI/user/default/workflows" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  comfyui
```

Follow logs until the UI is ready:

```bash
docker logs -f comfyui
```

Expected ready line:

```
To see the GUI go to: http://0.0.0.0:8188
```

Press `Ctrl+C` to leave the log view.

> [!NOTE]
> Startup may print benign warnings (CUDA-hooks diagnostics, package version skew, missing optional audio/ONNX GPU wheels). Treat the `To see the GUI go to: ...` line as the ready signal.

## Step 6. UI workflows vs API graphs

| Location | Format | Use |
|----------|--------|-----|
| `assets/workflows/*.json` (mounted into the UI workflow folder) | UI workflow (`nodes` / `links`) | **Load** in the web UI, then **Queue Prompt** |
| `assets/workflow_api/*.api.json` | API prompt graph | `POST /prompt`, `curl`, automation |

Loading an `.api.json` file with **Load** shows **"Error: the workflow does not contain any nodes"** — expected; those files are for the HTTP API only.

Optional HTTP API example (from the playbook root, ComfyUI on port 8188):

```bash
PROMPT=$(python3 -c "import json; d=json.load(open('assets/workflow_api/flux-text-to-image.api.json')); print(json.dumps({k:v for k,v in d.items() if str(k).isdigit()}, separators=(',',':')))")
curl -sS http://127.0.0.1:8188/prompt \
  -X POST \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":${PROMPT}}" | python3 -m json.tool
```

## Step 7. Validate the install

```bash
curl -I http://localhost:8188
```

Expected: HTTP 200. Open `http://<HARDWARE_IP>:8188` in a browser (`<HARDWARE_IP>` is the IP of your hardware platform).

**FLUX text-to-image (Tier 1):** **Load** `flux-text-to-image.json`, enter a prompt in **CLIP Text Encode**, click **Queue Prompt**. Expect roughly 15–30 seconds at default settings on high-memory hardware.

## Step 8. Image and video workflows

*Requires the matching model tier from Step 4.*

### Text-to-video with Wan 2.1 (Tier 1)

Load `wan-text-to-video.json`. Default graph targets ~720p, 81 frames (~5 s). Generation can take several minutes. Reduce frame count for faster iteration. Convert animated WEBP output to MP4 with `ffmpeg` if needed.

### Intermediate workflows (Tier 2)

- `hidream-text-to-image.json` — HiDream-I1 Full (17B) with four text encoders
- `wan-image-to-video.json` — place a source image in `input/` first
- `flux-to-wan-pipeline.json` — FLUX still → Wan I2V in one graph
- `cosmos-video2world.json` — NVIDIA Cosmos-Predict2 Video2World from an input image

### Advanced workflows (Tier 3)

- `hunyuan-1080p-video.json` — 1080p-class video (height divisible by 16; default 1920×1056). Needs very large GPU memory (~100–120 GB at default settings).
- `flux-controlnet.json` — Canny-conditioned FLUX; place a reference image in `input/`

## Step 9. Optional — cleanup and rollback

> [!WARNING]
> Cleanup removes the container and optionally downloaded models. Generated outputs in `output/` are preserved unless you delete them.

```bash
docker stop comfyui
docker rm comfyui
docker rmi comfyui          # optional
## sudo may be required if the container wrote files as root:
## sudo rm -rf models/
```

To avoid root-owned files on future runs, add `--user "$(id -u):$(id -g)"` to `docker run` (host UID must be able to write mounted directories).

## Troubleshooting

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| PyTorch CUDA not available (Image Gen Quick Start) | Incorrect CUDA wheels or missing drivers | Verify `nvidia-smi` and `nvcc --version`, reinstall PyTorch with the CUDA index URL from that tab |
| "permission denied" when running docker (Video Gen Workflow) | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Container fails to start with GPU error (Video Gen Workflow) | NVIDIA Container Toolkit not configured | Run `nvidia-ctk runtime configure --runtime=docker` and restart Docker |
| ComfyUI web UI not accessible | Firewall, wrong IP, or server not ready | Check host terminal output (Image Gen Quick Start) or `docker logs comfyui` (Video Gen Workflow); open port 8188; use `http://<HARDWARE_IP>:8188` |
| Model download fails | Network, disk space, or auth | Check connectivity, free disk, and (Video Gen Workflow) a valid `HF_TOKEN` plus accepted model licenses |
| HuggingFace download fails with 401 (Video Gen Workflow) | Invalid or missing HF token | Export a valid token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| "Model file not found" when running a workflow (Video Gen Workflow) | Weights missing or volume mount wrong | Confirm files under `./models/` and the `-v` mounts on `docker run` |
| Out of GPU / unified memory during generation | Resolution, frames, or model too large | Use a smaller model, lower resolution, or fewer frames; see Video Gen Workflow tier table |
| Workflow loads but nodes show red "missing" (Video Gen Workflow) | Custom node not installed | Use ComfyUI-Manager → Install Missing Custom Nodes, or rebuild the image |
| Web UI: **"Error: the workflow does not contain any nodes"** on **Load** | File is API format, not a UI workflow | Load `assets/workflows/<name>.json`; use `assets/workflow_api/<name>.api.json` only with `POST /prompt` |
| `device >= 0 && device < num_gpus INTERNAL ASSERT FAILED` (Video Gen Workflow) | `--gpus all` on a multi-GPU system | Use `--gpus '"device=N"'` for the intended GPU index from `nvidia-smi` |
| NGC image pull requires authentication (Video Gen Workflow) | NGC registry login required | Run `docker login nvcr.io` with your NGC API key |
| Very slow generation, low GPU utilization | Process not on GPU | Image Gen Quick Start: confirm CUDA in the venv. Video Gen Workflow: `docker exec comfyui nvidia-smi` |
| Memory pressure on unified-memory hardware even within capacity | Buffer cache not released to the GPU | Flush the buffer cache (see note below) |
| Container exits with `ModuleNotFoundError: torchaudio` or torchaudio ABI / undefined-symbol errors (Video Gen Workflow) | Image missing the playbook torchaudio stub, or a real torchaudio wheel was layered on NGC PyTorch | Rebuild from the shipped `assets/Dockerfile`. Do **not** `pip install torchaudio` inside the container |
| Custom-node build or DWPose warns about `onnxruntime` / no GPU providers on ARM64 (Video Gen Workflow) | `onnxruntime-gpu` has no aarch64 wheel on PyPI | The shipped Dockerfile substitutes CPU `onnxruntime`. Preprocessors run on CPU; treat as informational unless nodes fail to load |
| Startup shows `aimdo` hook failures or `urllib3` / `charset_normalizer` version warnings (Video Gen Workflow) | NGC base-image diagnostics / dependency skew | Benign if the ready line `To see the GUI go to: ...` appears; ComfyUI still works |

> [!NOTE]
> Some hardware platforms such as DGX Spark use Unified Memory Architecture (UMA), which shares memory dynamically between GPU and CPU. If you hit memory errors even when within total capacity, flush the buffer cache:
>
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

> [!NOTE]
> Video Gen Workflow logs: `docker logs -f comfyui`. Most missing-model and node errors appear there with clear messages.
