# Image & Video Generation with ComfyUI

> Generate images and videos with FLUX, Wan 2.1, HunyuanVideo, and Cosmos on DGX Station


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [HiDream-I1 image generation](#hidream-i1-image-generation)
  - [Wan 2.1 image-to-video](#wan-21-image-to-video)
  - [FLUX → Wan combined pipeline](#flux-wan-combined-pipeline)
  - [Cosmos-Predict2 Video2World](#cosmos-predict2-video2world)
  - [HunyuanVideo 1080p generation](#hunyuanvideo-1080p-generation)
  - [ControlNet with FLUX](#controlnet-with-flux)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

ComfyUI is a node-based visual interface for building image and video generation workflows using diffusion models. Instead of a single text box, you connect processing nodes — model loaders, text encoders, samplers, decoders — into a graph that gives full control over every generation step.

- **Node-based workflows** let you build, modify, and share complex generation pipelines visually.
- **Multi-model support** covers the latest architectures: FLUX for images, Wan 2.1 and HunyuanVideo for video, and NVIDIA Cosmos for world generation.
- **Full precision on GB300** — with 252 GB of HBM3e, you can run 12–17B image models and 13–14B video models at bf16 with no quantization or offloading, which is impossible on consumer hardware.

## What you'll accomplish

Deploy ComfyUI on DGX Station and run image and video generation workflows using six state-of-the-art models:

- **FLUX.1 [dev]** (12B) — high-quality text-to-image generation
- **HiDream-I1 Full** (17B) — the largest open image model, with four text encoders including Llama-3.1-8B
- **Wan 2.1 T2V/I2V 14B** — text-to-video and image-to-video at 720p
- **HunyuanVideo** (13B) — 1080p video generation leveraging the full GB300 memory (~100–120 GB VRAM)
- **NVIDIA Cosmos-Predict2** (14B) — NVIDIA's world foundation model for video-to-world generation

You will also learn advanced techniques including ControlNet-guided generation and combined image-to-video pipelines.

## What to know before starting

- Basic Docker container usage
- Familiarity with generative AI concepts (prompts, diffusion models) is helpful but not required

## Prerequisites

- NVIDIA DGX Station with GB300 GPU
- Docker installed: `docker --version`
- NVIDIA Container Toolkit configured: `nvidia-smi` should show the GB300
- HuggingFace account with access token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- At least 200 GB free disk space for model weights
- Network access to HuggingFace and GitHub

## Ancillary files

All required assets can be found [in the ComfyUI playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/station-comfyui/).

- `assets/Dockerfile` — Builds the ComfyUI container image from NGC PyTorch base (ARM64)
- `assets/scripts/download-models.sh` — Downloads all model weights from Hugging Face using the **`hf`** CLI (`huggingface-hub` package)
- `assets/workflows/*.json` — Eight **UI** workflows (ComfyUI 0.4 graph with `nodes` / `links`) for **Load** in the web UI
- `assets/workflow_api/*.api.json` — The same eight graphs in **API** format for `/prompt` and automation (`curl`, scripts)
- `assets/scripts/api_to_ui_workflow.py` — Regenerates UI JSON from API JSON if you edit a graph programmatically

## Time & risk

* **Duration:** 45 minutes (excluding model downloads, which may take 30–60 minutes depending on network speed)
* **Risks:**
  * Model downloads require HuggingFace authentication and substantial bandwidth (~150 GB total)
  * Port 8188 must be accessible for the ComfyUI web interface
* **Rollback:** Stop and remove the Docker container. Delete the `models/` directory to reclaim disk space.
* **Last Updated:** 05/07/2026
  * Re-validated end-to-end on GB300: clean image build (`comfyui-gb300`, ~24 GB), container starts and serves on port 8188, all 8 mounted UI workflows enumerate correctly, `/object_info` returns 1092 node types, `/prompt` validation rejects on missing-model with clean errors. Documented benign startup warnings (`aimdo` CUDA-hook fallback, `urllib3` / `charset_normalizer` version skew) so users do not chase non-issues.
  * 05/06/2026 — first publication; fixed walkthrough issues found on GB300: torchaudio shim for NGC PyTorch ABI mismatch, aarch64 onnxruntime swap, model-filename collisions (HiDream VAE → `ae_hidream.safetensors`, HunyuanVideo CLIP → `clip_l_hunyuan.safetensors`), `--gpus device=0` default, `df -h /` prereq, `~/.local/bin` PATH guidance, FLUX node list aligned with the actual graph, `.webp` output (not MP4), HF token via env not CLI, container output `chown` cleanup hint.

## Instructions

## Step 1. Verify your environment

Confirm Docker, GPU access, and available disk space.

```bash
docker --version
nvidia-smi
df -h /
```

- **Docker**: Must be running (version 24+ recommended).
- **nvidia-smi**: Should list the GB300 GPU with 252 GB HBM3e.
- **Disk space**: At least 200 GB free on `/` for model weights and the Docker image. On DGX Station `/home` is on the root filesystem, so checking `/` covers both. You can download fewer models by choosing a tier (see Step 4).

If you haven't already, add your user to the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Set up environment variables

Set your HuggingFace token so the download script and container can access gated models.

```bash
## HuggingFace token (required). Run this in the SAME shell that will
## launch `bash assets/scripts/download-models.sh` in Step 4 — the script
## reads $HF_TOKEN from the environment and exits early if it is unset.
## Get a token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_huggingface_token"
```

Some models (FLUX.1, HiDream-I1) require accepting the model license on HuggingFace before downloading. Visit each model page and click "Agree and access" if prompted:
- [FLUX.1 dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [HiDream-I1 Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full)

## Step 3. Clone the playbook and build the container

Clone the playbook repository and build the ComfyUI Docker image. The image is built on top of the NGC PyTorch container, which is already optimized for the GB300's ARM64 architecture.

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/station-comfyui
```

Build the container image:

```bash
docker build -t comfyui-gb300 -f assets/Dockerfile .
```

The build clones ComfyUI, installs dependencies (preserving the NGC-optimized PyTorch), and pre-installs custom nodes for video generation, ControlNet, and IP-Adapter. This takes approximately 5–10 minutes.

## Step 4. Download models

This playbook uses models organized into three tiers. Download only what you need, or download everything.

| Tier | Models | Disk space | Peak VRAM (approx.) | Workflows enabled |
|------|--------|------------|---------------------|-------------------|
| **1 — Getting Started** | FLUX.1 dev, Wan 2.1 T2V 14B | ~70 GB | ~80 GB (Wan 720p clip) | Text-to-image, text-to-video |
| **2 — Intermediate** | + HiDream-I1, Wan 2.1 I2V, Cosmos-Predict2 | ~180 GB | ~100 GB (FLUX→Wan two-model graph) | + HiDream image gen, image-to-video, FLUX→Wan pipeline, Cosmos Video2World |
| **3 — Advanced** | + HunyuanVideo, FLUX ControlNet (Canny) | ~230 GB | ~120 GB (Hunyuan 1080p / long clips) | + 1080p video, ControlNet-guided generation |

Peak VRAM depends on resolution, frame count, and precision; values above are **order-of-magnitude** for the default graphs in this playbook on a **GB300 (252 GB HBM3e)**.

Install the Hugging Face Hub CLI (provides the **`hf`** command) if you do not already have it. The CLI installs to `~/.local/bin/`, which is **not on the default non-interactive PATH**, so add it before continuing:

```bash
pip3 install --break-system-packages huggingface-hub
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
hf --version   # confirms PATH is correct
```

Run the download script with your chosen tier (default downloads all):

```bash
## Download Tier 1 only (Getting Started):
bash assets/scripts/download-models.sh 1

## Download all tiers:
bash assets/scripts/download-models.sh
```

Model downloads can take 30–60 minutes depending on network speed. The script uses the Hugging Face Hub **`hf download`** command (from the `huggingface_hub` package). If a download fails, the script **exits with an error** and prints which file was expected — check your token, network, and that you have accepted gated model licenses on Hugging Face.

After Tier 1 completes, verify weights landed under `models/`:

```bash
ls -la ./models/diffusion_models/
ls -la ./models/text_encoders/ | head
```

## Step 5. Launch ComfyUI

Start the ComfyUI container with all model and output directories mounted as volumes. On DGX Station, identify the GB300 GPU index with `nvidia-smi` and use `--gpus '"device=N"'` to target it. If the GB300 is your only GPU, `--gpus all` also works.

```bash
## Find the GB300 device index (look for "GB300" in the Name column)
nvidia-smi --query-gpu=index,name --format=csv,noheader
```

The default `--gpus '"device=0"'` works on single-GPU stations where the GB300 is index 0. If `nvidia-smi` reports the GB300 at a different index (for example index 1 on dual-GPU stations with an RTX PRO 6000 + GB300), substitute that index in the command below.

```bash
## device=0 by default; replace with the GB300 index from the command above
docker run -d \
  --name comfyui \
  --gpus '"device=0"' \
  --ipc host \
  --ulimit memlock=-1 \
  -p 8188:8188 \
  -v "$(pwd)/models:/opt/ComfyUI/models" \
  -v "$(pwd)/output:/opt/ComfyUI/output" \
  -v "$(pwd)/input:/opt/ComfyUI/input" \
  -v "$(pwd)/assets/workflows:/opt/ComfyUI/user/default/workflows" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  comfyui-gb300
```

Check startup logs:

```bash
docker logs -f comfyui
```

Expected output includes custom node loading messages and:
```
To see the GUI go to: http://0.0.0.0:8188
```

Press `Ctrl+C` to exit the log view. Open a web browser and navigate to `http://<STATION_IP>:8188` where `<STATION_IP>` is your DGX Station's IP address.

> [!NOTE]
> The startup logs include several **benign** warnings you can ignore: `aimdo: ... funchook_prepare(cuMemFree_v2) failed` (NGC PyTorch's CUDA hooks tool falling back to no-op), `urllib3 / charset_normalizer doesn't match a supported version`, `torchaudio missing` (covered by the import-only stub — no playbook workflow uses audio VAE), `DWPose: Onnxruntime not found ... switch to OpenCV with CPU device` (aarch64 has no `onnxruntime-gpu` wheel; CPU preprocessing still works), and `accelerate / GPTQModel / optimum / bitsandbytes not installed` from the HiDream sampler. The real "ready" signal is the `To see the GUI go to: ...` line above; treat anything else as suspect.

#### UI workflows vs API graphs (important)

ComfyUI uses **two different JSON shapes**:

| Location | Format | Use |
|----------|--------|-----|
| `assets/workflows/*.json` mounted at `user/default/workflows/` | **UI workflow** (has `"nodes"` and `"links"`) | **Load** in the web UI, edit in the canvas, then **Queue Prompt** |
| `assets/workflow_api/*.api.json` (on the host repo, not mounted into the default workflow folder) | **API prompt graph** (flat node ids → `class_type` / `inputs`) | **`POST /prompt`**, `curl`, automation |

If you open an **`.api.json`** file with **Load**, the UI shows **"Error: the workflow does not contain any nodes"** — that is expected; those files are not UI workflows.

**Optional — run the same graph via HTTP API** (from the playbook root, with ComfyUI listening on port 8188). Strip any non-node keys (for example `_comment` in some API files), minify to one line, and POST:

```bash
PROMPT=$(python3 -c "import json; d=json.load(open('assets/workflow_api/flux-text-to-image.api.json')); print(json.dumps({k:v for k,v in d.items() if str(k).isdigit()}, separators=(',',':')))")
curl -sS http://127.0.0.1:8188/prompt \
  -X POST \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":${PROMPT}}" | python3 -m json.tool
```

The response includes a `prompt_id` you can correlate with server logs and the `output/` folder.

**ComfyUI interface orientation:**
- **Canvas** — The central area where you build and view node workflows.
- **Queue Prompt** — The button (top right) that runs the current workflow.
- **Load** — Load a **UI** workflow from `flux-text-to-image.json`, `wan-text-to-video.json`, etc. (listed in the workflow sidebar under the mounted folder).
- **Manager** — Access ComfyUI-Manager to install additional custom nodes.

## Step 6. Image generation with FLUX.1 dev

*Requires: Tier 1 models*

Load the pre-built FLUX text-to-image workflow. In ComfyUI, click **Load** and select **`flux-text-to-image.json`** (UI format). Do **not** use the `*.api.json` files in `assets/workflow_api/` with Load — they are for the HTTP API only.

**What this workflow does:**

The workflow connects these nodes in sequence:

1. **UNETLoader** — Loads the FLUX.1 dev 12B transformer (~24 GB in bf16) with `weight_dtype=default`.
2. **DualCLIPLoader** — Loads CLIP-L and T5-XXL text encoders that convert your prompt into conditioning vectors.
3. **CLIP Text Encode** — Takes your text prompt and produces positive conditioning.
4. **FluxGuidance** — Applies FLUX's guidance value (default 3.5) to the conditioning.
5. **EmptySD3LatentImage** — Creates a blank latent at your chosen resolution (default: 1024x1024).
6. **ModelSamplingFlux** + **BasicScheduler** + **KSamplerSelect** + **BasicGuider** + **RandomNoise** — Configure FLUX's flow-matching schedule (20 steps, `euler`/`simple`).
7. **SamplerCustomAdvanced** — The diffusion sampling loop that denoises the latent.
8. **VAE Decode** — Converts the latent back into a pixel image.
9. **Save Image** — Writes the result to the `output/` directory.

**Try it:**

1. Find the **CLIP Text Encode** node and enter a prompt, for example: `A majestic snow leopard resting on a cliff at golden hour, photorealistic, 8k detail`
2. Click **Queue Prompt**.
3. The image generates in approximately 15–30 seconds. Results appear in the `output/` directory and in the preview node.

Experiment with different prompts, resolutions (512x512 up to 2048x2048), and step counts. FLUX.1 dev produces high-quality results even at 20 steps.

## Step 7. Video generation with Wan 2.1

*Requires: Tier 1 models*

Load `wan-text-to-video.json` from the workflow browser.

**What this workflow does:**

1. **Load Diffusion Model** — Loads the Wan 2.1 T2V 14B model (~28 GB in bf16).
2. **CLIPLoader** — Loads the UMT5-XXL text encoder for Wan.
3. **CLIP Text Encode** — Encodes your video description prompt.
4. **EmptyHunyuanLatentVideo** — Creates a blank video latent (default: 720p, 81 frames at ~16 fps ≈ 5 seconds). Wan reuses this latent format.
5. **KSampler** — Diffusion sampling over the video latent. This is the slowest step — expect 3–5 minutes for a 5-second clip on the GB300.
6. **VAE Decode** — Converts latents to video frames.
7. **SaveAnimatedWEBP** — Encodes frames into an animated WEBP file.

**Try it:**

1. Enter a prompt: `A drone shot flying over a misty mountain forest at sunrise, cinematic`
2. Click **Queue Prompt**.
3. Generation takes 3–10 minutes at 720p with 81 frames. Monitor GPU memory with `nvidia-smi` in another terminal — the 14B model at 720p uses approximately 65–80 GB of the GB300's 252 GB HBM3e.
4. The output **`.webp`** (animated WEBP from `SaveAnimatedWEBP`) appears in the `output/` directory. To convert to MP4, use `ffmpeg -i output/wan_t2v_output_00001_.webp output/wan_t2v_output.mp4`.

**Tips:**
- Reduce frame count (e.g., 49 frames ≈ 3 seconds) for faster iteration.
- Wan 2.1 responds well to cinematic, descriptive prompts with camera movement descriptions.

## Step 8. Intermediate workflows

*Requires: Tier 2 models*

This step introduces four additional workflows. Each builds on the basics from Steps 6–7.

### HiDream-I1 image generation

Load `hidream-text-to-image.json`.

HiDream-I1 Full is a **17B parameter** image model that uses **four text encoders** — CLIP-L, CLIP-G, T5-XXL, and Llama-3.1-8B-Instruct. The Llama encoder gives it exceptional prompt understanding, especially for complex or nuanced descriptions.

The full pipeline uses approximately **60–65 GB** in bf16 — well within the GB300's capacity but impossible on most GPUs.

**Try it:** Use a detailed, complex prompt to see the difference from FLUX — for example: `An astronaut riding a horse on Mars, with Earth visible in the sky, oil painting style by Rembrandt, dramatic chiaroscuro lighting`

### Wan 2.1 image-to-video

Load `wan-image-to-video.json`.

This workflow takes an **input image** and animates it into a video clip. Place your source image in the `input/` directory before running.

1. The **LoadImage** node reads from `input/`.
2. The **Wan 2.1 I2V 14B** model generates motion that is consistent with the source image.

**Try it:** Generate an image with FLUX first (Step 6), copy it from `output/` to `input/`, then animate it.

### FLUX → Wan combined pipeline

Load `flux-to-wan-pipeline.json`.

This workflow chains two models in a single graph:
1. **FLUX.1 dev** generates a high-quality still image from your text prompt.
2. The image is passed directly to **Wan 2.1 I2V 14B**, which animates it into a video.

This avoids manually moving files between workflows. Both models load into GPU memory simultaneously (~95 GB total in bf16).

### Cosmos-Predict2 Video2World

Load `cosmos-text-to-video.json`.

**NVIDIA Cosmos-Predict2 14B** is NVIDIA's world foundation model for Video2World generation. It takes an input image and generates a physically plausible video extending from that scene. Place your source image in the `input/` directory before running.

The Cosmos VAE is extremely efficient — it can encode/decode 1280x704 at 121 frames without tiling.

**Try it:** Use an image from a previous FLUX generation as the start frame, with a prompt describing the motion: `A red ball rolling down a wooden ramp and bouncing off a wall, physics simulation, realistic lighting`

## Step 9. Advanced workflows

*Requires: Tier 3 models*

### HunyuanVideo 1080p generation

Load `hunyuan-1080p-video.json`.

This is the **true GB300 showcase**. HunyuanVideo's 13B model generating at 1080p resolution uses approximately **100–120 GB of VRAM** — impossible on any consumer or professional GPU, but well within the GB300's 252 GB.

- Default: 1920x1056, 49 frames (~3 seconds). Note: height must be divisible by 16 for HunyuanVideo's latent space, so 1056 is used instead of 1080.
- Generation time: 2–5 minutes for 49 frames, longer for more.
- Monitor with `nvidia-smi` — you should see 100+ GB GPU memory usage.

**Try it:** `A time-lapse of cherry blossoms falling in a Japanese garden with a koi pond, 4K cinematic`

### ControlNet with FLUX

Load `flux-controlnet.json`.

ControlNet lets you **guide image generation with structural conditioning** — edges, depth maps, or pose skeletons extracted from a reference image.

1. Place a reference image in `input/`.
2. The **Canny Edge Preprocessor** extracts edge structure from the reference.
3. The **FLUX.1 Canny Dev** model (a full FLUX variant fine-tuned for canny conditioning) generates an image following that structure while applying the text prompt's style and content.
4. Both the preprocessed canny image and the final output are saved for comparison.

**Use cases:** Architectural visualization, consistent character poses, style transfer while preserving composition.

## Step 10. Cleanup

Stop and remove the ComfyUI container:

```bash
docker stop comfyui
docker rm comfyui
```

> [!NOTE]
> Files in `output/` and `models/` are written by the container as root, so removing them from the host shell needs `sudo` (e.g. `sudo rm -rf models/`). To avoid this in future runs, add `--user "$(id -u):$(id -g)"` to the `docker run` command in Step 5 — note that this requires the host UID to have write access to all mounted directories.

Optionally remove the Docker image:

```bash
docker rmi comfyui-gb300
```

Optionally remove downloaded models to reclaim disk space:

```bash
rm -rf models/
```

Generated images and videos in `output/` are preserved on the host regardless of container state.

## Troubleshooting

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| "permission denied" when running docker | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Container fails to start with GPU error | NVIDIA Container Toolkit not configured | Run `nvidia-ctk runtime configure --runtime=docker` and restart Docker |
| ComfyUI web UI not accessible | Firewall blocking port or wrong IP | Verify with `docker logs comfyui`, check that port 8188 is open, use `http://<STATION_IP>:8188` |
| "Model file not found" when running workflow | Model not downloaded or wrong path | Verify models are in `./models/` and the volume mount is correct in the docker run command |
| HuggingFace download fails with 401 | Invalid or missing HF token | Verify `HF_TOKEN` is exported and valid at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| CUDA out of memory during video generation | Frame count or resolution too high | Reduce frame count or resolution. At 720p with Wan 2.1 14B, keep clips under 5 seconds initially |
| CUDA out of memory during 1080p HunyuanVideo | Model + video tensors exceed GPU memory | Use fewer frames (e.g., 49 instead of 97). HunyuanVideo at 1080p needs ~100-120 GB |
| Workflow loads but nodes show red "missing" | Custom node not installed | Use ComfyUI-Manager (click Manager → Install Missing Custom Nodes) or rebuild the Docker image |
| Video output is a black screen | VAE decode issue or wrong model variant | Ensure you are using the correct model variant (T2V vs I2V) and the VAE is loaded |
| Very slow generation, GPU utilization low | PyTorch not using GPU or wrong CUDA version | Run `nvidia-smi` inside container: `docker exec comfyui nvidia-smi`. Ensure GPU is visible |
| "No module named ..." error on startup | Custom node dependency not installed | Exec into container and install: `docker exec comfyui pip install <module>` then restart |
| Docker build fails on ARM64 with `Could not find a version that satisfies the requirement onnxruntime-gpu` | `onnxruntime-gpu` has no aarch64 wheel on PyPI | Already handled by the shipped Dockerfile, which `sed`-substitutes `onnxruntime-gpu` → `onnxruntime` (CPU build) in every custom_node `requirements.txt` before `pip install`. If you see this error, you are building from a Dockerfile predating that fix — pull the latest assets and rebuild. |
| Docker build fails on ARM64 (other packages) | Some custom-node dependencies have no aarch64 wheel | Find the failing package in the build log. The custom-node install loop is wrapped in `\|\| true`, so the build still completes but the affected node will be missing modules at runtime. Either skip the node (remove its directory from `custom_nodes/` in the Dockerfile clone block) or install via ComfyUI-Manager after launch with a manually built wheel. |
| NGC image pull requires authentication | NGC registry needs login | Run `docker login nvcr.io` with your NGC API key |
| `device >= 0 && device < num_gpus INTERNAL ASSERT FAILED` on startup | Using `--gpus all` on a multi-GPU system causes a PyTorch assertion | Use `--gpus '"device=N"'` to target the GB300 specifically (check index with `nvidia-smi`) |
| `No HiDream models available` warning on startup | HiDream custom node reports no models found | This is a warning, not an error. It clears once HiDream model files are downloaded (Tier 2) |
| Web UI: **"Error: the workflow does not contain any nodes"** when using **Load** | The file is **API** format (flat `node_id → {class_type, inputs}`), not a UI workflow | In the playbook, use **`assets/workflows/<name>.json`** in the Load dialog (under **user/default/workflows** inside the container). For **`curl`** / HTTP API, use **`assets/workflow_api/<name>.api.json`** inside `{"prompt": ...}`. |
| `huggingface-cli: command not found` or download script errors | Deprecated CLI name | Install `huggingface_hub` and use **`hf download`** (the script does this automatically). |
| Download script exits but `models/diffusion_models/` is empty | Silent failure in older scripts or wrong token | Re-run with `bash -x assets/scripts/download-models.sh 1`; confirm `HF_TOKEN` and license acceptance on Hugging Face. The script now **fails fast** if a file is missing after `hf download`. |
| Container exits on startup with **`ModuleNotFoundError: torchaudio`** | Container was built from a Dockerfile predating the torchaudio shim | Rebuild the image: `docker build -t comfyui-gb300 -f assets/Dockerfile .`. The shipped Dockerfile creates an import-only `torchaudio` stub (NGC PyTorch's custom NVFP4 ABI is incompatible with PyPI torchaudio wheels). Lightricks audio VAE workflows are not supported in this image; no other workflow needs torchaudio. |
| `OSError: ... undefined symbol: torch_dtype_float4_e2m1fn_x2` from torchaudio | Real torchaudio installed on top of NGC PyTorch | Same fix as above — rebuild from the shipped Dockerfile. Do **not** `pip install torchaudio` manually inside the container. |
| `DWPose: Onnxruntime not found or doesn't come with acceleration providers, switch to OpenCV with CPU device` | Expected on aarch64. PyPI has no `onnxruntime-gpu` wheel for arm64; the Dockerfile substitutes the CPU `onnxruntime` package | Informational warning, not an error. DWPose preprocessing runs on CPU (slower than GPU) but produces correct output. |
| `aimdo: ... funchook_prepare(cuMemFree_v2) failed: 8 Failed to allocate memory in unused regions` at startup | NGC PyTorch's CUDA-hooks diagnostic tool (`aimdo`) cannot install hooks under default container caps and falls back to no-op | Benign. ComfyUI works normally; the message is informational from the NGC base image. No action required. |
| `RequestsDependencyWarning: urllib3 (...) or charset_normalizer (...) doesn't match a supported version!` at startup | Version skew between `requests` and the NGC-pinned `urllib3` / `charset_normalizer` wheels | Benign. ComfyUI's HTTP traffic still works. Suppress with `PYTHONWARNINGS=ignore::requests.RequestsDependencyWarning` if it bothers you. |

> [!NOTE]
> ComfyUI logs are visible with `docker logs -f comfyui`. Most errors (missing models, node failures) are reported in these logs with clear messages.
