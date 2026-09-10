# Generate Images and Video with ComfyUI

> FLUX.2 image and LTX-2 video workflows you run locally with full creative control


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Visual generative AI lets you create photorealistic images and coherent video clips from text (and image) prompts — with creative control that cloud generators often limit. Running these workflows **locally on your hardware platform** keeps assets under your control, avoids per-token cloud costs, and shortens the iteration loop that real creative projects need.

**ComfyUI** is an open-source, node-based app for advanced image and video generation. You install it, load templates for models such as **FLUX.2** and **LTX-2**, download model weights on demand, and build or save workflows as reusable graphs.

- **Local creative control** — refine prompts and graphs at your own pace without cloud round-trips
- **Template-first start** — Getting Started and All Templates paths cover text-to-image and image-to-video
- **VRAM-aware model choices** — pick precision and settings that fit your GPU memory

## What you'll accomplish

You'll install ComfyUI on your hardware platform, run a starter text-to-image workflow, generate images with **FLUX.2-Dev**, generate video with **LTX-2**, and optionally combine both into one custom workflow.

## What to know before starting

**Required:**

- Comfort installing and launching desktop applications on your hardware platform
- Enough free disk space for large model weight downloads (tens of GB per model)

**Optional:**

- Familiarity with generative AI prompting (subject, framing, style, mood)
- Experience with node-based tools or graph editors

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **RTX or RTX PRO** | Windows (ComfyUI desktop / portable); Linux paths noted for outputs | Dedicated VRAM (size varies by GPU) | ComfyUI from [comfy.org](https://comfy.org) · starter text-to-image template · FLUX.2-Dev · LTX-2 Image to Video · prefer lower-precision weights that fit your VRAM | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Enough GPU memory for the model and resolution you choose (frontier video models need more VRAM as resolution, length, and steps increase)
- Enough storage for ComfyUI plus model weights (individual models can exceed 30 GB)

**Software requirements**

- Network access to download ComfyUI and model weights (for example from Hugging Face via ComfyUI’s download dialog)
- A web browser (ComfyUI UI and optional image preview)
- For the Windows install path in **Instructions**: use the ComfyUI download from [comfy.org](https://comfy.org)

## Time & risk

- **Estimated time:** 13 MIN for install and first starter image; longer on first run of FLUX.2 or LTX-2 while model weights download
- **Risk level:** Low to Medium
  - Large model downloads can fail on unstable networks
  - High resolution, frame count, or step counts can exhaust GPU memory or slow generation sharply
- **Rollback:** Quit ComfyUI and uninstall or delete the ComfyUI install folder; delete downloaded model weights and saved workflows you no longer need
- **Last Updated:** 08/03/2026
  - Local ComfyUI workflow for FLUX.2 image and LTX-2 video generation on supported hardware platforms

## Instructions

## Step 1. Install and launch ComfyUI

Visit [comfy.org](https://comfy.org) to download and install ComfyUI for Windows, then launch ComfyUI on your hardware platform.

> [!NOTE]
> This playbook follows the desktop / portable ComfyUI path from [comfy.org](https://comfy.org). It does **not** invent host Python, Docker, or shell install commands. For a Linux host/container ComfyUI path on other hardware platforms, see [Generate Images and Videos with ComfyUI](https://build.nvidia.com/playbooks/comfyui).

## Step 2. Create a first image with the starter template

1. Click **Templates**, then **Getting Started**, and choose **1.1 Starter - Text to Image**.
2. Connect the model node to the **Save Image** node so the graph forms a complete pipeline.
3. Press the blue **Run** button and watch the green node highlights as your hardware platform generates the first image.

Change the prompt and run again to iterate.

## Step 3. Generate images with FLUX.2-Dev

1. Open **Templates** → **All Templates** and search for **FLUX.2 Dev Text to Image**.
2. Select the template. ComfyUI loads the connected nodes (the workflow).
3. When prompted, download the model weights. Weight files (`.safetensors`) save into the correct ComfyUI folders automatically. FLUX.2 can exceed 30 GB depending on the version — allow time and disk space.
4. Save the workflow: open the top-left hamburger menu and choose **Save**. Press **W** to show or hide the Workflows list.

If you close the download dialog before weights finish:

1. Press **W** to open **Workflows**
2. Select the workflow again so ComfyUI reloads it and re-prompts for any missing weights

**Prompt tips for FLUX.2-Dev**

- Start with clear subject, setting, style, and mood — for example: “Cinematic closeup of a vintage race car in the rain, neon reflections on wet asphalt, high contrast, 35mm photography.” Short-to-medium prompts are usually easier to control than long storylike prompts when learning.
- Add constraints for framing (“wide shot” / “portrait”), detail (“high detail, sharp focus”), and realism (“photorealistic” / “stylized illustration”).
- If results feel busy, remove adjectives instead of adding more.
- Avoid negative prompting — describe what you want.

Learn more in the [Black Forest Labs FLUX.2 prompting guide](https://docs.bfl.ai/guides/prompting_guide_flux2).

**Save locations**

Right-click the **Save Image** node to open the image in a browser or save it elsewhere. Default ComfyUI output folders are typically:

- Windows (standalone / portable): under the ComfyUI install (for example an `output` folder beside the app)
- Windows (desktop application): under the AppData directory
- Linux: under the ComfyUI install location

## Step 4. Generate video with LTX-2

Use the **LTX-2 Image to Video** template. Lightricks’ LTX-2 is an audio-video model for controllable, storyboard-style generation in ComfyUI. Unlike the starter and FLUX.2 text-to-image templates, this workflow combines an **image** and a **text prompt**.

1. Download the **LTX-2 Image to Video** template and its model weights when prompted.
2. Use an image from your FLUX.2-Dev run (or another still) as the image input.
3. Write the prompt like a short shot description, not a full movie script.

A walkthrough video is available here: [LTX-2 in ComfyUI](https://youtu.be/ifxOXmL351I?si=xf2_j01BAQlUhfhY0).

**Prompt tips for LTX-2**

Write a single flowing paragraph in the present tense, or a simple script-style format with scene headings, action, character names, and dialogue. Aim for four to six sentences that:

- Establish shot and scene (wide / medium / closeup, lighting, color, textures, atmosphere)
- Describe action as a clear sequence, with visible character traits, body language, and camera moves
- Add audio (ambient sound, music, dialogue in quotation marks)

Match detail to shot scale — closeups need more precise character and texture detail than wide shots. Useful prompt ingredients include camera movement language, shot type, pacing, atmosphere, style, lighting, emotion, and voice/audio.

**VRAM and quality**

LTX-2 uses significant VRAM. Memory use rises with resolution, frame rate, length, and steps. ComfyUI can stream weights to system memory when GPU memory runs short — that can keep a run going at the cost of speed. Constrain resolution, length, and steps for reasonable generation times on your hardware platform.

Learn more in the [Quick Start Guide for LTX-2 In ComfyUI](https://www.nvidia.com/en-us/geforce/news/rtx-ai-video-generation-guide/).

## Step 5. Combine FLUX.2-Dev and LTX-2 in one workflow (optional)

To avoid hopping between workflows when turning a still into video:

1. Open your saved **FLUX.2-Dev Text to Image** workflow.
2. Ctrl+left-click the FLUX.2-Dev Text to Image node to copy it.
3. In the **LTX-2 Image to Video** workflow, paste with Ctrl+V.
4. Drag from the FLUX.2-Dev node **IMAGE** output to the **Resize Image/Mask** input so a blue connector appears.
5. Save under a new name, then prompt for image and video in one graph.

## Step 6. Cleanup (optional)

Cleanup is optional rollback — not required to finish the playbook.

1. Quit ComfyUI
2. Delete saved workflows you no longer need from the Workflows list or on disk
3. Remove downloaded model weights you do not want to keep (large `.safetensors` files under your ComfyUI model folders)
4. Uninstall or delete the ComfyUI application folder if you want a full rollback

## Step 7. Next steps

1. Explore additional ComfyUI templates for advanced image and video models
2. Follow the [NVIDIA Blueprint for 3D-guided generative AI](https://github.com/NVIDIA-AI-Blueprints/3d-guided-genai-rtx) for 3D-guided image and video pipelines
3. Share work and get help in the [Stable Diffusion subreddit](https://www.reddit.com/r/StableDiffusion/) and [ComfyUI Discord](https://discord.com/invite/comfyorg)
4. For Linux host/container ComfyUI setups on other supported hardware catalogs, see [Generate Images and Videos with ComfyUI](https://build.nvidia.com/playbooks/comfyui)

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| ComfyUI will not install or launch from comfy.org | Incomplete download, blocked installer, or unsupported OS for this desktop path | Re-download from [comfy.org](https://comfy.org), confirm you are on a supported Windows install path for this playbook, and retry launch |
| Model weight download dialog closed early / workflow fails on missing weights | Required `.safetensors` files were not finished downloading | Press **W**, reopen the workflow, and complete the download prompts; confirm enough free disk space (FLUX.2 can exceed 30 GB) |
| Out of memory or very slow LTX-2 / FLUX.2 runs | Resolution, length, frame rate, steps, or precision exceed available GPU memory | Lower resolution, frame count, length, or steps; choose lower-precision weights that fit your VRAM; close other GPU apps; allow weight streaming to system memory only if you accept slower runs |
| Generated image or video quality is poor or chaotic | Prompt is too long, contradictory, or underspecified | Shorten the prompt; state subject, framing, style, and mood clearly; for LTX-2 use a short shot description with camera and audio cues (see **Instructions**) |
| Cannot find saved images on disk | Looking outside ComfyUI’s default output folders | Check the install `output` folder (portable), AppData paths (Windows desktop app), or the ComfyUI install location (Linux); or right-click **Save Image** to open/save explicitly |
| Nodes will not connect / Run does nothing | Graph is incomplete (model not linked to Save Image or required inputs missing) | Re-open the template, connect required outputs to inputs until the pipeline is complete, then press **Run** |

> [!NOTE]
> On discrete-GPU hardware platforms, GPU memory is separate from system RAM. CUDA out-of-memory or stalled generation usually means the workload exceeds VRAM: reduce resolution, length, steps, or model precision; enable ComfyUI weight streaming only if you accept slower performance; close other GPU applications.

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
