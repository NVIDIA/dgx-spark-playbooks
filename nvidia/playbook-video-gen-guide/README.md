# Generate Controlled Video with ComfyUI

> Composition-locked storyboards to 4K with LTX-2.3 and RTX Video Super Resolution

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Text-only video prompts give limited control over composition, camera angle, and subject motion. This workflow helps you guide the output step by step on your **hardware platform**: generate 3D scene assets, lock composition with depth-guided image generation, turn first and last frames into video, then upscale to 4K.

You start by generating a 3D scene to guide composition, turn that layout into photorealistic keyframes, use those images as first and last frames for video generation, then upscale with the **RTX Video Super Resolution** node in ComfyUI. The result is a high-resolution clip that follows your composition, camera angle, and subject motion.

## What you'll accomplish

You'll run a local storyboard-to-video workflow on your **hardware platform** that can:

- Generate 3D scene assets from text (standalone or in Blender)
- Produce composition-locked first and last frames with depth-guided image generation (FLUX.1)
- Generate video between those frames with LTX-2.3 in ComfyUI
- Upscale the result toward 4K with RTX Video Super Resolution

Creators can use any stage alone. For the full pipeline, complete each stage before starting the next so system resources stay available.

## What to know before starting

**Required:**

- Comfort with Windows desktop apps (Blender, ComfyUI, PowerShell or Command Prompt)
- Familiarity with generative AI concepts (prompts, CFG / guidance, diffusion or video models)

**Optional:**

- Prior ComfyUI experience ([How to Get Started With Visual Generative AI on NVIDIA RTX PCs](https://blogs.nvidia.com/blog/rtx-ai-garage-comfyui-tutorial/))
- Basic Blender scene layout (cameras, assets, animation scrubbing)

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **RTX or RTX PRO** | Windows 11 | 16 GB+ dedicated VRAM; 64 GB system RAM recommended | Blender + ComfyUI; LTX-2.3 FirstFrame/LastFrame template; RTX Video Super Resolution | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- At least 16 GB dedicated GPU memory (NVIDIA GeForce RTX 5070 Ti, NVIDIA RTX PRO 2000 Blackwell, or higher recommended)
- 64 GB system RAM recommended

**Software requirements**

- Windows 11
- Blender (source workflow uses Blender 4.2 for asset import and Blender 4.5 LTS with the ComfyUI Blender AI node)
- ComfyUI, including access to the template browser and ComfyUI Manager
- Network access to download blueprint dependencies and models from the linked GitHub repositories
- Browser access to local UIs (for example, `http://127.0.0.1:7860` for the 3D Object Generation blueprint)

**Workflow components** (install and setup live in the linked blueprints — this playbook does not ship a local install bundle):

| Stage | What it does |
| :---- | :---- |
| [3D Object Generation blueprint](https://github.com/NVIDIA-AI-Blueprints/3d-object-generation) | Describe objects, preview, and pick assets (Llama 3.1 8B, NVIDIA SANA, Microsoft TRELLIS) |
| [3D Guided Generative AI blueprint](https://github.com/NVIDIA-AI-Blueprints/3d-guided-genai-rtx) | Lay out the scene in Blender and generate start/end frames from the viewport with FLUX.1 (non-commercial; contact Black Forest Labs for commercial use) |
| [LTX-2.3 FirstFrame/LastFrame + RTX Video upscaler](https://github.com/NVIDIA-AI-Blueprints/3d-guided-genai-rtx/tree/main/example_workflows) | Turn keyframes into video with LTX-2.3, then upscale with RTX Video Super Resolution (ComfyUI template browser when available, or the example workflows on GitHub) |

## Time & risk

- **Estimated time:** 18 MIN (first run is longer when downloading models and blueprint dependencies)
- **Risk level:** Medium
  - Large model and blueprint downloads can fail if network connectivity is unstable
  - GPU memory pressure can appear if other GPU workloads are running during image or video generation
  - FLUX.1 in the guided blueprint is non-commercial unless you have a commercial license from Black Forest Labs
- **Rollback:** Close Blender and ComfyUI, deactivate any conda environments you created for the blueprints, and delete local blueprint clones or generated assets you no longer need
- **Last Updated:** 08/03/2026
  - Storyboard-to-4K video workflow with 3D-guided composition, LTX-2.3 first/last frame generation, and RTX Video Super Resolution upscaling

## Instructions

> [!NOTE]
> Install and dependency setup for each blueprint live in the GitHub repositories linked below. This playbook covers the creative workflow after those blueprints are installed. Concrete package pins, container tags, or a playbook-local asset clone are **not** included here.

## Step 1. Install the blueprints and ComfyUI pieces

Follow the setup instructions in each repository before you continue:

1. [3D Object Generation blueprint](https://github.com/NVIDIA-AI-Blueprints/3d-object-generation)
2. [3D Guided Generative AI blueprint](https://github.com/NVIDIA-AI-Blueprints/3d-guided-genai-rtx) (includes the ComfyUI Blender AI node path used in later steps)
3. LTX-2.3 FirstFrame/LastFrame + RTX Video upscaler workflows via the ComfyUI template browser (when available) or [example workflows on GitHub](https://github.com/NVIDIA-AI-Blueprints/3d-guided-genai-rtx/tree/main/example_workflows)

If you are new to ComfyUI, review [How to Get Started With Visual Generative AI on NVIDIA RTX PCs](https://blogs.nvidia.com/blog/rtx-ai-garage-comfyui-tutorial/).

## Step 2. Generate scene assets and build your scene

Generate assets with the 3D Object Generation blueprint, either standalone or in Blender 4.2.

Standalone launch (PowerShell or Command Prompt), after you have followed the blueprint install path:

```
cd C:\3d-object-generation
conda activate 3dwithtrellis311
python app.py
```

When the app is running, open [http://127.0.0.1:7860](http://127.0.0.1:7860) and generate assets. Type a description of the scene you want to build (for example, “spaceship bridge”).

Run generation several times to build a collection of assets. You can also model directly in Blender or import props from elsewhere. Keep all assets in the same folder.

Open the sample Blender file that ships with the blueprint, remove the sample props and set decoration, then use the Asset Importer add-on to pull your content into Blender. You may need a scale factor (10x is a common starting point).

Camera angle, scene depth, and subject position in this layout carry through into the generated video.

## Step 3. Set up Blender for image generation and make your first keyframe

Open Blender 4.5 LTS and open the 3D scene you built or edited. With the ComfyUI Blender AI node (ComfyUI x Blender) add-on installed from the guided blueprint, you should see it on the right side of the viewport.

Before you press Launch/Connect to ComfyUI, confirm the ComfyUI nodes are populated for both the first-frame and last-frame graphs:

- UNET Loader — `unet Name`
- DualCLIPLoader — `clip_name1`, `clip_name2`
- KSampler — `sampler_name`, `scheduler`

Press the Launch/Connect to ComfyUI button on the add-on, wait 30–60 seconds for ComfyUI to load, pick your composition for the first frame, and press **Run**. The image saves according to the SaveImage node in the graph.

The graph builds a depth map from your Blender scene and combines it with your text prompt to generate a photorealistic image that matches your layout and perspective. Image generation uses FLUX.1 Depth, accelerated by NVFP4 on supported hardware.

Refine your prompt until the composition looks right. That image is your first frame.

## Step 4. Generate your last keyframe

In Blender, if the scene is animated, scrub to the end pose you want. If the scene is static, place a second camera and move objects to the end composition.

Then:

1. Change the 3D Guided add-on top menu to last frame, and add a text prompt
2. Change the ComfyUI window top menu to last frame
3. Edit the output file name so you can distinguish the last frame

Press **Run** to create the last frame.

## Step 5. Generate video with LTX-2.3

In ComfyUI, search the template browser for LTX and open the FirstFrame/LastFrame template. Load your first and last frame images into the corresponding input nodes. Write a video prompt that describes the motion between the frames as a short paragraph (natural language, not a tag list).

Example structure for a controlled camera move:

*“Cinematic 1960s Supermarionation style. Two marionette pilots operate a retro cockpit… The camera performs a steady forward dolly-in… Outside, a rigid, static miniature space station… High-contrast studio lighting, visible model textures, and vintage 35mm film grain.”*

Adjust CFG to change prompt adherence. Raising CFG (for example from 1 toward 4) aligns the generation more tightly to the prompt and can reduce creative variation.

A short negative prompt can help, though it is optional. Prompting tips: [Prompting guide for LTX-2](https://ltx.io/model/model-blog/prompting-guide-for-ltx-2).

Iterate at 1280×704; when you like the motion, try 1920×1088. LTX needs pixel dimensions divisible by 32 (hence values such as 704 and 1088).

## Step 6. Upscale to 4K with RTX Video Super Resolution

On supported hardware platforms, connect the **RTX Video Super Resolution** node to scale the output:

1. Search `RTX` in ComfyUI Manager and install the RTX Video Super Resolution node
2. Search `RTX` in the Node Library and drag **RTX Video Super Resolution** into the graph
3. Connect VAE Decode **IMAGE** out to RTX Video **Images In**, and RTX Video **upscale_images** out to Create Video **Images In**

For 4K, choose 3× for a ~1280×720 source or 2× for 1920×1088. Stay on ULTRA quality unless you need faster performance.

## Step 7. Cleanup (optional)

When you are finished:

1. Close ComfyUI and Blender sessions
2. Deactivate conda environments you started for the blueprints
3. Remove generated images, videos, or blueprint clones you do not want to keep

Cleanup is optional.

## Step 8. Next steps

- Re-run Steps 3–5 with different camera moves or prompts while keeping the same 3D layout
- Tune LTX settings (steps, guidance, frame count) using the guidance in **Troubleshooting**
- For broader ComfyUI image and video workflows on other hardware platforms, see [Generate Images and Videos with ComfyUI](https://build.nvidia.com/playbooks/generate-images-comfyui)

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Generated 3D object has a clean silhouette but messy textures | Source image had a complex or cluttered background, strong directional shadows, or a non-ideal framing | Use 1:1 aspect ratio images with a plain or removed background, neutral even lighting, and a clear front-facing or three-quarter view; pre-remove busy backgrounds before TRELLIS |
| 3D object geometry or texture quality is low | Sparse Structure / Latent Sampling steps or CFG too low for the prompt | Raise Sparse Structure Sampling Steps for cleaner topology; raise Latent Sampling Steps for surface detail; increase CFG on both; start from defaults and increase if the output does not match the prompt |
| SANA previews are slow or lower quality than expected | Preview resolution not matched to SANA’s sweet spot | Use 1024×1024 for quality; use 512 or 768 for faster iteration previews |
| Need faster image iteration than FLUX.1 | Full FLUX.1 Depth path is heavier than needed for early layout tests | In the Load Model node, try a smaller model such as SDXL; use positive/negative prompts and Wildcards in the ComfyUI graph for lighting variation |
| LTX-2.3 output looks soft or takes too long while iterating | Resolution or frame count too high for draft passes | Iterate at 1280×720 (or 1280×704) and keep sequences under 257 frames; raise to 1920×1088 when ready for final quality |
| LTX motion ignores the prompt or looks unnatural | Guidance / step settings out of a useful range | Use 20–30 steps while iterating and 40+ for finals; set Guidance Scale around 3.0–3.5 |
| Last frame does not match the provided end image | End-frame guide strength or timing too weak; long clips degrade end-frame adherence | Raise last-frame strength to 1.0; try last-frame position index `-12` instead of `-1`; keep sequences near 5 seconds (121 frames) |
| Output video is completely black | Invalid frame count, missing crop guides, or missing text encoder | Use frame counts that follow `(N×8)+1` (49, 65, 97, 121, …); for FirstFrame/LastFrame workflows add LTXVCropGuides before VAE decode; confirm the Gemma text encoder loaded |
| Subject appearance drifts mid-video | Model limitation on long or multi-motion clips | Keep clips to about 5 seconds; describe one clear motion; reduce CFG to about 3.0–3.5; for recurring characters, a subject LoRA improves consistency |
| Upscale quality or speed is off | Upscale factor or quality level not matched to source resolution | Set Upscale Factor 1–4 from input size to target (for ~720p→4K use 3); set Quality Level to 4 for maximum edge sharpening unless you need speed |
| Prompt or negative prompt fights the keyframes | Prompt restates what is already in the images, or negative list is too long | Describe change and motion, not the static look; keep negatives focused (for example: morphing, distortion, warping, flicker, jitter, blur, artifacts); LTX-2.3 does not require a negative prompt |
| GPU out-of-memory or very slow generation | Workload exceeds dedicated VRAM, or other GPU apps are running | Close other GPU applications; lower resolution, frame count, or upscale factor; monitor with Task Manager / GPU tools on Windows |

> [!NOTE]
> On discrete-GPU hardware platforms, GPU memory is separate from system RAM. Out-of-memory usually means the workload exceeds VRAM: reduce resolution, frame count, or concurrent apps. Confirm no other process is holding GPU memory before retrying.

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
