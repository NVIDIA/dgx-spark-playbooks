# Build a Multi-GPU AI PC

> Homogeneous dual-GPU setups for llama.cpp tensor parallel and ComfyUI multi-GPU diffusion

## Table of Contents

- [Overview](#overview)
  - [Llama.cpp for LLMs](#llamacpp-for-llms)
  - [ComfyUI for Creative AI](#comfyui-for-creative-ai)
  - [Quick Spec Reference](#quick-spec-reference)
  - [Choosing Your Components](#choosing-your-components)
  - [Build Tips and Common Pitfalls](#build-tips-and-common-pitfalls)

---

## Overview

## How to Build a Multi-GPU AI PC: A Practical Guide

Many people are interested in exploring local generative AI for privacy and to avoid token limits. The challenge is that the latest models consume a lot of memory, and a lot of compute. To tackle this, some users have started building multi-GPU setups.

In this guide, we show you how to build your own multi-GPU PC, and how to configure your AI apps to take advantage of it.

## Why is this now possible?

AI enthusiasts have been using multi-GPU PCs to run local AI for years, by leveraging the AI frameworks that are run on datacenters, like vLLM or SGLang, to pair up GPUs. Until recently, the consumer tools that most AI enthusiasts use - Llama.cpp for LLMs and ComfyUI for creative AI - were not optimized for multi-GPU setups.

NVIDIA has partnered with the llama.cpp and ComfyUI community to bring optimized performance for multi-GPU setups on NVIDIA GPUs:

* Llama.cpp has added Tensor Parallel - a technique that allows you to spread a model across 2 GPUs and use both for compute. This effectively doubles the performance compared to the previously available method, Pipeline Parallelization, which only allowed you to share VRAM but not benefit from dual compute. This tensor parallel technique fully utilizes both GPUs for up to ~2X memory capacity and up to ~1.8X compute performance.

* ComfyUI has added CFG, a method to use both GPUs to compute the AI work during a diffusion process. Users can also choose what part of the pipeline is loaded on each GPU to benefit from the additional VRAM offered by a second GPU, avoiding having to load models in and out of memory for a further performance boost. These techniques have resulted in up to 2x compute across two GPUs.

## Configuring Your PC for Multi-GPU AI

If you already have a PC with 2 equivalent GPUs, you are ready to start using the new optimizations! Follow the instructions below to learn how to configure your apps to best leverage your 2 GPUs. If you haven’t made the move yet, you can skip over to the next section for tips on how to build your dual-GPU system.

**Supported GPUs:** Any NVIDIA RTX Ampere GPU (e.g. RTX 30 Series) or newer can be utilized, but the GPUs must be homogeneous (i.e., 2x 3090, 2x 5070 Ti, etc.); mixed GPU types are not supported. For the best experience, we recommend using 2 GPUs connected over a PCIe network.

### Llama.cpp for LLMs

* If you are using LM Studio:
   1. Under Runtime Settings (Ctrl+Shift+R), ensure you have a CUDA runtime selected
   2. Open the hardware settings (Ctrl+Shift+H) and from the dropdown menu, select "tensor parallelism" as the split strategy

* If you are using Llama.cpp directly:
   1. Get a CUDA-enabled build of llama.cpp. On Windows, go to the llama.cpp [GitHub release page](https://github.com/ggml-org/llama.cpp/releases/) and download the latest CUDA build. On Linux, the prebuilt releases do not include a CUDA build (they are CPU, Vulkan, ROCm, SYCL, or OpenVINO only and cannot split work across NVIDIA GPUs), so build from source with the CUDA Toolkit, CMake, and git installed:

      ```bash
      git clone https://github.com/ggml-org/llama.cpp
      cd llama.cpp
      cmake -B build -DGGML_CUDA=ON
      cmake --build build --config Release -j
      ```

      The binaries are written to `build/bin/`.
   2. Run `llama-cli` or `llama-server` with the following commands.

      - **Pipeline parallel**

         ```bash
         <app> -m <model_path> -sm layer -fa 1
         ```

      - **Tensor parallel**

         ```bash
         <app> -m <model_path> -sm tensor -fa 1
         ```

   3. Automatic parameter fitting is not supported with `-sm tensor` yet. If you hit OOM errors, reduce the `-np` value for `llama-server`, or reduce the `-c` or `-ngl` values.
   4. Refer to [multi-GPU readme](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) in llama.cpp repo for more details.

### ComfyUI for Creative AI

The new **MultiGPU CFG Split Node in ComfyUI** allows diffusion processing to occur using multiple GPUs installed in the same system. Performance results are workflow dependent, but speedups of up to 1.95x have been measured in common workflows.

**Supported Models:** The following models have been tested as working, speedups will vary by model and workflow.

* LTX-2.3
* WAN 2.2
* FLUX.2 Klein - Base Versions
* Z-Image
* Stable Diffusion 3.5 Large
* Hunyuan Video
* Qwen-Image-Edit-2511
* Hunyuan-3D-v2.1
* SDXL

**MultiGPU CFG Split Node:** The max_gpus field should be set to the maximum number of identical GPUs installed in the system.

**Node placement:** The MultiGPU CFG Split needs to be placed between the Model Load node and Sampling node. If other nodes are connected to the model output of the model loader node, the MultiGPU CFG should be the last node in the chain before the Sampling node.

**Workflow Requirements:** The node works by splitting the diffusion workflow at the CFG level; because of this, the CFG in the workflow must be greater than CFG = 1. Distilled workflows (or similar workflows) which require a CFG = 1 will not show a performance benefit when using the MultiGPU CFG Split node to run on multiple GPUs.

#### Verifying Multi-GPU Utilization

When running a workflow with MultiGPU CFG Split enabled, you can look at the Windows Task Manager and select the performance category.

You should see activity on both installed GPUs while the sampler is running in the workflow.

## How to Build Your Dual GPU PC

### Quick Spec Reference

For readers who want to skim and go, this is the recommended setup:

| Component | Good | Best |
| :---- | :---- | :---- |
| **GPU 1** | NVIDIA GeForce RTX 5080 | NVIDIA GeForce RTX 5090 |
| **GPU 2** | NVIDIA GeForce RTX 5080 | NVIDIA GeForce RTX 5090 |
| **Motherboard** | x8/x8 PCIe Gen 5 | x16/x16 PCIe Gen 5 |
| **CPU** | AMD Ryzen 9 / Intel Core i9 | AMD Threadripper |
| **RAM** | 64GB (32GB minimum) | 128GB |
| **Storage** | Single PCIe Gen 5 NVMe (4TB+) | Single PCIe Gen 5 NVMe (4TB+) |
| **Power supply** | 1200–1400W, 80 Plus Gold | 1800W+, 80 Plus Gold |
| **Case** | Large ATX or open-frame ATX | Large ATX or open-frame ATX |

The detailed reasoning behind each component is in the next section.

### Choosing Your Components

Each subsection below covers the shared technical context first, then splits the recommendation by path.

#### GPUs

Large language models load weights into GPU memory and stay there while running. Assuming models are quantized to 4-bit, a 30-billion-parameter LLM needs at least 24GB of VRAM. A 70-billion-parameter model needs 40GB or more. A 120-billion-parameter model needs roughly 70GB once context is included, which is more than the 64GB of two NVIDIA GeForce RTX 5090 GPUs; for a model that size, use a smaller or more aggressively quantized model, or accept slower partial CPU offload. Diffusion image and video models behave the same way.

NVIDIA GeForce RTX 50 Series GPUs ship with high-speed GDDR7 memory at 28–30 Gbps and add native FP4 support. FP4 quantization shrinks models by 60–70% while keeping quality near the original. NVIDIA GeForce RTX 40 Series GPUs support FP8 quantization, which delivers similar size reductions at slightly lower compression. Both let larger models fit in less VRAM.

GPUs must match. Local LLM inference runs at the speed of the slower GPU, and FP4 acceleration only works on NVIDIA GeForce RTX 50 Series. Two NVIDIA GeForce RTX 5090 GPUs deliver the most VRAM (64GB combined) and the most compute. Two NVIDIA GeForce RTX 5080 GPUs deliver 32GB combined at 37% lower power draw per GPU.

#### Motherboard

The motherboard decides how fast the two GPUs can talk to each other. PCIe Gen 5 doubles Gen 4 bandwidth at roughly 4GB per second per lane — useful headroom for AI workloads moving large model weights between GPUs. For the best performance, we recommend at least x8 PCIe Gen 5 lanes; anything less can create a bottleneck with some models. Most consumer motherboards have one PCIe x16 slot and several secondary slots wired for x1 or x4. Those secondary slots are not multi-GPU slots, and the location of the x8 slot varies between boards — check the manual carefully before buying. Bandwidth gets shared with M.2 NVMe drives on most consumer boards, so adding a second SSD can drop a GPU slot to x4 without warning.

x8/x8 PCIe Gen 5 is ideal for best performance. x16/x16 PCIe Gen 5 is the best option, since both GPUs run heavy AI work simultaneously and benefit from full bandwidth.

**Examples of good motherboards (AMD)**

- ASUS ROG Crosshair X870E Extreme
- ASUS ROG Crosshair X870E APEX
- ASUS ROG Crosshair X870E Hero
- ASUS ProArt X870E-Creator WiFi
- Gigabyte X870E Aorus Master X3D Ice
- Gigabyte X870E Aorus Xtreme AI TOP
- MSI MEG X870E Godlike

**Examples of good motherboards (Intel)**

- ASUS ROG Maximus Z890 Extreme
- ASUS ROG Maximus Z890 Apex
- ASUS ProArt Z890-Creator WiFi
- Gigabyte Z890 Aorus Xtreme AI TOP
- Gigabyte Z890 Aorus Master AI TOP
- Gigabyte Z890 Aorus Tachyon Ice
- Gigabyte Z890 Aero D/G
- MSI MEG Z890 Godlike
- MSI MEG Z890 Ace
- MSI MEG Z890 Unify X
- MSI MEG Z890 Carbon WiFi

#### CPU

The CPU provides the PCIe lanes both GPUs depend on. The latest AMD Ryzen and Intel Core CPUs both support PCIe Gen 5 and run AI workloads at similar speed. Look for at least 20 PCIe lanes — enough for two GPUs and one NVMe SSD with dedicated bandwidth for each. AMD's latest CPUs support up to 28 lanes, which leaves more headroom for a second SSD.

AMD Ryzen 9 or Intel Core i9 minimum. AMD Threadripper is the best option — it provides the PCIe lane count for full x16/x16 GPU operation plus additional NVMe drives, and the core count to keep up with parallel AI workloads.

#### RAM

System RAM holds models that get streamed into GPU memory and stages data between drives and GPUs. 32GB is the floor for any multi-GPU build.

64GB minimum. Builds running 100B-parameter LLMs or fine-tuning workloads should run 128GB. Tensor-parallel inference and FSDP2 fine-tuning stage large activation tensors through system memory.

#### Storage

Modern AI models are large and load slowly from disk. A single PCIe Gen 5 NVMe SSD handles most workloads. Combining multiple NVMe SSDs in a RAID array does not provide a meaningful performance boost for AI workloads, and adding M.2 drives can silently steal PCIe lanes from a GPU slot, so a single high-capacity drive is the better choice.

Single PCIe Gen 5 NVMe SSD, 4TB or larger. Quantized 70B and 120B LLMs each occupy 40–80GB of disk; serious model collections fill 2TB drives quickly.

#### Power Supply

A single NVIDIA GeForce RTX 5090 draws up to 575W under load. The PSU must be CEM 5.1 compliant, and each GPU requires a dedicated 12V-2x6 cable from the power supply — splicing or daisy-chaining cables across GPUs causes connector damage and instability. Always size up: running a power supply at 95% load shortens its life.

1600W minimum for two NVIDIA GeForce RTX 5090s plus an AMD Ryzen 9 9950X CPU. 1800W recommended for headroom. Paired RTX 5080s need 1200–1400W. 80 Plus Gold or better.

#### Case and Cooling

Two GPUs generate roughly twice the heat of one and need somewhere for it to go. A full-tower ATX or open-frame ATX chassis provides the volume. Examples of good cases include the Corsair 7000 and 9000 series, NZXT H9 Flow, Phanteks Enthoo Pro, and Thermaltake AX700 series. Air-cooled NVIDIA GeForce RTX 5090s stacked next to each other run hot — the top GPU starves the bottom GPU of intake.

### Build Tips and Common Pitfalls

* A multi-GPU PC has more failure modes than a single-GPU build. Most of them are avoidable.
* Seat the 12V-2x6 power cable fully into each GPU before installing the card in the motherboard slot. Plugging the power cable in with the GPU already mounted often leaves the connector partially seated, which causes thermal events under load. Connect the power outside the chassis, then install.
* Leave at least 35mm of slack on each GPU power cable. Tight bends fatigue the connector over time. Excessive bending damages the contacts and can cause the power cable to overheat.
* Update the motherboard BIOS before first boot. PCIe compatibility, CPU support, and memory training improvements ship in BIOS updates. A stale BIOS often manifests as one GPU not being detected.
* Verify PCIe lane allocation in BIOS after building. Adding a second M.2 NVMe drive can silently drop a GPU slot from x8 to x4 on most consumer motherboards. The BIOS PCIe status page shows the actual lane count assigned to each slot.
* Cable management is cooling. Loose cables block airflow between GPUs. Route cables behind the motherboard tray. Use motherboards that place fan headers and front-panel I/O on the back side of the PCB to keep the GPU corridor clear.
* Do not mix GPU power supplies. Both GPUs must be powered by the same PSU. Splitting GPU power across two PSUs creates ground-loop and timing issues.
