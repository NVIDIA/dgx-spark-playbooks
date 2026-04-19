---
name: dgx-spark-spark-reachy-photo-booth
description: AI augmented photo booth using the DGX Spark and Reachy Mini. — on NVIDIA DGX Spark. Use when setting up spark-reachy-photo-booth on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/spark-reachy-photo-booth/README.md -->
# Spark & Reachy Photo Booth

> AI augmented photo booth using the DGX Spark and Reachy Mini.

![Teaser](assets/teaser.jpg)

Spark & Reachy Photo Booth is an interactive and event-driven photo booth demo that combines the **DGX Spark™** with the **Reachy Mini** robot to create an engaging multimodal AI experience. The system showcases:

- **A multi-modal agent** built with the `NeMo Agent Toolkit`
- **A ReAct loop** driven by the `openai/gpt-oss-20b` LLM powered by `TensorRT-LLM`
- **Voice interaction** based on `nvidia/riva-parakeet-ctc-1.1B` and `hexgrad/Kokoro-82M`
- **Image generation** with `black-forest-labs/FLUX.1-Kontext-dev` for image-to-image restyling

**Outcome**: You'll deploy a complete photo booth system on DGX Spark running multiple inference models locally — LLM, image generation, speech recognition, speech generation, and computer vision — all without cloud dependencies. The Reachy robot interacts with users through natural conversation, captures photos, and generates custom images based on prompts, demonstrating real-time multimodal AI processing on edge hardware.

**Full playbook**: `/Users/jkneen/Documents/GitHub/dgx-spark-playbooks/nvidia/spark-reachy-photo-booth/README.md`
<!-- GENERATED:END -->
