# Corporate Bullshit Agentic Generator (CBAG)

> Generate a lip-synced talking-head video of someone delivering AI-written corporate buzzword nonsense — 100% locally on your DGX Spark. It's a remake of the old known project with an AI skew and demostrate voice cloning and fake video also for cybersecurity purposes.

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

CBAG is a fully local, GPU-native demo that turns a topic and a persona into a short video of a "person" delivering confident corporate nonsense. It chains four model families on a single DGX Spark:

1. **Buzzword grammar** — a deterministic generator seeds the nonsense (no GPU).
2. **LLM refine** — a local Ollama model (default `qwen2.5:3b`) rewrites it into an on-persona monologue (visionary CEO, McKinsey consultant, startup founder, thought leader).
3. **Voice** — Kokoro default voices, or your **own cloned voice** from a ~10 s sample (Qwen3-TTS).
4. **Talking-head video** — ComfyUI + Sonic + Stable Video Diffusion animate a portrait (a default face, or your uploaded photo) to lip-sync the speech.

Everything runs in one `docker compose` stack, exposed as a small web app. No cloud, no API tokens — a fun, honest stress-test of the Spark's unified memory (a 20B-class LLM, two speech models, and a video diffusion stack, warm at once).

## What you'll accomplish

You'll install CBAG on your DGX Spark and generate a talking-head video from a topic + persona in your browser, optionally in your own cloned voice.

## What to know before starting

- Basic command line and `git` usage
- Familiarity with Docker / Docker Compose and GPU containers
- That model weights are downloaded from their official sources on first install

## Prerequisites

**Hardware Requirements:**
- NVIDIA Grace Blackwell GB10 Superchip system (DGX Spark / Dell GB10), 128 GB unified memory
- At least ~50 GB available storage (container images + model weights)

**Software Requirements:**
- DGX OS / Ubuntu 24.04 with CUDA 13
- Docker + Docker Compose v2 + NVIDIA Container Toolkit: `docker compose version`
- Git: `git --version`
- A web browser with access to `https://<SPARK_IP>:8443`

## Ancillary files

All assets live in the CBAG repository (Apache-2.0): **https://github.com/vitorallo/CBAG**

Model weights are **not** bundled — `scripts/fetch-models.sh` downloads each from its official source at install time and verifies it by SHA-256. See the repo's `MODELS.md` for every model, its source, and its license.

> **Licensing note:** the talking-head video models (Stable Video Diffusion and Sonic) are licensed for **non-commercial** use only, so the talking-head feature of this demo is non-commercial. The text and voice stages use permissive (Apache-2.0/MIT) models. See `MODELS.md`.

## Time & risk

* **Estimated time:** 30–45 minutes (first build + model downloads)
* **Risk level:** Medium
  * Model downloads are large (~15–20 GB total) and may fail on flaky networks — `build.sh` is idempotent, just re-run it.
  * Port 443 is often taken by the pre-installed Dell demo, so CBAG serves HTTPS on **8443**.
* **Rollback:** `./scripts/down.sh` stops the stack; `./scripts/down.sh --purge` also removes the model volumes.
* **Last Updated:** 06/21/2026
  * Initial release.

## Instructions

## Step 1. Verify system prerequisites

Confirm your DGX Spark meets the requirements.

```bash
uname -m                  # aarch64
docker compose version    # Compose v2
nvidia-smi                # GPU detected
df -h .                   # ~50 GB free
```

## Step 2. Clone CBAG

```bash
git clone https://github.com/vitorallo/CBAG.git
cd CBAG
```

## Step 3. Build (preflight + auto-configure + images + weights)

```bash
./scripts/build.sh
```

This auto-detects your box's LAN/Tailscale IPs and writes `.env`, generates a self-signed TLS cert for those addresses, builds the container images, and downloads all model weights from their official sources (each verified by SHA-256). No Hugging Face token is required for the default sources.

## Step 4. Start the stack

```bash
./scripts/run.sh
```

It waits for the services to become healthy and prints the URL, e.g. `https://<your-spark-ip>:8443`.

## Step 5. Generate

Open the printed **https://** URL. On first visit, accept the self-signed certificate once (Advanced → Proceed) — HTTPS is required so the in-browser microphone (for voice cloning) works.

Pick a **topic** and a **persona**, then **Generate**. The text appears first, then the audio, then the talking-head video. Optionally upload a portrait (≥512 px) or record/upload a voice sample to clone your own voice.

Stop the stack any time with `./scripts/down.sh`.

## Troubleshooting

- **Port 443 already in use** — expected on a Spark (the pre-installed Dell demo owns it); CBAG serves HTTPS on **8443**.
- **Microphone is blocked** — you must use the `https://…:8443` URL (a secure context), not `http://…:8000`.
- **First voice-clone is slow** — the Qwen3-TTS model (~4 GB) downloads once on first use; the default Kokoro voices are instant.
- **A weight download failed** — re-run `./scripts/build.sh` (it skips what's already present and verifies SHA-256).
- **Render takes a while** — talking-head render time scales with audio length; choose a *short* length for a snappy demo.
