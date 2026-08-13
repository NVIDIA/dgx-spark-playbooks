# Wan2GP (WanGP) — AI video generation on DGX Spark

[Wan2GP](https://github.com/deepbeepmeep/Wan2GP) ("WanGP", by DeepBeepMeep) is a
memory-efficient front end for a large family of open video models — Wan 2.1 / 2.2,
LTX-2, Hunyuan Video, Qwen Image, Flux, and the **InfiniteTalk / MultiTalk**
audio-driven talking-head models. This playbook installs it natively on a DGX
Spark and reuses the same Blackwell PyTorch stack NVIDIA's ComfyUI playbook
provides, so the fragile ARM + CUDA 13 pieces are already solved.

## What you get

- A dedicated `~/wan2gp-env` virtualenv (your ComfyUI install is untouched).
- PyTorch 2.13.0+cu130 from the official cu130 wheel index (sm_121 / Blackwell).
- Wan2GP cloned to `~/Wan2GP`, with every dependency landmine handled for aarch64.
- A one-command launcher that serves the WanGP web UI on your LAN.

## Prerequisites

- DGX Spark (GB10) running DGX OS, Python 3.12.
- ~40 GB free disk for the environment; model weights download on first use and
  can be tens of GB more depending on which models you run.
- Outbound internet access (PyPI, the PyTorch cu130 index, and Hugging Face).

## Install

```bash
bash assets/setup.sh
```

The script installs system packages, creates the venv, installs the Blackwell
PyTorch wheels, clones Wan2GP, and layers its dependencies on top. The final step
prints a check confirming PyTorch sees the GB10 GPU (`cuda available: True`,
`sm_121`).

## Launch

```bash
bash assets/launch.sh
```

Then open the UI from any machine on your network:

```
http://<your-spark-hostname>.local:7860
```

Pick a model in the top dropdowns. For a talking-head clip, choose
**Infinitetalk** (or **Multitalk**), then a **Single Speaker** or **Multi Speakers**
variant, supply a reference image and one audio track per speaker, and Generate.
Model weights download automatically on first use.

## aarch64 / Blackwell specifics (why this playbook exists)

Installing Wan2GP straight from its `requirements.txt` fails on the Spark for a
handful of ARM-specific reasons. This playbook resolves each one:

- **PyTorch**: installed from the cu130 index rather than the requirements' pins,
  which resolve to x86 CUDA-12 wheels.
- **Attention**: SageAttention / FlashAttention / xformers have no ARM wheels and
  are skipped — Wan2GP falls back to PyTorch **sdpa** attention automatically.
- **torchcodec / onnxruntime-gpu**: Wan2GP pins fast-moving nightly builds that
  rotate off the index; the playbook installs current, ARM-compatible builds
  instead (torchcodec from cu130; CPU onnxruntime, since onnxruntime-gpu has no
  ARM wheel).
- **decord**: no ARM wheel exists; it is skipped (only used for video-*input*
  decoding — image + audio talking-head workflows do not need it).
- **rembg**: installed as the CPU build (its `[gpu]` extra requires
  onnxruntime-gpu).
- **audio-separator**: installed explicitly (needed by InfiniteTalk / MultiTalk
  vocal extraction; its default extra also pulls onnxruntime-gpu).
- **transformers**: pinned to `4.54.0` (Wan2GP's target). transformers 5.x ships
  a `higgs_audio_v2_tokenizer` that collides with Wan2GP's bundled OmniVoice TTS.
- **python3.12-dev / build-essential**: installed so Triton can JIT-compile
  kernels instead of rolling back to CPU.

## Performance on the Spark

The Spark's strength is capacity (128 GB unified memory), not raw memory
bandwidth, and video diffusion is bandwidth-bound — so expect generation to be
slower than a high-end discrete GPU. That is inherent to the LPDDR5X memory, not a
misconfiguration. The payoff is running large models and long clips that would not
fit on a smaller consumer card.

Two Spark-specific tips make a large difference:

- **Launch with `--profile 3`.** WanGP's default profile (4) *offloads* the model
  and streams weights between "reserved RAM" and "VRAM" every step. On the Spark's
  unified memory those are the same physical LPDDR5X, so that shuttle is wasted
  work — it dominates per-step time and can trigger out-of-memory failures on
  larger models. `--profile 3` keeps the whole model resident (easily within
  128 GB) and is dramatically faster:

  ```bash
  bash assets/launch.sh --profile 3
  ```

  (WanGP also has a TeaCache step-cache for extra speed; enable it in the UI's
  Advanced Mode if your version exposes it — the `--teacache` CLI flag is not
  available in all WanGP versions.)

- **If you hit an out-of-memory error mid-generation,** flush the Linux buffer
  cache, which can consume the unified memory pool on the Spark (a documented DGX
  Spark behaviour):

  ```bash
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  ```

Background-removal and audio-separation helpers run on CPU (onnxruntime-gpu has no
ARM build), which is fine for those auxiliary steps.

## Credits

Wan2GP is created and maintained by DeepBeepMeep:
https://github.com/deepbeepmeep/Wan2GP. This playbook only automates its
installation on DGX Spark.
