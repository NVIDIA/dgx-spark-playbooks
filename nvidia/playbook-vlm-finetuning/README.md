# Fine-Tune Vision Language Models

> Image and video understanding adapted to your own specialized tasks

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook shows how to fine-tune Vision-Language Models (VLMs) for image and video understanding on your **hardware platform**. You adapt multimodal models to specialized visual tasks with LoRA, preference optimization, and interactive Streamlit UIs for training and side-by-side inference.

Two recipes are included:

- **Image VLM fine-tuning** — Qwen2.5-VL-7B for wildfire detection from satellite imagery, using GRPO (Generalized Reward Preference Optimization)
- **Video VLM fine-tuning** — InternVL3-8B for dangerous driving analysis and structured metadata generation from driving videos

Both paths use Docker for a reproducible environment with the libraries needed for image and video workflows.

## What you'll accomplish

You'll have fine-tuned VLM checkpoints for image and/or video tasks on your **hardware platform**, with interactive UIs for training and comparing base vs fine-tuned models.

- Fine-tune Qwen2.5-VL for wildfire detection with GRPO and LoRA
- Fine-tune InternVL3 for dangerous driving analysis and structured video metadata
- Run Streamlit demos for base and fine-tuned inference side by side
- Use a shared Docker image for both image and video recipes

## What to know before starting

**Required:**

- Basic Linux command line and Docker container usage
- Familiarity with Hugging Face model downloads and tokens
- Fine-tuning concepts: LoRA, checkpoints, and GPU memory constraints

**Optional:**

- Experience with vision-language models and multimodal prompts
- Dataset prep for image classification or video + JSONL metadata
- Weights & Biases for training monitoring

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Docker image `vlm_demo` built from playbook assets | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient memory for 7B–8B VLM training and inference with vision encoders loaded
- Enough free disk for model downloads and datasets (plan for tens of GB)
- No other heavy GPU workloads running during training or inference

**Software requirements**

- NVIDIA Docker / NVIDIA Container Toolkit: `docker --version` and `nvidia-smi` inside a GPU container
- Network access to download models from Hugging Face and datasets (for example Kaggle)
- Hugging Face access token (`HF_TOKEN`) for model downloads
- Web browser access to ports `8501` (Streamlit) and `8888` (Jupyter, video recipe)
- Weights & Biases account for training monitoring (optional but recommended)

## Ancillary files

All required assets can be found [in this playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-vlm-finetuning/).

- `assets/Dockerfile` / `assets/launch.sh` — Build and run the shared VLM fine-tuning container
- `assets/ui_image/` — Streamlit app, configs, and scripts for image VLM (Qwen2.5-VL + GRPO)
- `assets/ui_video/` — Streamlit app, training notebook, and configs for video VLM (InternVL3)
- `assets/README.md` — Asset-level overview of both recipes

## Time & risk

- **Estimated time:** 2 HOURS for setup plus a short image training run with the default UI settings; video fine-tuning for usable quality typically needs a much longer run (often a day or more depending on dataset and hyperparameters)
- **Risk level:** Medium
  - Docker permission issues may require a group change and new login session
  - Large model and dataset downloads need significant disk space and time
  - Training sustains high GPU and memory use; residual cache can cause pressure after UI or notebook sessions
  - Image dataset prep may require a Kaggle login and accepting dataset terms; video datasets need correct folder layout and `metadata.jsonl`
- **Rollback:** Stop and remove Docker containers; delete downloaded models, datasets, and checkpoints under the mounted assets directory if needed
- **Last Updated:** 08/03/2026
  - Fine-tune image and video VLMs with Docker-based Streamlit and notebook workflows on supported hardware platforms

## Instructions

> [!NOTE]
> These instructions target **Linux** with a GPU-enabled Docker container. Run training and Streamlit steps inside the container unless noted otherwise. The same container supports both the image and video recipes.

## Step 1. Configure Docker permissions

To manage containers without `sudo`, your user must be in the `docker` group. If you skip this step, run Docker commands with `sudo`.

Open a terminal and test Docker access:

```bash
docker ps
```

If you see a permission-denied error (for example, permission denied while trying to connect to the Docker daemon socket), add your user to the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Clone the repository

Clone the playbook assets and open the assets directory:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd client-hardware-playbooks/nvidia/playbook-vlm-finetuning/assets
```

## Step 3. Build the Docker container

Export your Hugging Face token, then build the image. Warnings during the build are expected and can be ignored.

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
docker build --build-arg HF_TOKEN=$HF_TOKEN -t vlm_demo .
```

## Step 4. Run the Docker container

```bash
sh launch.sh
cd /vlm_finetuning
```

> [!NOTE]
> The same Docker container and launch commands work for both image and video VLM recipes. The image includes dependencies such as FFmpeg, Decord, and libraries used by both workflows.

Choose **Option A** (image) or **Option B** (video) below. You can run both over time in the same container, but stop Streamlit or Jupyter and free memory before switching heavy workloads.

## Step 5. [Option A] Image VLM fine-tuning (wildfire detection)

#### 5.1. Model download

```bash
hf download Qwen/Qwen2.5-VL-7B-Instruct
```

If you already have a fine-tuned checkpoint, place it in the `saved_model/` folder. Checkpoint numbers can differ. For a comparative analysis against the base model only, skip ahead to **Fine-tuned model inference**.

#### 5.2. Download the wildfire dataset

This recipe uses a wildfire prediction dataset with satellite and aerial imagery for binary classification (wildfire vs no wildfire).

```bash
mkdir -p ui_image/data
cd ui_image/data
```

Open the [Wildfire Prediction Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset) on Kaggle. Use **Download Via** → **cURL**, copy the curl command, then run it in the container.

> [!NOTE]
> You must be logged into Kaggle and may need to accept the dataset terms before the download link works.

```bash
## Paste and run the curl command from Kaggle, then unzip
unzip -qq wildfire-prediction-dataset.zip
rm wildfire-prediction-dataset.zip
cd ..
```

#### 5.3. Base model inference

Start the Streamlit demo to evaluate the base model before fine-tuning:

```bash
streamlit run Image_VLM.py
```

Open `http://localhost:8501/` (or `http://<HARDWARE_IP>:8501` from another device).

On first access, the backend starts vLLM servers for the base model. A spinner appears while vLLM comes up; this can take up to about 15 minutes.

Scroll to the **Image Inference** section. A sample satellite image is pre-loaded. Enter a prompt and select **Generate**. The left panel shows the base model; the right panel stays empty until a fine-tuned model is available. Example prompt:

`Identify if this region has been affected by a wildfire`

The base model typically struggles on this domain-specific task. Use GRPO fine-tuning next to improve accuracy and add structured reasoning.

#### 5.4. GRPO fine-tuning

With the Streamlit demo running, scroll to the **GRPO Training** section.

Configure fine-tuning method and LoRA parameters:

- **Finetuning Method:** Full Finetuning or LoRA
- **LoRA Parameters:** rank (8–64) and alpha (8–64)

Enable the VLM layers you want to train. For best quality, leave all options on (this increases training time).

Training parameters:

- **Steps:** 1–1000
- **Batch Size:** 1, 2, 4, 8, or 16
- **Learning Rate:** 1e-6 to 1e-2
- **Optimizer:** AdamW or Adafactor

GRPO reward settings:

- **Format Reward:** 2.0 (proper reasoning format)
- **Correctness Reward:** 5.0 (correct answers)
- **Number of Generations:** 4 (preference optimization)

Select **Start Finetuning**. Allow about 15 minutes for the model to load and for metadata to appear in the UI. Loss, step, and GRPO rewards update in a live table as training runs.

The default configuration is a reasonable starting point: about 100 steps can take up to about 2 hours. Longer runs (for example around 1000 steps) can take much longer and may improve accuracy further.

When training finishes, the script merges LoRA weights into the base model. Merging can take about 5 minutes after the last step.

To interrupt training, use **Stop Finetuning**. Use that control only to interrupt a run; it does not guarantee that checkpoints are stored or that LoRA adapters are fully merged.

After you stop training, the UI brings up vLLM servers for the base model and the newly fine-tuned model.

#### 5.5. Fine-tuned model inference

Compare the base and fine-tuned models. If Streamlit is not already running:

```bash
streamlit run Image_VLM.py
```

Wait about 15 minutes for the vLLM servers to come up if they are not already loaded.

In **Image Inference**, enter a prompt and select **Generate**. Example:

`Identify if this region has been affected by a wildfire`

With sufficient training, the fine-tuned model should reason in markdown and end with a concise bolded answer.

## Step 6. [Option B] Video VLM fine-tuning (driver behavior analysis)

Inside the same container, open the video UI directory:

```bash
cd /vlm_finetuning/ui_video
```

#### 6.1. Prepare your video dataset

Structure the dataset as follows. `metadata.jsonl` must contain one structured JSON record per video:

```text
dataset/
├── videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── metadata.jsonl
```

#### 6.2. Model download

```bash
hf download OpenGVLab/InternVL3-8B
```

#### 6.3. Base model inference

Evaluate base InternVL3-8B before fine-tuning:

```bash
## cd into /vlm_finetuning/ui_video if you are not already there
streamlit run Video_VLM.py
```

Open `http://localhost:8501/` (or `http://<HARDWARE_IP>:8501`).

On first access, the backend loads the base model through Hugging Face. A spinner appears; loading can take up to about 10 minutes.

Select a video from the dashcam gallery (green open icon). The video plays for reference. Enter a prompt and select **Generate**. The left panel shows the base model; the right panel stays empty until a fine-tuned model is available. Example prompt:

`Analyze the dashcam footage for unsafe driver behavior`

Before training, stop the Streamlit demo with `Ctrl+C` in the terminal.

> [!NOTE]
> To clear buffer cache after stopping the Streamlit UI (outside the container):
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

#### 6.4. Run the training notebook

```bash
cd train
jupyter notebook video_vlm.ipynb
```

Open Jupyter at `http://localhost:8888` (or `http://<HARDWARE_IP>:8888`). Set the dataset path in the notebook:

```python
dataset_path = "/path/to/your/dataset"
```

Key training knobs:

- **Model:** InternVL3-8B
- **Video Frames:** 12 to 16 frames per video
- **Sampling Mode:** Uniform temporal sampling
- **LoRA:** Parameter-efficient updates for large-scale fine-tuning
- **Hyperparameters:** Adjustable suite for video VLM fine-tuning

Usable video quality often needs a long training window (on the order of a day or more) because of spatio-temporal sequence processing. Monitor metrics in the notebook as training runs.

When finished, shut down the Jupyter kernel and stop the Jupyter server with `Ctrl+C`.

> [!NOTE]
> To clear buffer cache after stopping Jupyter (outside the container):
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

#### 6.5. Fine-tuned model inference

Compare base and fine-tuned models. If Streamlit is not already running:

```bash
## cd back to /vlm_finetuning/ui_video if needed
streamlit run Video_VLM.py
```

Open `http://localhost:8501/`. With sufficient training, the fine-tuned model should surface salient events and emit structured output that matches the schema you trained—suitable for export into a video analytics database. Try additional gallery videos as needed.

## Step 7. Cleanup (optional)

Stop running Streamlit or Jupyter with `Ctrl+C`. Exit the container, then remove the local image if you no longer need it:

```bash
docker rmi vlm_demo
```

> [!WARNING]
> Removing the image deletes the local Docker build. Rebuild with the Dockerfile in `assets/` before running this playbook again. Downloaded models, datasets, and checkpoints under the mounted assets directory are separate; delete those only if you intend to free disk space.

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every platform listed in this playbook.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| "permission denied" when running Docker | All hardware platforms | User not in the `docker` group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Container fails to start with GPU error | All hardware platforms | NVIDIA Container Toolkit not configured | Configure the NVIDIA Container Toolkit for Docker and confirm `nvidia-smi` works in a GPU container |
| Hugging Face download fails or auth error | All hardware platforms | Missing or invalid `HF_TOKEN` | Export a valid Hugging Face token; pass `--build-arg HF_TOKEN=$HF_TOKEN` when building; run `hf auth login` inside the container if needed |
| Kaggle dataset curl fails | All hardware platforms | Not logged in or terms not accepted | Sign in to Kaggle, accept dataset terms, regenerate the cURL download command, and retry |
| Streamlit unreachable on port 8501 | All hardware platforms | App not running or port blocked | Confirm `streamlit run` is active; open `http://localhost:8501` or `http://<HARDWARE_IP>:8501` |
| Jupyter unreachable on port 8888 | All hardware platforms | Notebook server not running or port blocked | Confirm `jupyter notebook` is running in `ui_video/train`; open `http://localhost:8888` or `http://<HARDWARE_IP>:8888` |
| vLLM / model load spinner runs a long time | All hardware platforms | First-time model load or cold start | Wait up to about 15 minutes for image (vLLM) or about 10 minutes for video; confirm GPU is free and disk has the model weights |
| Training OOM / memory pressure during train or infer | All hardware platforms | Other GPU jobs, residual cache, or workload exceeds available memory | Stop other GPU processes; bring down Streamlit or Jupyter before the next heavy step; flush buffer cache (see note); lower batch size, frames, or LoRA scope |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
