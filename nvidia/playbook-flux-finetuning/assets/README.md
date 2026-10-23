# FLUX.2 Klein Fine-tuning with LoRA

This project demonstrates fine-tuning FLUX.2 Klein using concept LoRA (Low-Rank Adaptation) with AI Toolkit for custom image generation. The demo includes training on custom concepts and inference through ComfyUI.

## Overview

The project includes:
- **FLUX.2 Klein Fine-tuning**: LoRA-based fine-tuning with AI Toolkit
- **Custom Concept Training**: Train on "tjtoy" toy and "sparkgpu" GPU
- **ComfyUI Integration**: Intuitive workflows for inference with custom models
- **Docker Support**: Complete containerized environment

## Contents
1. [Model Download](#1-model-download)
2. [Base Model Inference](#2-base-model-inference)
3. [Dataset Preparation](#3-dataset-preparation)
4. [Training](#4-training)
5. [Finetuned Model Inference](#5-finetuned-model-inference)

## 1. Model Download

### 1.1 Huggingface Authentication

You will have to be granted access to the FLUX.2 Klein model since it is gated. Go to the [Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) for a 64 GB system or the [Klein 9B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) for a 128 GB system to accept the terms and gain access to the checkpoints.

If you do not have a `HF_TOKEN` already, follow the instructions [here](https://huggingface.co/docs/hub/en/security-tokens) to generate one. Authenticate your system by replacing your generated token in the following command.

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
```

### 1.2 Download the pre-trained checkpoints

```bash
cd flux-finetuning/assets

# script to download (can take about a total of 15-60 mins, based on your internet speed)
sh download.sh
```

The script detects physical memory and downloads the required FLUX.2 Klein model, Qwen text encoder, and VAE for inference.
- **64 GB system**: FLUX.2 Klein 4B and Qwen3-4B
- **128 GB system**: FLUX.2 Klein 9B and Qwen3-8B
- `flux2-vae.safetensors`

Verify that your `models/` directory follows this structure after downloading the checkpoints.

```
models/
├── diffusion_models/
│   └── flux-2-klein-base.safetensors
├── loras/
├── text_encoders/
│   └── qwen_3.safetensors
└── vae/
    └── flux2-vae.safetensors
```

AI Toolkit downloads its training copy of the model into `hf_cache/` when training starts.

### 1.3 (Optional) Using fine-tuned checkpoints

If you already have fine-tuned LoRAs, place them inside `models/loras`. If you do not have one yet, proceed to the [Training](#training) section for more details.

## 2. Base Model Inference

Let's begin by generating an image using the base FLUX.2 Klein model on 2 concepts we am interested in, Toy Jensen and DGX Spark.

### 2.1 Spin up the docker container

```bash
# Build the inference docker image
docker build -f Dockerfile.inference -t flux2-comfyui .

# Launch the ComfyUI container (ensure you are inside flux-finetuning/assets)
sh launch_comfyui.sh
```
Access ComfyUI at `http://localhost:8188` to generate images with the base model. Do not select any pre-existing template.

### 2.2 Load the base workflow

Find the workflow section on the left-side panel of ComfyUI (or press `w`). Upon opening it, you should find two existing workflows loaded up. For the base Flux model, let's load the `base_flux2.json` workflow. After loading the json, you should see ComfyUI load up the workflow.

### 2.3 Fill in the prompt for your generation

Provide your prompt in the `Prompt` block. For example, we will use `tjtoy toy holding sparkgpu gpu in a datacenter`. You can expect the generation to take a few minutes since it is compute intesive to create high-resolution 1024px images.

For the provided prompt and random seed, the base Flux model generated the following image. Although the generation has good quality, it fails to understand the custom characters and concepts we would like to generate.

<figure>
  <img src="flux_assets/before_workflow.png" alt="Base model workflow" width="1000"/>
  <figcaption>Base FLUX.2 Klein model workflow without custom concept knowledge</figcaption>
</figure>

After playing around with the base model, you have 2 possible next steps.
* If you already have fine-tuned LoRAs placed inside `models/loras/`, please skip to [Load the finetuned workflow](#52-load-the-finetuned-workflow) section.
* If you wish to train a LoRA for your custom concepts, first make sure that the ComfyUI inference container is brought down before proceeding to train. You can bring it by interrupting the terminal with `Ctrl+C` keystroke.

> **Note**: To clear out any extra occupied memory from your system, execute the following command outside the container after interrupting the ComfyUI server.
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## 3. Dataset Preparation

Let's prepare our dataset to perform concept LoRA fine-tuning on FLUX.2 Klein. However, if you wish to continue with the provided dataset of Toy Jensen and DGX Spark, feel free to skip to the [Training](#training) section. This dataset is a collection of public assets accessible via Google Images.

### 3.1 Data collection

You will need to prepare a dataset of all the concepts you would like to generate, and about 5-10 images for each concept. For this example, we would like to generate images with 2 concepts.

#### TJToy Concept
- **Trigger phrase**: `tjtoy toy`
- **Training images**: 6 high-quality images of custom toy figures
- **Use case**: Generate images featuring the specific toy character in various scenes

#### SparkGPU Concept
- **Trigger phrase**: `sparkgpu gpu`
- **Training images**: 7 images of custom GPU hardware
- **Use case**: Generate images featuring the specific GPU design in different contexts

### 3.2 Format the dataset

Create a folder for each concept with it's corresponding name, and place it inside the `flux_data` directory. In our case, we have used `sparkgpu` and `tjtoy` as our concepts, and placed a few images inside each of them. After preparing the dataset, the structure inside `flux_data` should mimic the following.

```
flux_data/
├── concept_1/
│   ├── 1.png
│   ├── 2.jpg
    └── ...
└── concept_2/
    ├── 1.jpeg
    ├── 2.jpg
    └── ...
```

### 3.3 Update the data config

Now, let's modify the matching file under `config/` to reflect the concepts chosen. Ensure that you update/create entries for each of your concept, by modifying the `folder_path`, `default_caption`, and `trigger_word` fields. For better performance in finetuning, it is a good practice to append a class token to your concept name (like `toy` or `gpu`).

## 4. Training

### 4.1 Build the docker image

```bash
# Build the inference docker image
docker build -f Dockerfile.train -t flux2-train .
```

### 4.2 Setup the training command

Launch training by executing the following command. This train command automatically stores checkpoints in the `models/loras/` directory.

```bash
sh launch_train.sh
```

The launcher selects Klein 4B on 64 GB systems and Klein 9B on 128 GB systems. Both profiles run 1,000 optimizer steps and create the following stable LoRA path for the supplied workflow.

```
models/loras/flux2_klein_tjtoy_sparkgpu.safetensors
```

Feel free to play around with the other hyperparameters in the selected file under `config/` to find the best settings for your dataset. Some notable parameters to tune include:
- **Network Type**: LoRA with dimension and alpha 256
- **Learning Rate**: 0.0001 with AdamW 8-bit
- **Steps**: 1,000
- **Batch Size**: 2
- **Resolution**: 1024x1024
- **Mixed Precision**: bfloat16
- **Optimizations**: Torch compile, cached latents, and cached text embeddings

## 5. Finetuned Model Inference

Now let's generate images using our finetuned LoRAs!

### 5.1 Spin up the docker container

```bash
# Build the inference docker image, if you haven't already
docker build -f Dockerfile.inference -t flux2-comfyui .

# Launch the ComfyUI container (ensure you are inside flux-finetuning/assets)
sh launch_comfyui.sh
```
Access ComfyUI at `http://localhost:8188` to generate images with the finetuned model. Do not select any pre-existing template.

### 5.2 Load the finetuned workflow

Find the workflow section on the left-side panel of ComfyUI (or press `w`). Upon opening it, you should find two existing workflows loaded up. For the finetuned Flux model, let's load the `finetuned_flux2.json` workflow. After loading the json, you should see ComfyUI load up the workflow.

### 5.3 Fill in the prompt for your generation

Provide your prompt in the `Prompt` block. Now let's incorporate our custom concepts into our prompt for the finetuned model. For example, we will use `tjtoy toy holding sparkgpu gpu in a datacenter`. You can expect the generation to take a few minutes since it is compute intesive to create high-resolution 1024px images.

For the provided prompt and random seed, the finetuned Flux model generated the following image. Unlike the base model, we can see that the finetuned model can generate multiple concepts in a single image.

<figure>
  <img src="flux_assets/after_workflow.png" alt="After Fine-tuning" width="1000"/>
  <figcaption>Finetuned FLUX.2 Klein model with custom concept knowledge</figcaption>
</figure>

### 5.4 (Optional) Tuning your generations

ComfyUI exposes several fields to tune and change the look and feel of the generated images. Here are some parameters to look out for in the workflow.

1. **LoRA weights**: Change your trained LoRA file in the `Load LoRA Model Only` plugin, and even tune its strengths
2. **Adjust resolution**: Modify the `Width` and `Height` controls for other resolutions
3. **Random seed**: Change the noise seed in the `RandomNoise` plugin for alternative images with the same prompt
4. **Tune sampling**: Modify the sampler, scheduler and steps as necessary


## Credits

This project uses the following open-source repositories:
- [AI Toolkit](https://github.com/ostris/ai-toolkit) repository by `ostris` for FLUX.2 Klein fine-tuning.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI.git) repository by `comfyanonymous` for FLUX.2 Klein inference.
