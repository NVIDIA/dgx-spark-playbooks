#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
CMD="accelerate launch \
    --num_processes=1 --num_machines=1 --mixed_precision=bf16 \
    --main_process_ip=127.0.0.1 --main_process_port=29500 \
    --num_cpu_threads_per_process=2 \
    flux_train_network.py \
    --pretrained_model_name_or_path=models/checkpoints/flux1-dev.safetensors \
    --clip_l=models/text_encoders/clip_l.safetensors \
    --t5xxl=models/text_encoders/t5xxl_fp16.safetensors \
    --ae=models/vae/ae.safetensors \
    --dataset_config=flux_data/data.toml \
    --output_dir=models/loras/ \
    --prior_loss_weight=1.0 \
    --output_name=flux_dreambooth \
    --save_model_as=safetensors \
    --network_module=networks.lora_flux \
    --network_dim=256 \
    --network_alpha=256 \
    --learning_rate=1.0 \
    --optimizer_type=Prodigy \
    --lr_scheduler=cosine_with_restarts \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --sdpa \
    --max_train_epochs=100 \
    --save_every_n_epochs=25 \
    --mixed_precision=bf16 \
    --guidance_scale=1.0 \
    --timestep_sampling=flux_shift \
    --model_prediction_type=raw \
    --torch_compile \
    --persistent_data_loader_workers \
    --cache_latents \
    --cache_latents_to_disk \
    --cache_text_encoder_outputs \
    --cache_text_encoder_outputs_to_disk"

docker run -it \
    --rm \
    --gpus all \
    --ipc=host \
    --net=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd)/flux_data:/workspace/sd-scripts/flux_data \
    -v $(pwd)/models/vae:/workspace/sd-scripts/models/vae \
    -v $(pwd)/models/loras:/workspace/sd-scripts/models/loras \
    -v $(pwd)/models/checkpoints:/workspace/sd-scripts/models/checkpoints \
    -v $(pwd)/models/text_encoders:/workspace/sd-scripts/models/text_encoders \
    flux-train \
    bash -c "$CMD"