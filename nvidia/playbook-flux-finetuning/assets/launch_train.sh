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
set -eu

: "${HF_TOKEN:?Set HF_TOKEN before starting training.}"

assets_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$assets_dir/hf_cache" "$assets_dir/models/loras"

. "$assets_dir/select_memory_profile.sh"
FLUX_MEMORY_PROFILE=$(select_flux_memory_profile)
export FLUX_MEMORY_PROFILE

case "$FLUX_MEMORY_PROFILE" in
  128gb)
    config_file=flux2_klein_9b.yaml
    lora_file=flux2_klein_9b_tjtoy_sparkgpu/flux2_klein_9b_tjtoy_sparkgpu.safetensors
    ;;
  64gb)
    config_file=flux2_klein_4b.yaml
    lora_file=flux2_klein_4b_tjtoy_sparkgpu/flux2_klein_4b_tjtoy_sparkgpu.safetensors
    ;;
esac

echo "Using FLUX_MEMORY_PROFILE=$FLUX_MEMORY_PROFILE ($config_file)."

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --net=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN \
  -e HF_HOME=/workspace/ai-toolkit/hf_cache \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -v "$assets_dir/config:/workspace/ai-toolkit/config/playbook:ro" \
  -v "$assets_dir/flux_data:/workspace/ai-toolkit/flux_data" \
  -v "$assets_dir/hf_cache:/workspace/ai-toolkit/hf_cache" \
  -v "$assets_dir/models/loras:/workspace/ai-toolkit/output" \
  flux2-train \
  python run.py "config/playbook/$config_file"

ln -sfn "$lora_file" "$assets_dir/models/loras/flux2_klein_tjtoy_sparkgpu.safetensors"
