#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
set -eu

: "${HF_TOKEN:?Set HF_TOKEN before downloading the gated FLUX.2 Klein checkpoints.}"

. ./select_memory_profile.sh
FLUX_MEMORY_PROFILE=$(select_flux_memory_profile)
export FLUX_MEMORY_PROFILE

mkdir -p models/diffusion_models models/text_encoders models/vae models/loras

download_if_needed() {
  url="$1"
  file="$2"
  if [ -f "$file" ]; then
    echo "$file already exists, skipping."
  else
    curl -C - -L -H "Authorization: Bearer $HF_TOKEN" -o "$file" "$url"
  fi
}

flux2_url="https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files"

case "$FLUX_MEMORY_PROFILE" in
  128gb)
    klein_url="https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files"
    diffusion_url="https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/resolve/main/flux-2-klein-base-9b.safetensors"
    diffusion_file="models/diffusion_models/flux-2-klein-base-9b.safetensors"
    text_encoder_url="$klein_url/text_encoders/qwen_3_8b.safetensors"
    text_encoder_file="models/text_encoders/qwen_3_8b.safetensors"
    ;;
  64gb)
    klein_url="https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files"
    diffusion_url="$klein_url/diffusion_models/flux-2-klein-base-4b.safetensors"
    diffusion_file="models/diffusion_models/flux-2-klein-base-4b.safetensors"
    text_encoder_url="$klein_url/text_encoders/qwen_3_4b.safetensors"
    text_encoder_file="models/text_encoders/qwen_3_4b.safetensors"
    ;;
esac

echo "Using FLUX_MEMORY_PROFILE=$FLUX_MEMORY_PROFILE."
download_if_needed "$diffusion_url" "$diffusion_file"
download_if_needed "$text_encoder_url" "$text_encoder_file"
download_if_needed "$flux2_url/vae/flux2-vae.safetensors" \
  "models/vae/flux2-vae.safetensors"

ln -sfn "$(basename "$diffusion_file")" \
  "models/diffusion_models/flux-2-klein-base.safetensors"
ln -sfn "$(basename "$text_encoder_file")" \
  "models/text_encoders/qwen_3.safetensors"
