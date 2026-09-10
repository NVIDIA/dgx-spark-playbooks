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
download_if_needed() {
  url="$1"
  file="$2"
  if [ -f "$file" ]; then
    echo "$file already exists, skipping."
  else
    curl -C - -L -H "Authorization: Bearer $HF_TOKEN" -o "$file" "$url"
  fi
}

download_if_needed "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors" "models/vae/ae.safetensors"
download_if_needed "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors" "models/checkpoints/flux1-dev.safetensors"
download_if_needed "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" "models/text_encoders/clip_l.safetensors"
download_if_needed "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" "models/text_encoders/t5xxl_fp16.safetensors"