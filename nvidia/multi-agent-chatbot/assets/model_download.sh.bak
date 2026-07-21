#!/usr/bin/env bash
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
set -euo pipefail

ROOT_DIR="$(pwd)"
MODELS_DIR="$ROOT_DIR/models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

download_if_needed() {
  url="$1"
  file="$2"
  if [ -f "$file" ]; then
    echo "$file already exists, skipping."
  else
    curl -C - -L -o "$file" "$url"
  fi
}

download_if_needed "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q8_0.gguf" "deepseek-coder-6.7b-instruct.Q8_0.gguf"

download_if_needed "https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/resolve/main/Qwen3-Embedding-4B-Q8_0.gguf" "Qwen3-Embedding-4B-Q8_0.gguf"

# Comment next three lines if you want to use gpt-oss-20b
download_if_needed "https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-mxfp4-00001-of-00003.gguf" "gpt-oss-120b-mxfp4-00001-of-00003.gguf"

download_if_needed "https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-mxfp4-00002-of-00003.gguf" "gpt-oss-120b-mxfp4-00002-of-00003.gguf"

download_if_needed "https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-mxfp4-00003-of-00003.gguf" "gpt-oss-120b-mxfp4-00003-of-00003.gguf"

# Uncomment next line if you want to use gpt-oss-20b
# download_if_needed "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-mxfp4.gguf" "gpt-oss-20b-mxfp4.gguf"

echo "All models downloaded."
