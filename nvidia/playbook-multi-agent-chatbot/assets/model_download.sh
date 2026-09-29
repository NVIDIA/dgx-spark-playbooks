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

validate_gguf() {
  local file="$1"
  local magic

  if [[ "$file" != *.gguf || ! -f "$file" ]]; then
    echo "Error: $file must be an existing .gguf file." >&2
    return 1
  fi

  magic="$(LC_ALL=C head -c 4 -- "$file")"

  if [ "$magic" != "GGUF" ]; then
    echo "Error: $file does not have a valid GGUF magic header." >&2
    return 1
  fi

  return 0
}

download_if_needed() {
  local url="$1"
  local file="$2"

  if [ -e "$file" ]; then
    if validate_gguf "$file"; then
      echo "$file already exists and is valid, skipping."
      return 0
    fi

    echo "Error: $file already exists but is not a valid GGUF file." >&2
    echo "Remove the invalid file and run the script again." >&2
    return 1
  fi

  if ! curl \
    --fail \
    --location \
    --output "$file" \
    "$url"; then
    echo "Error: failed to download $file." >&2
    echo "Removing the incomplete file: $file" >&2
    rm -f -- "$file"
    return 1
  fi

  if ! validate_gguf "$file"; then
    echo "Error: downloaded file is not a valid GGUF file: $file" >&2
    echo "Removing the invalid downloaded file: $file" >&2
    rm -f -- "$file"
    return 1
  fi

  return 0
}

download_if_needed "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q8_0.gguf" "deepseek-coder-6.7b-instruct.Q8_0.gguf"

download_if_needed "https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/resolve/main/Qwen3-Embedding-4B-Q8_0.gguf" "Qwen3-Embedding-4B-Q8_0.gguf"

# Comment next line if you want to use gpt-oss-20b
download_if_needed "https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-MXFP4.gguf" "gpt-oss-120b-MXFP4.gguf"

# Uncomment next line if you want to use gpt-oss-20b
# download_if_needed "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf" "gpt-oss-20b-MXFP4.gguf"

echo "All models downloaded."
