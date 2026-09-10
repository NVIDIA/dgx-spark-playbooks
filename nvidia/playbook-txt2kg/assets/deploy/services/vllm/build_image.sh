#!/bin/bash
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

# Use latest stable vLLM release for better compute capability 12.1 support
# Clone the vLLM GitHub repo and use latest stable release.
git clone https://github.com/vllm-project/vllm.git /tmp/vllm-tutorial
cd /tmp/vllm-tutorial
git checkout $(git describe --tags --abbrev=0)

# Build the docker image using official vLLM Dockerfile.
DOCKER_BUILDKIT=1 docker build . \
        --file docker/Dockerfile \
        --target vllm-openai \
        --build-arg CUDA_VERSION=12.8.1 \
        --build-arg max_jobs=8 \
        --build-arg nvcc_threads=2 \
        --build-arg RUN_WHEEL_CHECK=false \
        --build-arg torch_cuda_arch_list="10.0+PTX;12.1" \
        --build-arg vllm_fa_cmake_gpu_arches="100-real;121-real" \
        -t vllm/vllm-openai:deploy

# Clean up
cd /
rm -rf /tmp/vllm-tutorial
