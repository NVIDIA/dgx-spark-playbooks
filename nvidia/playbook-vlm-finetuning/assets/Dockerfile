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

# Dockerfile stage to build GPU-accelerated ffmpeg
FROM nvcr.io/nvidia/pytorch:25.09-py3 AS ffmpeg-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        yasm \
        libx264-dev \
        libfaac-dev \
        libmp3lame-dev \
        libtheora-dev \
        libvorbis-dev \
        libxvidcore-dev \
        libxext-dev \
        libxfixes-dev \
        build-essential \
        git \
        pkg-config && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc-11 g++-11

ENV PATH=/usr/local/cuda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    NVCC=/usr/local/cuda/bin/nvcc \
    CC=/usr/bin/gcc-11 \
    CXX=/usr/bin/g++-11 \
    CUDAHOSTCXX=/usr/bin/g++-11 \
    FFMPEG_VERSION=4.4.6 \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN mkdir -p /deps && \
    cd /deps && \
    wget -q https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz && \
    tar -xf ffmpeg-${FFMPEG_VERSION}.tar.xz && \
    rm ffmpeg-${FFMPEG_VERSION}.tar.xz && \
    cd /deps/ffmpeg-${FFMPEG_VERSION} && \
    apt-get update && \
    apt-get install -y libdrm-dev && \
    ./configure \
        --prefix=/usr/local \
        --enable-nonfree \
        --enable-shared \
        --disable-static \
        --enable-libdrm \
        --enable-v4l2-m2m \
        --enable-gpl \
        --enable-libx264 \
        --extra-cflags="-I/usr/include/aarch64-linux-gnu" \
        --extra-ldflags="-L/usr/lib/aarch64-linux-gnu/tegra" \
        --nvccflags="-ccbin=/usr/bin/g++-11" \
        || (echo "---- tail ffbuild/config.log ----" && tail -n 200 ffbuild/config.log && exit 1) && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    echo "✅ FFmpeg installed:" && \
    ffmpeg -hide_banner -version | head -n 8 && \
    rm -rf /var/lib/apt/lists/*

# Dockerfile stage to compile decord from source
FROM nvcr.io/nvidia/pytorch:25.09-py3 AS decord-builder
COPY --from=ffmpeg-builder /usr/local /usr/local

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      cmake \
      ninja-build \
      pkg-config \
      python3-dev \
      python3-pip \
      gcc-11 \
      g++-11 \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc-11 \
    CXX=/usr/bin/g++-11 \
    PATH=/usr/local/bin:/lib/aarch64-linux-gnu:${PATH} \
    LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/:/lib/aarch64-linux-gnu/:/usr/local/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/lib/aarch64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}

RUN ln -sf /usr/lib/aarch64-linux-gnu/libnvcuvid.so.1 /usr/lib/aarch64-linux-gnu/libnvcuvid.so && \
    cp /usr/lib/aarch64-linux-gnu/libnvcuvid.so* /usr/local/cuda/lib64/ || true && \
    ln -sf /usr/lib/aarch64-linux-gnu/libnvcuvid.so.1 /usr/local/cuda/lib64/libnvcuvid.so && \
    echo "/usr/lib/aarch64-linux-gnu" > /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig && \
    python3 -m pip install --no-cache-dir --upgrade pip wheel build numpy && \
    python3 -m pip install --no-cache-dir --upgrade pip ninja && \
    apt-get update && \
    apt-get install -y --no-install-recommends libnvidia-decode-575 libnvidia-encode-575

RUN cd /workspace && \
    git clone --recursive https://github.com/dmlc/decord && \
    cmake -S /workspace/decord -B /workspace/decord/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_CUDA=ON \
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
        -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
        -DFFMPEG_ROOT=/usr/local \
        -DUSE_VIDEO_CODEC=OFF && \
    cd /workspace/decord/build && \
    ninja -j"$(nproc)" && \
    cd /workspace/decord/python && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel build && \
    python3 -m build --wheel

# Dockerfile for demo
FROM nvcr.io/nvidia/pytorch:25.09-py3
COPY --from=ffmpeg-builder /usr/local /usr/local
COPY --from=decord-builder /workspace/decord/python/dist/*.whl /tmp/wheels/

ARG HF_TOKEN

RUN pip install --no-cache-dir /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels && \
    apt-get update && \
    apt-get install -y libdrm2 libdrm-dev libx264-dev && \
    pip install streamlit timm wandb && \
    hf auth login --token $HF_TOKEN

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-13.0/
ENV CUDA_PATH=$CUDA_HOME
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
ENV CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

# install triton from source for latest blackwell support
RUN git clone https://github.com/triton-lang/triton.git && \
    cd triton && \
    git checkout c5d671f91d90f40900027382f98b17a3e04045f6 && \
    pip install -r python/requirements.txt && \
    pip install . && \
    cd ..

# install xformers from source for blackwell support
RUN git clone https://github.com/facebookresearch/xformers && \
    cd xformers && \
    git checkout 5146f2ab37b2163985c19fb4e8fbf6183e82f8ce && \
    git submodule update --init --recursive && \
    export TORCH_CUDA_ARCH_LIST="12.1" && \
    python setup.py install && \
    cd ..
# install unsloth without depedencies so we can build them from source
RUN pip install unsloth==2025.9.11 unsloth_zoo==2025.9.14 bitsandbytes==0.48.0

CMD ["/bin/bash"]


# docker run \
#     --rm \
#     --gpus=all \
#     --ipc=host \
#     --net=host \
#     --ulimit memlock=-1 \
#     --ulimit stack=67108864 \
#     -w $(pwd) \
#     -v $(pwd):$(pwd) \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     nvcr.io/nvidia/vllm:25.09-py3 \
#     vllm serve "unsloth/Qwen2.5-VL-7B-Instruct" --port "8000" --served-model-name "base-model" --max-model-len 16384 --gpu-memory-utilization 0.3 --async-scheduling --enable_prefix_caching