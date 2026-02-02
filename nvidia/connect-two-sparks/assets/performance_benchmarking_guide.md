# DGX Spark User Performance Guide

This repository contains benchmarking information, setup instructions, and example runs for evaluating AI workloads on NVIDIA DGX Spark.

It covers a wide range of frameworks and workloads including large language models (LLMs), diffusion models, fine-tuning, and more using tools such as TensorRT-LLM, vLLM, SGLang, Llama.cpp, and others.

Before running any benchmarks, ensure the following prerequisites are met for your selected workload:

## Prerequisites

- Access to a DGX Spark (or 2x DGX Spark)
- Docker and NVIDIA Container Toolkit
- A valid Hugging Face Token

## This guide includes benchmarking instructions for:

### Single Spark
- **[TensorRT-LLM (TRT-LLM)](#tensorrt-llm-trt-llm)**
  - [Offline](#offline-benchmark)
  - [Online](#online-benchmark)
- **[vLLM](#vllm)**
  - [Offline](#offline-benchmark-1)
  - [Online](#online-benchmark-1)
- **[SGLang](#sglang)**
  - [Offline](#offline-benchmark-2)
  - [Online](#online-benchmark-2)
- **[Llama.cpp](#llamacpp)**
  - [Offline](#offline-benchmark-3)
  - [Online](#online-benchmark-3)
- **[Image generation](#image-generation)** (Flux and SDXL)
- **[Fine-tuning](#fine-tuning)**

### Dual Spark
- [Measure bandwidth for Dual Spark setup](#measure-bw-between-dual-sparks)
- [Measure RDMA latency between Dual Sparks](#measure-rdma-latency-between-dual-sparks)

---

# Single Spark

## TensorRT-LLM (TRT-LLM)

### What this measures

- **Offline**: Raw model throughput/latency under synthetic load without HTTP stack, no scheduler overhead.
- **Online**: End-to-end serving performance through trtllm-serve (HTTP + scheduler + KV cache).

For more details, visit [trtllm-bench](https://github.com/NVIDIA/TensorRT-LLM/tree/main/benchmarks/cpp) and [trtllm-serve](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/serve) official documentation.

### Prerequisites (applies to Offline & Online)

#### 1) Docker permissions
```bash
sudo usermod -aG docker $USER
newgrp docker
```

#### 2) Set environment variables
```bash
# -------------------------------
# Environment Setup
# -------------------------------
export HF_TOKEN="<your_huggingface_token>"       # optional if model is public
export MODEL_HANDLE="openai/gpt-oss-20b"
export ISL=128
export OSL=128
export MAX_TOKENS=$((ISL + OSL))
```

### Offline Benchmark

This runs trtllm-bench with a deterministic synthetic dataset matching your ISL/OSL.

```bash
# -------------------------------
# TensorRT-LLM Offline Benchmark
# -------------------------------
docker run \
  --rm -it \
  --gpus all \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e ISL="$ISL" \
  -e OSL="$OSL" \
  -e MAX_TOKENS="$MAX_TOKENS" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \
  bash -lc '
set -e

# 1) Download model (cached in /root/.cache/huggingface)
hf download "$MODEL_HANDLE"

# 2) Prepare synthetic dataset (fixed ISL/OSL)
python benchmarks/cpp/prepare_dataset.py \
  --tokenizer "$MODEL_HANDLE" \
  --stdout token-norm-dist \
  --num-requests 1 \
  --input-mean "$ISL" --input-stdev 0 \
  --output-mean "$OSL" --output-stdev 0 \
  > /tmp/dataset.txt

# 3) Optional tuning config
cat > /tmp/extra-llm-api-config.yml <<EOF
kv_cache_config:
  dtype: "auto"
cuda_graph_config:
  enable_padding: true
EOF

# 4) Run offline benchmark
trtllm-bench -m "$MODEL_HANDLE" throughput \
  --dataset /tmp/dataset.txt \
  --backend pytorch \
  --tp 1 \
  --max_num_tokens "$MAX_TOKENS" \
  --concurrency 1 \
  --max_batch_size 1 \
  --kv_cache_free_gpu_mem_fraction 0.95 \
  --extra_llm_api_options /tmp/extra-llm-api-config.yml
'
```

### Online Benchmark

#### Terminal 1 -  run the TRT-LLM server
```bash
# -------------------------------
# Launch TensorRT-LLM OpenAI server
# -------------------------------
docker run \
  --rm -it \
  --gpus=all \
  --ipc=host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \
  bash -lc '
trtllm-serve serve "$MODEL_HANDLE" \
  --host 0.0.0.0 \
  --port 8000 \
  --backend pytorch \
  --max_num_tokens '"$MAX_TOKENS"' \
  --max_batch_size 1 \
  --kv_cache_free_gpu_memory_fraction 0.9 \
  --tp_size 1 \
  --ep_size 1 \
  --trust_remote_code
'
```

#### Terminal 2 - run the client (vLLM's built-in bench)
```bash
export MODEL_HANDLE="openai/gpt-oss-20b"
export ISL=128
export OSL=128
export MAX_TOKENS=$((ISL + OSL))

# -------------------------------
# Launch Benchmark Client
# -------------------------------
docker run \
  --rm -it \
  --gpus all \
  --ipc host \
  --network host \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e ISL="$ISL" \
  -e OSL="$OSL" \
  nvcr.io/nvidia/vllm:25.12-py3 \
  bash -lc '
vllm bench serve \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model "$MODEL_HANDLE" \
  --dataset-name random \
  --num-prompts 1 \
  --random-input-len "$ISL" \
  --random-output-len "$OSL" \
  --percentile-metrics ttft,tpot,itl,e2el \
  --max-concurrency 1 \
  --request-rate inf
'
```

---

## vLLM

### What this measures

- **Offline**: Raw model throughput/latency under synthetic load without HTTP stack, no scheduler overhead.
- **Online**: End-to-end serving performance through vLLM (HTTP + scheduler + KV cache).

For more details, visit [vllm benchmarking official documentation](https://docs.vllm.ai/en/latest/getting_started/benchmarking.html)

### Prerequisites (applies to Offline & Online)

#### 1) Docker permissions

```bash
sudo usermod -aG docker $USER
newgrp docker
```

#### 2) Set environment variables
```bash
# -------------------------------
# Environment Setup
# -------------------------------
export HF_TOKEN="<your_huggingface_token>"       # optional if model is public
export MODEL_HANDLE="openai/gpt-oss-20b"
export ISL=128
export OSL=128
export MAX_TOKENS=$((ISL + OSL))
```

### Offline Benchmark

This runs vllm bench throughput directly inside the vLLM container with a synthetic random dataset.

```bash
# -------------------------------
# vLLM Offline Benchmark
# -------------------------------
docker run \
  --rm -it \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e ISL="$ISL" \
  -e OSL="$OSL" \
  -e MAX_TOKENS="$MAX_TOKENS" \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
  nvcr.io/nvidia/vllm:25.12-py3 \
  bash -lc '
pip install -q datasets && \
vllm bench throughput \
  --model "$MODEL_HANDLE" \
  --dataset-name random \
  --num-prompts 1 \
  --input-len $ISL \
  --output-len $OSL \
  --max-model-len $MAX_TOKENS \
  --gpu-memory-utilization 0.8
'
```

### Online Benchmark

#### Terminal 1 - run the vLLM server
```bash
# -------------------------------
# Launch vLLM OpenAI server
# -------------------------------
docker run \
  --rm -it \
  --gpus all \
  --ipc host \
  --network host \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e MAX_TOKENS="$MAX_TOKENS" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  nvcr.io/nvidia/vllm:25.12-py3 \
  bash -lc '
vllm serve "$MODEL_HANDLE" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len $MAX_TOKENS \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
'
```

#### Terminal 2 - run the client benchmark
```bash
export MODEL_HANDLE="openai/gpt-oss-20b"
export ISL=128
export OSL=128
export MAX_TOKENS=$((ISL + OSL))

# -------------------------------
# Launch Benchmark Client
# -------------------------------
docker run \
  --rm -it \
  --gpus all \
  --ipc host \
  --network host \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e ISL="$ISL" \
  -e OSL="$OSL" \
  nvcr.io/nvidia/vllm:25.12-py3 \
  bash -lc '
vllm bench serve \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model "$MODEL_HANDLE" \
  --dataset-name random \
  --num-prompts 1 \
  --random-input-len $ISL \
  --random-output-len $OSL \
  --percentile-metrics ttft,tpot,itl,e2el \
  --max-concurrency 1 \
  --request-rate inf
'
```

---

## SGLang

### What this measures

- **Offline**: Raw model throughput/latency under synthetic load without HTTP stack, no scheduler overhead.
- **Online**: End-to-end serving performance through SGLang (HTTP + scheduler + KV cache).

For more details, visit [SGLang benchmarking official documentation](https://sgl-project.github.io/references/benchmark.html)

### Prerequisites (applies to Offline & Online)

#### 1) Docker permissions

```bash
sudo usermod -aG docker $USER
newgrp docker
```

#### 2) Set environment variables
```bash
# -------------------------------
# Environment Setup
# -------------------------------
export HF_TOKEN="<your_huggingface_token>"       # optional if model is public
export MODEL_HANDLE="openai/gpt-oss-20b"
export ISL=128
export OSL=128
export MAX_TOKENS=$((ISL + OSL))
```

### Offline Benchmark

This runs the official SGLang offline throughput benchmark to measure raw model execution performance without launching a server.

```bash
# -------------------------------
# SGLang Offline Benchmark
# -------------------------------
docker run \
  --rm -it \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  nvcr.io/nvidia/sglang:25.12-py3 \
  bash -lc '
python3 -m sglang.bench_offline_throughput \
  --model-path "$MODEL_HANDLE" \
  --dataset-name random \
  --num-prompts 1
'
```

### Online Benchmark

#### Terminal 1 - run the SGLang server
```bash
# -------------------------------
# Launch SGLang HTTP Server
# -------------------------------
docker run \
  --rm -it \
  --gpus all \
  --ipc host \
  --network host \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  nvcr.io/nvidia/sglang:25.12-py3 \
  bash -lc '
python3 -m sglang.launch_server \
  --model-path "$MODEL_HANDLE" \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --tp 1 \
  --attention-backend triton \
  --mem-fraction-static 0.75
'
```

#### Terminal 2 - run the client benchmark
```bash
# -------------------------------
# SGLang Online Benchmark Client
# -------------------------------
docker run \
  --rm -it \
  --network host \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  nvcr.io/nvidia/sglang:25.12-py3 \
  bash -lc '
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model "$MODEL_HANDLE" \
  --dataset-name random \
  --num-prompts 1 \
  --random-input-len '"$ISL"' \
  --random-output-len '"$OSL"'
'
```

---

## Llama.cpp

### What this measures

- **Offline**: Raw model throughput/latency under synthetic load without HTTP stack, no scheduler overhead.
- **Online**: End-to-end serving performance through llama-server (HTTP + scheduler + KV cache).

For more details, visit [Github Discussion](https://github.com/ggml-org/llama.cpp/discussions)

### Prerequisites (applies to Offline & Online)

**Note:**  
DGX Spark uses a long-term supported (LTS) base software stack, so the host OS, driver, and CUDA toolkit are updated together on a fixed release cadence. To access the latest CUDA features and performance improvements, users should run NVIDIA NGC containers (PyTorch, vLLM, TensorRT-LLM, etc.), which are validated for DGX Spark and include newer CUDA toolkits without modifying the host system. If required, users may also install CUDA directly via Debian packages; however, this approach is not recommended for most users and falls outside the supported DGX OS stack.

#### 1) Launch the latest pytorch container from NGC.
```bash
export HF_TOKEN="<your_huggingface_token>"       # optional if model is public
docker run --rm -it \
  --gpus all \
  --ipc=host \
  -p 8080:8080 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME":/home/nvidia \
  -w /home/nvidia \
  nvcr.io/nvidia/pytorch:25.12-py3
```

#### 2) Clone and build the latest Llama.cpp
```bash
# -------------------------------
# Clone and build Llama.cpp
# -------------------------------
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

sudo apt-get update
sudo apt-get install -y libcurl4-openssl-dev cmake g++ make

# Build with CUDA support for NVIDIA GPUs (adjust arch as needed)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121a" -DGGML_CUDA_CUB_3DOT2=on

cmake --build build --config Release -j
```

#### 3) Download model weights
```bash
# -------------------------------
# Download GGUF model
# -------------------------------
cd models

# Example: GPT-OSS-20B
curl -L -o gpt-oss-20b-mxfp4.gguf https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-mxfp4.gguf
```

#### 4) Set environment variables
```bash
# -------------------------------
# Environment Setup
# -------------------------------

export MODEL_HANDLE="gpt-oss-20b-mxfp4.gguf"
export ISL=128
export OSL=128
export MAX_TOKENS=$((ISL + OSL))
```

### Offline Benchmark
```bash
# -------------------------------
# Llama.cpp Offline Benchmark
# -------------------------------
./build/bin/llama-bench \
  -m models/$MODEL_HANDLE \
  -t $(nproc) \
  -p $ISL \
  -n $OSL \
  -ngl 99 \
  -dio 1 \
  -fa 1
```

### Online Benchmark

#### Terminal 1 - run the server
```bash
# -------------------------------
# Launch Llama.cpp Server
# -------------------------------
./build/bin/llama-server \
  --model models/$MODEL_HANDLE \
  --ctx-size $MAX_TOKENS \
  --n-predict $OSL \
  --threads $(nproc) \
  --host 0.0.0.0 \
  --port 8080 \
  -fa 1 \
  --backend-sampling
```

#### Terminal 2 - run the client
```bash
# -------------------------------
# Launch Benchmark Client
# -------------------------------
curl -s -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"What is the capital of France?\",
    \"temperature\": 0.5,
    \"stream\": false
  }" \
  http://127.0.0.1:8080/completion | jq .
```

---

## Image Generation

This benchmark evaluates diffusion model performance using TensorRT-based pipelines for:
- Flux.1 Schnell
- SDXL 1.0

You will measure image generation latency and throughput for text-to-image workloads on DGX Spark.

### Prerequisites

#### 1) Docker permissions
```bash
sudo usermod -aG docker $USER
newgrp docker
```

#### 2) Set environment variables
```bash
# -------------------------------
# Environment Setup
# -------------------------------
export HF_TOKEN="<your_huggingface_token>"
```

#### 3) Launch latest PyTorch container
```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  -e HF_TOKEN="$HF_TOKEN" \
  -v $HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  nvcr.io/nvidia/pytorch:25.12-py3
```

#### 4) Inside the container
```bash
git clone https://github.com/NVIDIA/TensorRT.git -b main --single-branch
cd TensorRT/demo/Diffusion
export TRT_OSSPATH=$HOME/TensorRT/
cd $TRT_OSSPATH/demo/Diffusion
pip install nvidia-modelopt[onnx,hf]
sed -i '/^nvidia-modelopt\[.*\]=.*/d' requirements.txt
pip install -r requirements.txt

apt-get update && \
apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6
```

### Flux.1 Schnell:
```bash
# -------------------------------
# Flux.1 Schnell txt2img Benchmark
# -------------------------------
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token="$HF_TOKEN" \
  --version="flux.1-schnell" \
  --fp4 \
  --download-onnx-models \
  --batch-size 1 \
  --width 1024 \
  --height 1024 \
  --denoising-steps 4
```

### SDXL 1.0:
```bash
# -------------------------------
# SDXL 1.0 txt2img Benchmark
# -------------------------------
python3 demo_txt2img_xl.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token="$HF_TOKEN" \
  --version xl-1.0 \
  --download-onnx-models \
  --batch-size 2 \
  --width 1024 \
  --height 1024 \
  --denoising-steps 50
```

---

## Fine-tuning

### What this measures

This benchmark evaluates training performance (step time, throughput, memory usage) for different fine-tuning strategies on DGX Spark:
- LoRA fine-tuning for parameter-efficient adaptation of Llama 3 8B
- qLoRA fine-tuning for memory-efficient fine-tuning of Llama 3 70B
- Full fine-tuning of a smaller Llama 3 3B model

### Prerequisites

#### 1) Docker permissions
```bash
sudo usermod -aG docker $USER
newgrp docker
```

#### 2) Set environment variables
```bash
# -------------------------------
# Environment Setup
# -------------------------------
export HF_TOKEN="<your_huggingface_token>"
```

#### 3) Launch latest PyTorch container
```bash
docker run --rm -it \
  --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  --ipc=host \
  -v $HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  nvcr.io/nvidia/pytorch:25.12-py3
```

#### 4) Inside the container
```bash
# Install dependencies
pip install transformers peft datasets "trl==0.26.2" "bitsandbytes==0.49.1"

# Clone DGX Spark playbooks
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/pytorch-fine-tune/assets

# Force bitsandbytes to use CUDA 13.0 binary (CUDA 13.1 not yet supported)
export BNB_CUDA_VERSION=130
```

### LoRA Fine-tuning
```bash
# -------------------------------
# Llama 3 8B LoRA Fine-tuning
# -------------------------------
python Llama3_8B_LoRA_finetuning.py --use_torch_compile
```

### qLoRA Fine-tuning
```bash
# -------------------------------
# Llama 3 70B qLoRA Fine-tuning
# -------------------------------
python Llama3_70B_qLoRA_finetuning.py
```

### Full Fine-tuning
```bash
# -------------------------------
# Llama 3 3B Full Fine-tuning
# -------------------------------
python Llama3_3B_full_finetuning.py --use_torch_compile
```

---

# Dual Spark

## Measure BW Between Dual Sparks

DGX Spark systems support high-bandwidth, low-latency interconnects over QSFP ports.
Bandwidth between two Sparks can be validated at two different layers:
- [GPU collective bandwidth using NCCL](#gpu-collective-bandwidth-using-nccl)
- [Raw RDMA fabric bandwidth (RoCE)](#raw-rdma-fabric-bandwidth-roce)

### GPU collective bandwidth using NCCL
- Follow the instruction here - https://build.nvidia.com/spark/nccl/stacked-sparks

#### What this measures
- This test measures effective GPU collective communication bandwidth using NCCL.

### Raw RDMA fabric bandwidth (RoCE)

#### What this measures
- This test measures raw point-to-point RDMA bandwidth over the QSFP-connected CX-7 NICs using RoCE.

### Prerequisites

#### 1) Install perftest tools (on both Sparks)
```bash
sudo apt install perftest
```

#### 2) Ensure one QSFP cable connects Spark-1 ↔ Spark-2 directly

### Setup

#### Step 1 – Identify devices and logical ports

Run on both Spark systems:
```bash
ibdev2netdev
```

**Example output (from Spark-1):**
```
nvidia@spark-5c2d:~$ ibdev2netdev
rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Down)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Down)
```
You will use the **Up** interfaces for IP assignment.

**Example output (from Spark-2):**
```
nvidia@spark-bd26:~$ ibdev2netdev
rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Down)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Down)
```
You will use the **Up** interfaces for IP assignment.

#### Step 2 - Assign Manual IPs

Assign unique subnets to each active port.

**Note:** Repeat this step after reboot if NetworkManager clears them.

**Spark-1 (HOST)**
```bash
# Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp1s0f0np0:
      addresses:
        - 192.168.200.12/24
      dhcp4: no
    enP2p1s0f0np0:
      addresses:
        - 192.168.201.12/24
      dhcp4: no
EOF

sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```
**Note:** Interfaces may differ; use the ones marked **Up**.

**Spark-2 (CLIENT)**
```bash
# Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp1s0f0np0:
      addresses:
        - 192.168.200.13/24
      dhcp4: no
    enP2p1s0f0np0:
      addresses:
        - 192.168.201.13/24
      dhcp4: no
EOF

sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```
**Note:** Interfaces may differ; use the ones marked **Up**.

#### Step 3 – Run Bandwidth Test

Open two terminals on each Spark (4 total).

**Note:** Make sure ports 12000 and 12001 are open and not in use.

**Spark-1 (HOST) - Terminal 1**
```bash
ib_write_bw -d rocep1s0f0 -i 1 -p 12000 -F --report_gbits --run_infinitely
```
**Note:** Replace device names with your actual **Up** interfaces. (replace `rocep1s0f0` with your `<HOST_NIC1_INTERFACE>`)

**Spark-1 (HOST) - Terminal 2**
```bash
ib_write_bw -d roceP2p1s0f0 -i 1 -p 12001 -F --report_gbits --run_infinitely
```
**Note:** Replace device names with your actual **Up** interfaces. (replace `roceP2p1s0f0` with your `<HOST_NIC2_INTERFACE>`)

**Spark-2 (CLIENT) - Terminal 1**
```bash
ib_write_bw -d rocep1s0f0 -i 1 -p 12000 -F --report_gbits 192.168.200.12 --run_infinitely
```
**Note:** Replace device names with your actual **Up** interfaces. (replace `rocep1s0f0` with your `<CLIENT_NIC1_INTERFACE>`)

**Spark-2 (CLIENT) - Terminal 2**
```bash
ib_write_bw -d roceP2p1s0f0 -i 1 -p 12001 -F --report_gbits 192.168.201.12 --run_infinitely
```
**Note:** Replace device names with your actual **Up** interfaces. (replace `roceP2p1s0f0` with your `<CLIENT_NIC2_INTERFACE>`)

#### STEP 4 – Monitor Bandwidth

**Example client output:**

**Client-1 Output**
```
nvidia@spark-bd26:~$ ib_write_bw -d rocep1s0f0 -i 1 -p 12000 -F --report_gbits 192.168.200.12 --run_infinitely
  WARNING: BW peak won't be measured in this run.
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF            Device         : rocep1s0f0
 Number of qps   : 1              Transport type : IB
 Connection type : RC             Using SRQ      : OFF
 PCIe relax order: ON
 ibv_wr* API     : ON
 TX depth        : 128
 CQ Moderation   : 1
 Mtu             : 1024[B]
 Link type       : Ethernet
 GID index       : 3
 Max inline data : 0[B]
 rdma_cm QPs     : OFF
 Data ex. method : Ethernet
---------------------------------------------------------------------------------------
 local address:  LID 0000 QPN 0x0129 PSN 0x57279d RKey 0x184300 VAddr 0x00ec99bedad000
 GID:            00:00:00:00:00:00:00:00:00:00:255:255:192:168:200:13
 remote address: LID 0000 QPN 0x0129 PSN 0x531b7  RKey 0x184300 VAddr 0x00ffeec955d000
 GID:            00:00:00:00:00:00:00:00:00:00:255:255:192:168:200:12
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]    MsgRate[Mpps]
 65536      882805           0.00               92.57                0.176554
 65536      882802           0.00               92.57                0.176554
 65536      882791           0.00               92.57                0.176554
 65536      882791           0.00               92.56                0.176552
 65536      882821           0.00               92.57                0.176555
```

**Client-2 Output**
```
nvidia@spark-bd26:~$ ib_write_bw -d roceP2p1s0f0 -i 1 -p 12001 -F --report_gbits 192.168.201.12 --run_infinitely
 WARNING: BW peak won't be measured in this run.
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF            Device         : roceP2p1s0f0
 Number of qps   : 1              Transport type : IB
 Connection type : RC             Using SRQ      : OFF
 PCIe relax order: ON
 ibv_wr* API     : ON
 TX depth        : 128
 CQ Moderation   : 1
 Mtu             : 1024[B]
 Link type       : Ethernet
 GID index       : 3
 Max inline data : 0[B]
 rdma_cm QPs     : OFF
 Data ex. method : Ethernet
---------------------------------------------------------------------------------------
 local address:  LID 0000 QPN 0x01a9 PSN 0x5e41f9 RKey 0x1a03ed VAddr 0x00f374277dd000
 GID:            00:00:00:00:00:00:00:00:00:00:255:255:192:168:201:13
 remote address: LID 0000 QPN 0x01a9 PSN 0x8ab8e7  RKey 0x1a0300 VAddr 0x00f285f5f1d000
 GID:            00:00:00:00:00:00:00:00:00:00:255:255:192:168:201:12
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]    MsgRate[Mpps]
 65536      927940           0.00               97.28                0.185548
 65536      927790           0.00               97.28                0.185549
 65536      927766           0.00               97.28                0.185550
 65536      927754           0.00               97.28                0.185545
 65536      927804           0.00               97.29                0.185557
 65536      927807           0.00               97.28                0.185554
```

**Total throughput = 92.57 + 97.28 = 189.85 Gbps**

## Measure RDMA Latency Between Dual Sparks

### What this measures

This test measures one-way RDMA latency between two DGX Spark systems over the same QSFP RoCE links used for bandwidth testing.

### Prerequisites

Before running the latency tests, complete Step 1 (Identify devices and logical ports) and
Step 2 (Assign Manual IPs) from the [Measure BW Between Dual Sparks](#measure-bw-between-dual-sparks) section above.

### Step 1.1 – Run RDMA Write Latency Test

This measures RDMA write latency on a single QSFP link.

Open two terminals (one per Spark).

**Spark-1 (HOST)**
```bash
ib_write_lat -d rocep1s0f0 -i 1 -p 13000 -F
```

Replace `rocep1s0f0` with your actual **Up** RDMA device from `ibdev2netdev`.

**Spark-2 (CLIENT)**
```bash
ib_write_lat -d rocep1s0f0 -i 1 -p 13000 -F 192.168.200.12
```

### Step 1.2 – Run RDMA Read Latency Test

This measures RDMA read latency on the second QSFP link.

Open two terminals (one per Spark).

**Spark-1 (HOST)**
```bash
ib_read_lat -d rocep1s0f0 -i 1 -p 13001 -F
```

**Spark-2 (CLIENT)**
```bash
ib_read_lat -d rocep1s0f0 -i 1 -p 13001 -F 192.168.201.12
```

Note: RDMA latency is a per-link metric and should be measured on a single QSFP link at a time. Latency values are not aggregated across multiple links.
