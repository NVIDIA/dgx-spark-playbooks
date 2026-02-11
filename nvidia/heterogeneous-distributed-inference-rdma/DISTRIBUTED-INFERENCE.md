# Distributed Inference Guide

> Deploy and run distributed AI inference across DGX Spark and Linux Workstation using vLLM and Ray

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

---

## Overview

## Basic idea

This guide walks you through deploying distributed inference across your heterogeneous RDMA cluster. Using Ray for orchestration and vLLM for inference, you can run large language models that exceed the memory capacity of any single GPU by distributing them across your DGX Spark and Linux workstation.

**Architecture:**
```
┌─────────────────────────────────┐    ┌───────────────────────────────────┐
│         DGX SPARK               │    │        WORKSTATION                │
│   (Grace Blackwell GB10)        │    │  (RTX 6000 Pro / RTX 5090)        │
│                                 │    │                                   │
│  ┌───────────────────────────┐  │    │  ┌───────────────────────────┐    │
│  │      vLLM Head Node       │  │    │  │      vLLM Worker          │    │
│  │   (API Server, Rank 0)    │  │    │  │   (Tensor Parallel)       │    │
│  └───────────────────────────┘  │    │  └───────────────────────────┘    │
│              │                  │    │              │                    │
│  ┌───────────────────────────┐  │    │  ┌───────────────────────────┐    │
│  │   Ray Head (6379)         │◄─┼────┼──│   Ray Worker              │    │
│  └───────────────────────────┘  │    │  └───────────────────────────┘    │
│              │                  │    │              │                    │
│  ┌───────────────────────────┐  │    │  ┌───────────────────────────┐    │
│  │   NCCL over RDMA          │◄─┼════┼──│   NCCL over RDMA          │    │
│  │   192.168.200.1           │  │    │  │   192.168.200.2           │    │
│  └───────────────────────────┘  │    │  └───────────────────────────┘    │
└─────────────────────────────────┘    └───────────────────────────────────┘
```

## What you'll accomplish

- Configure SSH and hostname resolution between nodes
- Test NCCL communication over RDMA
- Deploy RDMA-enabled Docker containers
- Establish a Ray cluster across both systems
- Run distributed inference with vLLM
- Benchmark performance across different configurations

## What to know before starting

- Familiarity with Docker and container networking
- Understanding of distributed computing concepts (Ray, tensor parallelism)
- Basic knowledge of LLM inference serving

## Prerequisites

- Completed [RDMA Network Setup](README.md) with validated 90+ Gbps bandwidth
- Docker installed on both systems: `docker --version`
- NVIDIA Container Toolkit installed
- Hugging Face account for model access (some models require authentication)

> [!NOTE]
> **Why we use the `nvcr.io/nvidia/vllm` container:** This tutorial uses the official NVIDIA vLLM container image (`nvcr.io/nvidia/vllm:25.09-py3`) on both nodes. This is important because:
> - **Version consistency:** Ray cluster is very sensitive to Python and Ray version mismatches between nodes. The container guarantees identical versions on both DGX Spark (ARM64) and Workstation (AMD64).
> - **Pre-installed dependencies:** NCCL, RDMA libraries, and all required packages are already configured.
> - **Multi-architecture support:** The same image tag works on both ARM64 (DGX Spark) and AMD64 (Workstation) architectures.
> - **vLLM ready:** No additional installation needed - just pull and run.

## Time & risk

- **Duration:** 1-2 hours including testing

- **Risk level:** Low - uses containers, non-destructive

- **Rollback:** Stop containers to revert

- **Last Updated:** 01/23/2026

---

## Instructions

## Step 1. Configure Hostnames

Add hostname aliases on both systems:

```bash
## Add hostname resolution on both DGX Spark and Workstation
sudo tee -a /etc/hosts > /dev/null <<EOF
192.168.200.1 dgx-spark
192.168.200.2 workstation
EOF
```

## Step 2. Set Up SSH Access

Install SSH server if needed (common on workstations):

```bash
## Check SSH server status
sudo systemctl status ssh

## If not installed:
sudo apt update
sudo apt install openssh-server
sudo systemctl start ssh
sudo systemctl enable ssh
```

Configure passwordless SSH between nodes:

On DGX Spark:
```bash
## Check if SSH key exists
ls ~/.ssh/id_*.pub

## If no key exists, generate one:
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519

## Copy key to workstation
ssh-copy-id <your-username>@workstation
```

On Workstation:
```bash
## Check if SSH key exists
ls ~/.ssh/id_*.pub

## If no key exists, generate one:
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519

## Copy key to DGX Spark
ssh-copy-id <your-username>@dgx-spark
```

Verify passwordless SSH:

```bash
## From DGX Spark
ssh <your-username>@workstation hostname
## Expected output: workstation

## From Workstation
ssh <your-username>@dgx-spark hostname
## Expected output: dgx-spark
```

---

## Step 3. Test NCCL Communication

Create the NCCL test script on both systems:

```bash
## Create test script
cat > test_nccl.py << 'EOF'
import os
import torch
import torch.distributed as dist
import argparse

def test_nccl_communication():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--master_addr', type=str, default='192.168.200.1')
    parser.add_argument('--master_port', type=str, default='29500')
    args = parser.parse_args()

    os.environ['RANK'] = str(args.rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['NCCL_SOCKET_IFNAME'] = 'enp1s0f0np0'

    print(f"Initializing process group - Rank: {args.rank}, World Size: {args.world_size}")
    print(f"Master: {args.master_addr}:{args.master_port}")

    dist.init_process_group(backend='nccl', rank=args.rank, world_size=args.world_size)
    print(f"Process group initialized - Rank: {dist.get_rank()}/{dist.get_world_size()}")

    device = torch.device('cuda:0')
    tensor = torch.ones(10, device=device) * (args.rank + 1)
    print(f"Rank {args.rank} - Before allreduce: {tensor}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {args.rank} - After allreduce: {tensor}")
    print(f"Expected result: {torch.ones(10) * (1 + 2)}")

    dist.destroy_process_group()
    print(f"Rank {args.rank} - Test completed successfully!")

if __name__ == "__main__":
    test_nccl_communication()
EOF
```

Run NCCL test in Docker containers:

On DGX Spark (start first):
```bash
docker run -it --runtime=nvidia --gpus all --network host --ipc=host \
  --privileged --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /dev/infiniband:/dev/infiniband -v /sys:/sys:ro \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0:1 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -v $(pwd):/workspace \
  nvcr.io/nvidia/vllm:25.09-py3 python /workspace/test_nccl.py --rank 0
```

On Workstation (connect to DGX):
```bash
docker run -it --runtime=nvidia --gpus all --network host --ipc=host \
  --privileged --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /dev/infiniband:/dev/infiniband -v /sys:/sys:ro \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0:1 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -v $(pwd):/workspace \
  nvcr.io/nvidia/vllm:25.09-py3 python /workspace/test_nccl.py --rank 1
```

**Success indicators:**
- Output shows: `NCCL INFO Using network IBext_v10`
- All-reduce operation completes successfully
- Final tensors show expected sum values (3.0 for each element)

---

## Step 4. Start RDMA-Enabled Containers

On DGX Spark:
```bash
docker run -it --runtime=nvidia --gpus all --network host --ipc=host --shm-size=10g \
  --privileged \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /dev/infiniband:/dev/infiniband \
  -v /sys:/sys:ro \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0:1 \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e RAY_USE_MULTIPLE_IPS=0 \
  -e RAY_NODE_IP_ADDRESS=192.168.200.1 \
  -e RAY_OVERRIDE_NODE_IP=192.168.200.1 \
  -e VLLM_HOST_IP=192.168.200.1 \
  nvcr.io/nvidia/vllm:25.09-py3 bash
```

On Workstation:
```bash
docker run -it --runtime=nvidia --gpus all --network host --ipc=host --shm-size=10g \
  --privileged \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /dev/infiniband:/dev/infiniband \
  -v /sys:/sys:ro \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0:1 \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e RAY_USE_MULTIPLE_IPS=0 \
  -e RAY_NODE_IP_ADDRESS=192.168.200.2 \
  -e RAY_OVERRIDE_NODE_IP=192.168.200.2 \
  nvcr.io/nvidia/vllm:25.09-py3 bash
```

**Key parameters explained:**
- `--runtime=nvidia`: Required for GPU access
- `--network host`: Uses host networking (required for RDMA)
- `--privileged`: Needed for InfiniBand device access
- `--ulimit memlock=-1`: Unlimited memory locking for RDMA
- `-v /dev/infiniband:/dev/infiniband`: Mounts RDMA devices
- `NCCL_IB_HCA=rocep1s0f0:1`: Tells NCCL to use specific RDMA device
- `RAY_USE_MULTIPLE_IPS=0`: Prevents Ray IP detection issues

---

## Step 5. Establish Ray Cluster

On DGX Spark container (head node):
```bash
ray start --head \
  --node-ip-address=192.168.200.1 \
  --port=6379 \
  --dashboard-host=192.168.200.1 \
  --dashboard-port=8265 \
  --num-gpus=1
```

Verify head node:
```bash
ray status
```

Expected output:
```
======== Autoscaler status: 2026-01-10 19:43:05.517578 ========
Node status
---------------------------------------------------------------
Active:
 1 node_xxxxx
Resources
---------------------------------------------------------------
Total Usage:
 0.0/20.0 CPU
 0.0/1.0 GPU
```

On Workstation container (worker node):
```bash
ray start \
  --address=192.168.200.1:6379 \
  --node-ip-address=192.168.200.2 \
  --num-gpus=1
```

> [!NOTE]
> Adjust `--num-gpus` based on your workstation configuration. In our case, we had 2 GPUs (RTX 6000 Pro + RTX 5090) but only used 1 for this tutorial.

Verify cluster formation:
```bash
ray status
```

Expected output (should show 2+ total GPUs depending on your setup):
```
======== Autoscaler status: 2026-01-10 19:46:26.274139 ========
Node status
---------------------------------------------------------------
Active:
 1 node_xxxxx (head)
 1 node_xxxxx (worker)
Resources
---------------------------------------------------------------
Total Usage:
 0.0/68.0 CPU
 0.0/2.0 GPU
```

---

## Step 6. Run Validation Test (4B Model)

Start small model for validation on DGX Spark container:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.8 \
  --host 192.168.200.1 \
  --port 8000
```

Test from another terminal:
```bash
curl -X POST "http://192.168.200.1:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "Test distributed inference:",
    "max_tokens": 500
  }'
```

---

## Step 7. Run FP8 Quantized Model (30B)

FP8 quantization provides excellent memory efficiency with good performance:

```bash
## Stop previous model (Ctrl+C), then start FP8 30B model
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.8 \
  --host 192.168.200.1 \
  --port 8000
```

**Benefits of FP8:**
- Memory efficiency: Reduced footprint compared to BF16
- Performance: 341+ tok/s demonstrated
- Hardware compatibility: Fully supported on Blackwell GB10

---

## Step 8. Run Large Model (72B)

This step demonstrates the real power of distributed inference: running a model that **exceeds the memory capacity of any single GPU**.

| Component | Available VRAM | Sufficient for 72B? |
|-----------|---------------|---------------------|
| DGX Spark | 128 GB | No (~136GB needed) |
| RTX 6000 Pro | 96 GB | No (~136GB needed) |
| **Combined Cluster** | **224 GB** | **Yes** |

The Qwen2.5-72B-Instruct model requires ~136GB in BF16 precision - impossible to run on either GPU alone. This is where our RDMA cluster shines, aggregating memory across both systems.

Memory-optimized configuration for 136GB model:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.85 \
  --host 192.168.200.1 \
  --port 8000 \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --disable-sliding-window \
  --enforce-eager
```

**Memory optimization parameters:**
- `--gpu-memory-utilization 0.85`: Uses 85% of GPU memory
- `--max-model-len 2048`: Limits context length to save memory
- `--max-num-seqs 8`: Reduces concurrent sequences
- `--disable-sliding-window`: Disables memory-intensive sliding window attention
- `--enforce-eager`: Uses eager execution (saves memory)

Test 72B model:
```bash
curl -X POST "http://192.168.200.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain the benefits of RDMA for AI workloads in one paragraph."}
    ],
    "max_tokens": 500
  }'
```

---

## Step 9. Monitor RDMA Traffic

Monitor RDMA activity during inference:

```bash
## Run on both systems (separate terminals)
watch -n 0.5 "
echo '=== RDMA Counters ===';
echo -n 'TX: '; cat /sys/class/infiniband/rocep1s0f0/ports/1/counters/port_xmit_data;
echo -n 'RX: '; cat /sys/class/infiniband/rocep1s0f0/ports/1/counters/port_rcv_data;
echo 'Timestamp: '; date;
"
```

During inference, you'll see counters increasing as tensors are communicated between GPUs.

---

## Performance Benchmarks

### Benchmark Commands

**Single-node testing:**
```bash
## On RTX 6000 Pro or DGX Spark
vllm bench latency --model Qwen/Qwen3-30B-A3B-Thinking-2507 --input-len 512 --output-len 2000 --num-iters 10
vllm bench throughput --model Qwen/Qwen3-30B-A3B-Thinking-2507 --input-len 512 --output-len 2000 --num-prompts 20
```

**Distributed testing:**
```bash
## 30B Model
vllm bench serve --host 192.168.200.1 --port 8000 --random-input-len 512 --random-output-len 2000 --num-prompts 20 --request-rate 2 --model Qwen/Qwen3-30B-A3B-Thinking-2507

## 72B Model
vllm bench serve --host 192.168.200.1 --port 8000 --random-input-len 256 --random-output-len 1500 --num-prompts 20 --request-rate 2 --model Qwen/Qwen2.5-72B-Instruct
```

### Performance Results Summary

| Configuration | Avg Latency | Output Throughput | Total Throughput |
|---------------|-------------|-------------------|------------------|
| **RTX 6000 Pro (Single)** | 36.87s | 679.88 tok/s | 853.90 tok/s |
| **DGX Spark (Single)** | 213.12s | 105.10 tok/s | 132.00 tok/s |
| **Distributed RDMA** | 191.09s | 205.83 tok/s | 259.41 tok/s |

### What This Demonstrates

The key achievement of this tutorial is successfully running distributed inference across heterogeneous hardware (DGX Spark ARM64 + Linux Workstation AMD64) over RDMA. The distributed setup aggregates GPU memory from both systems, enabling models that wouldn't fit on either device alone.

### FP8 30B Model Results

```
============ Serving Benchmark Result ============
Successful requests:                     20
Benchmark duration (s):                  115.36
Output token throughput (tok/s):         341.15
Total Token throughput (tok/s):          429.89
Mean TTFT (ms):                          171.00
Mean TPOT (ms):                          53.08
==================================================
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Ray worker can't connect to head | Firewall blocking port 6379 | `sudo ufw allow 6379/tcp` |
| NCCL timeout during model load | RDMA not working | Verify `ib_send_bw` test passes |
| "Placement group" errors | Ray cluster not formed | Check `ray status` on both nodes |
| OOM during 72B model load | Insufficient memory optimization | Add `--max-model-len 2048 --enforce-eager` |
| SSH connection refused | SSH server not running | `sudo systemctl start ssh` |
| Container can't access RDMA | Missing device mount | Ensure `-v /dev/infiniband:/dev/infiniband` |
| Wrong IP in Ray cluster | Multiple network interfaces | Set `RAY_USE_MULTIPLE_IPS=0` |
| Slow inference performance | NCCL using wrong interface | Verify `NCCL_SOCKET_IFNAME=enp1s0f0np0` |

---

## Credits

This playbook was contributed by **Csaba Kecskemeti** | [DevQuasar](https://devquasar.com/).

For a detailed walkthrough and additional context, see the original article:
[Distributed Inference Cluster: DGX Spark + RTX 6000 Pro](https://devquasar.com/ai/edge-ai/distributed-inference-cluster-dgx-spark-rtx-6000-pro/)
