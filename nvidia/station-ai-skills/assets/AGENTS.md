# DGX Station Essential Constraints

This file gives your coding agent the critical constraints it needs to avoid breaking things on NVIDIA DGX Station. When you need a step-by-step workflow, invoke the bundled skills: `vllm-setup`, `sglang-setup`, `mig-configure`, `dgx-diagnose`. In Codex, install them into `$CODEX_HOME/skills` and mention them as `$vllm-setup` or plain text like "use vllm-setup"; in Claude Code or Gemini CLI, type `/<name>`; in Cursor, reference the rule by name.

## System architecture (quick reference)

- **GB300 GPU** — Blackwell Ultra (SM103), up to 279 GB HBM3e, 20 PFLOPS sparse FP4. This is the AI compute GPU.
- **Grace CPU** — 72-core ARM Neoverse V2, up to 496 GB LPDDR5x.
- **RTX PRO 6000** — Discrete display GPU (PCIe, non-coherent). For graphics only.
- **NVLink C2C** — Coherent CPU-GPU link. CPU + GPU memory = up to 775 GB total.
- The GB300 is typically device **1** and RTX PRO is device **0**. Always verify: `nvidia-smi --query-gpu=index,name --format=csv,noheader`

## Critical constraint: mixed coherency

**CUDA cannot handle mixed-coherency GPUs in the same process.** The GB300 uses hardware-coherent memory (ATS) while the RTX PRO uses non-coherent memory (HMM via PCIe). A single CUDA context can use one or the other, not both.

**Never use `--gpus all`** — it will cause CUDA assert failures.

## GPU targeting

There are three ways to target the GB300:

**1. By device index** (most common):
```bash
export CUDA_VISIBLE_DEVICES=1        # bare metal
docker run --gpus '"device=1"' ...   # Docker
```

**2. By coherency modality:**
```bash
export CUDA_DEVICE_MODALITY=ATS      # GB300 (coherent)
export CUDA_DEVICE_MODALITY=NONATS   # RTX PRO (non-coherent)
```

**3. By driver application profiles** in `~/.nv/nvidia-application-profiles-rc`:
```json
{
  "rules": [
    { "pattern": { "feature": "cmdline", "matches": "my_app" }, "profile": "UseATSGpuInMixedCoherencySystems" }
  ]
}
```

## Display and graphics

- The GB300 does not support X display. Display runs on RTX PRO only.
- **Do not run `nvidia-xconfig -a`** — it generates an invalid config.
- If CUDA initializes before Vulkan in a process, it may bind to the GB300, causing `VK_ERROR_INITIALIZATION_FAILED`. Run CUDA and Vulkan in separate processes.

## Memory

- GB300 HBM is in the system memory pool (NUMA node 1). `malloc` may allocate there.
- Use `numactl --membind=0` for CPU-only processes that shouldn't touch GPU memory.
- CPU can cache accesses to GB300 memory, but GB300 cannot cache accesses to CPU memory.

## Software versions

| Component | Validated version | Notes |
|-----------|-------------------|-------|
| NVIDIA Driver | 590.48.01 | Check with `nvidia-smi` |
| CUDA (driver) | 13.1 | Containers bring their own runtime |
| vLLM container | `nvcr.io/nvidia/vllm:26.01-py3` | **Avoid 25.10** (FlashInfer buffer overflow) |
| SGLang container | `lmsysorg/sglang:latest-cu130` | cu130 required for SM103 |
| CUDA base image | `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04` | For custom containers |
| Ubuntu | 24.04 | Preinstalled |

## Common pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `--gpus all` CUDA assert failure | Mixed coherency | Use `--gpus '"device=N"'` for the GB300 |
| vLLM 25.10 FlashInfer crash | Known DGX Station bug | Use `vllm:26.01-py3` or newer |
| SGLang CUDA errors | Wrong CUDA for Blackwell | Use `sglang:latest-cu130` |
| Model runs on RTX PRO | Wrong device index | Verify with `nvidia-smi --query-gpu=index,name --format=csv,noheader` |
| `nvidia-smi -mig 1` "In use" | GPU processes running | `sudo fuser -v /dev/nvidia*` |
| NVLink errors after disabling MIG | Fabric Manager stopped | `sudo systemctl start nvidia-fabricmanager` |
| `malloc` lands in GPU memory | HBM in system pool | `numactl --membind=0` |
| X crash after `nvidia-xconfig -a` | Invalid mixed-coherency config | Restore from `/etc/X11/xorg.conf.nvidia-xconfig-original` |
| Vulkan `VK_ERROR_INITIALIZATION_FAILED` | CUDA bound GB300 first | Separate CUDA and Vulkan into different processes |
| HuggingFace 401 | Missing HF_TOKEN | Pass inline: `-e HF_TOKEN="hf_..."` |
| Port conflict | Port already in use | `lsof -i :PORT`, use different port |
