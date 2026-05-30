
# DGX Station Diagnostics

Diagnose common DGX Station issues. Run through the checks below to identify the problem.

## Step 1. Gather system state

Run these commands and analyze the output:

```bash
# GPU status
nvidia-smi

# GPU device list with indices
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# Driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1

# MIG state
nvidia-smi -i 1 -q 2>/dev/null | grep -i "MIG Mode" || echo "Could not query MIG on device 1"

# Fabric Manager
systemctl is-active nvidia-fabricmanager

# GPU processes
sudo fuser -v /dev/nvidia* 2>/dev/null || echo "No GPU processes found"

# Docker containers using GPUs
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" 2>/dev/null
```

## Step 2. Match symptoms to known issues

Based on the gathered state and the user's reported problem, check for these known issues:

### CUDA crashes with `--gpus all`
**Cause:** Mixed coherency — GB300 (ATS) and RTX PRO (non-ATS) cannot share a CUDA context.
**Fix:** Use `--gpus '"device=N"'` targeting only the GB300.

### Model running on wrong GPU (RTX PRO instead of GB300)
**Check:** The device index in the docker command vs actual GPU indices.
**Fix:** Verify with `nvidia-smi --query-gpu=index,name --format=csv,noheader` and correct the `--gpus` flag.

### vLLM crash / FlashInfer buffer overflow
**Check:** Container version — `docker inspect vllm-server | grep Image`
**Fix:** Use `nvcr.io/nvidia/vllm:26.01-py3`. Version 25.10 has a known FlashInfer bug on DGX Station.

### SGLang CUDA errors
**Check:** Container tag — must be `cu130` for Blackwell SM103.
**Fix:** Use `lmsysorg/sglang:latest-cu130`.

### CUDA OOM despite 279 GB HBM
**Check:** `--max-model-len` / `--context-length` and memory utilization settings.
**Fix:** Reduce context length or lower `--gpu-memory-utilization` / `--mem-fraction-static`.

### `nvidia-smi -mig 1` returns "In use by another client"
**Check:** `sudo fuser -v /dev/nvidia*` — GPU processes must be stopped first.
**Fix:** Stop all GPU workloads, then retry.

### NVLink errors after disabling MIG
**Check:** `systemctl is-active nvidia-fabricmanager`
**Fix:** `sudo systemctl start nvidia-fabricmanager`

### X server crash after nvidia-xconfig -a
**Fix:** `sudo cp /etc/X11/xorg.conf.nvidia-xconfig-original /etc/X11/xorg.conf`

### Vulkan VK_ERROR_INITIALIZATION_FAILED
**Cause:** CUDA initialized before Vulkan, binding to GB300.
**Fix:** Run CUDA and Vulkan workloads in separate processes. For Vulkan apps: `__GL_DeviceModalityPreference=2 ./your_app`

### HuggingFace 401 / token errors
**Fix:** Pass token inline: `-e HF_TOKEN="hf_..."`. Don't rely on shell export for background Docker tasks.

### Port already in use
**Check:** `lsof -i :<PORT>`
**Fix:** Stop the conflicting process or use a different host port: `-p 8001:8000`.

## Step 3. Report findings

Tell the user:
1. What the issue is
2. Why it happens (root cause)
3. The specific command to fix it
4. How to verify the fix worked
