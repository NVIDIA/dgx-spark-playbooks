---
name: mig-configure
description: Configure NVIDIA MIG (Multi-Instance GPU) partitions on the DGX Station GB300, including enabling MIG mode, choosing a profile layout, creating instances, and retrieving MIG UUIDs. Use when the user asks to partition the GB300, set up MIG, run multiple models in isolation on one GPU, or reconfigure existing MIG instances.
metadata:
  publisher: nvidia
  hardware: DGX Station GB300
---

# MIG Configuration on DGX Station

Configure MIG (Multi-Instance GPU) partitions on the DGX Station GB300.

## Steps

1. **Find the GB300 GPU index.** Run:
   ```bash
   nvidia-smi --query-gpu=index,name --format=csv,noheader
   ```

2. **Check current MIG state:**
   ```bash
   nvidia-smi -i <GB300_INDEX> -q | grep -i "MIG Mode"
   ```

3. **If MIG is already enabled, show current instances:**
   ```bash
   nvidia-smi mig -lgi -i <GB300_INDEX>
   nvidia-smi mig -lci -i <GB300_INDEX>
   ```
   If the user wants to reconfigure, destroy existing instances first (step 6).

4. **If MIG is not enabled, enable it.** All GPU processes must be stopped first:
   ```bash
   # Check for running GPU processes
   sudo fuser -v /dev/nvidia*

   # Enable MIG
   sudo nvidia-smi -i <GB300_INDEX> -mig 1

   # Verify
   nvidia-smi -i <GB300_INDEX> -q | grep -i "MIG Mode"
   ```

5. **Show available profiles and help the user choose a layout:**
   ```bash
   nvidia-smi mig -lgip -i <GB300_INDEX>
   ```

   Common GB300 MIG profiles:

   | Profile | ID | Memory | Use case |
   |---------|----|--------|----------|
   | 1g.35gb | 19 | ~35 GB | Small models (7-8B), dev/test |
   | 1g.35gb+me | 20 | ~35 GB | Same + media extensions |
   | 1g.70gb | 15 | ~70 GB | Slightly larger inference |
   | 2g.70gb | 14 | ~70 GB | Medium models (14-30B) |
   | 3g.139gb | 9 | ~139 GB | Large models (70B quantized) |
   | 4g.139gb | 5 | ~139 GB | Large models, more compute |
   | 7g.278gb | 0 | ~278 GB | Full GPU as single instance |

   Suggest layouts based on the user's workload. Examples:
   - **Two models (70B + 8B):** `3g.139gb + 2g.70gb + 2g.70gb` → IDs `9,14,14`
   - **Many small models:** `7 × 1g.35gb` → IDs `19,19,19,19,19,19,19`
   - **One large model with isolation:** `7g.278gb` → ID `0`

   Ask the user what models they want to run before suggesting a layout.

6. **Create (or recreate) instances:**

   If reconfiguring, destroy existing instances first:
   ```bash
   sudo nvidia-smi mig -dci -i <GB300_INDEX>
   sudo nvidia-smi mig -dgi -i <GB300_INDEX>
   ```

   Then create the new layout:
   ```bash
   sudo nvidia-smi mig -cgi <PROFILE_IDS> -C -i <GB300_INDEX>
   ```

7. **Get the MIG device UUIDs:**
   ```bash
   nvidia-smi -L
   ```
   Note the `MIG-<uuid>` entries — these are used to target specific MIG instances.

8. **Show the user how to use MIG devices:**
   ```bash
   # Bare metal
   export CUDA_VISIBLE_DEVICES=MIG-<uuid>

   # Docker
   docker run --gpus '"device=MIG-<uuid>"' ...
   ```

9. **Report the final layout** to the user with UUIDs and suggested docker commands for each instance.

## Disabling MIG

If the user wants to return to full-GPU mode:

```bash
# Stop all workloads using MIG instances first
sudo nvidia-smi mig -dci -i <GB300_INDEX>
sudo nvidia-smi mig -dgi -i <GB300_INDEX>
sudo nvidia-smi -i <GB300_INDEX> -mig 0

# Ensure Fabric Manager is running for NVLink re-initialization
sudo systemctl start nvidia-fabricmanager
```
