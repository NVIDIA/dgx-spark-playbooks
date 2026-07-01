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

   | ID | Profile name (driver-dependent) | Approx. memory | Use case |
   |----|----------------------------------|----------------|----------|
   | 19 | `1g.35gb` (59x) · `1g.31gb` (61x) | ~30 GB | Small models (7-8B), dev/test |
   | 20 | `1g.35gb+me` · `1g.31gb+me` | ~30 GB | Same + media extensions |
   | 15 | `1g.70gb` | ~68 GB | Slightly larger inference |
   | 14 | `2g.70gb` | ~68 GB | Medium models (14-30B) |
   | 9  | `3g.139gb` (59x) · `3g.126gb` (61x) | ~137 GB | Large models (70B quantized) |
   | 5  | `4g.139gb` · `4g.126gb` | ~137 GB | Large models, more compute |
   | 0  | `7g.278gb` (59x) · `7g.251gb` (61x) | ~276 GB | Full GPU as single instance |

   > **Profile names depend on your driver version; the profile IDs do not.** Always read the exact
   > names and sizes on your box with `nvidia-smi mig -lgip -i <GB300_INDEX>`, and create instances by
   > **ID**. (Driver 59x reports the `…35gb/139gb/278gb` names; 61x reports `…31gb/126gb/251gb` for the
   > same IDs.)

   Suggest layouts based on the user's workload (use the stable IDs). Examples:
   - **Two models (70B + smaller):** one `3g` + two `1g.70gb` → IDs `9,15,15`
   - **Many small models:** three `1g` → IDs `19,19,19`
   - **One large model with isolation:** the full `7g` → ID `0`

   > **MIG layouts are constrained by fixed memory-slice placement, not just total memory** — never
   > sum nominal GB and assume any combination fits. A `3g + 2g + 2g` layout (`9,14,14`) is **not**
   > realizable, for example, because the second `2g` has no legal placement after a `3g`. And
   > `nvidia-smi mig -lgip` `Free/Total` tracks **compute** (GPC) slices, so it overstates the number
   > of instances you can actually create (QA observed only 3 creatable `1g` instances on a 61x
   > Station even though `Free/Total` reported 7). Always validate a specific layout with
   > `nvidia-smi mig -lgipp` before relying on it.

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
```

> **Do not run `nvidia-fabricmanager` on DGX Station.** It has a single GB300 over NVLink-C2C (no
> NVSwitch fabric), so Fabric Manager is not installed and `systemctl start nvidia-fabricmanager`
> fails with "Unit not found." NVLink-C2C re-initializes automatically after MIG is disabled. If MIG
> mode is stuck in a "pending" state, reset the GPU instead: `sudo nvidia-smi -i <GB300_INDEX> --gpu-reset`.
