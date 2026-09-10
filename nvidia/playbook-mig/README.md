# Set Up Multi-Instance GPU (MIG)

> Isolated GPU instances with dedicated memory and compute

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)
  - [MIG reconfiguration (day-2 operations)](#mig-reconfiguration-day-2-operations)
  - [Profile selection guidance](#profile-selection-guidance)
  - [Post-disable verification](#post-disable-verification)

---

## Overview

## Basic idea

**Multi-Instance GPU (MIG)** lets you partition a single NVIDIA B300 GPU on your hardware platform into multiple smaller GPU instances. Each instance has dedicated memory and compute, so you can run multiple workloads or users on one physical GPU without sharing memory. This playbook walks you through enabling MIG, creating a B300 MIG layout, and using the instances from bare-metal apps or containers.

MIG is controlled via `nvidia-smi`: you enable MIG mode, then create GPU and compute instances using B300 profile IDs (for example, 1g.35gb, 2g.70gb, 7g.278gb). When you no longer need partitioning, you disable MIG to restore full-GPU and NVLink P2P.

## What you'll accomplish

You'll have MIG enabled and configured on the B300 GPUs on your **hardware platform** and know how to use the instances.

- **Enable MIG** on all B300 GPUs or on a per-GPU basis
- **Create a MIG layout** using B300 profile IDs (with a known-good example for multiple GPUs)
- **Verify** the layout with `nvidia-smi -L` and `sudo nvidia-smi mig -lgi` / `-lci`
- **Run workloads** by setting `CUDA_VISIBLE_DEVICES` to a MIG UUID or by using the container/Kubernetes flows from the MIG User Guide
- **Disable MIG** when you need full-GPU mode and NVLink again

## What to know before starting

**Required:**

- Basic Linux command line and use of `sudo`
- Familiarity with `nvidia-smi` and GPU indices

**Optional:**

- Understanding of `CUDA_VISIBLE_DEVICES` and containers if you plan to run workloads on MIG instances

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Station** | DGX OS (Linux) | Large HBM + Grace DRAM | MIG via `nvidia-smi`; B300 profile IDs (for example, 1g.35gb) | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- NVIDIA GB300 (B300) GPUs on the hardware platform
- No additional storage requirement for MIG configuration itself

**Software requirements**

- NVIDIA driver and `nvidia-smi` installed and working: `nvidia-smi`. Use a driver that supports MIG on B300 (see [Troubleshooting](troubleshooting.md) if `nvidia-smi -mig 1` reports that MIG mode is not supported)
- Root or sudo access to run `nvidia-smi -mig 1`, `-mig 0`, and `nvidia-smi mig -cgi ... -C`
- For containers/Kubernetes: nvidia-container-toolkit and MIG support as described in the [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)

## Ancillary files

No local ancillary files are required. All steps use `nvidia-smi` and MIG commands on your hardware platform. For container and Kubernetes setup, see the [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) (Getting Started with MIG and Kubernetes sections).

## Time & risk

- **Estimated time:** 15 MIN (layout design may take longer if you customize profiles per GPU)
- **Risk level:** Low to Medium
  - Enabling or disabling MIG requires sudo and affects all workloads on that GPU
  - Disabling MIG removes all MIG instances; ensure Fabric Manager is running so NVLink/NVSwitch re-initialize correctly
- **Rollback:** Destroy all MIG instances with `sudo nvidia-smi mig -dci -i N` and `sudo nvidia-smi mig -dgi -i N` for each GPU index N, then run `sudo nvidia-smi -mig 0` to disable MIG and return to a single full-GPU instance per GB300. Ensure **Fabric Manager** is running after disabling MIG: `sudo systemctl status nvidia-fabricmanager` (start if needed: `sudo systemctl start nvidia-fabricmanager`)
- **Last Updated:** 08/03/2026
  - Enable MIG, create B300 layouts, run workloads on MIG instances, and restore full-GPU mode on supported hardware platforms

## Instructions

## Step 1. Prerequisites and verify B300 GPUs

Ensure your hardware platform has B300 GPUs (GB300), a driver that supports MIG (see [Troubleshooting](troubleshooting.md)), and that `nvidia-smi` is available. You need root or sudo to enable MIG and create instances.

**Before enabling MIG:** All GPU processes must be stopped. Desktop environments (for example, GNOME, Xwayland), NVIDIA services (for example, nvsm_core, nvidia-pe, nv-hostengine), or workloads like vLLM can hold the GPU and cause "In use by another client" when you run MIG commands. Check what is using the GPUs:

```bash
sudo fuser -v /dev/nvidia*
```

Stop or suspend any processes that are using the GPUs before proceeding to Step 2.

```bash
nvidia-smi
nvidia-smi -L
```

Expected output should list one or more **NVIDIA GB300** devices. If you see GB300 GPUs, you can proceed to enable MIG.

## Step 2. Enable MIG mode on the B300 GPUs

Ensure no GPU processes are running (see Step 1). Enable MIG for all GPUs or for a specific GPU. This must be done with elevated privileges.

**Enable MIG on all GPUs:**

```bash
sudo nvidia-smi -mig 1
```

**Or enable MIG on a single GPU (for example, GPU 0 only):**

```bash
sudo nvidia-smi -i 0 -mig 1
```

**Expected output:** Success typically shows no error message; the command returns to the prompt. If you see "In use by another client", stop all GPU processes (desktop, services, containers) and run `sudo fuser -v /dev/nvidia*` to confirm nothing is using the GPUs, then retry.

If MIG mode shows **Pending** after enablement (for example, in `nvidia-smi -q | grep -i mig`), wait a short time and run the command again, or reboot the system to allow the driver to apply the MIG state.

Enabling MIG partitions each B300 into multiple GPU Instances; you will create and assign profiles in the next steps.

## Step 3. Verify MIG mode and inspect B300 profiles

Confirm that MIG mode is enabled:

```bash
nvidia-smi -q | grep -i mig
## or for a specific GPU:
nvidia-smi -i 0 -q | grep -i "MIG Mode"
```

Expected output should show MIG Mode: **Enabled**.

List the GPU Instance Profiles available on a B300 (for example, GPU 0). These profile IDs are used when creating MIG instances:

```bash
nvidia-smi mig -lgip -i 0
```

On GB300 you should see profiles such as (exact memory sizes may match your driver; IDs are used in commands):

- MIG 1g.35gb (ID 19)
- MIG 1g.35gb+me (ID 20)
- MIG 1g.70gb (ID 15)
- MIG 2g.70gb (ID 14)
- MIG 3g.139gb (ID 9)
- MIG 4g.139gb (ID 5)
- MIG 7g.278gb (ID 0)

Note the **IDs**; you will pass them to `-cgi` when creating the layout.

## Step 4. Create a MIG layout (example for B300)

Create GPU and compute instances using the profile IDs from Step 3. The basic pattern is:

```bash
sudo nvidia-smi mig -cgi <profile_id,profile_id,...> -C -i <gpu_index>
```

This example assumes a **6-GPU** hardware platform. If you have fewer GPUs (for example, 1 or 2), run only the `-cgi` lines for the GPU indices that exist on your system (for example, `-i 0` and `-i 1` only). Each GPU can have any combination of profiles that fits within its capacity:

```bash
## GPU 0: 7 × 1g.35gb
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -C -i 0

## GPU 1: 4 × 1g.70gb
sudo nvidia-smi mig -cgi 15,15,15,15 -C -i 1

## GPU 2: 3 × 2g.70gb
sudo nvidia-smi mig -cgi 14,14,14 -C -i 2

## GPU 3: 2 × 3g.139gb
sudo nvidia-smi mig -cgi 9,9 -C -i 3

## GPU 4: 1 × 4g.139gb
sudo nvidia-smi mig -cgi 5 -C -i 4

## GPU 5: 1 × 7g.278gb (full GPU as a single MIG instance)
sudo nvidia-smi mig -cgi 0 -C -i 5
```

You can choose any valid combination of profile IDs per GPU that fits within the GB300’s capacity; the above is a known-good example.

## Step 5. Verify MIG instances

Check the resulting MIG device layout:

```bash
nvidia-smi -L
```

You should see each physical GPU (for example, **NVIDIA GB300**) followed by its MIG devices:

```
GPU 0: NVIDIA GB300 (UUID: GPU-...)
  MIG 1g.35gb Device 0: (UUID: MIG-...)
  MIG 1g.35gb Device 1: (UUID: MIG-...)
  ...
GPU 1: NVIDIA GB300 (UUID: GPU-...)
  MIG 1g.70gb Device 0: (UUID: MIG-...)
  ...
```

To list GPU instances and compute instances (requires sudo):

```bash
sudo nvidia-smi mig -lgi     # list GPU instances
sudo nvidia-smi mig -lci     # list compute instances
```

## Step 6. Using the MIG devices

**Bare-metal CUDA applications:** set `CUDA_VISIBLE_DEVICES` to a MIG device UUID (from `nvidia-smi -L`):

```bash
export CUDA_VISIBLE_DEVICES=MIG-<uuid>
./your_app
```

**Verify a MIG instance is visible:** From the same shell where you set `CUDA_VISIBLE_DEVICES`, run `nvidia-smi`. You should see only the single MIG device (for example, one "MIG 1g.35gb" device):

```bash
export CUDA_VISIBLE_DEVICES=MIG-<uuid-from-nvidia-smi-L>
nvidia-smi
```

**Containers (Docker):** Use the MIG device UUID in the `--gpus` option. Example:

```bash
docker run --gpus '"device=MIG-<uuid>"' nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
```

Replace `<uuid>` with a full MIG UUID from `nvidia-smi -L`. For Kubernetes and nvidia-container-toolkit workflows, see the [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) (Getting Started with MIG and Kubernetes sections).

## Step 7. Disabling MIG and restoring full GPU

When you need full NVLink P2P and a single full-GPU instance again, you must **destroy all MIG instances first**, then disable MIG. If you run `sudo nvidia-smi -mig 0` without destroying instances, it will fail with "In use by another client."

**1. Destroy compute instances and GPU instances on each GPU.** For each GPU index that has MIG instances, run (replace `N` with the GPU index, for example 0, 1, … 5 for a 6-GPU system):

```bash
## Destroy all compute instances on GPU N (required before destroying GPU instances)
sudo nvidia-smi mig -dci -i N

## Destroy all GPU instances on GPU N
sudo nvidia-smi mig -dgi -i N
```

Repeat for every GPU that has MIG instances. Example for a 6-GPU system:

```bash
for i in 0 1 2 3 4 5; do sudo nvidia-smi mig -dci -i $i; sudo nvidia-smi mig -dgi -i $i; done
```

**2. Disable MIG mode on all GPUs:**

> [!WARNING]
> This returns each GB300 to a single full-GPU instance. Any workloads using MIG UUIDs must be stopped first and will need to be reconfigured or restarted.

```bash
sudo nvidia-smi -mig 0
```

**3. Verify MIG is fully disabled:**

```bash
nvidia-smi -q | grep -A2 "MIG Mode"
```

Expected output should show `Current: Disabled` for each GPU.

On hardware platforms with NVLink/NVSwitch fabric, ensure **Fabric Manager** is running after disabling MIG so NVLinks and NVSwitch fabric are re-initialized (see [Troubleshooting](troubleshooting.md)).

## Troubleshooting

| Symptom | Cause | Fix |
|--------|--------|-----|
| `nvidia-smi -mig 1` fails or "MIG mode not supported" | Driver does not support MIG on this GPU, or GPU is not MIG-capable | Use a driver that supports MIG on GB300 (see the [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) for supported versions). Check `nvidia-smi -q` for driver and GPU model. |
| "In use by another client" when running `-mig 1`, `-cgi`, or `-mig 0` | GPU is held by another process or MIG instances still exist | **For enable/create:** Stop all GPU processes (desktop, vLLM, nvsm_core, nvidia-pe, nv-hostengine, and similar). Run `sudo fuser -v /dev/nvidia*` to see what is using the GPUs; stop those processes and retry. **For disable:** Destroy all MIG instances first: run `sudo nvidia-smi mig -dci -i N` then `sudo nvidia-smi mig -dgi -i N` for each GPU index N that has instances, then run `sudo nvidia-smi -mig 0`. |
| `nvidia-smi mig -cgi ... -C -i N` fails (for example, "Invalid combination") | Profile combination exceeds GPU capacity or invalid IDs | Run `nvidia-smi mig -lgip -i N` and use only listed profile IDs. Ensure the sum of instance sizes does not exceed the GB300's capacity for that GPU. |
| MIG instances not visible after creation | Instances not created or wrong GPU index | Run `nvidia-smi -L` and `sudo nvidia-smi mig -lgi` to confirm. Re-run the `-cgi` commands for the correct `-i <gpu_index>`. |
| App doesn't see MIG device when using `CUDA_VISIBLE_DEVICES=MIG-<uuid>` | Wrong UUID or app not using `CUDA_VISIBLE_DEVICES` | Get UUIDs from `nvidia-smi -L`. Export `CUDA_VISIBLE_DEVICES=MIG-<uuid>` in the same shell before launching the app. |
| "Insufficient Permissions" when running `nvidia-smi mig -lgi` or `-lci` | Listing instances requires root | Use `sudo nvidia-smi mig -lgi` and `sudo nvidia-smi mig -lci`. |
| After `nvidia-smi -mig 0`, NVLink or fabric issues | Fabric Manager not re-initializing | Ensure Fabric Manager is running after disabling MIG: `sudo systemctl status nvidia-fabricmanager`; start if needed with `sudo systemctl start nvidia-fabricmanager`. See the [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) for details. |
| Permission denied when running `nvidia-smi -mig` or `mig -cgi` | Need root for MIG operations | Use `sudo` for `nvidia-smi -mig 1/0`, `nvidia-smi mig -cgi ... -C`, `-dci`, and `-dgi`. |

### MIG reconfiguration (day-2 operations)

To change the MIG layout (for example, add or remove instances, or switch profiles), destroy existing instances on the affected GPU(s), then create the new layout:

1. **Destroy compute instances and GPU instances** on each GPU you want to reconfigure (replace `N` with the GPU index):
   ```bash
   sudo nvidia-smi mig -dci -i N
   sudo nvidia-smi mig -dgi -i N
   ```
2. **Create the new layout** with `sudo nvidia-smi mig -cgi <profile_ids> -C -i N` as in the Instructions (Step 4).

Workloads using the old MIG UUIDs must be stopped before destroying instances; they will need to be restarted with the new UUIDs from `nvidia-smi -L` after recreation.

### Profile selection guidance

| Profile (typical name) | Use case |
|------------------------|----------|
| 1g.35gb (ID 19) | Small inference, dev/test, many concurrent small jobs |
| 1g.70gb (ID 15) | Slightly larger inference or light training |
| 2g.70gb (ID 14) | Medium inference or small training |
| 3g.139gb (ID 9) | Larger inference or medium training |
| 4g.139gb (ID 5) | Heavy inference or moderate training |
| 7g.278gb (ID 0) | Full-GPU as single MIG instance; max memory per partition |

Exact profile names may vary by driver (for example, 1g.34gb vs 1g.35gb); use the **profile IDs** from `nvidia-smi mig -lgip -i 0` in your `-cgi` commands.

### Post-disable verification

After running `sudo nvidia-smi -mig 0`, confirm MIG is fully disabled:

```bash
nvidia-smi -q | grep -A2 "MIG Mode"
```

Expected output should show `Current: Disabled` for each GPU. If you still see MIG devices in `nvidia-smi -L`, destroy any remaining instances with `-dci`/`-dgi` per GPU, then run `-mig 0` again.

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
