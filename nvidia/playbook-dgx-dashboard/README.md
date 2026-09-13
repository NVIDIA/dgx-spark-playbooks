# Monitor Your AI Compute with DGX Dashboard

> Notebook access, GPU status, and update controls in one place

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

DGX Dashboard is a web application that runs locally on your hardware platform. It provides a graphical interface for system updates, resource monitoring, and an integrated JupyterLab environment. You can open it from the desktop app launcher or reach it remotely through NVIDIA Sync or an SSH tunnel. When you work remotely, the dashboard is the easiest way to apply system package and firmware updates.

## What you'll accomplish

You'll access DGX Dashboard on your hardware platform, launch a JupyterLab instance with a pre-configured Python environment, monitor GPU utilization, manage system updates, and run a sample AI workload.

## What to know before starting

**Required:**

- Basic terminal usage for SSH connections and port forwarding
- Familiarity with Python environments and Jupyter notebooks

**Optional:**

- NVIDIA Sync installed on your local machine for one-click remote dashboard access

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | DGX Dashboard at `http://localhost:11000`; JupyterLab via dashboard | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Hardware platform powered on, networked, and reachable for local desktop or SSH access

**Software requirements**

- NVIDIA DGX OS on the hardware platform
- Web browser access to port `11000` (local or tunneled)
- NVIDIA Sync installed (recommended remote access) or an SSH client configured for manual tunneling

## Ancillary files

All required assets can be found [in the playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-dgx-dashboard/).

- `assets/jupyter-cell.py` — Sample Stable Diffusion XL notebook cell for verifying GPU-backed generation in JupyterLab

## Time & risk

- **Estimated time:** 30 MIN (longer on first JupyterLab launch and first model download)
- **Risk level:** Low
  - Normal dashboard and JupyterLab usage makes no permanent system changes
  - System updates upgrade packages and firmware and trigger a reboot — save work before updating
- **Rollback:** Stop JupyterLab instances in the dashboard; delete the JupyterLab working directory if needed. After system updates, restore from a system backup or recovery media if you must roll back.
- **Last Updated:** 07/31/2026
  - Access DGX Dashboard locally or remotely, launch JupyterLab, monitor GPU use, apply updates, and run a sample workload

## Instructions

> [!TIP]
> For remote access, NVIDIA Sync is the recommended path. Complete remote-access / NVIDIA Sync setup for your hardware platform before using Option B below. Option C covers manual SSH tunnels if you prefer not to use NVIDIA Sync.

## Step 1. Access DGX Dashboard

Choose one of the following methods to open the DGX Dashboard web interface.

**Option A: Desktop shortcut (local access)**

If you have local access to your hardware platform:

1. Log into the desktop environment on your hardware platform
2. Open the app launcher (for example, from the bottom-left corner of the screen)
3. Click the **DGX Dashboard** shortcut
4. The dashboard opens in your default web browser at `http://localhost:11000`

**Option B: NVIDIA Sync (recommended for remote access)**

If you have NVIDIA Sync installed on your local machine:

1. Click the NVIDIA Sync icon in your system tray
2. Select your hardware platform from the device list
3. Click **Connect**
4. Click **DGX Dashboard** to launch the dashboard
5. The dashboard opens in your default web browser at `http://localhost:11000` using an automatic SSH tunnel

**Option C: Manual SSH tunnels**

For remote access without NVIDIA Sync, configure an SSH tunnel to the hardware platform.

Open a tunnel for the Dashboard server (port `11000`) and for JupyterLab if you want remote notebook access. Each user account has a different assigned JupyterLab port.

1. Check your assigned JupyterLab port by SSH-ing into your hardware platform and running:

```bash
cat /opt/nvidia/dgx-dashboard-service/jupyterlab_ports.yaml
```

2. Look for your username and note the assigned port number.
3. Create an SSH tunnel that includes both ports:

```bash
ssh -L 11000:localhost:11000 -L <ASSIGNED_PORT>:localhost:<ASSIGNED_PORT> <USERNAME>@<HARDWARE_IP>
```

Replace `<USERNAME>` with your hardware platform username and `<HARDWARE_IP>` with the hardware platform's reachable IP address. Replace `<ASSIGNED_PORT>` with the port number from the YAML file.

Open your web browser and navigate to `http://localhost:11000`.

## Step 2. Log into DGX Dashboard

Once the dashboard loads in your browser:

1. Enter your hardware platform system username
2. Enter your system password
3. Click **Login**

You should see the main dashboard with panels for JupyterLab management, system monitoring, and settings.

## Step 3. Launch JupyterLab

Create and start a JupyterLab environment:

1. Click the **Start** button in the right panel
2. Monitor the status as it transitions through: Starting → Preparing → Running
3. Wait for the status to show **Running** (first launch may take several minutes)
4. If JupyterLab does not open automatically (for example, a pop-up was blocked), click **Open In Browser**

When starting, a default working directory (`/home/<USERNAME>/jupyterlab`) is created and a virtual environment is set up automatically. Review installed packages in the `requirements.txt` file created in that working directory.

To use a different working directory later, click **Stop**, change the path, then click **Start** again to create a new isolated environment.

## Step 4. Test with a sample AI workload

Verify your setup by running a Stable Diffusion XL image generation example:

1. In JupyterLab, create a new notebook: **File → New → Notebook**
2. Select **Python 3 (ipykernel)**
3. Add a new cell and paste the following code (the same cell is provided in `assets/jupyter-cell.py`):

```python
import warnings
warnings.filterwarnings('ignore', message='.*cuda capability.*')
import tqdm.auto
tqdm.auto.tqdm = tqdm.std.tqdm

from diffusers import DiffusionPipeline
import torch
from PIL import Image
from IPython.display import display

## --- Model setup ---
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    variant="fp16" if dtype == torch.float16 else None,
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

## --- Prompt setup ---
prompt = "a cozy modern reading nook with a big window, soft natural light, photorealistic"
negative_prompt = "low quality, blurry, distorted, text, watermark"

## --- Generation settings ---
height = 1024
width = 1024
steps = 30
guidance = 7.0

## --- Generate ---
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    guidance_scale=guidance,
    height=height,
    width=width,
)

## --- Save to file ---
image: Image.Image = result.images[0]
display(image)
image.save("sdxl_output.png")
print("Saved image as sdxl_output.png")
```

4. Run the cell (Shift+Enter or the Run button)
5. The notebook downloads the model and generates an image (first run may take several minutes)

## Step 5. Monitor GPU utilization

While the image generation is running:

1. Switch back to the DGX Dashboard tab in your browser
2. Observe the GPU telemetry data in the monitoring panels

## Step 6. Stop JupyterLab

When finished with your session:

1. Return to the main DGX Dashboard tab
2. Click **Stop** in the JupyterLab panel
3. Confirm the status changes from **Running** to **Stopped**

## Step 7. Manage system updates

If updates are available, a banner appears or the Settings page indicates them.

From the Settings page, under the **Updates** tab:

1. Click **Update** to open the confirmation dialog
2. Click **Update Now** to start the update
3. Wait for the update to complete and the hardware platform to reboot

> [!WARNING]
> System updates upgrade packages and firmware (when available) and trigger a reboot. Save your work before proceeding.

## Step 8. Cleanup

To clean up resources after this walkthrough:

1. Stop any running JupyterLab instances via the dashboard
2. Delete the JupyterLab working directory if you no longer need it

> [!WARNING]
> If you ran system updates, the only rollback is to restore from a system backup or recovery media.

No permanent changes are made to the system during normal dashboard usage.

## Step 9. Next steps

With DGX Dashboard configured, you can:

- Create additional JupyterLab environments for different projects
- Use the dashboard for ongoing system maintenance and updates
- Explore other playbooks that build on local notebooks and GPU-backed workloads

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| User can't run updates | User not in the sudo group | Add the user to the sudo group: `sudo usermod -aG sudo <USERNAME>`, then log out and back in so group membership applies |
| JupyterLab won't start | Issue with the current virtual environment | Change the working directory in the JupyterLab panel and start a new instance |
| SSH tunnel connection refused | Incorrect IP or port | Verify the hardware platform IP and ensure the SSH service is running |
| GPU not visible in monitoring | Driver or GPU visibility issue | Check GPU status with `nvidia-smi` |

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
