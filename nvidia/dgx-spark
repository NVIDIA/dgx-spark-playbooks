# DGX Dashboard

> Monitor your DGX system and launch JupyterLab

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

The DGX Dashboard is a web application that runs locally on DGX Spark devices, providing a graphical interface for system updates, resource monitoring, and an integrated JupyterLab environment. Users can access the dashboard locally from the app launcher or remotely through NVIDIA Sync or SSH tunneling. The dashboard is the easiest way to update system packages and firmware when working remotely.

## What you'll accomplish

You will learn how to access and use the DGX Dashboard on your DGX Spark device. By the end of this walkthrough, you will be able to launch JupyterLab instances with pre-configured Python environments, monitor GPU performance, manage system updates, and run a sample AI workload using Stable Diffusion. You'll understand multiple access methods including desktop shortcuts, NVIDIA Sync, and manual SSH tunneling.

## What to know before starting

- Basic terminal usage for SSH connections and port forwarding
- Understanding of Python environments and Jupyter notebooks

## Prerequisites

**Hardware Requirements:**
-  NVIDIA Grace Blackwell GB10 Superchip System

**Software Requirements:**
- NVIDIA DGX OS
- NVIDIA Sync installed (for remote access method) or SSH client configured

## Ancillary files

- Python code snippet for SDXL found [here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/dgx-dashboard/assets/jupyter-cell.py)


## Time & risk

* **Duration:** 15-30 minutes for complete walkthrough including sample AI workload
* **Risk level:** Low - Web interface operations with minimal system impact
* **Rollback:** Stop JupyterLab instances through dashboard interface; no permanent system changes made during normal usage.
* **Last Updated:** 11/21/2025
  * Minor copyedits

## Instructions

## Step 1. Access DGX Dashboard

Choose one of the following methods to access the DGX Dashboard web interface:

**Option A: Desktop shortcut (local access)**

If you have local access to your DGX Spark device:

1. Log into the Ubuntu Desktop environment on your DGX Spark device
2. Open the Ubuntu app launcher by clicking on the bottom left corner of the screen
3. Click on the DGX Dashboard shortcut in the app launcher
4. The dashboard will open in your default web browser at `http://localhost:11000`

**Option B: NVIDIA Sync (recommended for remote access)**

If you have NVIDIA Sync installed on your local machine:

1. Click the NVIDIA Sync icon in your system tray
2. Select your DGX Spark device from the device list
3. Click "Connect"
4. Click "DGX Dashboard" to launch the dashboard
5. The dashboard will open in your default web browser at `http://localhost:11000` using an automatic SSH tunnel

Don't have NVIDIA Sync? [Install it here](/spark/connect-to-your-spark/sync)

**Option C: Manual SSH tunnels**

For manual remote access without NVIDIA Sync you must first [manually configure an SSH tunnel](/spark/connect-to-your-spark/manual-ssh).

You must open a tunnel for the Dashboard server (port 11000) and for JupyterLab if you want to access it remotely. Each user account will have a different assigned port number for JupyterLab.

1. Check your assigned JupyterLab port by SSH-ing into your DGX Spark and running the following command:

```bash
cat /opt/nvidia/dgx-dashboard-service/jupyterlab_ports.yaml
```

2. Look for your username and note the assigned port number.
3. Create a new SSH tunnel including both ports:

```bash
ssh -L 11000:localhost:11000 -L <ASSIGNED_PORT>:localhost:<ASSIGNED_PORT> <USERNAME>@<SPARK_DEVICE_IP>
```
Replace `<USERNAME>` with your DGX Spark device username and `<SPARK_DEVICE_IP>` with the device's IP address.

Replace `<ASSIGNED_PORT>` with the port number from the YAML file.

Open your web browser and navigate to `http://localhost:11000`.


## Step 2. Log into DGX Dashboard

Once the dashboard loads in your browser:

1. Enter your DGX Spark system username in the username field
2. Enter your system password in the password field
3. Click "Login" to access the dashboard interface

You should see the main dashboard with panels for JupyterLab management, system monitoring, and settings.

## Step 3. Launch JupyterLab instance

Create and start a JupyterLab environment:

1. Click the "Start" button in the right panel
2. Monitor the status as it transitions through: Starting → Preparing → Running
3. Wait for the status to show "Running" (this may take several minutes on the first launch)
4. Once "Running", if JupyterLab does not automatically open in your browser (a pop-up was blocked), you can click the "Open In Browser" button

When starting, a default working directory (/home/<USERNAME>/jupyterlab) is created and a virtual environment is set up automatically. You can
review the packages installed by looking at the `requirements.txt` file that is created in the working directory.

In the future, you can change the working directory, creating a new isolated environment, by clicking the "Stop" button, changing the path to the new working directory and then clicking the "Start" button again.

## Step 4. Test with sample AI workload

Verify your setup by running a simple Stable Diffusion XL image generation example:

1. In JupyterLab, create a new notebook: File → New → Notebook
2. Click "Python 3 (ipykernel)" to create the notebook
3. Add a new cell and paste the following code:

```python
import warnings
warnings.filterwarnings('ignore', message='.*cuda capability.*')
import tqdm.auto
tqdm.auto.tqdm = tqdm.std.tqdm

from diffusers import DiffusionPipeline
import torch
from PIL import Image
from datetime import datetime
from IPython.display import display

## --- Model setup ---
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    variant="fp16" if dtype==torch.float16 else None,
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
image.save(f"sdxl_output.png")
print(f"Saved image as sdxl_output.png")
```

4. Run the cell (Shift+Enter or click the Run button)
5. The notebook will download the model and generate an image (first run may take several minutes)

## Step 5. Monitor GPU utilization

While the image generation is running:

1. Switch back to the DGX Dashboard tab in your browser
2. Observe the GPU telemetry data in the monitoring panels

## Step 6. Stop JupyterLab instance

When finished with your session:

1. Return to the main DGX Dashboard tab
2. Click the "Stop" button in the JupyterLab panel
3. Confirm the status changes from "Running" to "Stopped"

## Step 6. Manage system updates

If system updates are available it will be indicated by a banner or on the Settings page.

From the Settings page, under the "Updates" tab:

1. Click "Update" to open the confirmation dialog
2. Click "Update Now" to initiate the update process
3. Wait for the update to complete and your device to reboot

> [!WARNING]
> System updates will upgrade packages, firmware (if available), and trigger a reboot. Save your work before proceeding.

## Step 7. Cleanup and rollback

To clean up resources and return your system to its original state:

1. Stop any running JupyterLab instances via dashboard
2. Delete the JupyterLab working directory

> [!WARNING]
> If you ran system updates, the only rollback is to restore from a system backup or recovery media.

No permanent changes are made to the system during normal dashboard usage.

## Step 8. Next steps

Now that you have DGX Dashboard configured, you can:

- Create additional JupyterLab environments for different projects
- Use the dashboard to manage system maintenance and updates

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| User can't run updates | User not in sudo group | Add user to sudo group: `sudo usermod -aG sudo <USERNAME>`; then run `newgrp docker`|
| JupyterLab won't start | Issue with current virtual environment | Change the working directory in the JupyterLab panel and start a new instance |
| SSH tunnel connection refused | Incorrect IP or port | Verify Spark device IP and ensure SSH service is running |
| GPU not visible in monitoring | Driver issues | Check GPU status with `nvidia-smi` |


For latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
