# Accelerate Single-Cell RNA Data Analysis

> Run a Scanpy-style pipeline end to end on a GPU

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Single-cell RNA sequencing (scRNA-seq) lets researchers study gene activity in each cell on its own, exposing variation, cell types, and cell states that bulk methods hide. These large, high-dimensional datasets take heavy compute to handle.

This playbook shows an end-to-end GPU-powered workflow for scRNA-seq using [RAPIDS-singlecell](https://rapids-singlecell.readthedocs.io/en/latest/), a RAPIDS-powered library in the [scverse® ecosystem](https://github.com/scverse). It follows the familiar [Scanpy API](https://scanpy.readthedocs.io/en/stable/) and lets researchers run data preprocessing, quality control (QC) and cleanup, visualization, and investigation faster than CPU tools by working with sparse count matrices directly on the GPU.

## What you'll accomplish

You'll run a Scanpy-style single-cell RNA analysis pipeline end to end on the GPU on your **hardware platform**. The workflow covers:

1. GPU-accelerated data loading and preprocessing
2. QC cells visually to understand the data
3. Filter unusual cells
4. Remove unwanted sources of variation
5. Cluster and visualize PCA and UMAP data
6. Batch correction and analysis using Harmony, k-nearest neighbors, UMAP, and t-SNE
7. Explore biological information with differential expression analysis and trajectory analysis

## What to know before starting

**Required:**

- Experience working in a Linux terminal
- Basic understanding of Docker containers
- Familiarity with Jupyter Notebook or JupyterLab
- Basic Python knowledge

**Optional:**

- Familiarity with Scanpy or other scverse single-cell workflows
- Background in genomics or single-cell RNA analysis concepts

**Terms to know:**

- The rapids-singlecell library mirrors the Scanpy API from scverse, so users familiar with the standard CPU workflow can adapt to GPU acceleration through CuPy and NVIDIA RAPIDS cuML and cuGraph.
- **Algorithmic precision:** Unlike Scanpy's CPU implementation, which uses approximate nearest-neighbor search, this GPU implementation computes the exact graph; small differences in results are expected and valid.
- **Parameter sensitivity:** When performing t-SNE, the number of nearest neighbors must be at least 3× to avoid distortion.

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | RAPIDS 25.10 notebooks container (`nvcr.io/nvidia/rapidsai/notebooks:25.10-cuda13-py3.13`); JupyterLab on port 8888 | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Minimum 40 GB memory free for the Docker container and GPU-accelerated data processing
- At least 30 GB available storage for the Docker container and data files
- High-speed internet connection recommended

**Software requirements**

- Working NVIDIA and CUDA drivers: `nvidia-smi`
- Docker: `docker --version`
- Git: `git --version`
- Network access to pull the RAPIDS notebooks container, libraries, and the demo dataset (`dli_census.h5ad`)
- Web browser access to JupyterLab on port 8888 (or a forwarded local port)

## Ancillary files

All required assets can be found [in the playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-single-cell/). In the running playbook, they are available under the `playbook` folder.

- `assets/scRNA_analysis_preprocessing.ipynb` — Main playbook notebook
- `assets/README.md` — Quick start guide to the playbook environment (also available as `START_HERE.md` in JupyterLab)
- `assets/setup/start_playbook.sh` — Starts the playbook in a Docker container
- `assets/setup/setup_playbook.sh` — Configures the Docker container before you enter the JupyterLab environment
- `assets/setup/requirements.txt` — Library list used by setup commands to install into the playbook environment
- `cudf`, `cuml`, and `cugraph` folders — Additional RAPIDS example notebooks included in the Docker container when you start it

## Time & risk

- **Estimated time:** 15 MIN for first run (full notebook pipeline typically ~2–3 minutes after the container is ready)
- **Risk level:** Low
  - Minimal, as this workflow runs in a Docker container
  - Container pull, environment build, or dataset download may fail due to network issues
  - Large datasets may trigger out-of-memory (OOM) errors; kill or restart kernels to free GPU resources between stages
- **Rollback:** Stop the Docker container and remove the cloned repository to fully remove the installation. If an OOM error occurs, kill all kernels to free GPU memory and restart either the notebook or the entire playbook.
- **Last Updated:** 08/03/2026
  - Run a Scanpy-style scRNA-seq pipeline with RAPIDS-singlecell in a RAPIDS notebooks container

## Instructions

## Step 1. Verify your environment

Confirm GPU access, Git, and Docker on your hardware platform. Open a terminal, then run:

```bash
nvidia-smi
git --version
docker --version
```

- `nvidia-smi` prints information about your GPU. If it fails, your GPU is not properly configured.
- `git --version` prints something like `git version 2.43.0`. If Git is missing, install it and retry.
- `docker --version` prints something like `Docker version 28.3.3, build 980b856`. If Docker is missing, install it and retry. If you see a permission denied error, add your user to the docker group by running `sudo usermod -aG docker $USER && newgrp docker`.

## Step 2. Installation

Clone the playbook assets and start the containerized environment:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd client-hardware-playbooks/nvidia/playbook-single-cell/assets
bash ./setup/start_playbook.sh
```

`start_playbook.sh` will:

1. Pull the RAPIDS 25.10 notebooks Docker container
2. Build the environments needed for the playbook in the container using `setup_playbook.sh`
3. Start JupyterLab

Keep the terminal window open while using the playbook.

You can access JupyterLab in these ways:

1. At `http://127.0.0.1:8888` if running locally on the hardware platform
2. At `http://<HARDWARE_IP>:8888` if accessing the hardware platform over your network
3. By creating an SSH tunnel with `ssh -L 8888:localhost:8888 username@<HARDWARE_IP>`, then opening `http://127.0.0.1:8888` in a browser on your local machine

Once in JupyterLab, you'll see `scRNA_analysis_preprocessing.ipynb` and the folders `cudf`, `cuml`, `cugraph`, and `playbook`.

- `scRNA_analysis_preprocessing.ipynb` is the playbook notebook — open it by double-clicking the file
- `cudf`, `cuml`, and `cugraph` contain standard RAPIDS library example notebooks for further exploration
- `playbook` contains the playbook files. The contents of this folder are read-only inside a rootless Docker container

If you want to install any of the playbook notebooks on your own system, check the READMEs that accompany the notebook assets.

## Step 3. Run the notebook

In JupyterLab, run `scRNA_analysis_preprocessing.ipynb`.

Use `Shift + Enter` to run each cell at your own pace, or `Run > Run All` to run all cells.

After exploring `scRNA_analysis_preprocessing`, you can open other RAPIDS notebooks in the `cudf`, `cuml`, and `cugraph` folders and run them the same way.

## Step 4. Download your work

Because the Docker container is not privileged and cannot write back to the host system, use JupyterLab to download any files you want to keep after the container shuts down.

Right-click the file in the browser and select **Download**.

## Step 5. Cleanup (optional)

When you have downloaded your work, return to the terminal where you started the playbook.

In that terminal:

1. Press `Ctrl + C`
2. Quickly enter `y` and press `Enter` at the prompt, or press `Ctrl + C` again
3. The Docker container will shut down

> [!WARNING]
> This deletes all data that was not already downloaded from the Docker container. The browser window may still show cached files if it remains open.

## Step 6. Next steps

Once you're comfortable with this foundational workflow:

1. Explore additional single-cell notebooks in the [Single-Cell Analysis AI Blueprint](https://github.com/NVIDIA-AI-Blueprints/single-cell-analysis-blueprint/tree/main)
2. Apply RAPIDS-singlecell methods to larger datasets from [10x Genomics](https://www.10xgenomics.com/datasets) or [CZ CELLxGENE](https://cellxgene.cziscience.com/)
3. Learn more about the [scverse community](https://scverse.org/about/) and [how to join](https://scverse.org/join/)

## Step 7. Further support

For questions or issues about these notebooks, open an issue on the [Single-Cell Analysis AI Blueprint](https://github.com/NVIDIA-AI-Blueprints/single-cell-analysis-blueprint/tree/main) repository.

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every platform listed in this playbook.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| Docker is not found | All hardware platforms | Docker is not installed or not on `PATH` | Install Docker (for example: `curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`). You will be prompted for your password. |
| Docker command exits with a permissions error | All hardware platforms | Your user is not in the `docker` group | Run `sudo groupadd docker && sudo usermod -aG docker $USER`, then close the terminal, open a new one, and try again |
| Docker container download, environment build, or data download fails | All hardware platforms | Connectivity issue or a resource temporarily unavailable | Retry later; confirm network access to the container registry and dataset hosts |
| JupyterLab UI not reachable in the browser | All hardware platforms | Port 8888 blocked, wrong host, or missing SSH forward | Use `http://127.0.0.1:8888` or `http://<HARDWARE_IP>:8888`; if remote, forward with `ssh -L 8888:localhost:8888 username@<HARDWARE_IP>` |
| Out-of-memory (OOM) errors during notebook cells | All hardware platforms | GPU memory pressure from large datasets or leftover kernels | Kill or restart Jupyter kernels to free GPU memory; flush the UMA buffer cache if needed (see note below); restart the notebook or playbook |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
