# Single-cell RNA Sequencing

> An end-to-end GPU-powered workflow for scRNA-seq using RAPIDS

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Single-cell RNA sequencing (scRNA-seq) lets researchers study gene activity in each cell on its own, exposing variation, cell types, and cell states that bulk methods hide. But these large, high-dimensional datasets take heavy compute to handle.

This playbook shows an end-to-end GPU-powered workflow for scRNA-seq using [RAPIDS-singlecell](https://rapids-singlecell.readthedocs.io/en/latest/), a RAPIDS powered library in the [scverse® ecosystem](https://github.com/scverse). It follows the familiar [Scanpy API](https://scanpy.readthedocs.io/en/stable/) and lets researchers run the steps of data preprocessing, quality control (QC) and cleanup, visualization, and investigation faster than CPU tools by working with sparse count matrices directly on the GPU.

## What you'll accomplish

1. GPU-Accelerated Data Loading & Preprocessing
2. QC cells visually to understand the data
3. Filter unusual cells
4. Remove unwanted sources of variation 
5. Cluster and visualize PCA and UMAP data
6. Batch Correction and analysis using Harmony, k-nearest neighbors, UMAP, and tSNE
7. Explore the biological information from the data with differential expression analysis and trajectory analysis

The README elaborates on these steps.

## What to know before starting

- The rapids-singlecell library mimics the Scanpy API from scverse, allowing users familiar with the standard CPU workflow to easily adapt to GPU acceleration through cuPy and NVIDIA RAPIDS cuML and cuGraph.
- Algorithmic Precision: Unlike Scanpy's CPU implementation which uses approximate nearest neighbor search, this GPU implementation computes the exact graph; consequently, small differences in results are expected and valid.
- Parameter Sensitivity: When performing t-SNE, the number of nearest neighbors must be at least 3x to avoid distortion

## Prerequisites
**Hardware Requirements:**
- NVIDIA Grace Blackwell GB10 Superchip System (DGX Spark)
- Minimum 40GB Unified memory free for docker container and GPU accelerated data processing
- At least 30GB available storage space for docker container and data files
- High Speed network connectivity
- High speed internet connection recommended

**Software Requirements:**
- NVIDIA DGX OS
- Docker

## Ancillary files

All required assets can be found [in the Single-cell RNA Sequencing repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/single-cell/). In the running playbook, they will all be found under the `playbook` folder.

- `scRNA_analysis_preprocessing.ipynb` - Main playbook notebook.  
- `README.md` - Quick Start Guide to the Playbook Environment.  It will also be found in the main directory of the Jupyter Lab.  Please start there!
- `/setup/start_playbook.sh` - Script to start the install of the playbook in a Docker container
- `/setup/setup_playbook.sh` - Configures the Docker container before user enters JupyterLab environment
- `/setup/requirements.txt` - used as a list of libraries that commands in setup_playbook will install into the playbook environment 

## Time & risk
* **Estimated Time:** ~15 minutes for first run

  - Total Notebook Processing Time: Approximately 2-3 minutes for the full pipeline (~130 seconds recorded in demo).
  - Data Loading: ~1.7 seconds.
  - Preprocessing: ~21 seconds.
  - Post-processing (Clustering/Diff Exp): ~104 seconds.
  - Data: Internet access to download the docker container, libraries, and demo dataset (dli_census.h5ad).

* **Risks**

  - GPU Memory Constraints: The workflow is very GPU memory intensive. Large datasets may trigger Out Of Memory (OOM) errors.
  - Kernel Management: You may need to kill/restart kernels to free up GPU resources between workflow stages.
  - Rollback: If an OOM error occurs, kill all kernels to free GPU memory and restart either the specific notebook or the entire playbook.

* **Last Updated:** 01/02/2026
  * First Publication

## Instructions

## Step 1. Verify your environment

Let's first verify that you have a working GPU, git, and Docker.  Open up Terminal, then copy and paste in the below commands:

```bash
nvidia-smi
git --version
docker --version
```

- `nvidia-smi` will output information about your GPU.  If it doesn't, your GPU is not properly configured.
- `git --version` will print something like `git version 2.43.0`.  If you get an error saying that git is not installed, please reinstall it.
- `docker --version` will print something like `Docker version 28.3.3, build 980b856`.  If you get an error saying that Docker is not installed, please reinstall it. If you see a permission denied error, add your user to the docker group by running `sudo usermod -aG docker $USER && newgrp docker`.

## Step 2. Installation
Open up Terminal, then copy and paste in the below commands:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/single-cell/assets
bash ./setup/start_playbook.sh
```

start_playbook.sh will:

1. pull the RAPIDS 25.10 Notebooks Docker container
2. build all the environments needed for the playbook in the container using setup_playbook.sh
3. start JupyterLab

Please keep the Terminal window open while using the playbook.

You can access your JupyterLab server in two ways
1. at `http://127.0.0.1:8888` if running locally on the DGX Spark. 
2. at `http://<SPARK_IP>:8888` if using your DGX Spark headless over your network.

Once in JupyterLab, you'll be greeted with a directory containing scRNA_analysis_preprocessing.ipynb, and the folders `cuDF`, `cuML`, `cuGraph`, and `playbook`. 

- `scRNA_analysis_preprocessing.ipynb`is the playbook notebook.  You will want to open this by double clicking on the file.
- `cuDF`, `cuML`, `cuGraph` folders contain the standard RAPIDS library example notebooks to help you continue exploring.
- `playbook` contains the playbook files.  The contents of this folder are read-only inside of a rootless Docker Container.

If you want to install any of the playbook notebooks on your own system, check out the readmes within the folder that accompanies the notebook

## Step 3. Run the notebook

Once in JupyterLab, there all you have to do is run the `scRNA_analysis_preprocessing.ipynb`. You'll get both these playbook notebooks as well as the standard RAPIDS library example notebooks to help you get going.

You can use `Shift + Enter` to manually run each cell at your own pace, or `Run > Run All` to run all the cells.

Once you're done with exploring the `scRNA_analysis_preprocessing` notebook, you can explore other RAPIDS notebooks by going into the folders, selecting other notebooks, and doing the same thing.

## Step 4. Download your work

Since the docker container cannot privileged write back to the host system, you can use JupyterLab to download any files you may want to keep once the docker container is shut down.

Simply right click the file you want, in the browser, and click `Download` in the dropdown.

## Step 5. Cleanup

Once you have downloaded all your work, go back to the Terminal window where you started running the playbook.

In the Terminal window, 
1. Type `Ctrl + C`
2. Quickly either enter `y` and then hit `Enter` at the prompt or hit `Ctrl + C` again
3. The Docker container will proceed to shut down

> [!WARNING]
> This will delete ALL data that wasn't already downloaded from the Docker container.  The browser window may still show cached files if it is still open.

## Troubleshooting

<!-- 
TROUBLESHOOTING TEMPLATE: Although optional, this resource can significantly help users resolve common issues.
Replace all placeholder content in {} with your actual troubleshooting information.
Remove these comment blocks when you're done.

PURPOSE: Provide quick solutions to problems users are likely to encounter.
FORMAT: Use the table format for easy scanning. Add detailed notes when needed.
-->

| Symptom | Cause | Fix |
|---------|-------|-----|
| Docker is not found. | Docker may have been uninstalled, as it is preinstalled on your DGX Spark | Please install Docker using their convenience script here: `curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`. You will be prompted for your password. |
| Docker command unexpectedly exits with "permissions" error | Your user is not part of the `docker` group | Open Terminal and run these commands: `sudo groupadd docker && sudo usermod -aG docker $USER`.  You will be prompted for your password.  Then, close the Terminal, open a new one, and try again |
| Docker container download, environment build, or data download fails | There was either a connectivity issue or a resource may be temporarily unavailable. | You may need to try again later. If this persists, please post on the Spark user forum for support |




<!-- 
Space reserved for some common known issues that might be relevant to your project. Assess potential consequences before changing or deleting.
-->

> [!NOTE] 
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

For latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
