# Accelerate Data Science with CUDA-X Libraries

> Zero code changes for pandas and scikit-learn workflows on the GPU

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook includes two example notebooks that demonstrate GPU acceleration of key machine learning algorithms and core pandas operations using CUDA-X Data Science libraries:

- **NVIDIA cuDF:** Accelerates data preparation and core data processing of large strings datasets, with no code changes.
- **NVIDIA cuML:** Accelerates popular, compute-intensive machine learning algorithms in scikit-learn (LinearSVC), UMAP, and HDBSCAN, with no code changes.

CUDA-X Data Science (formerly RAPIDS) is an open-source library collection that accelerates the data science and data processing ecosystem. These libraries accelerate popular Python tools like scikit-learn and pandas with zero code changes, so you can maximize GPU performance with your existing code.

## What you'll accomplish

You'll accelerate popular machine learning algorithms and data analytics operations on the GPU on your **hardware platform**. You'll see how to accelerate popular Python tools with zero code changes, and how to run data science workflows on GPU-accelerated hardware.

## What to know before starting

**Required:**

- Familiarity with pandas and scikit-learn
- Familiarity with machine learning algorithms such as support vector machines, clustering, and dimensionality reduction
- Experience working in a terminal

**Optional:**

- Experience installing and using conda environments
- Familiarity with Jupyter Notebook

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Conda env `rapids-test` (RAPIDS 26.06, Python 3.12, CUDA 13.0); Jupyter on port 8888 | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient memory for large strings datasets and ML demos (see matrix)
- At least several GB available storage for conda packages and Kaggle dataset downloads

**Software requirements**

- CUDA 13 available: `nvcc --version` or `nvidia-smi`
- Conda (Miniconda or Anaconda): `conda --version`
- Kaggle API credentials (`kaggle.json`) for dataset download in the notebooks
- Network access to download conda packages and Kaggle datasets
- Web browser access to Jupyter on port 8888 (or a forwarded local port)

## Ancillary files

All required assets can be found [in the playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-cuda-x-data-science/).

- `assets/cudf_pandas_demo.ipynb` — Large strings data processing workflow with pandas code accelerated on GPU via cuDF
- `assets/cuml_sklearn_demo.ipynb` — Machine learning algorithms including LinearSVC, UMAP, and HDBSCAN accelerated on GPU via cuML

## Time & risk

- **Estimated time:** 30 MIN (setup); each notebook typically runs in a few minutes after dependencies and data are ready
- **Risk level:** Low
  - Data download may be slow or fail due to network issues
  - Kaggle API key setup may require retries if credentials are misplaced
- **Rollback:** No permanent system changes during normal usage; remove the conda environment to reset
- **Last Updated:** 08/03/2026
  - Install CUDA-X (RAPIDS) libraries and run cuDF pandas and cuML scikit-learn notebooks with zero code changes

## Instructions

## Step 1. Verify system requirements

Confirm CUDA access, install conda if needed, and prepare Kaggle API credentials on your hardware platform.

```bash
## Verify CUDA / GPU access
nvcc --version
nvidia-smi
```

Expected output should show CUDA 13 and GPU information from `nvidia-smi`.

- Install conda using [these instructions](https://docs.anaconda.com/miniconda/install/) if it is not already available.
- Create a Kaggle API key using [these instructions](https://www.kaggle.com/discussions/general/74235). You will place `kaggle.json` next to the notebooks in Step 4.

## Step 2. Install CUDA-X Data Science libraries

Create a new conda environment with the CUDA-X (RAPIDS) libraries and notebook dependencies:

```bash
conda create -n rapids-test -c rapidsai -c conda-forge -c nvidia  \
  rapids=26.06 python=3.12 'cuda-version=13.0' \
  jupyter hdbscan umap-learn
```

## Step 3. Activate the conda environment

```bash
conda activate rapids-test
```

## Step 4. Clone the playbook repository

Clone the playbook assets and open the assets directory:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd client-hardware-playbooks/nvidia/playbook-cuda-x-data-science/assets
```

Place the `kaggle.json` file created in Step 1 in this assets folder (same directory as the notebooks).

## Step 5. Run the notebooks

There are two notebooks in the playbook assets.

One runs a large strings data processing workflow with pandas code on the GPU:

```bash
jupyter notebook cudf_pandas_demo.ipynb
```

The other covers machine learning algorithms including UMAP and HDBSCAN:

```bash
jupyter notebook cuml_sklearn_demo.ipynb
```

Open a browser to `http://localhost:8888`. From another device, use `http://<HARDWARE_IP>:8888` where `<HARDWARE_IP>` is your hardware platform's reachable address.

If you access the hardware platform over SSH, forward the Jupyter port to your local browser:

```bash
ssh -N -L YYYY:localhost:XXXX username@remote_host
```

- `YYYY`: The local port you want to use (for example, 8888)
- `XXXX`: The port Jupyter Notebook is listening on on the remote machine (for example, 8888)
- `-N`: Prevents SSH from executing a remote command
- `-L`: Specifies local port forwarding

## Step 6. Cleanup (optional)

When you are finished, stop Jupyter (Ctrl+C in the terminal where it is running).

Optionally remove the conda environment:

```bash
conda deactivate
conda env remove -n rapids-test
```

## Step 7. Next steps

Apply zero-code-change acceleration to your own pandas and scikit-learn workflows:

1. Enable cuDF pandas accelerator mode in notebooks that already use pandas.
2. Enable cuML accelerator mode for scikit-learn, UMAP, and HDBSCAN workloads.
3. Compare CPU vs GPU runtimes on representative datasets for your workflow.
4. See the RAPIDS documentation linked under **Resources** for broader library coverage.

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every platform listed in this playbook.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| `nvidia-smi` or `nvcc` not found | All hardware platforms | Missing NVIDIA drivers or CUDA toolkit | Install NVIDIA drivers / CUDA for your hardware platform |
| Conda create fails resolving packages | All hardware platforms | Channel or version pin mismatch | Retry with the exact `conda create` command from Instructions; ensure network access to `rapidsai`, `conda-forge`, and `nvidia` channels |
| Kaggle download fails in a notebook | All hardware platforms | Missing or invalid `kaggle.json` | Place a valid `kaggle.json` in the same folder as the notebook and confirm the API key is active |
| Jupyter UI not reachable in the browser | All hardware platforms | Port 8888 blocked, wrong host, or missing SSH forward | Use `http://localhost:8888`, or `http://<HARDWARE_IP>:8888`; if remote, forward with `ssh -N -L 8888:localhost:8888 username@remote_host` |
| Notebooks run on CPU only / no GPU speedup | All hardware platforms | Accelerator mode not enabled or CUDA env mismatch | Follow the notebook setup cells; confirm the `rapids-test` env is active and `nvidia-smi` works |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
