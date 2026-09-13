# Accelerate Topic Modeling with BERTopic

> Interactive topic maps from large text collections with cuML acceleration

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Topic modeling helps you discover hidden themes in large document collections—but traditional methods crawl when datasets grow to millions of records. This playbook shows how to process large product-review corpora in minutes using GPU-accelerated BERTopic.

BERTopic combines transformer embeddings with clustering to extract human-readable topics from text. By swapping CPU-based UMAP and HDBSCAN with GPU-accelerated versions from **RAPIDS cuML**, you get the same results dramatically faster—no code changes required.

- **Drop-in GPU acceleration**: Load `cuml.accel` and your existing UMAP/HDBSCAN code runs on GPU automatically
- **Scale to millions**: Process datasets that would take hours on CPU in minutes on GPU
- **Interactive visualizations**: Explore topic distributions, relationships, and document clusters

## What you'll accomplish

You'll run a complete topic modeling pipeline on product reviews and generate interactive visualizations of discovered topics on your **hardware platform**.

By the end, you'll be able to:

- Use cuML's drop-in accelerators for UMAP and HDBSCAN
- Generate sentence embeddings at scale with SentenceTransformers
- Create topic visualizations including heatmaps, barcharts, and document datamaps

## What to know before starting

**Required:**

- Experience with Python and Jupyter notebooks
- Basic understanding of machine learning concepts (embeddings, clustering)
- Familiarity with pandas DataFrames

**Optional:**

- Experience installing and using conda environments
- Familiarity with Hugging Face model downloads and caching

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Station** | DGX OS (Linux) | Large HBM + Grace DRAM | Conda env `rapids-25.10` (cuDF/cuML 25.10, Python 3.11, CUDA 13.0); JupyterLab; optional Streamlit on port 8501 | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Minimum 64 GB GPU memory for processing very large document sets (tens of millions of reviews)
- At least 50 GB available storage for the dataset and embeddings

**Software requirements**

- Conda (Miniconda or Anaconda): `conda --version`
- CUDA 13.0 compatible drivers: `nvidia-smi`
- Network access to download the Amazon Reviews dataset (~14 GB compressed) and Python packages
- Web browser access to JupyterLab (and optionally Streamlit on port 8501)

## Ancillary files

All required assets can be found [in the playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-topic-modeling/).

- `assets/video_notebook_for_GPU_Accelerated_Machine_Learning_BERTopic_1M.ipynb` — Complete Jupyter notebook with the GPU-accelerated topic modeling pipeline
- `assets/topic_modeling_app.py` — Streamlit dashboard for interactive UMAP/HDBSCAN tuning
- `assets/run_app.sh` — Launcher for the Streamlit dashboard (defaults to the `rapids-25.10` conda env)

## Time & risk

- **Estimated time:** 45 MIN (includes environment setup, dataset download, and embedding generation)
- **Risk level:** Low
  - Large dataset download (~14 GB) may take time depending on network speed
  - Embedding generation requires significant GPU memory
- **Rollback:** Delete the downloaded dataset and any generated embedding files; optionally remove the conda environment
- **Last Updated:** 08/03/2026
  - Install RAPIDS cuML accelerators and run BERTopic topic modeling with interactive visualizations

## Instructions

## Step 1. Prepare the Hugging Face cache

Ensure the Hugging Face cache is writable so model downloads succeed on your hardware platform:

```bash
sudo chown -R $USER:$USER $HOME/.cache/huggingface 2>/dev/null || true
sudo chmod -R u+rwX $HOME/.cache/huggingface 2>/dev/null || true
mkdir -p $HOME/.cache/huggingface
```

If you see "Permission denied" when downloading models later, run the `chown`/`chmod` lines again for your user.

## Step 2. Install RAPIDS cuDF and cuML

Create a new conda environment with RAPIDS libraries for GPU-accelerated data processing:

```bash
conda create -n rapids-25.10 \
  -c rapidsai -c conda-forge \
  cudf=25.10 cuml=25.10 python=3.11 'cuda-version=13.0'
```

This installs cuDF (GPU DataFrame library) and cuML (GPU machine learning library) that provide drop-in acceleration for pandas and scikit-learn operations.

## Step 3. Activate the conda environment

```bash
conda activate rapids-25.10
```

## Step 4. Install machine learning packages

Install UMAP, HDBSCAN, BERTopic, and supporting libraries for topic modeling.
Note: `datamapplot` will upgrade dask/distributed — the next command pins them back.

```bash
python -m pip install \
  transformers datasets sentence-transformers \
  umap-learn hdbscan==0.8.40 bertopic matplotlib \
  scikit-learn==1.4.2 datamapplot streamlit
```

Pin dask/distributed back to RAPIDS-compatible versions:

```bash
python -m pip install "dask==2025.9.1" "distributed==2025.9.1"
```

These packages provide:

- **dask**: Parallel computing library
- **distributed**: Distributed task scheduler for dask
- **sentence-transformers**: Generate text embeddings
- **umap-learn / hdbscan**: Dimensionality reduction and clustering (GPU-accelerated via cuML)
- **bertopic**: Topic modeling framework
- **datamapplot**: Document visualization
- **streamlit**: Interactive dashboard for the topic explorer app (`run_app.sh`)

> [!NOTE]
> Pip may report dependency conflicts (for example, dask/distributed downgraded, or cuml/rapids-dask-dependency). BERTopic and the notebook can still run. If you need cuML and RAPIDS dask together, keep the conda default dask versions and install only the BERTopic stack via pip in a separate environment; see **Troubleshooting**.

## Step 5. Install visualization packages

Install JupyterLab and visualization libraries for interactive topic exploration:

```bash
conda install -c conda-forge \
    notebook=7.5.0 \
    jupyterlab=4.5.0 \
    ipywidgets=8.1.8 \
    jupyterlab-widgets=3.0.16 \
    bokeh=3.8.1 \
    colorcet=3.1.0 \
    datashader=0.18.2 \
    plotly=6.5.0
```

If conda reports `PackagesNotFoundError` for `jupyterlab-widgets`, install it with pip:

```bash
python -m pip install jupyterlab-widgets
```

## Step 6. Install compatible PyTorch

Install PyTorch with CUDA 13.0 support for GPU-accelerated embedding generation:

```bash
python -m pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## Step 7. Clone the repository and download the dataset

Clone the playbook repository and download the Amazon Electronics Reviews dataset:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd client-hardware-playbooks/nvidia/playbook-topic-modeling/assets
```

Download the dataset (~14 GB compressed):

```bash
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz
```

## Step 8. Pull Git LFS files (notebooks)

The notebook files are stored in Git LFS — without this step, JupyterLab will throw a `NotJSONError` when trying to open them.

```bash
conda install -c conda-forge git-lfs
git lfs install
git lfs pull
```

## Step 9. Launch JupyterLab

Start JupyterLab from the assets directory:

```bash
jupyter lab
```

Open a browser to the URL JupyterLab prints (typically `http://localhost:8888`). From another device, use `http://<HARDWARE_IP>:8888` where `<HARDWARE_IP>` is your hardware platform's reachable address.

If you access the hardware platform over SSH, forward the Jupyter port to your local browser:

```bash
ssh -N -L YYYY:localhost:XXXX username@remote_host
```

- `YYYY`: The local port you want to use (for example, 8888)
- `XXXX`: The port JupyterLab is listening on on the remote machine (for example, 8888)
- `-N`: Prevents SSH from executing a remote command
- `-L`: Specifies local port forwarding

## (Optional) Launch the interactive dashboard

As an alternative to the notebook, run the Streamlit dashboard for live UMAP/HDBSCAN tuning. From the assets directory (with the dataset already downloaded in Step 7):

```bash
./run_app.sh
```

Then open the URL it prints (default `http://localhost:8501`). The script auto-selects the `rapids-25.10` conda env; override with `PYTHON=/path/to/python ./run_app.sh` if needed.

## Step 10. Select the rapids-25.10 kernel

In JupyterLab, open the notebook `video_notebook_for_GPU_Accelerated_Machine_Learning_BERTopic_1M.ipynb`.

Select the **rapids-25.10** kernel from the kernel selector in the top right corner of the notebook interface.

## Step 11. Execute all cells

Run all cells in the notebook sequentially. The notebook will:

1. **Load data with cuDF**: GPU-accelerated pandas via `%load_ext cudf.pandas`
2. **Preprocess text**: Clean and normalize review text
3. **Generate embeddings**: Create sentence embeddings
4. **Enable GPU acceleration**: Load cuML accelerators via `%load_ext cuml.accel`
5. **Run BERTopic**: Cluster documents into topics using GPU-accelerated UMAP and HDBSCAN
6. **Visualize results**: Generate interactive topic visualizations

## Step 12. Explore the results

After the notebook completes, you'll have:

- **Topic information table**: Discovered topics with keywords and document counts
- **Topic visualization**: Interactive 2D map of topic relationships
- **Barchart**: Top keywords for each topic
- **Heatmap**: Topic similarity matrix
- **Document datamap**: Visual clustering of documents by topic

## Step 13. Cleanup (optional)

Remove the conda environment when finished:

```bash
conda deactivate
conda env remove -n rapids-25.10
```

Remove the downloaded dataset:

```bash
rm Electronics.jsonl.gz
```

Remove generated embedding files and the cloned playbook directory if you no longer need them:

```bash
## Optional: remove Hugging Face cache (embedding cache from the notebook)
rm -rf ~/.cache/huggingface

## From the parent of client-hardware-playbooks/, remove the cloned repo
rm -rf client-hardware-playbooks/
```

## Step 14. Next steps

Apply this workflow to your own datasets:

1. **Adjust data size**: Modify the `nrows` parameter when loading data to process smaller subsets
2. **Tune clustering**: Experiment with `min_cluster_size` and `min_samples` in HDBSCAN
3. **Try different embedding models**: Swap `all-MiniLM-L6-v2` for domain-specific models
4. **Export topics**: Save the topic model using `topic_model.save()` for later analysis
5. **Monitor GPU usage**: Run `nvidia-smi -l 1` to watch GPU utilization during processing

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every platform listed in this playbook.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| `Permission denied` when downloading Hugging Face models | All hardware platforms | Hugging Face cache not writable | Re-run the cache ownership commands in Instructions Step 1 for your user |
| Conda create fails resolving packages | All hardware platforms | Channel or version pin mismatch | Retry with the exact `conda create` command from Instructions; ensure network access to `rapidsai` and `conda-forge` |
| Pip reports dask/distributed or RAPIDS dependency conflicts | All hardware platforms | `datamapplot` upgrades dask beyond RAPIDS-compatible pins | Re-pin with `python -m pip install "dask==2025.9.1" "distributed==2025.9.1"`; BERTopic can still run with the reported conflicts |
| `PackagesNotFoundError` for `jupyterlab-widgets` | All hardware platforms | Package unavailable on the conda channel | Install with `python -m pip install jupyterlab-widgets` |
| `NotJSONError` when opening the notebook in JupyterLab | All hardware platforms | Git LFS pointer file not pulled | Run `git lfs install` and `git lfs pull` from the playbook assets directory |
| JupyterLab or Streamlit UI not reachable in the browser | All hardware platforms | Port blocked, wrong host, or missing SSH forward | Use `http://localhost:8888` (Jupyter) or `http://localhost:8501` (Streamlit); from another device use `http://<HARDWARE_IP>:<port>`; if remote, forward with `ssh -N -L 8888:localhost:8888 username@remote_host` |
| Notebook runs on CPU only / no GPU speedup | All hardware platforms | Wrong kernel or accelerator not loaded | Select the **rapids-25.10** kernel; confirm `%load_ext cudf.pandas` and `%load_ext cuml.accel` cells ran successfully; verify `nvidia-smi` |
| `CUDA out of memory` during embeddings or clustering | All hardware platforms | Workload exceeds available GPU memory | Reduce `nrows` when loading data; close other GPU applications; confirm free memory with `nvidia-smi` |
| Dataset download is slow or fails | All hardware platforms | Network interruption or remote host unavailable | Retry `wget` for `Electronics.jsonl.gz`; confirm outbound network access |

> [!NOTE]
> On discrete-GPU hardware platforms, GPU memory is separate from system RAM. CUDA out-of-memory usually means the workload exceeds VRAM: reduce the number of rows processed, lower batch size for embeddings, or close other GPU applications. Use `nvidia-smi` to confirm no other process is holding memory.

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
