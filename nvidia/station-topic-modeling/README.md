# Topic Modeling

> Extract insights from massive text datasets using cuML's GPU-accelerated BERTopic


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Topic modeling helps you discover hidden themes in large document collections—but traditional methods crawl when datasets grow to millions of records. This playbook shows how to process **40 million Amazon product reviews in minutes** using GPU-accelerated BERTopic.

BERTopic combines transformer embeddings with clustering to extract human-readable topics from text. By swapping CPU-based UMAP and HDBSCAN with GPU-accelerated versions from **RAPIDS cuML**, you get the same results dramatically faster—no code changes required.

- **Drop-in GPU acceleration**: Load `cuml.accel` and your existing UMAP/HDBSCAN code runs on GPU automatically
- **Scale to millions**: Process datasets that would take hours on CPU in minutes on GPU
- **Interactive visualizations**: Explore topic distributions, relationships, and document clusters

## What you'll accomplish

You'll run a complete topic modeling pipeline on 40 million product reviews and generate interactive visualizations of discovered topics. By the end, you'll be able to:

- Use cuML's drop-in accelerators for UMAP and HDBSCAN
- Generate sentence embeddings at scale with SentenceTransformers
- Create topic visualizations including heatmaps, barcharts, and document datamaps

## What to know before starting

- Experience with Python and Jupyter notebooks
- Basic understanding of machine learning concepts (embeddings, clustering)
- Familiarity with pandas DataFrames

## Prerequisites

**Hardware Requirements:**
- NVIDIA DGX Station with GB300 GPU
- Minimum 64GB GPU memory for processing 40M documents
- At least 50GB available storage for dataset and embeddings

**Software Requirements:**
- Conda (Miniconda or Anaconda): `conda --version`
- CUDA 13.0 compatible drivers: `nvidia-smi`
- Network access to download the Amazon Reviews dataset (~14GB compressed)

## Ancillary files

All required assets are in the playbook directory `nvidia/station-topic-modeling/assets` (see [Instructions](https://build.nvidia.com/station/topic-modeling/instructions), Step 7). Key file:

- `video_notebook_for_GPU_Accelerated_Machine_Learning_BERTopic_RTX6000_40M.ipynb` - Complete Jupyter notebook with GPU-accelerated topic modeling pipeline (filename reflects original demo hardware; the notebook runs on GB300 and other NVIDIA GPUs)


## Time & risk

* **Estimated time:** 45 minutes (includes environment setup, dataset download, and embedding generation)
* **Risk level:** Low
  * Large dataset download (~14GB) may take time depending on network speed
  * Embedding generation requires significant GPU memory
* **Rollback:** Delete the downloaded dataset and any generated embedding files to restore state
* **Last Updated:** 03/02/2026
  * First Publication

## Instructions

## Step 1. (DGX Station) Hugging Face cache permissions

Ensure the Hugging Face cache is writable so model downloads succeed:

```bash
sudo chown -R $USER:$USER $HOME/.cache/huggingface 2>/dev/null || true
sudo chmod -R u+rwX $HOME/.cache/huggingface 2>/dev/null || true
mkdir -p $HOME/.cache/huggingface
```

If you see "Permission denied" when downloading models later, run the `chown`/`chmod` lines with your username (e.g. `nvidia`).

## Step 2. Install RAPIDS cuDF and cuML

Create a new conda environment with RAPIDS libraries for GPU-accelerated data processing.

```bash
conda create -n rapids-25.10 \
  -c rapidsai -c conda-forge \
  cudf=25.10 cuml=25.10 python=3.11 'cuda-version=13.0'
```

## Step 3. Activate the conda environment

```bash
conda activate rapids-25.10
```

## Step 4. Install machine learning packages

Install UMAP, HDBSCAN, BERTopic, and supporting libraries: 
Note: `datamapplot` will upgrade dask/distributed — the next step pins them back.

```bash
python -m pip install \
  transformers datasets sentence-transformers \
  umap-learn hdbscan==0.8.40 bertopic matplotlib \
  scikit-learn==1.4.2 datamapplot streamlit "nbformat>=4.2.0" ipykernel
```

Register the conda environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name rapids-25.10 --display-name "Python (rapids-25.10)"
```

Pin dask/distributed to RAPIDS-compatible versions (`datamapplot` upgrades them):

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
> Pip may report dependency conflicts (e.g. dask/distributed downgraded, cuml/rapids-dask-dependency). BERTopic and the notebook can still run. If you need cuML and RAPIDS dask together, consider keeping the conda default dask versions and installing only the BERTopic stack via pip in a separate env; see **Troubleshooting**.

## Step 5. Install visualization packages

Install JupyterLab and visualization libraries directly into the conda environment:

```bash
python -m pip install jupyterlab ipywidgets jupyterlab-widgets bokeh colorcet datashader plotly
```

## Step 6. Install compatible PyTorch

Install PyTorch with CUDA 13.0 support for GPU-accelerated embedding generation.

```bash
python -m pip install torch==2.9.0 torchvision torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
```

## Step 7. Clone the repository and download the dataset

Clone the playbook repository and download the Amazon Electronics Reviews dataset.

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/station-topic-modeling/assets
```

Download the dataset (~6GB compressed):

```bash
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz
```

## Step 8. Pull Git LFS files (notebooks)

The notebook files are stored in Git LFS. Without this step, JupyterLab will throw a `NotJSONError` when opening them.

```bash
conda install -c conda-forge git-lfs
git lfs install
git lfs pull
```

## Step 9. Launch JupyterLab

From the assets directory:

```bash
jupyter lab
```

## Step 10. Select the rapids-25.10 kernel

In JupyterLab, open `video_notebook_for_GPU_Accelerated_Machine_Learning_BERTopic_1M.ipynb` and select the **Python (rapids-25.10)** kernel from the kernel selector in the top right.

## Step 11. Execute all cells

Run all cells sequentially. The notebook will:
1. **Load data with cuDF** — GPU-accelerated pandas via `%load_ext cudf.pandas`
2. **Preprocess text** — clean and normalize review text
3. **Generate embeddings** — create sentence embeddings with SentenceTransformers
4. **Enable GPU acceleration** — load cuML accelerators via `%load_ext cuml.accel`
5. **Run BERTopic** — cluster documents into topics using GPU-accelerated UMAP and HDBSCAN
6. **Visualize results** — generate interactive topic visualizations

## Step 12. Explore the results

After the notebook completes, you'll have:
* **Topic information table** — discovered topics with keywords and document counts
* **Topic visualization** — interactive 2D map of topic relationships
* **Barchart** — top keywords for each topic
* **Heatmap** — topic similarity matrix
* **Document datamap** — visual clustering of documents by topic

## (Optional) Launch the interactive dashboard

As an alternative to the notebook, run the Streamlit dashboard for live UMAP/HDBSCAN
tuning. From the assets directory (with the dataset already downloaded in Step 7):

```bash
chmod +x run_app.sh
./run_app.sh
```

Then open the URL it prints in the terminal (default `http://localhost:8501`).

## Step 13. Cleanup (optional)

Remove the conda environment when finished:

```bash
conda deactivate
conda env remove -n rapids-25.10
```

Remove the downloaded dataset and generated files:

```bash
rm Electronics.jsonl.gz
rm -rf ~/.cache/huggingface
rm -rf dgx-spark-playbooks/
```

## Next steps

* **Adjust data size**: Modify `nrows` parameter when loading data to process smaller subsets
* **Tune clustering**: Experiment with `min_cluster_size` and `min_samples` in HDBSCAN
* **Try different embedding models**: Swap `all-MiniLM-L6-v2` for domain-specific models
* **Export topics**: Save the topic model using `topic_model.save()` for later analysis
* **Monitor GPU usage**: Run `nvidia-smi -l 1` to watch GPU utilization during processing

## Troubleshooting

### The Document datamap is blank (other visualizations work)

The datamap is the only visualization rendered with **WebGL (deck.gl)** instead of Plotly, so it
fails in ways the other views don't.

If it is blank until you nudge a sidebar slider, you are on an older copy of the dashboard that laid
the views out with `st.tabs`. Streamlit renders every tab body on every run, so the deck.gl canvas
drew once while its tab was still hidden and never redrew when you switched to it. The dashboard now
picks views with a radio selector and renders only the selected one, which mounts the canvas
visible. Pull the latest `assets/topic_modeling_app.py`.

If it is blank no matter what, the browser can't load the renderer. By default datamapplot fetches
deck.gl, d3, Arrow, jQuery and Google Fonts from a CDN *in your browser* at view time:

* Keep **Bundle JS + fonts into the page (offline mode)** ticked. This inlines the libraries into
  the page so the browser never contacts `unpkg.com`. The first offline render downloads and caches
  the bundle to `~/.local/share/datamapplot/`, so the machine running the app needs network access
  once.
* Switch the renderer to **Static image** for a matplotlib version that needs no WebGL at all.
* Click **Download standalone HTML** and open the file directly — the browser console reports the
  actual error.

Confirm WebGL2 is available at `chrome://gpu`. Remote-desktop sessions and software-rendering
setups frequently have it disabled.

### The static datamap takes minutes to render

Static rendering places every topic label with matplotlib, which costs roughly 0.4s per distinct
label. On a 100k-document fit the sample typically contains ~250 topics, so labelling all of them
takes around two minutes; the top 25 take about three seconds.

Use the **Label the top N topics** slider that appears next to the **Static image** renderer.
Topics outside the top N are still plotted, just without a label. The interactive renderer places
labels dynamically in the browser and stays under a second regardless of topic count.

### Pip reports dependency conflicts after Step 4

`datamapplot` upgrades `dask`/`distributed` past what RAPIDS expects, and pip warns about
`cuml`/`rapids-dask-dependency`. The pin in Step 4 restores compatible versions:

```bash
python -m pip install "dask==2025.9.1" "distributed==2025.9.1"
```

BERTopic, the notebook, and the dashboard all run with these warnings present. If you need cuML and
RAPIDS dask together for other work, keep the conda default dask versions and install the BERTopic
stack via pip into a separate environment.

### JupyterLab throws `NotJSONError` when opening the notebook

The notebooks are Git LFS pointers until they're pulled. Re-run Step 8 from the repository root:

```bash
git lfs install
git lfs pull
```

### "Permission denied" while downloading models

The Hugging Face cache isn't writable by your user. Re-run Step 1 with your actual username in
place of `$USER` (on DGX Station this is often `nvidia`):

```bash
sudo chown -R $USER:$USER $HOME/.cache/huggingface
sudo chmod -R u+rwX $HOME/.cache/huggingface
```

### Out of memory during embedding generation

Embedding is the most memory-hungry step. Lower the document count before re-running — `nrows` in
the notebook, or **Number of documents** in the dashboard sidebar. Watch usage with `nvidia-smi -l 1`
while the step runs.

### The dashboard starts with the wrong Python, or the port is taken

`run_app.sh` resolves the `rapids-25.10` conda env by default. Override any part of that with
environment variables:

```bash
PYTHON=/path/to/python ./run_app.sh   # use a specific interpreter
CONDA_ENV=my-env ./run_app.sh         # use a different conda env
PORT=8502 ./run_app.sh                # bind a different port
```

### Refitting in the dashboard is slow after changing the document count

Preprocessed text and embeddings are cached per document count under `assets/.cache/`. Changing
**Number of documents** invalidates that cache and re-runs the embedding step; every other
parameter only re-runs UMAP → HDBSCAN → c-TF-IDF. Delete `assets/.cache/` to force a full rebuild.
