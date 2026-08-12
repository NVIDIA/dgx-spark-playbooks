# Build Your Own Transaction Foundation Model

> Run the NVIDIA transaction foundation model developer example on a single DGX Spark

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook adapts the [NVIDIA transaction foundation model developer example](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model) for a single NVIDIA DGX Spark. It preserves the upstream five-notebook workflow while replacing the original `nvcr.io/nvidia/nemo:25.09.01` setup with a pinned GB10 container.

The upstream setup assumptions can fail on GB10 even when basic imports and small cuDF tests succeed:

- `nvcr.io/nvidia/nemo:25.09.01` inherits the PyTorch 25.06 software stack, including RAPIDS 25.04. Runtime upgrades can leave old top-level RAPIDS packages mixed with newer shared libraries, and the full TabFormer `cudf.read_csv` load can segfault in native code.
- Inline `%pip install` notebook cells mutate the environment instead of using one reproducible dependency set.
- The NeMo container does not provide every notebook dependency, including the required Hugging Face, XGBoost, plotting, and `torchdata` packages.
- Starting Jupyter before checking the environment and Git LFS checkpoint moves setup failures into later notebook steps.

This playbook instead builds one ARM64 image from `nvcr.io/nvidia/nemo-automodel:26.06.00`, installs a coherent RAPIDS 26.06 CUDA 13 stack with `uv`, preserves NVIDIA's PyTorch build, and validates the environment and checkpoint before Jupyter starts.

![Transaction foundation model architecture](https://raw.githubusercontent.com/NVIDIA-AI-Blueprints/transaction-foundation-model/main/assets/architecture_diagram.png)

## What you'll accomplish

You will:

1. Build a pinned GB10 container for the upstream transaction foundation model notebooks.
2. Validate ARM64, CUDA, RAPIDS, PyTorch, NeMo AutoModel, and the model checkpoint.
3. Run the five-notebook workflow from baseline training through embedding-based fraud detection.
4. Apply a documented Parquet staging workaround if the full TabFormer CSV triggers a cuDF parser failure.

## What to know before starting

- Working with Docker and GPU-enabled containers
- Running commands in a Linux terminal
- Using Jupyter notebooks
- Basic familiarity with model training, embeddings, and tabular data workflows

## Prerequisites

- One NVIDIA DGX Spark or GB10 partner system with 128 GB unified memory
- Linux ARM64 / `aarch64`
- DGX OS based on Ubuntu 24.04, updated through your approved workflow
- Docker with NVIDIA Container Runtime
- Git and Git LFS
- Access to GitHub, NGC, PyPI, and `https://pypi.nvidia.com`
- At least 100 GB of free disk space for the image, dataset, notebooks, and generated artifacts

## Ancillary files

The required build files are in the [`assets`](assets/) directory:

- [`Dockerfile.gb10`](assets/Dockerfile.gb10) builds the ARM64 NeMo AutoModel and RAPIDS environment.
- [`requirements-gb10.txt`](assets/requirements-gb10.txt) pins the notebook and RAPIDS dependencies.

## Time & risk

- **Estimated time:** Not yet benchmarked end-to-end. The first container build, dataset download, and notebook execution dominate the runtime.
- **Risks:** The workflow downloads large container and dataset artifacts. Package or notebook changes upstream may require revalidation. Full TabFormer CSV parsing may need the documented Parquet workaround.
- **Rollback:** The container uses `--rm`, so exiting removes it. The custom image and cloned upstream repository can be removed separately if no longer needed.
- **Last Updated:** 08/12/2026
  - Initial publication for DGX Spark GB10

## Instructions

## Step 1. Validate the DGX Spark host

Run these commands on the host:

```bash
uname -m
nvidia-smi
docker --version
nvidia-ctk --version
git --version
git lfs version
df -h .
```

Confirm that `uname -m` returns `aarch64`, `nvidia-smi` reports the GB10 GPU, Docker and NVIDIA Container Runtime are installed, Git and Git LFS are available, and the working filesystem has at least 100 GB free. On DGX Spark, `nvidia-smi` may report framebuffer `Memory-Usage` as not supported.

Validate GPU access from a minimal CUDA container:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
```

If Docker requires `sudo`, add your user to the Docker group and start a shell with the new group membership:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

## Step 2. Get the playbook and build the GB10 image

From the directory where you want to keep the playbook and upstream repositories, clone this repository:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks.git
cd dgx-spark-playbooks
```

If you already cloned the playbook repository, skip the `git clone` command and change into that existing checkout.

If NGC authentication is required, log in before the build pulls the base image:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC_API_KEY>
```

Build the image using the supplied assets:

```bash
docker build --platform linux/arm64 \
  -f nvidia/transaction-foundation-model/assets/Dockerfile.gb10 \
  -t tfm-gb10:nemo-automodel-26.06 \
  nvidia/transaction-foundation-model/assets
```

The image uses:

- `nvcr.io/nvidia/nemo-automodel:26.06.00`
- `uv` for Python package installation
- RAPIDS `26.6.*` CUDA 13 wheels with `cuda-toolkit==13.2.*`
- `torchdata` installed with `--no-deps` so the NVIDIA PyTorch build is not replaced

## Step 3. Clone the upstream repository

Return to the parent directory, then clone the developer example alongside the playbook repository:

```bash
cd ..
git clone https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model.git
cd transaction-foundation-model
```

Do not start the notebooks yet. Validate the Python stack, cuDF path, and Git LFS checkpoint first.

The upstream notebooks are:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_dataset_baseline.ipynb` | Load the TabFormer dataset, create temporal splits, and train an XGBoost baseline. |
| 2 | `02_seq_preproc_tokenization.ipynb` | Build the RAPIDS-accelerated tokenizer pipeline. |
| 3 | `03_foundation_model_training.ipynb` | Run a short single-GB10 NeMo AutoModel pretraining demo. |
| 4 | `04_inference_embedding_extraction.ipynb` | Load the checkpoint, run inference, extract embeddings, and visualize with UMAP. |
| 5 | `05_xgboost_fraud_detection.ipynb` | Compare fraud detection using raw features, embeddings, and combined features. |

## Step 4. Launch and validate the container

From the cloned `transaction-foundation-model` directory, launch the image with the repository mounted at `/workspace`:

```bash
docker run --platform linux/arm64 --gpus all --rm -it \
  --name tfm-gb10 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --shm-size=16g \
  -p 8888:8888 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  tfm-gb10:nemo-automodel-26.06
```

Inside the container, pull the real checkpoint files and validate the environment:

```bash
git config --global --add safe.directory /workspace
git lfs install
git lfs pull --include="models/decoder-foundation-model/**"

uv pip check --system

python - <<'PY'
from pathlib import Path
from safetensors import safe_open
import platform
import torch
import cudf, cuml, cupy, xgboost
import transformers, torchdata
import nemo_automodel

print("machine:", platform.machine())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("cudf:", cudf.__version__)
print("cuml:", cuml.__version__)
print("cupy:", cupy.__version__)
print("xgboost:", xgboost.__version__)
print("transformers:", transformers.__version__)
print("torchdata:", torchdata.__version__)
print("nemo_automodel:", getattr(nemo_automodel, "__version__", "imported"))

smoke = Path("/tmp/cudf_gb10_smoke.csv")
smoke.write_text("a,b\n1,2\n3,4\n")
print(cudf.read_csv(str(smoke)).to_pandas())

weights = Path("/workspace/models/decoder-foundation-model/model-00001-of-00001.safetensors")
print("checkpoint size:", weights.stat().st_size)
with safe_open(str(weights), framework="pt") as f:
    print("checkpoint tensor count:", len(f.keys()))
PY
```

The checkpoint should be about 55 MB. If validation reports a file of only a few hundred bytes or raises `SafetensorError`, rerun the `git lfs pull` command before continuing.

## Step 5. Start Jupyter

Inside the container, start the notebook server:

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Open the printed URL on the DGX Spark desktop. For a remote DGX Spark, create an SSH tunnel from your local workstation:

```bash
ssh -L 8888:localhost:8888 <user>@<dgx-spark-host>
```

Then open the printed token URL through `http://localhost:8888/?token=...`.

## Step 6. Run and validate the baseline notebook

Skip the upstream `%pip install` cells because they would mutate the validated image. If an additional package is required, install it with `!uv pip install --system --break-system-packages <package>` and restart the kernel.

Run `01_dataset_baseline.ipynb`. It downloads TabFormer, loads the raw CSV, creates temporal splits, and trains the XGBoost baseline.

If the notebook completes, continue to Step 7. If its kernel exits while reading the full CSV, the dataset should already be present. Open a second host terminal and enter the running container:

```bash
docker exec -it tfm-gb10 bash
```

Confirm the dataset path and reproduce the exact CSV read:

```bash
python - <<'PY'
from pathlib import Path
import cudf

raw_csv = Path("/workspace/data/TabFormer/raw/card_transaction.v1.csv")
print("exists:", raw_csv.exists(), raw_csv)
print("sample:", cudf.read_csv(str(raw_csv), nrows=1000).shape)
gdf = cudf.read_csv(str(raw_csv))
print("full:", gdf.shape)
print(gdf.head(3).to_pandas())
PY
```

If the sample succeeds but the full read segfaults on RAPIDS 26.06, reconnect to the container if necessary and convert the raw CSV to Parquet once with pandas:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

raw_csv = Path("/workspace/data/TabFormer/raw/card_transaction.v1.csv")
parquet_out = raw_csv.with_suffix(".parquet")
df = pd.read_csv(raw_csv)
df.to_parquet(parquet_out, index=False)
print(parquet_out)
PY
```

Return to notebook 01 and change its load cell to:

```python
raw_gdf = cudf.read_parquet(str(RAW_CSV.with_suffix(".parquet")))
```

This moves only the raw CSV parse to CPU. Downstream splitting, tokenization, UMAP, and dataframe work continue to use RAPIDS. Restart the kernel and finish notebook 01 before continuing.

## Step 7. Run the remaining notebooks

After notebook 01 completes, run the remaining notebooks in order:

1. Run `02_seq_preproc_tokenization.ipynb` to build the tokenized transaction corpus.
2. Run `03_foundation_model_training.ipynb` for the 30-step single-GB10 demo. Do not use its `torchrun --nproc-per-node=8` scale-out example on one Spark.
3. Run `04_inference_embedding_extraction.ipynb` to load `models/decoder-foundation-model/` and extract embeddings. Start with `BATCH_SIZE = 1024`; reduce it to `512` or `256` if needed.
4. Run `05_xgboost_fraud_detection.ipynb` to compare the fraud detection models.

## Step 8. Tune or customize the workflow

- Keep the custom image pinned for reproducibility.
- Use native ARM64 images. Do not run a container without a `linux/arm64` manifest through emulation for this workflow.
- Keep notebook 03 on one GPU for the Spark demo.
- If the 30-step training demo hits memory pressure, reduce both notebook 03 batch-size overrides from `16` to `8`.
- Monitor `/proc/meminfo`, system tools, and workload behavior alongside CUDA reports because DGX Spark uses unified memory.
- Keep the supplied DGX Spark power adapter connected during performance-sensitive runs.
- Adapt `src/tokenizer/` for different transaction schemas.
- Update `configs/pretrain_financial_decoder.yaml` to change the decoder configuration.
- Replace XGBoost with another classifier that accepts fixed-length feature vectors.
- Move long, full pretraining runs to larger multi-GPU infrastructure or a validated multi-Spark setup.

The upstream model uses the following architecture:

| Parameter | Value |
|-----------|-------|
| Architecture | Llama-style decoder-only transformer |
| Parameters | ~29M |
| Hidden size | 512 |
| Layers | 8 |
| Attention | Grouped Query Attention, 8 query heads and 2 KV heads |
| Context window | 8,192 tokens with RoPE; training uses 4,096-token sequences |
| Activation | SwiGLU |
| Normalization | RMSNorm |
| Vocabulary | ~6,251 domain-specific tokens |

## Step 9. Cleanup and rollback

Stop Jupyter with `Ctrl+C`, then exit the container. Because it was launched with `--rm`, Docker removes the stopped container.

To remove only the custom image:

```bash
docker image rm tfm-gb10:nemo-automodel-26.06
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `no matching manifest for linux/arm64` | The selected image is not built for ARM64 | Inspect the manifest. This playbook uses `nvcr.io/nvidia/nemo-automodel:26.06.00`, which was verified for ARM64 on 08/06/2026. |
| Docker cannot access the GPU | NVIDIA Container Runtime or Docker permissions issue | Run the CUDA validation container, check `nvidia-ctk --version`, and retry with `sudo docker ...` if needed. |
| `nvidia-smi` reports `Memory-Usage: Not Supported` | Expected DGX Spark integrated-GPU behavior | Continue if CUDA workloads see the GPU; monitor memory with OS tools and workload behavior. |
| `cudf.read_csv` segfaults in `nvcr.io/nvidia/nemo:25.09.01` | The image inherits RAPIDS 25.04, and runtime upgrades can leave a mixed native stack | Rebuild and use `tfm-gb10:nemo-automodel-26.06`. |
| A sample CSV read succeeds but the full TabFormer read segfaults | Possible cuDF parser issue for the dataset or environment | Use the Step 6 pandas-to-Parquet staging workaround. |
| The resolver reports conflicts between RAPIDS versions | Old packages and new shared libraries are mixed | Rebuild the custom image from scratch; do not reuse a mutated container. |
| `ModuleNotFoundError` for `cudf`, `cuml`, or `nemo_automodel` | Wrong image or failed dependency installation | Rebuild `Dockerfile.gb10`, then rerun Step 4 validation. |
| `ModuleNotFoundError: transformers` | Hugging Face runtime is missing | Rebuild with the supplied requirements, or run `uv pip install --system --break-system-packages transformers==4.53.3`. |
| `ModuleNotFoundError: torchdata` | NeMo dataloader dependency is missing | Rebuild, or run `uv pip install --system --break-system-packages --no-deps "torchdata>=0.11,<0.12"`. |
| `SafetensorError: Error while deserializing header: header too large` | The checkpoint is a Git LFS pointer or partial file | Pull `models/decoder-foundation-model/**` with Git LFS and validate it with `safe_open`. |
| XGBoost falls back to CPU | XGBoost build or CUDA visibility issue | Confirm `xgboost==3.3.0` or newer, `torch.cuda.is_available()`, and `XGB_DEVICE = "cuda"`. |
| `git lfs pull` fails due to ownership | Host and container repository ownership differ | Run `git config --global --add safe.directory /workspace` inside the container. |
| Jupyter does not open remotely | Port 8888 is not forwarded | Use `ssh -L 8888:localhost:8888 <user>@<dgx-spark-host>` and open the token URL locally. |

For current platform issues, review the [DGX Spark known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).

## References

- [Transaction Foundation Model repository](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model)
- [Install NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/latest/get-started/installation)
- [NeMo software component versions](https://docs.nvidia.com/nemo/megatron-bridge/latest/releases/software-versions.html)
- [DGX Spark porting dependencies](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/dependencies.html)
- [NVIDIA Container Runtime for Docker](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)
- [RAPIDS platform support](https://docs.rapids.ai/platform-support/)
- [RAPIDS installation guide](https://docs.rapids.ai/install/)
- [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
- [XGBoost on PyPI](https://pypi.org/project/xgboost/)

## License

Unless otherwise noted, the upstream transaction foundation model contents are licensed under the [Apache License, Version 2.0](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model/blob/main/LICENSE). Third-party software and datasets are governed by their respective licenses and terms.

## Terms of Use

The upstream project states that it is not accepting contributions. Report security vulnerabilities or NVIDIA AI concerns through [NVIDIA's security vulnerability process](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
