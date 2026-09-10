# Accelerate Portfolio Optimization with cuOpt

> Tail-risk modeling and backtesting across thousands of assets

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook demonstrates an end-to-end GPU-accelerated workflow using NVIDIA cuOpt and NVIDIA cuML to solve large-scale portfolio optimization problems, using the Mean-CVaR (Conditional Value-at-Risk) model, in near real-time.

Portfolio Optimization (PO) involves solving high-dimensional, non-linear numerical optimization problems to balance risk and return. Modern portfolios often contain thousands of assets, making traditional CPU-based solvers too slow for advanced workflows. By moving the computational heavy lifting to the GPU, this solution dramatically reduces computation time.

## What you'll accomplish

You'll implement a Mean-CVaR portfolio optimization pipeline on your **hardware platform**, with tools for performance evaluation, strategy backtesting, benchmarking, and visualization. The workflow includes:

- **GPU-accelerated optimization:** Leveraging NVIDIA cuOpt LP/MILP solvers
- **Data-driven risk modeling:** Implementing CVaR as a scenario-based risk measure that models tail risks without making assumptions about asset return distributions
- **Scenario generation:** Using GPU-accelerated Kernel Density Estimation (KDE) via NVIDIA cuML to model return distributions
- **Real-world constraint management:** Implementing constraints including concentration limits, leverage constraints, turnover limits, and cardinality constraints
- **Comprehensive backtesting:** Evaluating portfolio performance with tools for testing rebalancing strategies

## What to know before starting

**Required:**

- Basic familiarity with Terminal and Linux command line
- Basic understanding of Docker containers
- Basic knowledge of Jupyter Notebooks and JupyterLab
- Basic Python knowledge
- Basic knowledge of data science and machine learning concepts
- Basic knowledge of the stock market and equities

**Optional:**

- Background in financial services, especially quantitative finance and portfolio management
- Moderate experience programming algorithms and strategies in Python using machine learning concepts

**Terms to know:**

- **CVaR vs. mean-variance:** Unlike traditional mean-variance models, this workflow uses Conditional Value-at-Risk (CVaR) to capture nuances of risk, specifically tail risk or scenario-specific stresses.
- **Linear programming:** CVaR reformulates the risk-return tradeoff as a scenario-based linear program where the problem size scales with the number of scenarios, which is why GPU acceleration is critical.
- **Benchmarking:** The pipeline includes built-in tools to streamline benchmarking against standard CPU-based libraries to validate performance gains.

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
- Network access to pull the RAPIDS notebooks container and related packages
- Web browser access to JupyterLab on port 8888 (or a forwarded local port)

## Ancillary files

All required assets can be found [in the playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-portfolio-optimization/). In the running playbook, they are available under the `playbook` folder.

- `assets/cvar_basic.ipynb` — Main playbook notebook
- `assets/README.md` — Quick start guide to the playbook environment
- `assets/setup/start_playbook.sh` — Starts the playbook in a Docker container
- `assets/setup/setup_playbook.sh` — Configures the Docker container before you enter the JupyterLab environment
- `assets/setup/pyproject.toml` — Library list used by setup commands to install into the playbook environment
- `cudf`, `cuml`, and `cugraph` folders — Additional RAPIDS example notebooks included in the Docker container when you start it

## Time & risk

- **Estimated time:** 20 MIN for first run (full notebook pipeline typically ~7 minutes after the container is ready)
- **Risk level:** Low
  - Minimal, as this workflow runs in a Docker container
  - Container pull or environment build may fail due to network issues
- **Rollback:** Stop the Docker container and remove the cloned repository to fully remove the installation
- **Last Updated:** 08/03/2026
  - Run a Mean-CVaR portfolio optimization pipeline with cuOpt and cuML in a RAPIDS notebooks container

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
- `docker --version` prints something like `Docker version 28.3.3, build 980b856`. If Docker is missing, install it and retry.

## Step 2. Installation

Clone the playbook assets and start the containerized environment:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd client-hardware-playbooks/nvidia/playbook-portfolio-optimization/assets
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

Once in JupyterLab, you'll see `cvar_basic.ipynb` and the folders `cudf`, `cuml`, and `cugraph`.

- `cvar_basic.ipynb` is the playbook notebook — open it by double-clicking the file
- `cudf`, `cuml`, and `cugraph` contain standard RAPIDS library example notebooks for further exploration
- `playbook` contains the playbook files. The contents of this folder are read-only inside a rootless Docker container

If you want to install any of the playbook notebooks on your own system, check the READMEs that accompany the notebook assets.

## Step 3. Run the notebook

In JupyterLab, run `cvar_basic.ipynb`.

Before you start running cells, **change the kernel to "Portfolio Optimization"** as described in the notebook. Failure to do so will cause errors by the second code cell. If you already started with the wrong kernel, select the correct kernel, restart the kernel, and try again.

Use `Shift + Enter` to run each cell at your own pace, or `Run > Run All` to run all cells.

After exploring `cvar_basic`, you can open other RAPIDS notebooks in the `cudf`, `cuml`, and `cugraph` folders and run them the same way.

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

Once you're comfortable with this foundational workflow, explore these advanced portfolio optimization topics in any order in the [NVIDIA AI Blueprints quantitative portfolio optimization repository](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/):

- [`efficient_frontier.ipynb`](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/tree/main/notebooks/efficient_frontier.ipynb) — Efficient frontier analysis
  - Generate the efficient frontier by solving multiple optimization problems
  - Visualize the risk-return tradeoff across different portfolio configurations
  - Compare portfolios along the efficient frontier
  - Leverage GPU acceleration to compute multiple optimal portfolios quickly

- [`rebalancing_strategies.ipynb`](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/tree/main/notebooks/rebalancing_strategies.ipynb) — Dynamic portfolio rebalancing
  - Time-series backtesting framework
  - Testing rebalancing strategies (periodic, threshold-based, and more)
  - Evaluating the impact of transaction costs on portfolio performance
  - Analyzing strategy performance over different market conditions
  - Comparing multiple rebalancing approaches

- For more on formulating portfolio optimization problems with similar risk–return frameworks, see the [DLI course: Accelerating Portfolio Optimization](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-DS-09+V1)

## Step 7. Further support

For questions or issues, visit:

- [GitHub Issues](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/issues)

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every platform listed in this playbook.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| Docker is not found | All hardware platforms | Docker is not installed or not on `PATH` | Install Docker (for example: `curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`). You will be prompted for your password. |
| Docker command exits with a permissions error | All hardware platforms | Your user is not in the `docker` group | Run `sudo groupadd docker && sudo usermod -aG docker $USER`, then close the terminal, open a new one, and try again |
| Docker container download, environment build, or data download fails | All hardware platforms | Connectivity issue or a resource temporarily unavailable | Retry later; confirm network access to the container registry |
| JupyterLab UI not reachable in the browser | All hardware platforms | Port 8888 blocked, wrong host, or missing SSH forward | Use `http://127.0.0.1:8888` or `http://<HARDWARE_IP>:8888`; if remote, forward with `ssh -L 8888:localhost:8888 username@<HARDWARE_IP>` |
| Notebook errors by the second code cell | All hardware platforms | Wrong Jupyter kernel selected | Switch to the **Portfolio Optimization** kernel, restart the kernel, and re-run from the top |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
