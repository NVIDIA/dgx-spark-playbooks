# Portfolio Optimization

> GPU-Accelerated portfolio optimization using cuOpt and cuML

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

You will implement a pipeline that provides tools for performance evaluation, strategy backtesting, benchmarking, and visualization. The workflow includes:
- **GPU-Accelerated Optimization:** Leveraging NVIDIA cuOpt LP/MILP solvers 
- **Data-Driven Risk Modeling:** Implementing CVaR as a scenario-based risk measure that models tail risks without making assumptions about asset return distributions.
- **Scenario Generation:** Using GPU-accelerated Kernel Density Estimation (KDE) via NVIDIA cuML to model return distributions.
- **Real-World Constraint Management:** Implementing constraints including concentration limits, leverage constraints, turnover limits, and cardinality constraints.
- **Comprehensive Backtesting:** Evaluating portfolio performance with specific tools for testing rebalancing strategies.


## What to know before starting

- **Required Skills (you'll get it):**
  - Basic with Terminal and Linux command line
  - Basic understanding of Docker containers
  - Basic knowledge of using Jupyter Notebooks and Jupyter Lab
  - Basic Python knowledge
  - Basic knowledge of data science and machine learning concepts
  - Basic knowledge of what the stock market and stocks are

- **Optional Skills (you'll enjoy it):**
  - Background in Financial Services, especially in quantatitve finance and portfolio management
  - Moderate knowledge programming algorithms and strategies, in python, using machine learning concepts 

- **Terms to know:**
  - **CVaR vs. Mean-Variance:** Unlike traditional mean-variance models, this workflow uses Conditional Value-at-Risk (CVaR) to capture nuances of risk, specifically tail risk or scenario-specific stresses.
  - **Linear Programming:** CVaR reformulates the risk-return tradeoff as a scenario-based linear program where the problem size scales with the number of scenarios, which is why GPU acceleration is critical.
  - **Benchmarking:** The pipeline includes built-in tools to streamline the benchmarking process against standard CPU-based libraries to validate performance gains.

## Prerequisites

**Hardware Requirements:**
- NVIDIA Grace Blackwell GB10 Superchip System (DGX Spark)
- Minimum 40GB Unified memory free for docker container and GPU accelerated data processing
- At least 30GB available storage space for docker container and data files
- High speed internet connection recommended

**Software Requirements:**
- NVIDIA DGX OS with working NVIDIA and CUDA drivers
- Docker
- Git

## Ancillary files

All required assets can be found [in the Portfolio Optimization repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/portfolio-optimization/assets/).  In the running playbook, they will all be found under the `playbook` folder.

- `cvar_basic.ipynb` - Main playbook notebook.  
- `/setup/README.md` - Quick Start Guide to the Playbook Environment.
- `/setup/start_playbook.sh` - Script to start the install of the playbook in a Docker container
- `/setup/setup_playbook.sh` - Configures the Docker container before user enters jupyterlab environment
- `/setup/pyproject.toml` - used as a lists of libraries that commands in setup_playbook will install into the playbook environment 
- `cuDF, cuML, and cuGraph folders` - more example notebooks to continue your GPU Accelerated Data Science Journey.  These will be part of the Docker Container when you start it.

## Time & risk

* **Estimated Time** ~20 minutes for first run
  - Total Notebook Processing Time: Approximately 7 minutes for the full pipeline.

- **Risks:**
  - Minimal, as this is run in a Docker container.

* **Rollback:** Stop the Docker container and remove the cloned repository to fully remove the installation.

* **Last Updated:** 1/05/2026
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
- `docker --version` will print something like `Docker version 28.3.3, build 980b856`.  If you get an error saying that Docker is not installed, please reinstall it.

## Step 2. Installation
Open up Terminal, then copy and paste in the below commands:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks/nvidia/portfolio-optimization
cd dgx-spark-playbooks/nvidia/portfolio-optimization/assets
bash ./setup/start_playbook.sh
```

start_playbook.sh will:

1. pull the RAPIDS 25.10 Notebooks Docker container
2. build all the environments needed for the playbook in the container using `setup_playbook.sh`
3. start Jupyterlab

Please keep the Terminal window open while using the playbook.

You can access your Jupyterlab server in three ways
1. at `http://127.0.0.1:8888` if running locally on the DGX Spark. 
2. at `http://<SPARK_IP>:8888` if using your DGX Spark headless over your network.
3. by creating an SSH tunnel using `ssh -L 8888:localhost:8888 username@spark-IP` in Terminal and the going to `http://127.0.0.1:8888` in your browser on your host machine

Once in Jupyterlab, you'll be greeted with a directory containing `cvar_basic.ipynb`, and the folders `cudf`, `cuml` and `cugraph`. 

- `cvar_basic.ipynb` is the playbook notebook.  You will want to open this by double clicking on the file.
- `cudf`, `cuml`, `cugraph` folders contain the standard RAPIDS library example notebooks to help you continue exploring.
- `playbook` contains the playbook files.  The contents of this folder are read-only inside of a rootless Docker Container.

If you want to install any of the playbook notebooks on your own system, check out the readmes within the folder that accompanies the notebook

## Step 3. Run the notebook

Once in jupyterlab, you have to do is run the `cvar_basic.ipynb`. 

Before your start running the cells in the notebook, **please change the kernel to "Portfolio Optimization" as per the instructions in the notebook.**  Failure to do so will cause errors by the second code cell.  If you started already, you will have to set it to the correct kernel, then restart the kernel, and try again. 

You can use `Shift + Enter` to manually run each cell at your own pace, or `Run > Run All` to run all the cells.

Once you're done with exploring the `cvar_basic` notebook, you can explore other RAPIDS notebooks by going into the folders, selecting other notebooks, and doing the same thing.

## Step 4. Download your work

Since the docker container is not priviledged and cannot write back to the host system, you can use Jupyterlab to download any files you may want to keep once the docker container is shut down.

Simply right click the file you want, in the browser, and click `Download` in the drop down.

## Step 5. Cleanup

Once you have downloaded all your work, Go back to the Terminal window where you started running the playbook.

In the Terminal window:
1. Type `Ctrl + C`
2. Quickly either enter `y` and then hit `Enter` at the prompt or hit `Ctrl + C` again
3. The Docker container will proceed to shut down

> [!WARNING]
> This will delete ALL data that wasn't already downloaded from the Docker container.  The browser window may still show cached files if it is still open.

## Step 6. Next Steps

Once you're comfortable with this foundational workflow, please explore these advanced portfolio optimization topics in any order at the **[NVIDIA AI Blueprints](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/)**:

* **[`efficient_frontier.ipynb`](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/tree/main/notebooks/efficient_frontier.ipynb)** - Efficient Frontier Analysis

  This notebook demonstrates how to:
  - Generate the efficient frontier by solving multiple optimization problems
  - Visualize the risk-return tradeoff across different portfolio configurations
  - Compare portfolios along the efficient frontier
  - Leverage GPU acceleration to quickly compute multiple optimal portfolios

* **[`rebalancing_strategies.ipynb`](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/tree/main/notebooks/rebalancing_strategies.ipynb)** - Dynamic Portfolio Rebalancing

  This notebook introduces dynamic portfolio management techniques:
  - Time-series backtesting framework
  - Testing various rebalancing strategies (periodic, threshold-based, etc.)
  - Evaluating the impact of transaction costs on portfolio performance
  - Analyzing strategy performance over different market conditions
  - Comparing multiple rebalancing approaches

* If you'd further learn how to formulate portfolio optimization problems using similar risk–return frameworks, check out the **[DLI course: Accelerating Portfolio Optimization](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-DS-09+V1)**

## Step 7. Further Support

For questions or issues, please visit:
- [GitHub Issues](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/issues)
- [GitHub Discussions](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/discussions)

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
| Docker command unexpectedly exits with "permissions" error | Your user is not part of the `docker` group | Open Terminal and run these commands: `sudo groupadd docker $$ sudo usermod -aG docker $USER`.  You will be prompted for your password.  Then, close the Terminal, open a new one, and try again |
| Docker container download, environment build, or data download fails | There was either a connectivity issue or a resource may be temporariliy unavailable. | You may need to try again later. If this persist, please reach out to us! |




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
