# Install and Use Isaac Sim and Isaac Lab

> Build Isaac Sim and Isaac Lab from source for Spark

## Table of Contents

- [Overview](#overview)
- [Run Isaac Sim](#run-isaac-sim)
- [Run Isaac Lab](#run-isaac-lab)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Isaac Sim is a robotics simulation platform built on NVIDIA Omniverse that enables photorealistic, physically accurate simulations of robots and environments. It provides a comprehensive toolkit for robotics development, including physics simulation, sensor simulation, and visualization capabilities. Isaac Lab is a reinforcement learning framework built on top of Isaac Sim, designed for training and deploying RL policies for robotics applications.

Isaac Sim uses GPU-accelerated physics simulation to enable fast, realistic robot simulations that can run faster than real-time. Isaac Lab extends this with pre-built RL environments, training scripts, and evaluation tools for common robotics tasks like locomotion, manipulation, and navigation. Together, they provide an end-to-end solution for developing, training, and testing robotics applications entirely in simulation before deploying to real hardware.

## What you'll accomplish

You'll build Isaac Sim from source on your NVIDIA DGX Spark device and set up Isaac Lab for reinforcement learning experiments. This includes compiling the Isaac Sim engine, configuring the development environment, and running a sample RL training task to verify the installation.

## What to know before starting

- Experience building software from source using CMake and build systems
- Familiarity with Linux command line operations and environment variables
- Understanding of Git version control and Git LFS for large file management
- Basic knowledge of Python package management and virtual environments
- Familiarity with robotics simulation concepts (helpful but not required)

## Prerequisites

**Hardware Requirements:**
- NVIDIA Grace Blackwell GB10 Superchip System
- At least 50GB available storage space for Isaac Sim build artifacts and dependencies

**Software Requirements:**
- NVIDIA DGX OS
- GCC/G++ 11 compiler: `gcc --version` shows version 11.x
- Git and Git LFS installed: `git --version` and `git lfs version` succeed
- Network access to clone repositories from GitHub and download dependencies

## Ancillary files

All required assets can be found in the Isaac Sim and Isaac Lab repositories on GitHub:
- [Isaac Sim repository](https://github.com/isaac-sim/IsaacSim) - Main Isaac Sim source code
- [Isaac Lab repository](https://github.com/isaac-sim/IsaacLab) - Isaac Lab RL framework

## Time & risk

* **Estimated time:** 30 min (including build time which typically takes 10-15 minutes)
* **Risk level:** Medium
  * Large repository clones with Git LFS may fail due to network issues
  * Build process requires significant compilation time and may encounter dependency issues
  * Build artifacts consume substantial disk space
* **Rollback:** Isaac Sim build directory can be removed to free space. Git repositories can be deleted and re-cloned if needed.
* **Last Updated:** 1/06/2024
  * First Publication

## Run Isaac Sim

## Step 1. Install gcc-11 and git-lfs

Confirm that GCC/G++ 11 is being used before building using the following commands:
```bash
sudo apt update && sudo apt install -y gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200
sudo apt install git-lfs
gcc --version
g++ --version
```

## Step 2. Clone the Isaac Sim repository into your workspace

Clone Isaac Sim from the NVIDIA GitHub repository and set up Git LFS to pull large files.

> **Note:** For Isaac Sim 6.0.0 Early Developer Release, use:
> ```bash
> git clone --depth=1 --recursive --branch=develop https://github.com/isaac-sim/IsaacSim
> ```

```bash
git clone --depth=1 --recursive https://github.com/isaac-sim/IsaacSim
cd IsaacSim
git lfs install
git lfs pull
```

## Step 3. Build Isaac Sim

Build Isaac Sim and accept the license agreement.

```bash
./build.sh
```

You get this following message when build is successful: **BUILD (RELEASE) SUCCEEDED (Took 674.39 seconds)**


## Step 4. Recognize Isaac Sim for the system.

Be sure that you are inside Isaac Sim directory when running the following commands.

```bash
export ISAACSIM_PATH="${PWD}/_build/linux-aarch64/release"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

## Step 5. Run Isaac Sim

Launch Isaac Sim using the provided Python executable.

```bash
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
${ISAACSIM_PATH}/isaac-sim.sh
```

## Run Isaac Lab

## Step 1. Install Isaac Sim
If you haven't already done so, install [Isaac Sim](build.nvidia.com/spark/isaac/isaac-sim) first.

## Step 2. Clone the Isaac Lab repository into your workspace

Clone Isaac Lab from the NVIDIA GitHub repository.

```bash
git clone --recursive https://github.com/isaac-sim/IsaacLab
cd IsaacLab
```

## Step 3. Create a symbolic link to the Isaac Sim installation

Be sure that you have already installed Isaac Sim from [Isaac Sim](build.nvidia.com/spark/isaac/isaac-sim) before running the following command.

```bash
echo "ISAACSIM_PATH=$ISAACSIM_PATH"
```
Create a symbolic link to the Isaac Sim installation directory.
```bash
ln -sfn "${ISAACSIM_PATH}" "${PWD}/_isaac_sim"
ls -l "${PWD}/_isaac_sim/python.sh"
```

## Step 4. Install Isaac Lab.

```bash
./isaaclab.sh --install
```

## Step 5. Run Isaac Lab and Validate Humanoid Reinforcement Learning Training

Launch Isaac Lab using the provided Python executable. You can run the training in one of the following modes:

**Option 1: Headless Mode (Recommended for Faster Training)**

Runs without visualization and outputs logs directly to the terminal.

```bash
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-H1-v0 --headless
```

**Option 2: Visualization Enabled**

Runs with real-time visualization in Isaac Sim, allowing you to monitor the training process interactively.

```bash
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-H1-v0
```

## Troubleshooting

## Common issues for Isaac Sim

| Symptom                     | Cause                    | Fix                               |
|-----------------------------|--------------------------|-----------------------------------|
| Isaac Sim error compilation | gcc+11 is not by default | Be sure that gcc+11 is by default |
| Isaac Sim not executes      | Error libgomp.so.1       | Add export LD_PRELOAD             |
| Error in build              | old installation         | Remove .cache folder              |

## Common Issues for Isaac Lab
| Symptom                          | Cause | Fix |
|----------------------------------|--------|-----|
| Isaac Lab not executes           | Error libgomp.so.1       | Add export LD_PRELOAD     |
