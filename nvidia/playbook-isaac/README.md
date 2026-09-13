# Set Up Isaac Sim and Isaac Lab

> Realistic robot simulation and RL training before real-world deployment

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Isaac Sim is a robotics simulation platform built on NVIDIA Omniverse that enables photorealistic, physically accurate simulations of robots and environments. It provides a comprehensive toolkit for robotics development, including physics simulation, sensor simulation, and visualization capabilities. Isaac Lab is a reinforcement learning framework built on top of Isaac Sim, designed for training and deploying RL policies for robotics applications.

Isaac Sim uses GPU-accelerated physics simulation to enable fast, realistic robot simulations that can run faster than real-time. Isaac Lab extends this with pre-built RL environments, training scripts, and evaluation tools for common robotics tasks like locomotion, manipulation, and navigation. Together, they provide an end-to-end solution for developing, training, and testing robotics applications entirely in simulation before deploying to real hardware.

## What you'll accomplish

You'll build Isaac Sim from source on your **hardware platform** and set up Isaac Lab for reinforcement learning experiments. This includes compiling the Isaac Sim engine, configuring the development environment, and running a sample RL training task to verify the installation.

## What to know before starting

**Required:**

- Experience building software from source using CMake and build systems
- Familiarity with Linux command line operations and environment variables
- Understanding of Git version control and Git LFS for large file management
- Basic knowledge of Python package management and virtual environments

**Optional:**

- Familiarity with robotics simulation concepts

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Isaac Sim build from source (`linux-aarch64`); Isaac Lab RL sample (`Isaac-Velocity-Rough-H1-v0`) | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- At least 50 GB available storage for Isaac Sim build artifacts and dependencies

**Software requirements**

- OS matching the Supported hardware platforms matrix above
- GCC/G++ 11 compiler: `gcc --version` shows version 11.x
- Git and Git LFS installed: `git --version` and `git lfs version` succeed
- Network access to clone repositories from GitHub and download dependencies

## Ancillary files

All required assets can be found in the Isaac Sim and Isaac Lab repositories on GitHub:

- [Isaac Sim repository](https://github.com/isaac-sim/IsaacSim) — Main Isaac Sim source code
- [Isaac Lab repository](https://github.com/isaac-sim/IsaacLab) — Isaac Lab RL framework

## Time & risk

- **Estimated time:** 30 MIN (including build time, which typically takes 10–15 minutes)
- **Risk level:** Medium
  - Large repository clones with Git LFS may fail due to network issues
  - Build process requires significant compilation time and may encounter dependency issues
  - Build artifacts consume substantial disk space
- **Rollback:** Remove the Isaac Sim build directory to free space. Git repositories can be deleted and re-cloned if needed.
- **Last Updated:** 08/03/2026
  - Build Isaac Sim from source and set up Isaac Lab with a sample reinforcement learning training run on supported hardware platforms

## Instructions

## Step 1. Install gcc-11 and git-lfs

Confirm that GCC/G++ 11 is the default compiler before building:

```bash
sudo apt update && sudo apt install -y gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200
sudo apt install git-lfs
gcc --version
g++ --version
```

Expected output should show GCC/G++ version 11.x.

## Step 2. Clone the Isaac Sim repository

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

When the build succeeds, you should see a message similar to: **BUILD (RELEASE) SUCCEEDED (Took 674.39 seconds)**

## Step 4. Set Isaac Sim environment paths

Stay inside the Isaac Sim directory when running the following commands. The release path uses the `linux-aarch64` build layout for this hardware platform:

```bash
export ISAACSIM_PATH="${PWD}/_build/linux-aarch64/release"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

## Step 5. Run Isaac Sim

Launch Isaac Sim using the provided Python executable. Preload `libgomp` so the runtime finds the OpenMP library on this hardware platform:

```bash
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
${ISAACSIM_PATH}/isaac-sim.sh
```

Isaac Sim should start. Close it when you are ready to continue with Isaac Lab.

## Step 6. Clone the Isaac Lab repository

Clone Isaac Lab from the NVIDIA GitHub repository. Complete Steps 1–5 first so `ISAACSIM_PATH` points at a built Isaac Sim release.

> **Note:** For Isaac Lab Early Developer Release, use:
> ```bash
> git clone --recursive --branch=develop https://github.com/isaac-sim/IsaacLab
> ```

```bash
git clone --recursive https://github.com/isaac-sim/IsaacLab
cd IsaacLab
```

## Step 7. Create a symbolic link to the Isaac Sim installation

Confirm `ISAACSIM_PATH` is set, then create a symbolic link to the Isaac Sim installation directory:

```bash
echo "ISAACSIM_PATH=$ISAACSIM_PATH"
ln -sfn "${ISAACSIM_PATH}" "${PWD}/_isaac_sim"
ls -l "${PWD}/_isaac_sim/python.sh"
```

Expected output should show a valid symlink to `python.sh` under `_isaac_sim`.

## Step 8. Install Isaac Lab

Install X11 development libraries required to build `imgui_bundle` from source for Newton, then install Isaac Lab:

```bash
sudo apt update
sudo apt install -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev
./isaaclab.sh --install
```

## Step 9. Run Isaac Lab and validate humanoid reinforcement learning training

Launch Isaac Lab using the provided Python executable. You can run the training in one of the following modes:

**Option 1: Headless mode (recommended for faster training)**

Runs without visualization and outputs logs directly to the terminal.

```bash
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-H1-v0 --headless
```

**Option 2: Visualization enabled**

Runs with real-time visualization in Isaac Sim, allowing you to monitor the training process interactively.

```bash
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-H1-v0
```

Training should start and print progress logs for the `Isaac-Velocity-Rough-H1-v0` task.

## Step 10. Cleanup

Cleanup is optional rollback — not required to finish the playbook.

> [!WARNING]
> This removes the cloned Isaac Sim and Isaac Lab repositories and their build artifacts. Copy any custom work elsewhere first if you want to keep it.

```bash
cd ..
rm -rf IsaacSim IsaacLab
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Isaac Sim compilation fails | GCC/G++ 11 is not the default compiler | Install and select gcc-11/g++-11 (Instructions Step 1); confirm with `gcc --version` and `g++ --version` |
| Isaac Sim does not start | Missing or unresolved `libgomp.so.1` | `export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"` before launching |
| Build fails after a previous attempt | Stale build cache from an older installation | Remove the `.cache` folder under the Isaac Sim repository and rebuild |
| Isaac Lab does not start | Missing or unresolved `libgomp.so.1` | `export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"` before launching |
| Symlink to Isaac Sim is broken | `ISAACSIM_PATH` unset or points at the wrong release directory | Re-export `ISAACSIM_PATH` to `${PWD}/_build/linux-aarch64/release` from the Isaac Sim tree, then recreate `_isaac_sim` (Instructions Steps 4 and 7) |

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
