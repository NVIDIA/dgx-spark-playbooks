# Develop and Deploy Healthcare Robots with Isaac For Healthcare

> End-to-end development and deployment of healthcare robots on DGX Spark

## Table of Contents

- [Overview](#overview)
- [Part 1: Preparation](#part-1-preparation)
  - [Set Up Conda Environment](#set-up-conda-environment)
  - [Set Up Docker Environment](#set-up-docker-environment)
  - [Set Up the Scene](#set-up-the-scene)
  - [Calibrate the Robot](#calibrate-the-robot)
  - [Test Teleoperation](#test-teleoperation)
- [Part 2: Synthetic Data Generation](#part-2-synthetic-data-generation)
- [Part 3: Real-World Data Collection](#part-3-real-world-data-collection)
- [Part 4: GR00T N1.5 Fine-Tuning](#part-4-gr00t-n15-fine-tuning)
- [Part 5: Deploying Trained Robotic Policy](#part-5-deploying-trained-robotic-policy)

---

## Overview

## Basic idea

Robotics and physical AI are driving the next wave of AI breakthroughs. Developing physical AI requires [3 computers](https://blogs.nvidia.com/blog/three-computers-robotics/) — 1. A simulation computer to generate synthetic data and digital twins, bridging the data gap. 2. A training computer to build the necessary foundation and world models. 3. A runtime computer to handle real-time robotic inference and intelligent interactions.

This tutorial demonstrates the development and deployment of an autonomous healthcare robot using [NVIDIA Isaac For Healthcare](https://developer.nvidia.com/blog/introducing-nvidia-isaac-for-healthcare-an-ai-powered-medical-robotics-development-platform/) on a single [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), consolidating the 3-computers developer workflow onto one hardware platform. The example focuses on the [SO-101 robot](https://github.com/TheRobotStudio/SO-ARM100?tab=readme-ov-file) acting as a scrub nurse—a specialized nursing professional working directly in the sterile field during surgical procedures—to perform a crucial pick-and-place task — autonomously picking up a pair of surgical scissors and placing them into a surgical tray.

## What you'll accomplish

You'll complete the full development lifecycle of an autonomous healthcare robot on DGX Spark, covering the following stages:

- **Part 1 — Preparation.** Set up the hardware, software environments, and task environment.
- **Part 2 — Generating synthetic data with Isaac Sim.** Collect synthetic pick-and-place demonstrations using teleoperation in a simulated environment.
- **Part 3 — Collecting real-world data.** Collect real-world teleoperation data with the physical SO-101 robot.
- **Part 4 — Fine-tuning the GR00T N1.5 model.** Fine-tune a pretrained GR00T N1.5 model using the collected data.
- **Part 5 — Deploying trained robotic policy.** Deploy the fine-tuned model in both simulated and real-world environments.

## What to know before starting

- Experience with Linux command line
- Basic understanding of Docker containers
- Familiarity with Python and conda environments
- Basic knowledge of robotics concepts (teleoperation, calibration)
- Familiarity with machine learning concepts (helpful but not required)

## Prerequisites

**Hardware Requirements:**
- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) with FastOS version 1.91.+ (verify with `cat /etc/fastos-release`; upgrade if necessary following [steps here](https://docs.nvidia.com/dgx/dgx-spark/system-recovery.html#recovery-process-steps))
- [SO-101 Robot](https://github.com/TheRobotStudio/SO-ARM100?tab=readme-ov-file) with both leader & follower arms and wrist camera module (ensure mounting/fixation tools are included or acquired separately)
- USB-C splitter (needed since 4 USB connections are required and DGX Spark has only 3 available USB-C ports; use a high-quality splitter to minimize latency)
- OpenCV compatible USB web camera (for the room camera)
- Surgical tray (dimensions 24cm x 16cm x 5cm)
- Surgical scissors (length 18cm)
- Scene setup accessories — table, table cloth, and a camera stand/holder for the room camera

**Software Requirements:**
- NVIDIA DGX OS
- Miniconda: [installation guidelines](https://www.anaconda.com/docs/getting-started/miniconda/install#aws-graviton2%2Farm64)
- Docker (pre-installed on DGX OS)

## Ancillary files

All required assets can be found in the [NVIDIA Isaac-For-Healthcare-Workflows repository](https://github.com/isaac-for-healthcare/i4h-workflows).

- `workflows/so_arm_starter/` - Source code for the robotic scrub nurse example workflow
- `tools/env_setup_so_arm_starter.sh` - Environment setup script for the conda environment
- `workflows/so_arm_starter/docker/dgx.Dockerfile` - Dockerfile for the Docker environment

## Time & risk

* **Estimated time:** Approximately 2 days (GR00T N1.5 fine-tuning at 30,000 steps takes around 24 hours on DGX Spark; data collection and other setup steps require several additional hours)
* **Risk level:** Medium
  * Robot calibration must remain consistent throughout the tutorial; re-calibrating after data collection or training may require restarting the entire process
  * Large downloads and Docker builds may take significant time
  * Leader and follower arm power cords have different voltages—do not mix them up
* **Rollback:** Conda environment and Docker image can be removed to revert software changes. Collected datasets can be deleted from `~/.cache/huggingface/lerobot/`.

## Part 1: Preparation

## Step 1. Prepare Hardware and Accessories

Required components:

* [**NVIDIA DGX Spark**](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) — Verify that FastOS version is 1.91.+ with `cat /etc/fastos-release`; upgrade if necessary following [steps here](https://docs.nvidia.com/dgx/dgx-spark/system-recovery.html#recovery-process-steps).
* [**SO-101 Robot**](https://github.com/TheRobotStudio/SO-ARM100?tab=readme-ov-file) — Requires both leader & follower arms with wrist camera module. Ensure mounting/fixation tools are included or acquired separately.
* **USB-C Splitter** — Needed since 4 USB connections (2 USB-C for arms, 2 USB-A for cameras) are required and DGX Spark has only 3 available USB-C ports. Use a high-quality splitter to minimize latency.
* **OpenCV compatible USB web camera** — For the room camera.
* **Surgical Tray** — Dimensions 24cm x 16cm x 5cm.
* **Surgical Scissors** — Length 18cm.
* **Scene Setup Accessories** — Table, table cloth, and a camera stand/holder for the room camera.

## Step 2. Set Up Software Environments

Power on DGX Spark and open a terminal window.

Create a folder named `workspace` under your home directory, and clone the NVIDIA Isaac-For-Healthcare-Workflows repository `i4h-workflows` from GitHub:

```shell
mkdir ~/workspace
cd ~/workspace && git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
```

The source code for several Isaac For Healthcare example workflows is in this repository, including the robotic scrub nurse example at `<path-to-i4h-workflows>/workflows/so_arm_starter`.

This tutorial requires two separate software environments on DGX Spark:

1. A conda environment for most of the tasks.
2. A docker environment for all tasks that require Isaac-GR00T.

A separate docker environment was needed primarily because of the complexity in installing certain Isaac-GR00T dependencies, like `flash_attn`, on the DGX Spark's native arm64 OS.

### Set Up Conda Environment

First, ensure Miniconda is installed on DGX Spark. If not, follow the [installation guidelines here](https://www.anaconda.com/docs/getting-started/miniconda/install#aws-graviton2%2Farm64). Then, create a new conda environment and install the necessary dependencies for this tutorial:

```shell
conda create -n so_arm_starter python=3.11 -y
conda activate so_arm_starter
cd <path-to-i4h-workflows> && bash tools/env_setup_so_arm_starter.sh
```

Installation takes about 20 minutes and, when complete, prints a success message to the terminal.

```shell
==========================================
Environment setup script finished.
==========================================
```

After installation, **deactivate and reactivate the `so_arm_starter` environment** to apply configurations:

```shell
conda deactivate
conda activate so_arm_starter
```

After reactivating the conda environment, set the following environment variable:

```shell
export PYTHONPATH=<path-to-i4h-workflows>/workflows/so_arm_starter/scripts
```

To avoid manually setting the environment variable each time you activate `so_arm_starter`, optionally add the command to `~/.bashrc`. Source the file immediately after adding it to activate it in the current session.

### Set Up Docker Environment

To set up the docker environment, build a docker image using the `dgx.Dockerfile` provided under `<path-to-i4h-workflows>/workflows/so_arm_starter/docker`:

```shell
cd <path-to-i4h-workflows>/workflows/so_arm_starter/docker
docker build -t soarm-dgx -f dgx.Dockerfile .
```

The build takes about 20 minutes, creating a docker image named `soarm-dgx`.

## Step 3. Set Up the Task Environment

### Set Up the Scene

To set up the scrub nurse pick-and-place scene:

1. **Mount Arms:** Firmly mount the follower arm on the table and the leader arm nearby for comfortable teleoperation.
2. **Set Scene:** Place the table cloth, surgical tray, and scissors on the table. Use a non-reflective, dark table cloth to minimize reflections and maintain consistent background color. Fixate the table cloth to the table to prevent movement when the follower's gripper touches it. Ensure the tray and scissors are within easy reach of the follower arm's gripper.
3. **Mount Camera:** Mount the room camera above the table for a top-down view. While other positions (like a side-view) might offer better object localization, the top-down view minimizes environmental elements, focusing only on task-relevant objects for a more robust setup.

To finally adjust the table and room camera stand for optimal wrist and room camera views, power on the robot and cameras. Connect the following to the DGX Spark:

* Leader and follower arms (2x USB-C)
* Wrist camera (1x USB-A)
* Room camera (1x USB-A or USB-C)

Due to limited DGX Spark USB-C ports, a USB-C splitter (and optional USB-A/C converters) is needed. Power the leader and follower arms, **taking care not to mix up the power cords as voltages differ.** Use a camera tool (e.g., Cheese on DGX Spark) to check live feeds and finalize positioning.

### Calibrate the Robot

First, identify the device IDs for the two robot arms and the two cameras.

Open a new terminal on DGX Spark. Activate the `so_arm_starter` conda environment:

```shell
conda activate so_arm_starter
```

Execute the following command and follow the on-screen instructions to identify the device IDs of the leader arm and the follower arm:

```shell
python -m lerobot.find_port
```

On a Linux-based system, the device IDs are usually `/dev/ttyACM0` and `/dev/ttyACM1`.

Execute the following command to identify the wrist and room camera indices:

```shell
python -m lerobot.find_cameras
```

The console should list 2 cameras with their indices (e.g., `/dev/video0` and `/dev/video2`). This command also captures and saves the current camera frames as distinct PNG images in `outputs/captured_images/`, using camera indices in the filename for easy identification and verification of feeds.

Set access permissions for the robot arms before calibration by running:

```shell
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

Adjust device IDs as needed. **Execute these commands every time the robot disconnects from and reconnects to DGX Spark.**

Run the following commands in the terminal to calibrate the leader arm and the follower arm:

```shell
## Leader arm:
python -m lerobot.calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=so101_leader

## Follower arm:
python -m lerobot.calibrate  --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=so101_follower
```

Adjust device IDs and customize `--teleop.id` and `--robot.id` to set different device names if needed. Then, follow on-screen instructions and refer to the [video here](https://huggingface.co/docs/lerobot/so101#calibration-video) for proper calibration.

> [!WARNING]
> Maintain *one* single follower arm calibration for this tutorial. Re-calibrating after collecting data or training the GR00T model risks needing to restart everything, as subsequent steps rely on the initial calibration.

### Test Teleoperation

To complete the preparation, teleoperate the follower arm using the leader arm.

Run the following command to teleoperate without camera feeds:

```shell
python -m lerobot.teleoperate \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM1 \
--robot.id=so101_follower \
--teleop.type=so101_leader \
--teleop.port=/dev/ttyACM0 \
--teleop.id=so101_leader
```

Adjust the `--robot.port`, `--teleop.port`, `--robot.id` and `--teleop.id` arguments if needed.

Run the following command to teleoperate with camera feeds:

```shell
python -m lerobot.teleoperate \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM1 \
--robot.id=so101_follower \
--robot.cameras="{wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, room: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
--teleop.type=so101_leader \
--teleop.port=/dev/ttyACM0 \
--teleop.id=so101_leader \
--display_data=true
```

Adjust device IDs, names and camera indices if needed.

During teleoperation with camera feeds, the [Rerun viewer](https://rerun.io/) UI appears, showing real-time views from both cameras and the robot's motor action data.

## Part 2: Synthetic Data Generation

## Step 1. Launch Isaac Sim for Data Collection

Ensure the leader arm is powered on and connected to DGX Spark. Open a new terminal on DGX Spark, activate the `so_arm_starter` conda environment and set the `PYTHONPATH`:

```shell
conda activate so_arm_starter
export PYTHONPATH=<path-to-i4h-workflows>/workflows/so_arm_starter/scripts
```

Then, run the following command in the terminal:

```shell
python -m simulation.environments.teleoperation_record  \
 --port=/dev/ttyACM0 \
 --enable_cameras \
 --record \
 --dataset_path=./data-collection-sim/dataset.hdf5
```

If needed, adjust the leader arm device ID and modify the `--dataset_path` argument to save data elsewhere.

The command launches [Isaac Sim](https://developer.nvidia.com/isaac/sim), loading a scene with a follower arm, table, surgical scissors, and a tray. The initial load may take about 2 minutes; if Isaac Sim seems unresponsive, do not force quit—wait for it to load fully.

To change the simulated follower arm's color to match your physical robot, go to the `Stage` panel (right side of Isaac Sim) → `World` → `envs` → `env_0` → `robot` → `Looks` → `material_a_3d_printed`, then under the `Property` tab, adjust the `Albedo Color`.

The first command run requires leader arm calibration, even if previously done, due to a different program-specific calibration file. Your existing calibration remains unchanged.

## Step 2. Collect Synthetic Pick-and-Place Demonstrations

To teleoperate the robot in Isaac Sim and collect synthetic pick-and-place demonstrations:

* Press "B" to begin teleoperation; the robot moves to the initial position.
* Use the physical leader arm to control the virtual follower arm for the pick-and-place task.
* Press "N" to save a successful episode.
* Press "R" to restart without saving.
* Scissors position and angle are slightly randomized per new episode.
* Press Ctrl + C to quit.

Use these shortcuts for Isaac Sim viewport navigation:

* "F" key after clicking the robot to auto-focus.
* Middle mouse wheel to zoom.
* "ALT" + left mouse drag to change the view angle.
* Middle mouse wheel click + drag to move in the viewport.

Collecting around 70 synthetic episodes is sufficient for this tutorial.

## Step 3. Convert Data to LeRobot Format

After collecting the synthetic data, convert them to the Hugging Face [LeRobot](https://github.com/huggingface/lerobot) dataset format for fine-tuning the Isaac GR00T model:

```shell
python -m training.hdf5_to_lerobot \
--repo_id=spark/scrub-nurse-sim \
--hdf5_path=./data-collection-sim/dataset.hdf5 \
--task_description="Grip the scissors and put them into the tray."
```

Modify `--repo_id` and `--task_description` as needed, but ensure a meaningful task description. The resulting dataset, containing motor actions, wrist camera, and room camera recordings, is stored under `/home/$USER/.cache/huggingface/lerobot/<repo_id>`.

## Part 3: Real-World Data Collection

## Step 1. Set Up for Real-World Data Collection

Ensure the leader arm, follower arm, wrist camera, and room camera are connected to DGX Spark. On DGX Spark, open a new terminal, activate the `so_arm_starter` conda environment:

```shell
conda activate so_arm_starter
```

## Step 2. Collect Real-World Data Episodes

Run the following command to collect real-world data episodes as LeRobot dataset:

```shell
python -m lerobot.record \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM1 \
--robot.cameras="{wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, room: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
--robot.id=so101_follower \
--teleop.type=so101_leader \
--teleop.port=/dev/ttyACM0 \
--teleop.id=so101_leader \
--display_data=true \
--dataset.repo_id="spark/scrub-nurse-real" \
--dataset.num_episodes=20 \
--dataset.single_task="Grip the scissors and put them into the tray." \
--dataset.push_to_hub=false
```

Modify robot device IDs, names and camera indices to match yours. Ensure `--dataset.single_task` matches the task description for synthetic data collection. You can change `--dataset.repo_id` to alter the LeRobot dataset name. The dataset will be saved under `/home/$USER/.cache/huggingface/lerobot/<repo_id>`.

The command initiates the Rerun viewer and teleoperation for both arms. Follow these steps for pick-and-place demonstration recording:

* The recording starts immediately upon command execution for the current episode; be prepared or you'll need to re-record.
* Each episode's recording has three sequential states:
  1. **Demonstration recording** (60s) — Record the task.
  2. **Scene Reset** (60s) — Perform randomization, robot/object resets. Rerun displays signals, but no recording occurs.
  3. **Data Saving** (approx. 5s) — Saves recording to a LeRobot dataset. Rerun temporarily freezes; no recording occurs.
* Right Arrow (→) — skips to the next state. Cannot skip State 3 (saving stage); pressing it then could corrupt the episode.
* Left Arrow (←) (during State 1) — cancels the current recording, giving 60 seconds to reset the scene before recording restarts. Use this if you mess up.
* **ESC** — stops recording and saves all currently recorded content. Use after a completed successful episode to avoid including unwanted "garbage" data.
* Collecting multiple small, separate LeRobot datasets might be easier, and they can be combined for GR00T training later.

## Step 3. Prepare Datasets for Training

After creating the datasets, copy the `modality.json` file generated during synthetic data creation (e.g., `/home/$USER/.cache/huggingface/lerobot/spark/scrub-nurse-sim/meta/modality.json`) to each dataset's `meta` folder. This file is essential for GR00T model training.

Collecting 20 real-world episodes should be sufficient for this tutorial.

## Part 4: GR00T N1.5 Fine-Tuning

## Step 1. Launch Docker Container

Run the following command on DGX Spark to start a docker container:

```shell
docker run -it --gpus all --privileged --rm \
	--ipc=host \
	--network=host \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	--entrypoint=bash \
	-e "NVIDIA_VISIBLE_DEVICES=all" \
	-e "PYTHONPATH=<path-to-i4h-workflows>/workflows/so_arm_starter/scripts"\
	-v /dev:/dev \
	-v /home/"$USER"/.cache/huggingface/lerobot:/root/.cache/huggingface/lerobot \
	-v $(pwd):/workspace \
	-w /workspace \
soarm-dgx
```

We mount `/home/"$USER"/.cache/huggingface/lerobot` to the container so previous calibration files and datasets are accessible.

## Step 2. Download Pretrained Model

Download our pretrained GR00T N1.5 model [here](https://github.com/isaac-for-healthcare/i4h-workflows/blob/main/workflows/so_arm_starter/README.md#-running-workflows). The model was trained on 70 simulated and 5 real episodes. This model will likely require fine-tuning due to variations in your robot hardware, calibration, and task setup.

## Step 3. Run GR00T N1.5 Fine-Tuning

Run the following command to run GR00T N1.5 fine-tuning:

```shell
PYTHONWARNINGS="ignore::UserWarning" python -m training.gr00t_n1_5.train \
--dataset_path <dataset-1> <dataset-2> ... \
--output_dir /workspace/training-output/ \
--data_config so100_dualcam \
--base-model-path <pretrained-gr00t-model> \
--max-steps 30000 \
--save-steps 2000
```

Change `--base-model-path` to the pretrained model path. Experiment with `--max-steps` and `--save-steps`; we found 30,000 steps typically sufficient for convergence. On DGX Spark, 30,000 steps should take around 24 hours.

You can use Tensorboard to monitor the training progress.

## Part 5: Deploying Trained Robotic Policy

## Step 1. Convert Model to TensorRT Format

To get the optimal inference performance, let's convert the fine-tuned GR00T N1.5 model to [TensorRT](https://developer.nvidia.com/tensorrt) format.

Open a terminal window and create the same docker container as in Part 4. Then, run the following commands:

```shell
python -m policy_runner.gr00tn1_5.trt.export_onnx --ckpt_path <fine-tuned-gr00t-model-path>
bash <path-to-i4h-workflows>/workflows/so_arm_starter/scripts/policy_runner/gr00tn1_5/trt/build_engine.sh
```

This generates a `gr00t_engine` folder that contains the converted TensorRT model. Avoid running heavy compute or graphics tasks on DGX Spark during conversion.

## Step 2. Deploy in Isaac Sim

To deploy the trained policy model in Isaac Sim, an [RTI DDS](https://www.rti.com/products/dds-standard) license file is required for communication of different modules. Get a professional or evaluation license from [here](https://www.rti.com/get-connext).

Open a new terminal window and create the same docker container as in Part 4. First, set the `RTI_LICENSE_FILE` environment variable:

```shell
export RTI_LICENSE_FILE=<path-to-rti-license-file>
```

Then, run the following command:

```shell
python -m policy_runner.run_policy \
--ckpt_path=<fine-tuned-gr00t-model-path> \
--task_description="Grip the scissors and put them into the tray." \
--trt \
--trt_engine_path=<fine-tuned-gr00t-tensorrt-model>
```

This loads the GR00T model for inference in the background.

Open another terminal window. Activate the `so_arm_starter` conda environment and set `PYTHONPATH` and `RTI_LICENSE_FILE`:

```shell
conda activate so_arm_starter
export PYTHONPATH=<path-to-i4h-workflows>/workflows/so_arm_starter/scripts
export RTI_LICENSE_FILE=<path-to-rti-license-file>
```

Then, run the following command in the terminal:

```shell
python -m simulation.environments.sim_with_dds --enable_cameras
```

Isaac Sim will open up and load the pick-and-place scene, then the simulated robot will execute the task autonomously, driven by the GR00T N1.5 policy model.

## Step 3. Deploy in Real World

Ensure the follower arm, wrist camera, and room camera are connected to DGX Spark.

Launch the same docker container as in Part 4. Find and modify the configuration file under `<path-to-i4h-workflows>/workflows/so_arm_starter/scripts/holoscan_apps/soarm_robot_config.yaml` to update the follower arm's device ID, name, camera indices, and the fine-tuned GR00T model path. Then, run the following command:

```shell
python -m holoscan_apps.gr00t_inference_app \
--config <path-to-i4h-workflows>/workflows/so_arm_starter/scripts/holoscan_apps/soarm_robot_config.yaml
```

This command launches an efficient GR00T N1.5 inference application using [NVIDIA Holoscan SDK](https://github.com/nvidia-holoscan/holoscan-sdk). The follower arm will execute the task autonomously shortly after.

## Conclusion

This tutorial demonstrated the end-to-end workflow of developing and deploying an autonomous healthcare robot on a single **NVIDIA DGX Spark**. Leveraging **NVIDIA Isaac For Healthcare**, we consolidated the 3-computers workflow of synthetic data generation, GR00T N1.5 training, and robotic policy deployment onto one powerful hardware platform. This workflow highlights the efficiency of the DGX Spark for accelerating the physical AI development pipeline, making the creation and deployment of intelligent healthcare robots more streamlined and accessible.
