# NCCL for Multiple Sparks

> Install and test NCCL on two, three, or four Sparks

## Table of Contents

- [Overview](#overview)
- [Run on two Sparks](#run-on-two-sparks)
- [Run on three Sparks](#run-on-three-sparks)
- [Run on four Sparks](#run-on-four-sparks)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NCCL (NVIDIA Collective Communication Library) enables high-performance GPU-to-GPU communication
across multiple nodes. This walkthrough sets up NCCL for multi-node distributed training on
two, three, or four DGX Spark systems with Blackwell architecture. You'll configure networking,
build NCCL from source with Blackwell support, and validate communication between nodes.

## What you'll accomplish

You'll have a working multi-node NCCL environment that enables high-bandwidth GPU communication
across DGX Spark systems for distributed training workloads, with validated network performance
and proper GPU topology detection.

## What to know before starting

- Working with Linux network configuration and netplan
- Basic understanding of MPI (Message Passing Interface) concepts
- SSH key management and passwordless authentication setup

## Prerequisites

- Two, three, or four DGX Spark systems
- Completed the matching connection playbook for your node count:
  - 2 Sparks — [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks)
  - 3 Sparks — [Connect Three Sparks](https://build.nvidia.com/spark/connect-three-sparks/three-sparks-ring)
  - 4 Sparks — [Connect Multiple Sparks through a Switch](https://build.nvidia.com/spark/multi-sparks-through-switch/multi-sparks)
- NVIDIA driver installed: `nvidia-smi`
- CUDA toolkit available: `nvcc --version`
- Root/sudo privileges: `sudo whoami`

## Time & risk

* **Duration**: 30 minutes for setup and validation
* **Risk level**: Medium - involves network configuration changes
* **Rollback**: The NCCL & NCCL Tests repositories can be deleted from DGX Spark
* **Last Updated:** 12/15/2025
  * Use nccl latest version v2.30.7-1

## Run on two Sparks

## Quick start (scripts)

If you just want a working setup fast, use the helper scripts from GitHub. They
automate Steps 2-5 below. Complete Step 1 (network setup) first, then run
everything from **Node 1** (the launcher), passing each node's **management IP**
(the address you SSH to):

```bash
## 1. Download the helper scripts.
curl -fsSL https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/nccl/assets/setup.sh -o setup.sh
curl -fsSL https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/nccl/assets/launch.sh -o launch.sh

## 2. Build NCCL v2.30.7-1 and the test suite on both nodes.
bash setup.sh <NODE_2_IP>

## 3. Run the all_gather test across both nodes.
##    Assumes the Ethernet interface enP7s7. On Wi-Fi, prefix the command with
##    MGMT_IFNAME=wlP9s9 (your Wi-Fi interface) and use Wi-Fi IPs. See Step 4.
bash launch.sh --topology direct <NODE_1_IP> <NODE_2_IP>
```

To understand what the scripts do — or to debug — follow the manual steps below.

---

## Step 1. Configure network connectivity

Follow the network setup instructions from the [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks) playbook to establish connectivity between your DGX Spark nodes.

This includes:
- Physical QSFP cable connection
- Network interface configuration (automatic or manual IP assignment)
- Passwordless SSH setup
- Network connectivity verification

## Step 2. Build NCCL with Blackwell support

Execute these commands on both nodes to build NCCL from source with Blackwell
architecture support:

```bash
## Install dependencies and build NCCL
sudo apt-get update && sudo apt-get install -y libopenmpi-dev
git clone -b v2.30.7-1 https://github.com/NVIDIA/nccl.git ~/nccl/
cd ~/nccl/
make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"

## Set environment variables
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
```

## Step 3. Build NCCL test suite

Compile the NCCL test suite on **both nodes**:

```bash
## Clone and build NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests/
cd ~/nccl-tests/
make MPI=1
```

## Step 4. Confirm the CX-7 ports and note each node's management IP

```bash
## Check network port status
ibdev2netdev
```

Example output:
```text
rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Down)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Down)
```

For the test command you need each node's **management IP** (the regular
Ethernet address you SSH to). Find it on each node with:

```bash
ip addr show enP7s7
```

Take note of the management IP for **both nodes**.

> [!NOTE]
> These steps assume the wired **Ethernet** management interface (`enP7s7`). If
> your nodes use **Wi-Fi** instead (no Ethernet), replace `enP7s7` with your
> Wi-Fi interface (e.g. `wlP9s9` — confirm the name with `ip -o link show`) in
> Step 5, and use each node's **Wi-Fi IP** as its management IP. All nodes must
> use the same interface — either `enP7s7` on every node or `wlP9s9` on every
> node, not a mix.

## Step 5. Run NCCL communication test

> [!NOTE] 
> Full bandwidth can be achieved with just one QSFP cable.
> When two QSFP cables are connected, all four interfaces must be assigned IP addresses to obtain full bandwidth.

Run these commands on **Node 1** (the launcher); `mpirun` launches the test across all nodes over SSH. Replace the IP addresses and interface names with the ones you found in the previous step.

```bash
## Set network interface environment variables (use your management network interface)
export UCX_NET_DEVICES=enP7s7
export NCCL_SOCKET_IFNAME=enP7s7
export OMPI_MCA_btl_tcp_if_include=enP7s7

## Run the all_gather performance test across both nodes (replace the management IP addresses with the ones you found from the previous step)
mpirun -np 2 -H <management IP for Node 1>:1,<management IP for Node 2>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf
```

You can also test your NCCL setup with a larger buffer size to use more of your 200Gbps bandwidth.

```bash
## Set network interface environment variables (use your management network interface)
export UCX_NET_DEVICES=enP7s7
export NCCL_SOCKET_IFNAME=enP7s7
export OMPI_MCA_btl_tcp_if_include=enP7s7

## Run the all_gather performance test across both nodes
mpirun -np 2 -H <management IP for Node 1>:1,<management IP for Node 2>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

> [!NOTE]
> The IP addresses in the `mpirun` command are followed by `:1`. For example, `mpirun -np 2 -H 192.168.0.10:1,192.168.0.20:1`

## Step 6. Cleanup and rollback

```bash
## Rollback network configuration (if needed)
rm -rf ~/nccl/
rm -rf ~/nccl-tests/
```

## Step 7. Next steps
Your NCCL environment is ready for multi-node distributed training workloads on DGX Spark.
Now you can try running a larger distributed workload such as TRT-LLM or vLLM inference.

## Run on three Sparks

## Quick start (scripts)

If you just want a working setup fast, use the helper scripts from GitHub. They
automate Steps 2-5 below. Complete Step 1 (network setup) first, then run
everything from **Node 1** (the launcher), passing each node's **management IP**
(the address you SSH to):

```bash
## 1. Download the helper scripts.
curl -fsSL https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/nccl/assets/setup.sh -o setup.sh
curl -fsSL https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/nccl/assets/launch.sh -o launch.sh

## 2. Build NCCL v2.30.7-1 and the test suite on all three nodes.
bash setup.sh <NODE_2_IP> <NODE_3_IP>

## 3. Run the all_gather test across all three nodes.
##    Assumes the Ethernet interface enP7s7. On Wi-Fi, prefix the command with
##    MGMT_IFNAME=wlP9s9 (your Wi-Fi interface) and use Wi-Fi IPs. See Step 4.
bash launch.sh --topology ring <NODE_1_IP> <NODE_2_IP> <NODE_3_IP>
```

To understand what the scripts do — or to debug — follow the manual steps below.

---

## Step 1. Configure network connectivity

Follow the network setup instructions from the [Connect three Sparks](https://build.nvidia.com/spark/connect-three-sparks/three-sparks-ring) playbook to establish connectivity between your DGX Spark nodes.

This includes:
- Physical QSFP cable connection
- Network interface configuration (automatic or manual IP assignment)
- Passwordless SSH setup
- Network connectivity verification

## Step 2. Build NCCL with Blackwell support

Execute these commands on all three nodes to build NCCL from source with Blackwell
architecture support:

```bash
## Install dependencies and build NCCL
sudo apt-get update && sudo apt-get install -y libopenmpi-dev
git clone -b v2.30.7-1 https://github.com/NVIDIA/nccl.git ~/nccl/
cd ~/nccl/
make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"

## Set environment variables
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
```

## Step 3. Build NCCL test suite

Compile the NCCL test suite on **all three nodes**:

```bash
## Clone and build NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests/
cd ~/nccl-tests/
make MPI=1
```

## Step 4. Confirm the CX-7 ports and note each node's management IP

```bash
## Check network port status
ibdev2netdev
```

Example output:
```text
rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
```

For the test command you need each node's **management IP** (the regular
Ethernet address you SSH to). Find it on each node with:

```bash
ip addr show enP7s7
```

Take note of the management IP for **all three nodes**.

> [!NOTE]
> These steps assume the wired **Ethernet** management interface (`enP7s7`). If
> your nodes use **Wi-Fi** instead (no Ethernet), replace `enP7s7` with your
> Wi-Fi interface (e.g. `wlP9s9` — confirm the name with `ip -o link show`) in
> Step 5, and use each node's **Wi-Fi IP** as its management IP. All nodes must
> use the same interface — either `enP7s7` on every node or `wlP9s9` on every
> node, not a mix.

## Step 5. Run NCCL communication test

> [!NOTE] 
> Full bandwidth can be achieved with just one QSFP cable.
> When two QSFP cables are connected, all four interfaces must be assigned IP addresses to obtain full bandwidth.

Run these commands on **Node 1** (the launcher); `mpirun` launches the test across all nodes over SSH. Replace the IP addresses and interface names with the ones you found in the previous step.

```bash
## Set network interface environment variables (use your management network interface)
export UCX_NET_DEVICES=enP7s7
export NCCL_SOCKET_IFNAME=enP7s7
export OMPI_MCA_btl_tcp_if_include=enP7s7

## Ring-specific NCCL settings
export NCCL_IB_SUBNET_AWARE_ROUTING=1
export NCCL_NET_PLUGIN=none

## Run the all_gather performance test across all three nodes (replace the management IP addresses with the ones you found from the previous step)
mpirun -np 3 -H <management IP for Node 1>:1,<management IP for Node 2>:1,<management IP for Node 3>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf
```

You can also test your NCCL setup with a larger buffer size to use more of your 200Gbps bandwidth.

```bash
## Set network interface environment variables (use your management network interface)
export UCX_NET_DEVICES=enP7s7
export NCCL_SOCKET_IFNAME=enP7s7
export OMPI_MCA_btl_tcp_if_include=enP7s7

## Ring-specific NCCL settings
export NCCL_IB_SUBNET_AWARE_ROUTING=1
export NCCL_NET_PLUGIN=none

## Run the all_gather performance test across all three nodes
mpirun -np 3 -H <management IP for Node 1>:1,<management IP for Node 2>:1,<management IP for Node 3>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

> [!NOTE]
> The IP addresses in the `mpirun` command are followed by `:1`. For example, `mpirun -np 3 -H 192.168.0.10:1,192.168.0.20:1,192.168.0.30:1`

## Step 6. Cleanup and rollback

```bash
## Rollback network configuration (if needed)
rm -rf ~/nccl/
rm -rf ~/nccl-tests/
```

## Step 7. Next steps
Your NCCL environment is ready for multi-node distributed training workloads on DGX Spark.
Now you can try running a larger distributed workload such as TRT-LLM or vLLM inference.

## Run on four Sparks

## Quick start (scripts)

If you just want a working setup fast, use the helper scripts from GitHub. They
automate Steps 2-5 below. Complete Step 1 (network setup) first, then run
everything from **Node 1** (the launcher), passing each node's **management IP**
(the address you SSH to):

```bash
## 1. Download the helper scripts.
curl -fsSL https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/nccl/assets/setup.sh -o setup.sh
curl -fsSL https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/nccl/assets/launch.sh -o launch.sh

## 2. Build NCCL v2.30.7-1 and the test suite on all four nodes.
bash setup.sh <NODE_2_IP> <NODE_3_IP> <NODE_4_IP>

## 3. Run the all_gather test across all four nodes.
##    Assumes the Ethernet interface enP7s7. On Wi-Fi, prefix the command with
##    MGMT_IFNAME=wlP9s9 (your Wi-Fi interface) and use Wi-Fi IPs. See Step 4.
bash launch.sh --topology switch <NODE_1_IP> <NODE_2_IP> <NODE_3_IP> <NODE_4_IP>
```

To understand what the scripts do — or to debug — follow the manual steps below.

---

## Step 1. Configure network connectivity

Follow the network setup instructions from the [Connect Multiple Sparks through a Switch](https://build.nvidia.com/spark/multi-sparks-through-switch/multi-sparks) playbook to establish connectivity between your DGX Spark nodes.

This includes:
- Physical QSFP cable connection
- Network interface configuration (automatic or manual IP assignment)
- Passwordless SSH setup
- Network connectivity verification

## Step 2. Build NCCL with Blackwell support

Execute these commands on all four nodes to build NCCL from source with Blackwell
architecture support:

```bash
## Install dependencies and build NCCL
sudo apt-get update && sudo apt-get install -y libopenmpi-dev
git clone -b v2.30.7-1 https://github.com/NVIDIA/nccl.git ~/nccl/
cd ~/nccl/
make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"

## Set environment variables
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
```

## Step 3. Build NCCL test suite

Compile the NCCL test suite on **all four nodes**:

```bash
## Clone and build NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests/
cd ~/nccl-tests/
make MPI=1
```

## Step 4. Confirm the CX-7 ports and note each node's management IP

```bash
## Check network port status
ibdev2netdev
```

Example output:
```text
rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Down)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Down)
```

For the test command you need each node's **management IP** (the regular
Ethernet address you SSH to). Find it on each node with:

```bash
ip addr show enP7s7
```

Take note of the management IP for **all four nodes**.

> [!NOTE]
> These steps assume the wired **Ethernet** management interface (`enP7s7`). If
> your nodes use **Wi-Fi** instead (no Ethernet), replace `enP7s7` with your
> Wi-Fi interface (e.g. `wlP9s9` — confirm the name with `ip -o link show`) in
> Step 5, and use each node's **Wi-Fi IP** as its management IP. All nodes must
> use the same interface — either `enP7s7` on every node or `wlP9s9` on every
> node, not a mix.

## Step 5. Run NCCL communication test

> [!NOTE] 
> Full bandwidth can be achieved with just one QSFP cable.
> When two QSFP cables are connected, all four interfaces must be assigned IP addresses to obtain full bandwidth.

Run these commands on **Node 1** (the launcher); `mpirun` launches the test across all nodes over SSH. Replace the IP addresses and interface names with the ones you found in the previous step.

```bash
## Set network interface environment variables (use your management network interface)
export UCX_NET_DEVICES=enP7s7
export NCCL_SOCKET_IFNAME=enP7s7
export OMPI_MCA_btl_tcp_if_include=enP7s7

## Run the all_gather performance test across all four nodes (replace the management IP addresses with the ones you found from the previous step)
mpirun -np 4 -H <management IP for Node 1>:1,<management IP for Node 2>:1,<management IP for Node 3>:1,<management IP for Node 4>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf
```

You can also test your NCCL setup with a larger buffer size to use more of your 200Gbps bandwidth.

```bash
## Set network interface environment variables (use your management network interface)
export UCX_NET_DEVICES=enP7s7
export NCCL_SOCKET_IFNAME=enP7s7
export OMPI_MCA_btl_tcp_if_include=enP7s7

## Run the all_gather performance test across all four nodes
mpirun -np 4 -H <management IP for Node 1>:1,<management IP for Node 2>:1,<management IP for Node 3>:1,<management IP for Node 4>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

> [!NOTE]
> The IP addresses in the `mpirun` command are followed by `:1`. For example, `mpirun -np 4 -H 192.168.0.10:1,192.168.0.20:1,192.168.0.30:1,192.168.0.40:1`

## Step 6. Cleanup and rollback

```bash
## Rollback network configuration (if needed)
rm -rf ~/nccl/
rm -rf ~/nccl-tests/
```

## Step 7. Next steps
Your NCCL environment is ready for multi-node distributed training workloads on DGX Spark.
Now you can try running a larger distributed workload such as TRT-LLM or vLLM inference.

## Troubleshooting

## Common issues for running on multiple Sparks

| Issue | Cause | Solution |
|-------|-------|----------|
| mpirun hangs or times out | SSH connectivity issues | 1. Test basic SSH connectivity: `ssh <remote_ip>` should work without password prompts<br>2. Try a simple mpirun test: `mpirun -np 2 -H <IP for Node 1>:1,<IP for Node 2>:1 hostname`<br>3. Verify SSH keys are setup correctly for all nodes |
| Network interface not found | Wrong interface name or down status | Check interface status with `ibdev2netdev` and verify IP configuration |
| NCCL build fails | Missing dependencies such as OpenMPI or incorrect CUDA version | Verify CUDA installation and required libraries are present |
