# Set Up NCCL for Multi-Node GPU Communication

> Benchmarked interconnect bandwidth for distributed training across nodes

## Table of Contents

- [Overview](#overview)
- [Two nodes (direct)](#two-nodes-direct)
- [Three nodes (ring)](#three-nodes-ring)
- [Four nodes (switch)](#four-nodes-switch)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NCCL (NVIDIA Collective Communication Library) enables high-performance GPU-to-GPU communication across multiple nodes. This walkthrough sets up NCCL for multi-node distributed training on two, three, or four nodes of multi-node capable hardware. You'll configure networking, build NCCL from source with Blackwell support, and validate communication between nodes.

## What you'll accomplish

You'll have a working multi-node NCCL environment that enables high-bandwidth GPU communication across nodes for distributed training workloads, with validated network performance and proper GPU topology detection.

## What to know before starting

**Required:**

- Working with Linux network configuration and netplan
- Basic understanding of MPI (Message Passing Interface) concepts
- SSH key management and passwordless authentication setup

**Optional:**

- Familiarity with RoCE / high-speed interconnect troubleshooting on multi-node capable hardware

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended defaults, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Build NCCL from source (`v2.30.7-1`, Blackwell `sm_121`) | ✅ (high-speed interconnect) |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Two, three, or four nodes of multi-node capable hardware
- Multi-node networking and inter-device SSH configured by one of these paths:
  - **Recommended for supported two- to four-node DGX Spark clusters:** Complete the [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html). If Cluster Assistant reports success, the connection prerequisite is complete; do not repeat a manual connection playbook. Continue with the tab for your node count.
  - **Manual setup or troubleshooting:** Complete the [Connect multiple nodes for distributed workloads](https://build.nvidia.com/playbooks/connect-multiple-sparks) playbook.

**Software requirements**

- The same username on every node. When using Cluster Assistant, choose to standardize user information when prompted; the NCCL helper scripts assume matching usernames.
- NVIDIA driver installed: `nvidia-smi`
- CUDA toolkit available: `nvcc --version`
- Root/sudo privileges: `sudo whoami`

## Ancillary files

All required assets can be found [in this playbook's assets folder](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-nccl/assets).

- `setup.sh` — builds NCCL and the NCCL test suite on every node
- `launch.sh` — runs the all_gather performance test for direct, ring, or switch topologies

## Time & risk

- **Estimated time:** 30 MIN for setup and validation
- **Risk level:** Medium
  - Involves network configuration changes on multi-node capable hardware
- **Rollback:** Remove the NCCL and NCCL Tests repositories from each node (`~/nccl/`, `~/nccl-tests/`)
- **Last Updated:** 08/12/2026
  - Added NVIDIA Sync Cluster Assistant as the recommended network setup path and clarified that successful Cluster Assistant users should skip the manual connection playbooks

## Two nodes (direct)

## Step 1. Confirm network connectivity

Use [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) to connect your nodes. It is the recommended path: it handles the cabling checks, interface configuration, and passwordless SSH for you.

> [!TIP]
> If Cluster Assistant successfully created your cluster, this step is already complete. Continue below.

If you choose not to use Cluster Assistant, follow the network setup instructions from the [Connect multiple nodes for distributed workloads](https://build.nvidia.com/playbooks/connect-multiple-sparks) playbook instead.

The manual connection path includes:

- Physical QSFP cable connection
- Network interface configuration (automatic or manual IP assignment)
- Passwordless SSH setup
- Network connectivity verification

## Quick start (scripts)

If you just want a working setup fast, use the helper scripts. They automate Steps 2–5 below. Complete Step 1 first, then run everything from **Node 1** (the launcher), passing each node's **management IP** (the address you SSH to):

```bash
## 1. Download the helper scripts.
curl -fsSL "https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-nccl/assets/setup.sh" -o setup.sh
curl -fsSL "https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-nccl/assets/launch.sh" -o launch.sh

## 2. Build NCCL v2.30.7-1 and the test suite on both nodes.
bash setup.sh <NODE_2_IP>

## 3. Run the all_gather test across both nodes.
##    Assumes the Ethernet interface enP7s7. On Wi-Fi, prefix the command with
##    MGMT_IFNAME=wlP9s9 (your Wi-Fi interface) and use Wi-Fi IPs. See Step 4.
bash launch.sh --topology direct <NODE_1_IP> <NODE_2_IP>
```

To understand what the scripts do — or to debug — follow the manual steps below.

---

## Step 2. Build NCCL with Blackwell support

Execute these commands on both nodes to build NCCL from source with Blackwell architecture support:

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

## Step 4. Confirm interconnect ports and note each node's management IP

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

For the test command you need each node's **management IP** (the regular Ethernet address you SSH to). Find it on each node with:

```bash
ip addr show enP7s7
```

Take note of the management IP for **both nodes**.

> [!NOTE]
> These steps assume the wired **Ethernet** management interface (`enP7s7`) validated on multi-node capable hardware. If your nodes use **Wi-Fi** instead (no Ethernet), replace `enP7s7` with your Wi-Fi interface (e.g. `wlP9s9` — confirm the name with `ip -o link show`) in Step 5, and use each node's **Wi-Fi IP** as its management IP. All nodes must use the same interface — either `enP7s7` on every node or `wlP9s9` on every node, not a mix.

## Step 5. Run NCCL communication test

> [!NOTE]
> Full bandwidth can be achieved with just one QSFP cable.
> When two QSFP cables are connected, all four interfaces must be assigned IP addresses to obtain full bandwidth.

Run these commands on **Node 1** (the launcher); `mpirun` launches the test across all nodes over SSH. Replace the IP addresses and interface names with the ones you found in the previous step.

```bash
## Run the all_gather performance test across both nodes (replace the management IP addresses with the ones you found from the previous step)
mpirun -np 2 -H <management IP for Node 1>:1,<management IP for Node 2>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x UCX_NET_DEVICES=enP7s7 \
  -x NCCL_SOCKET_IFNAME=enP7s7 \
  -x OMPI_MCA_btl_tcp_if_include=enP7s7 \
  $HOME/nccl-tests/build/all_gather_perf
```

You can also test your NCCL setup with a larger buffer size to use more of your interconnect bandwidth.

```bash
## Run the all_gather performance test across both nodes
mpirun -np 2 -H <management IP for Node 1>:1,<management IP for Node 2>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x UCX_NET_DEVICES=enP7s7 \
  -x NCCL_SOCKET_IFNAME=enP7s7 \
  -x OMPI_MCA_btl_tcp_if_include=enP7s7 \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

> [!NOTE]
> The IP addresses in the `mpirun` command are followed by `:1`. For example, `mpirun -np 2 -H 192.168.0.10:1,192.168.0.20:1`

## Step 6. Cleanup and rollback

```bash
## Remove NCCL build artifacts (if needed)
rm -rf ~/nccl/
rm -rf ~/nccl-tests/
```

## Step 7. Next steps

Your NCCL environment is ready for multi-node distributed training workloads on multi-node capable hardware. Next, try a larger distributed workload such as TensorRT-LLM or vLLM inference.

## Three nodes (ring)

## Step 1. Confirm network connectivity

Use [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) to connect your nodes. It is the recommended path: it handles the cabling checks, interface configuration, and passwordless SSH for you.

> [!TIP]
> If Cluster Assistant successfully created your cluster, this step is already complete. Continue below.

If you choose not to use Cluster Assistant, follow the network setup instructions from the [Connect multiple nodes for distributed workloads](https://build.nvidia.com/playbooks/connect-multiple-sparks) playbook instead.

The manual connection path includes:

- Physical QSFP cable connection
- Network interface configuration (automatic or manual IP assignment)
- Passwordless SSH setup
- Network connectivity verification

## Quick start (scripts)

If you just want a working setup fast, use the helper scripts. They automate Steps 2–5 below. Complete Step 1 first, then run everything from **Node 1** (the launcher), passing each node's **management IP** (the address you SSH to):

```bash
## 1. Download the helper scripts.
curl -fsSL "https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-nccl/assets/setup.sh" -o setup.sh
curl -fsSL "https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-nccl/assets/launch.sh" -o launch.sh

## 2. Build NCCL v2.30.7-1 and the test suite on all three nodes.
bash setup.sh <NODE_2_IP> <NODE_3_IP>

## 3. Run the all_gather test across all three nodes.
##    Assumes the Ethernet interface enP7s7. On Wi-Fi, prefix the command with
##    MGMT_IFNAME=wlP9s9 (your Wi-Fi interface) and use Wi-Fi IPs. See Step 4.
bash launch.sh --topology ring <NODE_1_IP> <NODE_2_IP> <NODE_3_IP>
```

To understand what the scripts do — or to debug — follow the manual steps below.

---

## Step 2. Build NCCL with Blackwell support

Execute these commands on all three nodes to build NCCL from source with Blackwell architecture support:

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

## Step 4. Confirm interconnect ports and note each node's management IP

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

For the test command you need each node's **management IP** (the regular Ethernet address you SSH to). Find it on each node with:

```bash
ip addr show enP7s7
```

Take note of the management IP for **all three nodes**.

> [!NOTE]
> These steps assume the wired **Ethernet** management interface (`enP7s7`) validated on multi-node capable hardware. If your nodes use **Wi-Fi** instead (no Ethernet), replace `enP7s7` with your Wi-Fi interface (e.g. `wlP9s9` — confirm the name with `ip -o link show`) in Step 5, and use each node's **Wi-Fi IP** as its management IP. All nodes must use the same interface — either `enP7s7` on every node or `wlP9s9` on every node, not a mix.

## Step 5. Run NCCL communication test

> [!NOTE]
> Full bandwidth can be achieved with just one QSFP cable.
> When two QSFP cables are connected, all four interfaces must be assigned IP addresses to obtain full bandwidth.

Run these commands on **Node 1** (the launcher); `mpirun` launches the test across all nodes over SSH. Replace the IP addresses and interface names with the ones you found in the previous step.

```bash
## Run the all_gather performance test across all three nodes (replace the management IP addresses with the ones you found from the previous step)
mpirun -np 3 -H <management IP for Node 1>:1,<management IP for Node 2>:1,<management IP for Node 3>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x UCX_NET_DEVICES=enP7s7 \
  -x NCCL_SOCKET_IFNAME=enP7s7 \
  -x OMPI_MCA_btl_tcp_if_include=enP7s7 \
  -x NCCL_IB_SUBNET_AWARE_ROUTING=1 \
  -x NCCL_NET_PLUGIN=none \
  $HOME/nccl-tests/build/all_gather_perf
```

You can also test your NCCL setup with a larger buffer size to use more of your interconnect bandwidth.

```bash
## Run the all_gather performance test across all three nodes
mpirun -np 3 -H <management IP for Node 1>:1,<management IP for Node 2>:1,<management IP for Node 3>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x UCX_NET_DEVICES=enP7s7 \
  -x NCCL_SOCKET_IFNAME=enP7s7 \
  -x OMPI_MCA_btl_tcp_if_include=enP7s7 \
  -x NCCL_IB_SUBNET_AWARE_ROUTING=1 \
  -x NCCL_NET_PLUGIN=none \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

> [!NOTE]
> The IP addresses in the `mpirun` command are followed by `:1`. For example, `mpirun -np 3 -H 192.168.0.10:1,192.168.0.20:1,192.168.0.30:1`

## Step 6. Cleanup and rollback

```bash
## Remove NCCL build artifacts (if needed)
rm -rf ~/nccl/
rm -rf ~/nccl-tests/
```

## Step 7. Next steps

Your NCCL environment is ready for multi-node distributed training workloads on multi-node capable hardware. Next, try a larger distributed workload such as TensorRT-LLM or vLLM inference.

## Four nodes (switch)

## Step 1. Confirm network connectivity

Use [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) to connect your nodes. It is the recommended path: it handles the cabling checks, interface configuration, and passwordless SSH for you.

> [!TIP]
> If Cluster Assistant successfully created your cluster, this step is already complete. Continue below.

If you choose not to use Cluster Assistant, follow the network setup instructions from the [Connect multiple nodes for distributed workloads](https://build.nvidia.com/playbooks/connect-multiple-sparks) playbook instead.

The manual connection path includes:

- Physical QSFP cable connection
- Network interface configuration (automatic or manual IP assignment)
- Passwordless SSH setup
- Network connectivity verification

## Quick start (scripts)

If you just want a working setup fast, use the helper scripts. They automate Steps 2–5 below. Complete Step 1 first, then run everything from **Node 1** (the launcher), passing each node's **management IP** (the address you SSH to):

```bash
## 1. Download the helper scripts.
curl -fsSL "https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-nccl/assets/setup.sh" -o setup.sh
curl -fsSL "https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-nccl/assets/launch.sh" -o launch.sh

## 2. Build NCCL v2.30.7-1 and the test suite on all four nodes.
bash setup.sh <NODE_2_IP> <NODE_3_IP> <NODE_4_IP>

## 3. Run the all_gather test across all four nodes.
##    Assumes the Ethernet interface enP7s7. On Wi-Fi, prefix the command with
##    MGMT_IFNAME=wlP9s9 (your Wi-Fi interface) and use Wi-Fi IPs. See Step 4.
bash launch.sh --topology switch <NODE_1_IP> <NODE_2_IP> <NODE_3_IP> <NODE_4_IP>
```

To understand what the scripts do — or to debug — follow the manual steps below.

---

## Step 2. Build NCCL with Blackwell support

Execute these commands on all four nodes to build NCCL from source with Blackwell architecture support:

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

## Step 4. Confirm interconnect ports and note each node's management IP

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

For the test command you need each node's **management IP** (the regular Ethernet address you SSH to). Find it on each node with:

```bash
ip addr show enP7s7
```

Take note of the management IP for **all four nodes**.

> [!NOTE]
> These steps assume the wired **Ethernet** management interface (`enP7s7`) validated on multi-node capable hardware. If your nodes use **Wi-Fi** instead (no Ethernet), replace `enP7s7` with your Wi-Fi interface (e.g. `wlP9s9` — confirm the name with `ip -o link show`) in Step 5, and use each node's **Wi-Fi IP** as its management IP. All nodes must use the same interface — either `enP7s7` on every node or `wlP9s9` on every node, not a mix.

## Step 5. Run NCCL communication test

> [!NOTE]
> Full bandwidth can be achieved with just one QSFP cable.
> When two QSFP cables are connected, all four interfaces must be assigned IP addresses to obtain full bandwidth.

Run these commands on **Node 1** (the launcher); `mpirun` launches the test across all nodes over SSH. Replace the IP addresses and interface names with the ones you found in the previous step.

```bash
## Run the all_gather performance test across all four nodes (replace the management IP addresses with the ones you found from the previous step)
mpirun -np 4 -H <management IP for Node 1>:1,<management IP for Node 2>:1,<management IP for Node 3>:1,<management IP for Node 4>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x UCX_NET_DEVICES=enP7s7 \
  -x NCCL_SOCKET_IFNAME=enP7s7 \
  -x OMPI_MCA_btl_tcp_if_include=enP7s7 \
  $HOME/nccl-tests/build/all_gather_perf
```

You can also test your NCCL setup with a larger buffer size to use more of your interconnect bandwidth.

```bash
## Run the all_gather performance test across all four nodes
mpirun -np 4 -H <management IP for Node 1>:1,<management IP for Node 2>:1,<management IP for Node 3>:1,<management IP for Node 4>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x UCX_NET_DEVICES=enP7s7 \
  -x NCCL_SOCKET_IFNAME=enP7s7 \
  -x OMPI_MCA_btl_tcp_if_include=enP7s7 \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

> [!NOTE]
> The IP addresses in the `mpirun` command are followed by `:1`. For example, `mpirun -np 4 -H 192.168.0.10:1,192.168.0.20:1,192.168.0.30:1,192.168.0.40:1`

## Step 6. Cleanup and rollback

```bash
## Remove NCCL build artifacts (if needed)
rm -rf ~/nccl/
rm -rf ~/nccl-tests/
```

## Step 7. Next steps

Your NCCL environment is ready for multi-node distributed training workloads on multi-node capable hardware. Next, try a larger distributed workload such as TensorRT-LLM or vLLM inference.

## Troubleshooting

## Common issues for multi-node NCCL

| Symptom | Cause | Fix |
|---------|-------|-----|
| `mpirun` hangs or times out | SSH connectivity issues | 1. Test basic SSH connectivity: `ssh <remote_ip>` should work without password prompts<br>2. Try a simple mpirun test: `mpirun -np 2 -H <IP for Node 1>:1,<IP for Node 2>:1 hostname`<br>3. Verify SSH keys are set up correctly for all nodes |
| Network interface not found | Wrong interface name or down status | Check interface status with `ibdev2netdev` and verify IP configuration |
| NCCL build fails | Missing dependencies such as OpenMPI or incorrect CUDA version | Verify CUDA installation and required libraries are present |

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
