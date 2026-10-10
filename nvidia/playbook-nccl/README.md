# Set Up NCCL for Multi-Node GPU Communication

> Benchmarked interconnect bandwidth for distributed training across nodes

## Table of Contents

- [Overview](#overview)
- [Two nodes (direct)](#two-nodes-direct)
- [Three nodes (ring)](#three-nodes-ring)
- [Four nodes (switch)](#four-nodes-switch)
- [Five to eight nodes (switch, existing network)](#five-to-eight-nodes-switch-existing-network)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NCCL (NVIDIA Collective Communication Library) enables high-performance GPU-to-GPU communication across multiple nodes. This walkthrough sets up NCCL for multi-node distributed training on two to eight nodes of multi-node capable hardware. The two-, three-, and four-node sections include the existing setup flow; the five- to eight-node section validates an already configured switch network without changing network configuration.

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
- Two to eight nodes of multi-node capable hardware
- Multi-node networking and inter-device SSH configured by one of these paths:
  - **Recommended for supported two- to four-node DGX Spark clusters:** Complete the [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html). If Cluster Assistant reports success, the connection prerequisite is complete; do not repeat a manual connection playbook. Continue with the tab for your node count.
  - **Manual setup or troubleshooting:** Complete the [Connect multiple nodes for distributed workloads](https://build.nvidia.com/playbooks/connect-multiple-sparks) playbook.
  - **Five to eight nodes:** Configure and verify the switch network and inter-device SSH with your existing deployment process, then use the [five- to eight-node validation](#five-to-eight-nodes-switch-existing-network). NVIDIA Sync Cluster Assistant is limited to two to four devices.

**Software requirements**

- The same username on every node. When using Cluster Assistant, choose to standardize user information when prompted; the NCCL helper scripts assume matching usernames.
- NVIDIA driver installed: `nvidia-smi`
- CUDA toolkit available: `nvcc --version`
- Root/sudo privileges: `sudo whoami`

## Ancillary files

All required assets can be found [in this playbook's assets folder](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-nccl/assets).

- `setup.sh` — builds NCCL and the NCCL test suite on every node
- `launch.sh` — runs bounded `all_gather` or `all_reduce` tests for direct, ring, or switch topologies; switch validation accepts two to eight nodes
- `preflight.sh` — performs a read-only prerequisite check on every remote node before MPI starts
- `tests/test_launch.sh` — CPU-only mocked tests for command construction, validation, and timeout handling
- `tests/test_preflight.sh` — CPU-only mocked tests for preflight aggregation and fail-closed behavior

## Time & risk

- **Estimated time:** 30 MIN for setup and validation
- **Risk level:** Medium
  - Involves network configuration changes on multi-node capable hardware
- **Rollback:** Remove the NCCL and NCCL Tests repositories from each node (`~/nccl/`, `~/nccl-tests/`)
- **Last Updated:** 08/12/2026
  - Added NVIDIA Sync Cluster Assistant as the recommended network setup path and clarified that successful Cluster Assistant users should skip the manual connection playbooks

> [!WARNING]
> The `setup.sh` and `spark_cluster_setup` assets install packages and configure DGX OS networking. Do not run them on NixOS. On NixOS, use the NCCL, CUDA, Open MPI, and `nccl-tests` packages already installed by your system configuration, and use `launch.sh` only after the network and inter-device SSH are already configured.

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

## Five to eight nodes (switch, existing network)

This section is a validation path for five to eight systems connected through an already configured Ethernet switch. It does not provision the network, install an operating system, or claim support for NVIDIA Sync Cluster Assistant. Cluster Assistant currently supports two to four DGX Spark devices; configure larger clusters manually and verify them before running the workload.

Use this path only when all of the following are true:

- Every node is on the same switch network and has one reachable management address.
- Node 1 can use passwordless SSH to every other node, and the same user and software environment are present on every node. Operator-to-node SSH alone is not sufficient for `mpirun`.
- Each node has one GPU, a working CUDA/NCCL installation, Open MPI `mpirun`, and the matching `nccl-tests` binaries.
- You have a reservation or explicit coordination for the complete node set. A collective test occupies one GPU on every node and can affect network and GPU resources.

On NixOS, do not run `setup.sh`, `spark_cluster_setup.sh`, or the Ubuntu/Debian package-install commands in the earlier sections. Use your NixOS-managed packages and system configuration. Set `NCCL_TEST_BIN` to the NixOS-provided `all_gather_perf` or `all_reduce_perf` path, and set `CUDA_HOME`, `MPI_HOME`, and `NCCL_HOME` only when your environment needs explicit library paths. The launcher does not change network configuration.

On NixOS, the host NVIDIA driver and RDMA libraries may be outside the temporary Nix shell that provides Open MPI and `nccl-tests`. Preserve those host paths in `LD_LIBRARY_PATH`; a common layout is `LD_LIBRARY_PATH=/run/opengl-driver/lib:/run/current-system/sw/lib`. Use the paths exposed by the target system rather than copying them into the repository. The launcher prepends the selected CUDA, NCCL, and MPI paths while preserving the existing value.

This is a host-native launcher. `mpirun`, the selected NCCL test binary, CUDA/NCCL libraries, and the matching Open MPI runtime must be available to the launcher and at the same usable paths on every remote node. `NCCL_TEST_BIN` selects an existing executable; it does not install MPI, CUDA, NCCL, or `nccl-tests`, and it does not provide a container runtime. If those prerequisites live only inside a container, use a separately validated container launcher and record that runtime as part of the evidence instead of presenting the result as a host-native run.

For example, the test-binary override can be combined with the existing-network command:

```bash
NCCL_TEST_BIN=/path/to/all_gather_perf \
CUDA_HOME=/path/to/cuda MPI_HOME=/path/to/openmpi NCCL_HOME=/path/to/nccl \
bash launch.sh --topology switch --collective all_gather \
  --interface <management-interface> --timeout 300 \
  <node1> <node2> <node3> <node4> <node5> <node6> <node7> <node8>
```

Run the read-only preflight before reserving GPU work. It checks SSH, GPU visibility, `mpirun`, the selected NCCL test binary, the management interface, RDMA tooling, and any `NCCL_IB_HCA` devices without installing packages or changing node state:

```bash
NCCL_IB_HCA=mlx5_1:1,mlx5_3:1 \
bash preflight.sh --collective both --interface <management-interface> \
  <node1> <node2> <node3> <node4> <node5> <node6> <node7> <node8>
```

If Node 1 uses a non-default key to reach the other nodes, point both the preflight and launcher at that key with `MPI_SSH_IDENTITY_FILE=/path/to/key`. The key must already be authorized on every remote node; neither script copies keys or changes SSH configuration.

The launcher constrains both the NCCL bootstrap socket and the PRRTE/Open MPI control plane to `--interface`. This matters on DGX Spark hosts with multiple management, overlay, and fabric addresses: NCCL data traffic should still select the configured high-speed HCAs through `NCCL_IB_HCA`, while MPI control traffic stays on the reachable management network.

To inspect the exact MPI command without starting a workload, add `--dry-run` to `launch.sh`. Dry-run mode still validates topology, node count, duplicate addresses, interface syntax, and timeout syntax, but does not require local MPI or the test binary to be installed.

```bash
bash launch.sh --dry-run --topology switch --collective all_reduce \
  --interface <management-interface> --timeout 300 \
  <node1> <node2> <node3> <node4> <node5> <node6> <node7> <node8>
```

Before the first run, perform a read-only preflight on every node:

```bash
hostname
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
ip -o link show
ibdev2netdev
ssh <node> true
```

Choose one management interface name that is present and reachable on every node. The launcher uses that interface for MPI/NCCL bootstrap; NCCL should use the high-speed interconnect for collective data. If more than one HCA is present, set `NCCL_IB_HCA` to the exact device/port selection already validated by your deployment.

Start with a small bounded sweep and pass every node exactly once:

```bash
BEGIN=8M END=64M FACTOR=2 \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET \
bash launch.sh --topology switch --collective all_gather \
  --interface <management-interface> --timeout 300 \
  <node1> <node2> <node3> <node4> <node5> <node6> <node7> <node8>
```

Confirm the log shows `Initialized NET plugin IB` and the selected `mlx5_*` devices (or the equivalent site-specific HCAs). A `NET/Socket` selection means the run did not use the intended RDMA transport and should be investigated before treating the result as a fabric validation.

For an all-reduce correctness/performance check, use the same node list and change the collective:

```bash
BEGIN=8M END=64M FACTOR=2 \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET \
bash launch.sh --topology switch --collective all_reduce \
  --interface <management-interface> --timeout 300 \
  <node1> <node2> <node3> <node4> <node5> <node6> <node7> <node8>
```

The same command supports five, six, or seven nodes; the number of node arguments must match the requested rank count. The helper rejects duplicate nodes, unsupported counts, invalid addresses, and non-positive timeouts before launching MPI. It returns exit status 124 when the wall-clock limit is exceeded. After a timeout or failed run, verify that no `mpirun`, `all_gather_perf`, or `all_reduce_perf` processes remain on any node before reusing the reservation.

For a useful scaling check, repeat the same bounded commands with the same software, interface, HCA selection, and message sweep at 2, 4, and 8 nodes. Use disjoint reserved subsets when the full eight-node reservation is not available, and treat each result as a separate datapoint: a passing 2- or 4-node run does not establish eight-node behavior. At each size, run both `all_gather` and `all_reduce`, require every rank to complete with exit status zero, and record whether NCCL selected `IB` rather than falling back to `Socket`.

Keep a private run table with the node count, exact node identities, image or package provenance, kernel/driver/CUDA/NCCL/Open MPI versions, selected HCA and management interface, message sweep, exit status, transport evidence, and cleanup result. Publish only sanitized counts, versions, outcomes, and relevant NCCL log lines. Do not publish passwords, private keys, usernames, management addresses, or unredacted host inventories.

The default sweep is intentionally small. `all_gather` allocates output that grows with world size, so increase `END` only after correctness is established and the reservation owner approves the added memory and network load. For a larger follow-up, set `BEGIN` and `END` explicitly (for example, `256M`) rather than restoring the old unbounded or multi-gigabyte examples by accident.

Treat the output as successful only when the command exits zero, every rank completes, and the collective reports consistent results. With `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET`, inspect the log for `Using network IB`; `Using network Socket` indicates a fallback and is not evidence that the high-speed data path was used. The bootstrap interface name alone does not prove the collective transport. Capture the node count, exact node list in private records, software/kernel versions, selected HCA, message sweep, exit status, and cleanup result. Sanitize hostnames, IP addresses, usernames, and private paths before publishing evidence.

This validation extends the launcher and documentation to eight switch-connected nodes; it does not establish scaling beyond eight nodes, and it does not make NixOS or larger clusters supported by NVIDIA Sync Cluster Assistant.

## Troubleshooting

## Common issues for multi-node NCCL

| Symptom | Cause | Fix |
|---------|-------|-----|
| `mpirun` hangs or times out | SSH connectivity, wrong interface, or a failed remote rank | 1. Test basic SSH connectivity: `ssh <remote_ip>` should work without password prompts<br>2. Verify the chosen interface exists on every node and try a simple `mpirun ... hostname` check<br>3. The helper enforces `--timeout`; after status 124, verify remote cleanup before retrying |
| Network interface not found | Wrong interface name or down status | Check interface status with `ibdev2netdev` and verify IP configuration |
| NCCL reports `Using network Socket` | NCCL did not select the high-speed interconnect | Check HCA/link state and the exact `NCCL_IB_HCA` value; do not interpret management-interface reachability as transport validation |
| Duplicate or incorrect rank count | A node was repeated or the node list does not match the topology | Pass each node once; switch accepts two to eight nodes, direct exactly two, and ring exactly three |
| NCCL build fails | Missing dependencies such as OpenMPI or incorrect CUDA version | Verify CUDA installation and required libraries are present |

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
