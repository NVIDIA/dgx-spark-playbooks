# Heterogeneous Distributed Inference over RDMA

> Set up high-speed RDMA networking between DGX Spark (ConnectX-7) and a Linux Workstation (ConnectX-5) for distributed AI inference

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)
- [Credits](#credits)

---

## Overview

## Basic idea

This playbook guides you through setting up a heterogeneous distributed computing environment using RDMA (Remote Direct Memory Access) over Converged Ethernet (RoCE v2). You will connect a DGX Spark system with a Linux workstation equipped with a Mellanox ConnectX network adapter, enabling high-speed GPU-to-GPU communication for distributed AI workloads.

With RDMA enabled, data flows directly between GPU memories:

```
GPU memory → PCIe → NIC (mlx5) → wire → NIC → PCIe → GPU memory
```

**Key properties:**
- **No CPU copies:** Data bypasses system memory
- **No kernel networking stack:** Direct hardware-to-hardware communication
- **Ultra-low latency:** ~750 nanoseconds end-to-end
- **High message rate:** Up to 200M messages/second

## What you'll accomplish

- Enable low-latency, zero-copy GPU↔GPU communication between heterogeneous systems
- Configure RoCE v2 networking over 100 Gbps direct QSFP connection
- Validate RDMA performance (93+ Gbps achievable)
- Prepare both systems for multi-node inference and training with NCCL

## What to know before starting

- Basic understanding of Linux networking and command line
- Familiarity with network interface configuration (netplan)
- Understanding of PCIe and GPU computing concepts
- Basic knowledge of RDMA/InfiniBand terminology is helpful but not required

## Prerequisites

**Node A: DGX Spark**
- GPU: 128 GB unified memory (Grace Blackwell GB10)
- NIC: ConnectX-7 (QSFP56/QSFP112)
- OS: NVIDIA DGX OS (Ubuntu-based, ARM64)

**Node B: Linux Workstation**
- GPU: NVIDIA GPU with sufficient VRAM (e.g., RTX 6000 Pro, RTX 5090)
- NIC: ConnectX-5 or newer (e.g., MCX516A-CDAT for 100 GbE dual-port)
- OS: Ubuntu 20.04 / 22.04 / 24.04
- PCIe: Gen4 x16 slot recommended

**Physical Requirements:**
- One QSFP cable (QSFP56 ↔ QSFP28 compatible, 100 Gbps negotiated)
- Direct connection or dedicated switch

> [!NOTE]
> Interface names (e.g., `enp1s0f0np0`, `rocep1s0f0`) are system-specific and will differ on your hardware. Use these commands to identify your interfaces:
> ```bash
> ## Find RDMA device to network interface mapping
> ibdev2netdev
>
> ## List all network interfaces
> ip link show
>
> ## Show detailed RDMA device info
> ibv_devinfo
> ```

## Ancillary files

All required files for this playbook can be found [here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/heterogeneous-distributed-inference-rdma/)

- [**test_nccl.py**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/heterogeneous-distributed-inference-rdma/assets/test_nccl.py) - NCCL communication test script

## Time & risk

- **Duration:** 2-3 hours including validation and testing

- **Risk level:** Medium - involves network reconfiguration

- **Rollback:** Network changes can be reversed by removing netplan configs or IP assignments

- **Last Updated:** 01/23/2026

---

## Instructions

## Step 1. Understand the Architecture

Your distributed inference system uses **two separate communication planes**:

| Component | Purpose | Protocol | Latency |
|-----------|---------|----------|---------|
| **Control Plane (Ray)** | Orchestration, scheduling, actor management | TCP/IP (gRPC) | Milliseconds |
| **Data Plane (NCCL)** | High-speed GPU tensor transfers | RoCE v2 (RDMA) | Microseconds |

Both planes use the same 100 Gbps ConnectX network in this configuration.

**RoCE vs InfiniBand:**

| Mode | What it is | Notes |
|------|------------|-------|
| **RoCE v2 (Ethernet)** | RDMA over Ethernet | Recommended for this setup |
| **InfiniBand** | Native IB fabric | Requires IB switches |

> [!NOTE]
> If your ConnectX-5 is Ethernet-only (not VPI), RoCE v2 is the correct and only supported mode.

**Core software components (required on both nodes):**

| Component | Purpose | Notes |
|-----------|---------|--------|
| `mlx5_core` | Main NIC driver | Kernel module |
| `mlx5_ib` | RDMA support | Kernel module |
| `rdma-core` | Userspace RDMA stack | Package: rdma-core |
| `infiniband-diags` | Diagnostics (`ibstat`) | Package: infiniband-diags |
| `mstflint` | Firmware inspection | Package: mstflint |
| `NCCL` | Multi-GPU collectives | Built into PyTorch/frameworks |
| `GPUDirect RDMA` | GPU↔NIC zero-copy | Requires nvidia-peermem |

---

## Step 2. Set Up the Workstation (ConnectX-5)

**Hardware & BIOS checklist:**

1. Install the ConnectX card in a PCIe Gen3/4 x16 slot (CPU-direct, not via chipset)

2. **Cooling Requirements:** ConnectX-5 100GbE cards generate significant heat under load. Ensure adequate case airflow and monitor temperatures with `sensors | grep mlx`

3. **BIOS settings:**
   ```
   Above 4G Decoding: Enabled
   ASPM (Power Management): Disabled
   PCIe Speed: Auto / Gen4
   SR-IOV: Enabled (optional, for virtualization)
   ```

Verify PCIe detection:

```bash
## Check if ConnectX card is detected
lspci -nn | grep -i mellanox
```

Expected output:
```
03:00.0 Ethernet controller [0200]: Mellanox MT27800 [ConnectX-5] [15b3:1017]
03:00.1 Ethernet controller [0200]: Mellanox MT27800 [ConnectX-5] [15b3:1017]
```

## Step 3. Install Drivers on Workstation

Check if mlx5 drivers are already installed:

```bash
## Check for existing Mellanox drivers
lsmod | grep mlx5
```

**Option 1: Ubuntu Inbox Drivers (Recommended)**

```bash
## Update package list
sudo apt update

## Install kernel modules
sudo apt install linux-modules-extra-$(uname -r)

## Load drivers
sudo modprobe mlx5_core mlx5_ib
```

**Option 2: NVIDIA MLNX_OFED (If inbox drivers insufficient)**

```bash
## Download from: https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
wget https://content.mellanox.com/ofed/MLNX_OFED-24.01-0.3.3.1/MLNX_OFED_LINUX-24.01-0.3.3.1-ubuntu24.04-x86_64.tgz

## Extract and install
tar -xzf MLNX_OFED_LINUX-*.tgz
cd MLNX_OFED_LINUX-*
sudo ./mlnxofedinstall --upstream-libs --dpdk
sudo /etc/init.d/openibd restart
```

## Step 4. Install Required Packages on Workstation

```bash
## Update package list
sudo apt update

## Install RDMA and networking packages
sudo apt install -y \
  rdma-core \
  ibverbs-utils \
  rdmacm-utils \
  libibmad5 \
  infiniband-diags \
  perftest \
  mstflint \
  ethtool \
  ibutils
```

## Step 5. Verify Workstation RDMA Stack

Verify kernel drivers are loaded:

```bash
## Check loaded drivers
lsmod | grep mlx5
```

You must see `mlx5_core` and `mlx5_ib`. If missing, load them:

```bash
## Load drivers manually
sudo modprobe mlx5_core mlx5_ib

## Make permanent
echo 'mlx5_core' | sudo tee -a /etc/modules
echo 'mlx5_ib' | sudo tee -a /etc/modules
```

Validate RDMA stack:

```bash
## Show RDMA device info
ibv_devinfo
```

Expected output:
```
hca_id: mlx5_0
    transport:                  InfiniBand (0)
    fw_ver:                     16.35.2000
    node_guid:                  xxxx:xxxx:xxxx:xxxx
    vendor_id:                  0x02c9
    vendor_part_id:             4119
    phys_port_cnt:              1
```

```bash
## Show adapter status
ibstat
```

Validate PCIe bandwidth (replace `03:00.0` with your actual bus address):

```bash
## Check PCIe link speed and width
sudo lspci -s 03:00.0 -vv | grep -E "LnkCap|LnkSta"
```

Target output:
```
LnkCap: Port #0, Speed 16GT/s, Width x16
LnkSta: Speed 16GT/s (ok), Width x16 (ok)
```

---

## Step 6. Set Up DGX Spark (ConnectX-7)

**Fix repository signature issues (if needed):**

If you encounter GPG key errors:

```bash
## Remove problematic repository
sudo rm -f /etc/apt/sources.list.d/*ffmpeg* 2>/dev/null || true

## Download and install updated GPG key
curl -fsSL https://workbench.download.nvidia.com/stable/linux/gpgkey | \
gpg --dearmor | sudo tee /usr/share/keyrings/ai-workbench-desktop-key.gpg > /dev/null

## Update package list
sudo apt update
```

## Step 7. Install Required Packages on DGX Spark

```bash
## Update package list
sudo apt update

## Install RDMA packages
sudo apt install -y \
  infiniband-diags \
  rdma-core \
  ibverbs-utils \
  mstflint \
  perftest \
  ethtool
```

> [!NOTE]
> DOCA-OFED is **not required** for DGX Spark systems. The standard Ubuntu packages provide all necessary functionality.

## Step 8. Verify DGX Spark Interfaces

Verify network interfaces:

```bash
## Show network interfaces
ip link show | grep -E "enp|ib"
```

You should see ConnectX-7 ports like `enp1s0f0np0`, `enp1s0f1np1`, etc.

Verify RDMA interfaces:

```bash
## Show RDMA device to interface mapping
ibdev2netdev
```

Example output:
```
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Down)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Down)
```

Check PCIe topology:

```bash
## Show GPU and NIC topology
nvidia-smi topo -m
```

This shows how GPUs and NICs are interconnected via PCIe.

---

## Step 9. Connect the QSFP Cable

**Hot-plug vs Cold-plug:**
- Hot-plugging QSFP cables is safe with ConnectX-5/7 hardware
- Cold-plug recommended for first-time setup

**Connection procedure:**
1. Identify ports: DGX Spark has 2 physical QSFP ports with 4 logical interfaces
2. Connect QSFP cable between any available ports
3. Cable compatibility: QSFP56 ↔ QSFP28 works (100 Gbps negotiated)
4. Link detection: Should be automatic within 10-20 seconds

Verify physical link detection on DGX Spark:

```bash
## Check link status
ibdev2netdev
```

Expected output (after cable connection):
```
rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Down)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Down)
```

> [!NOTE]
> If none of the interfaces are showing as 'Up', please check the QSFP cable connection, reboot the systems and try again.

Verify on Workstation:

```bash
## Check link status
ibdev2netdev
ip link show | grep -E "enp|mlx"
```

---

## Step 10. Configure Network Interfaces

**Network Configuration:**
- **RDMA Network:** 192.168.200.0/24
- **DGX Spark:** 192.168.200.1
- **Workstation:** 192.168.200.2
- **MTU:** 9000 (jumbo frames for optimal RDMA performance)

> [!NOTE]
> The management IP addresses shown in examples (192.168.1.x) are placeholders. Replace these with your actual network IP addresses that you see when running `ip addr show`.

**Option 1: Temporary Configuration (Testing)**

> [!NOTE]
> These commands are temporary and will be lost on reboot!

On DGX Spark:
```bash
## Configure RDMA interface (use interface showing "Up" from ibdev2netdev)
sudo ip addr add 192.168.200.1/24 dev enp1s0f0np0
sudo ip link set enp1s0f0np0 up
sudo ip link set enp1s0f0np0 mtu 9000
```

On Workstation:
```bash
## Configure RDMA interface
sudo ip addr add 192.168.200.2/24 dev enp1s0f0np0
sudo ip link set enp1s0f0np0 up
sudo ip link set enp1s0f0np0 mtu 9000
```

**Option 2: Permanent Configuration (Production)**

First, identify your active internet interface on both systems:

```bash
## Find your internet interface
ip addr show | grep -A 2 "inet.*scope global"
ip link show | grep "state UP"
```

On DGX Spark:
```bash
## Create netplan configuration (REPLACE interface names with YOUR actual interfaces!)
sudo tee /etc/netplan/99-rdma.yaml > /dev/null <<EOF
network:
  version: 2
  renderer: networkd
  ethernets:
    enp1s0f0np0:
      addresses:
        - 192.168.200.1/24
      mtu: 9000
      dhcp4: false
    enP7s7:         # Replace with YOUR actual internet interface!
      dhcp4: true
  wifis:
    wlP9s9:         # WiFi - optional backup
      dhcp4: true
      access-points:
        "<your-wifi-ssid>":
          password: "<your-wifi-password>"
EOF

## Set permissions and apply
sudo chmod 600 /etc/netplan/99-rdma.yaml
sudo netplan apply
```

On Workstation:
```bash
## Create netplan configuration (REPLACE interface names with YOUR actual interfaces!)
sudo tee /etc/netplan/99-rdma.yaml > /dev/null <<EOF
network:
  version: 2
  renderer: networkd
  ethernets:
    enp1s0f0np0:
      addresses:
        - 192.168.200.2/24
      mtu: 9000
      dhcp4: false
    eno2np1:        # Replace with YOUR actual internet interface!
      dhcp4: true
EOF

## Set permissions and apply
sudo chmod 600 /etc/netplan/99-rdma.yaml
sudo netplan apply
```

> [!IMPORTANT]
> Before applying netplan, identify your active internet interface to avoid losing connectivity. Interface names may change after applying netplan (e.g., `mlx5_0` to `rocep1s0f0`). Always verify current device names with `ibdev2netdev`.

## Step 11. Verify Network Connectivity

Test basic connectivity:

```bash
## From DGX Spark
ping -c 4 192.168.200.2

## From Workstation
ping -c 4 192.168.200.1
```

Expected output:
```
PING 192.168.200.2 (192.168.200.2) 56(84) bytes of data.
64 bytes from 192.168.200.2: icmp_seq=1 time=0.xxx ms
...
4 packets transmitted, 4 received, 0% packet loss
```

---

## Step 12. Test RDMA Bandwidth

Identify correct device names:

```bash
## Check available RDMA devices
ibv_devinfo
ls /sys/class/infiniband/
```

**Device name mapping:**
- **DGX Spark:** Use `rocep1s0f0` or `roceP2p1s0f0`
- **Workstation:** Use `mlx5_0` or `mlx5_1` (or `rocep1s0f0` after persistent config)

Run bandwidth test:

On DGX Spark (server) - Start first:
```bash
## Start RDMA bandwidth test server
ib_send_bw -d rocep1s0f0
```

On Workstation (client) - Connect to server:
```bash
## Connect to server and run bandwidth test
ib_send_bw -d rocep1s0f0 192.168.200.1
```

Example successful output:
```
---------------------------------------------------------------------------------------
                    Send BW Test
 Dual-port       : OFF        Device         : rocep1s0f0
 Number of qps   : 1          Transport type : IB
 Connection type : RC         Using SRQ      : OFF
 Link type       : Ethernet
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[MB/sec]    BW average[MB/sec]   MsgRate[Mpps]
 65536      1000             11664.71            11664.25           0.186628
---------------------------------------------------------------------------------------
```

**Performance Analysis:**
- 11,664 MB/sec = ~93.3 Gbps
- Achieves >93% of 100 Gbps line rate - Excellent!
- Link type: Ethernet confirms RoCE v2 is working

**Performance expectations:**
- **>90 Gbps:** Excellent - Ready for production AI workloads
- **80-90 Gbps:** Good - Sufficient for most multi-node training
- **<80 Gbps:** Check MTU (should be 9000), cable quality, or PCIe slot

---

## Step 13. Configure Environment Variables for NCCL

Add to both systems (persistent across reboots):

```bash
## Add RDMA configuration to bashrc
echo '# RDMA Network Configuration' >> ~/.bashrc
echo 'export UCX_NET_DEVICES=enp1s0f0np0' >> ~/.bashrc
echo 'export NCCL_SOCKET_IFNAME=enp1s0f0np0' >> ~/.bashrc
echo 'export OMPI_MCA_btl_tcp_if_include=enp1s0f0np0' >> ~/.bashrc

## Apply to current session
source ~/.bashrc
```

Verification:
```bash
## Check environment variables
echo $UCX_NET_DEVICES
echo $NCCL_SOCKET_IFNAME
## Both should show: enp1s0f0np0
```

---

## Step 14. (Optional) Configure GPUDirect RDMA

**When needed:**
- High-frequency GPU-to-GPU transfers
- Zero-copy GPU memory access
- Maximum performance training workloads

**Configuration:**
```bash
## Install nvidia-peermem module
sudo apt install nvidia-peer-memory-dkms
sudo modprobe nvidia-peermem
```

---

## Step 15. Final Validation

At this point, you should have achieved:

- [ ] Physical link detected - `ibdev2netdev` shows "(Up)" status
- [ ] IP connectivity working - `ping 192.168.200.x` succeeds
- [ ] MTU set to 9000 - Jumbo frames enabled
- [ ] RDMA bandwidth >90 Gbps validated
- [ ] RoCE v2 confirmed - Link type: Ethernet
- [ ] Environment variables set for NCCL

Your RDMA setup is **fully operational** and ready for distributed AI workloads!

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ibdev2netdev` shows no devices | mlx5 drivers not loaded | `sudo modprobe mlx5_core mlx5_ib` |
| Interface shows "(Down)" after cable | Link not negotiated | Check cable, try different port, reboot |
| Ping fails between nodes | IP not configured or wrong interface | Verify `ip addr show`, check interface names |
| RDMA bandwidth <80 Gbps | MTU not set to 9000 | `sudo ip link set <interface> mtu 9000` |
| "mlx5_0 not found" error | Device name changed after netplan | Run `ibdev2netdev` to find current name |
| Permission denied on `/dev/infiniband` | Missing RDMA permissions | Run with `sudo` or add user to `rdma` group |
| GPG key errors on DGX Spark | Expired NVIDIA repository key | See Step 6 for fix |
| Lost internet after netplan apply | Wrong interface in netplan config | Identify correct interface with `ip link show` first |

---

## Next Steps

Continue to [**Distributed Inference Guide**](DISTRIBUTED-INFERENCE.md) to:
- Set up SSH and hostname configuration
- Configure NCCL for multi-node communication
- Deploy RDMA-enabled containers with Ray cluster
- Run distributed inference with vLLM
- Benchmark performance across configurations

---

## Credits

This playbook was contributed by **Csaba Kecskemeti** | [DevQuasar](https://devquasar.com/).

For a detailed walkthrough and additional context, see the original article:
[Distributed Inference Cluster: DGX Spark + RTX 6000 Pro](https://devquasar.com/ai/edge-ai/distributed-inference-cluster-dgx-spark-rtx-6000-pro/)

![DevQuasar](assets/devquasar-logo.png)
