# Connect Two DGX Sparks for Distributed Workloads

> Combined memory and compute over a direct high-speed link

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Configure two multi-node capable hardware platforms for high-speed inter-node communication using a direct 200GbE QSFP link. This setup enables distributed workloads by establishing network connectivity and passwordless SSH between nodes.

> [!TIP]
> **Recommended for DGX Spark:** Use [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) to configure and validate the cluster network. If Cluster Assistant reports success, do not repeat the manual network or SSH steps in this playbook; continue to your workload playbook.

## What you'll accomplish

You'll physically connect two hardware platforms with a QSFP cable, configure network interfaces for cluster communication, and establish passwordless SSH between nodes to create a functional distributed computing environment.

## What to know before starting

**Required:**

- Basic understanding of distributed computing concepts
- Experience with network interface configuration and netplan
- Experience with SSH key management

**Optional:**

- Familiarity with InfiniBand/RoCE tooling such as `ibdev2netdev`

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Direct QSFP interconnect + netplan/SSH setup | ✅ (200GbE QSFP direct link) |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Two nodes of multi-node capable hardware
- One QSFP cable for a direct 200GbE connection between the two devices
- The same username on both systems

**Software requirements**

- SSH access available to both systems
- Root or sudo access on both systems: `sudo whoami`
- Network tooling for interface discovery: `ibdev2netdev`
- For automatic SSH discovery (Instructions Step 4 Option 1): `avahi-utils` so `avahi-browse` is available

## Ancillary files

All required assets are in [this playbook's assets folder](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-two-sparks/assets).

- [`discover-sparks`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-two-sparks/assets/discover-sparks) — automatic node discovery and SSH key distribution
- [`performance_benchmarking_guide.md`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-two-sparks/assets/performance_benchmarking_guide.md) — optional performance benchmarking reference

## Time & risk

- **Estimated time:** 60 MIN including validation
- **Risk level:** Medium
  - Involves network reconfiguration on both nodes
  - Incorrect interface or IP assignment can interrupt cluster connectivity until rolled back
- **Rollback:** Network changes can be reversed by removing netplan configs or deleting temporary IP assignments (see Cleanup in **Instructions**)
- **Last Updated:** 08/12/2026
  - Added NVIDIA Sync Cluster Assistant as the recommended setup path and clarified when to skip manual configuration

## Instructions

> [!TIP]
> If [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) successfully configured this DGX Spark cluster, skip this manual setup and continue to your workload playbook.

## Step 1. Ensure the same username on both systems

On both systems, check the username and make sure it matches:

```bash
## Check current username
whoami
```

If usernames don't match, create a shared user (for example, `nvidia`) on both systems and log in with that user:

```bash
## Create nvidia user and add to sudo group
sudo useradd -m nvidia
sudo usermod -aG sudo nvidia

## Set password for nvidia user
sudo passwd nvidia

## Switch to nvidia user
su - nvidia
```

## Step 2. Physical hardware connection

Connect the QSFP cable between both nodes using any QSFP interface on each device. Use the same physical port on each device to avoid issues with later NCCL tests. This establishes the 200GbE direct connection required for high-speed inter-node communication.

After the cable is connected, confirm which interfaces are up. In this example, the interfaces showing as `Up` are **enp1s0f1np1** / **enP2p1s0f1np1** (each physical port has two logical interfaces).

```bash
## Check QSFP interface availability on both nodes
ibdev2netdev
```

Example output:

```bash
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
```

> [!NOTE]
> If none of the interfaces are showing as `Up`, check the QSFP cable connection, reboot the systems, and try again.
> Which interfaces show as `Up` depends on which port you use. Each physical port has two logical interfaces; for example, `enp1s0f1np1` and `enP2p1s0f1np1` refer to the same physical port.

## Step 3. Network interface configuration

Choose one option to set up the network interfaces. Option 1 and Option 2 are mutually exclusive.

> [!NOTE]
> Full bandwidth can be achieved with just one QSFP cable.
> When two QSFP cables are connected, all four interfaces must be assigned IP addresses to obtain full bandwidth.

**Option 1: Manual IP assignment with a netplan configuration file**

On node 1:

```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.10/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.10/24
      dhcp4: no
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

On node 2:

```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.11/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.11/24
      dhcp4: no
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

**Option 2: Manual IP assignment with the command line**

> [!NOTE]
> With this option, the IPs assigned to the interfaces will change if you reboot the system.

Use the interfaces that show as `(Up)`. In this example, we'll use **enp1s0f1np1** and **enP2p1s0f1np1**.

On node 1:

```bash
## Assign static IP and bring up interface.
sudo ip addr add 192.168.100.10/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up

sudo ip addr add 192.168.101.10/24 dev enP2p1s0f1np1
sudo ip link set enP2p1s0f1np1 up
```

Repeat the same process for node 2, using IP **192.168.100.11/24** and **192.168.101.11/24**. Confirm the correct interface names with `ibdev2netdev`.

```bash
## Assign static IP and bring up interface.
sudo ip addr add 192.168.100.11/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up

sudo ip addr add 192.168.101.11/24 dev enP2p1s0f1np1
sudo ip link set enP2p1s0f1np1 up
```

Verify the IP assignment on both nodes:

```bash
## Check the interfaces showing as "(Up)" in your output, e.g. enp1s0f1np1 and enP2p1s0f1np1
ip addr show enp1s0f1np1
ip addr show enP2p1s0f1np1
```

## Step 4. Set up passwordless SSH authentication

#### Option 1: Automatically configure SSH

Download the [`discover-sparks`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-two-sparks/assets/discover-sparks) script (or clone this playbook's assets), then run it from one of the nodes to automatically discover and configure SSH:

```bash
## Example: download the script into the current directory
curl -fsSL -o discover-sparks https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-connect-two-sparks/assets/discover-sparks
chmod +x discover-sparks

bash ./discover-sparks
```

Expected output is similar to the following (IPs and hostnames will differ). The first time you run the script, you'll be prompted for your password for each node.

```
Found: 192.168.100.10 (node-1.local)
Found: 192.168.100.11 (node-2.local)

Setting up shared SSH access across all nodes...
You may be prompted for your password on each node.

Shared SSH setup complete!
All nodes can now SSH to each other using the shared key (id_ed25519_shared).
```

> [!NOTE]
> If you encounter any errors, follow Option 2 below to configure SSH manually and debug the issue.
>
> The discovery script writes its SSH key under `~/.ssh/` and fails if that directory does not exist. Run `mkdir -p ~/.ssh && chmod 700 ~/.ssh` on both nodes first if you have never used SSH on them.

#### Option 2: Manually discover and configure SSH

Find the IP addresses for the ConnectX interfaces that are up. On both nodes, run the following command and note the addresses for the next step. You only need the IP address of one of the interfaces for configuring SSH keys (for example, **enp1s0f1np1**):

```bash
ip addr show enp1s0f1np1
```

Example output:

```
## In this example, we are using interface enp1s0f1np1.
user@node-1:~$ ip addr show enp1s0f1np1
    4: enp1s0f1np1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
        link/ether 3c:6d:66:cc:b3:b7 brd ff:ff:ff:ff:ff:ff
        inet 192.168.100.10/24 brd 192.168.100.255 scope global noprefixroute enp1s0f1np1
          valid_lft forever preferred_lft forever
        inet6 fe80::3e6d:66ff:fecc:b3b7/64 scope link
          valid_lft forever preferred_lft forever
```

In this example, the IP address for node 1 is **192.168.100.10**. Repeat the process for node 2.

On both nodes, run the following commands to enable passwordless SSH. Use any existing public key (for example `~/.ssh/id_ed25519.pub` or `~/.ssh/id_rsa.pub`):

```bash
## Copy your SSH public key to both nodes. Replace the IP addresses with the ones you found above.
ssh-copy-id -i ~/.ssh/id_ed25519.pub <username>@<IP for Node 1>
ssh-copy-id -i ~/.ssh/id_ed25519.pub <username>@<IP for Node 2>
```

## Step 5. Verify multi-node communication

Test basic multi-node functionality:

```bash
## Test hostname resolution across nodes
ssh <IP for Node 1> hostname
ssh <IP for Node 2> hostname
```

## Step 6. Cleanup and rollback

> [!WARNING]
> These steps reset network configuration on the interconnect interfaces. Only run them if you need to undo this playbook.

```bash
## Rollback network configuration (if using Option 1) on both nodes.
sudo rm /etc/netplan/40-cx7.yaml
sudo netplan apply

## Rollback network configuration (if using Option 2) on both nodes.
## Node 1:
sudo ip addr del 192.168.100.10/24 dev enp1s0f1np1  # Adjust the interface name to the one you used in step 3.
sudo ip addr del 192.168.101.10/24 dev enP2p1s0f1np1  # Adjust the interface name to the one you used in step 3.

## Node 2:
sudo ip addr del 192.168.100.11/24 dev enp1s0f1np1  # Adjust the interface name to the one you used in step 3.
sudo ip addr del 192.168.101.11/24 dev enP2p1s0f1np1  # Adjust the interface name to the one you used in step 3.
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Network unreachable" errors | Network interfaces not configured | Verify the netplan config and run `sudo netplan apply` |
| SSH authentication failures | SSH keys not properly distributed | Re-run `./discover-sparks` and enter passwords when prompted |
| Node 2 not visible in the cluster | Network connectivity issue | Verify the QSFP cable connection and check IP configuration with `ibdev2netdev` and `ip addr show` |
| Discovery script fails writing keys | `~/.ssh` directory missing | Run `mkdir -p ~/.ssh && chmod 700 ~/.ssh` on both nodes, then retry |
| No interfaces show as `Up` | Cable or port issue | Reseat the QSFP cable, use matching physical ports on both nodes, reboot, and re-check `ibdev2netdev` |

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
