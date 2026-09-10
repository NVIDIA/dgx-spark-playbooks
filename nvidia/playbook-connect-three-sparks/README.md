# Connect Three DGX Sparks in a Ring Topology

> Distributed workloads over direct node-to-node cabling without a switch

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Option 1: Automatically configure SSH](#option-1-automatically-configure-ssh)
  - [Option 2: Manually discover and configure SSH](#option-2-manually-discover-and-configure-ssh)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Configure three nodes of multi-node capable hardware in a ring topology for high-speed
inter-node communication using 200GbE direct QSFP connections. This setup enables
distributed workloads across three nodes by establishing network connectivity and
configuring SSH authentication.

> [!TIP]
> **Recommended for DGX Spark:** Use [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) to configure and validate the three-node ring. If Cluster Assistant reports success, do not repeat the manual network or SSH steps in this playbook; continue to your workload playbook.

## What you'll accomplish

You'll physically connect three multi-node capable hardware platforms with QSFP cables,
configure network interfaces for cluster communication, and establish passwordless SSH
between nodes to create a functional distributed computing environment.

## What to know before starting

**Required:**

- Basic understanding of distributed computing concepts
- Working with network interface configuration and netplan
- Experience with SSH key management

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Three-node ring over QSFP (200GbE); netplan + passwordless SSH | ✅ (QSFP ring) |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Three multi-node capable hardware platforms
- Three QSFP cables for direct 200GbE connection between the devices in a ring topology. Use a [recommended QSFP cable](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/qsfp-cable-0-4m-for-dgx-spark/) or similar.
- The same username on all systems

**Software requirements**

- SSH access available to all systems
- Root or sudo access on all systems: `sudo whoami`
- All systems updated to the latest OS and firmware (see Resources)
- Network tooling for interface discovery: `ibdev2netdev`
- `avahi-utils` on all systems, for the automatic SSH setup option: `avahi-browse --version`

## Ancillary files

All required assets are in [this playbook's assets folder](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-three-sparks/assets).

- [`discover-sparks`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-three-sparks/assets/discover-sparks) — automatic node discovery and SSH key distribution
- [Cluster setup scripts](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup) — automatic network configuration, validation, and NCCL sanity test

## Time & risk

- **Estimated time:** 60 MIN including validation
- **Risk level:** Medium
  - Involves network reconfiguration on all three nodes
- **Rollback:** Network changes can be reversed by removing netplan configs or IP assignments (see Cleanup in Instructions)
- **Last Updated:** 08/12/2026
  - Added NVIDIA Sync Cluster Assistant as the recommended setup path and clarified when to skip manual configuration

## Instructions

> [!TIP]
> If [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) successfully configured this DGX Spark cluster, skip this manual setup and continue to your workload playbook.

## Step 1. Ensure the same username on all systems

On all systems check the username and make sure it's the same:

```bash
## Check current username
whoami
```

If usernames don't match, create a new user (for example, `nvidia`) on all systems and log in with the new user:

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

Connect the QSFP cables between the three multi-node capable hardware platforms in a ring topology.
Here, Port0 is the CX7 port next to the Ethernet port and Port1 is the CX7 port further away from it.

1. Node1 (Port0) to Node2 (Port1)
2. Node2 (Port0) to Node3 (Port1)
3. Node3 (Port0) to Node1 (Port1)

> [!NOTE]
> Double-check that the connections are correct; otherwise network configuration may fail.

This establishes the 200GbE direct connection required for high-speed inter-node communication.
Upon connection between the three nodes, you will see an output like the one below on all nodes: in this example the interface showing as `Up` is **enp1s0f0np0** / **enP2p1s0f0np0** and **enp1s0f1np1** / **enP2p1s0f1np1** (each physical port has two logical interfaces).

Example output:

```bash
## Check QSFP interface availability on all nodes
nvidia@node-1:~$ ibdev2netdev
rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
```

> [!NOTE]
> If all of the interfaces are not showing as `Up`, check the QSFP cable connection, reboot the systems, and try again.

## Step 3. Network interface configuration

Choose one option to set up the network interfaces. The options are mutually exclusive. Option 1 is recommended to avoid the complexity of manual network setup.

> [!NOTE]
> Each CX7 port provides full 200GbE bandwidth.
> In a three-node ring topology all four interfaces on each node must be assigned an IP address to form a symmetric cluster.

**Option 1: Automatic IP assignment with script**

A script is available [here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup) which automates the following:

1. Interface network configuration for all nodes
2. Passwordless authentication between the nodes
3. Verification of multi-node communication
4. NCCL bandwidth tests

> [!NOTE]
> If you use the script steps below, you can skip the rest of the setup instructions in this playbook.

Use the steps below to run the script:

```bash
## Clone the repository
git clone https://github.com/NVIDIA/dgx-spark-playbooks

## Enter the script directory
cd dgx-spark-playbooks/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup

## Check the README.md for steps to run the script and configure the cluster networking
```

**Option 2: Manual IP assignment with the netplan configuration file**

On node 1:

```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.0.1/24
    enP2p1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.1.1/24
    enp1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.2.1/24
    enP2p1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.3.1/24
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
    enp1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.4.1/24
    enP2p1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.5.1/24
    enp1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.0.2/24
    enP2p1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.1.2/24
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

On node 3:

```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.2.2/24
    enP2p1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.3.2/24
    enp1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.4.2/24
    enP2p1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.5.2/24
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

## Step 4. Set up passwordless SSH authentication

### Option 1: Automatically configure SSH

Run the [`discover-sparks`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-three-sparks/assets/discover-sparks) script from one of the nodes to discover the other nodes on the ring and distribute a shared SSH key:

```bash
## Get this playbook's assets if you don't already have them
git clone https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-three-sparks
cd nvidia/playbook-connect-three-sparks/assets

bash ./discover-sparks
```

Expected output is similar to the below, with different IPs and node names. You may see more than one IP for each node as four interfaces (**enp1s0f0np0**, **enP2p1s0f0np0**, **enp1s0f1np1**, and **enP2p1s0f1np1**) have IP addresses assigned. This is expected and does not cause any issues. The first time you run the script, you'll be prompted for your password on each node.

```
Found: 192.168.0.1 (node-1.local)
Found: 192.168.0.2 (node-2.local)
Found: 192.168.3.2 (node-3.local)
Generating shared SSH key for all nodes...

Setting up shared SSH access across all nodes...
You may be prompted for your password on each node.
  ✓ Added shared public key to local authorized_keys
Configuring 192.168.0.1...
  ✓ Successfully configured 192.168.0.1 with shared key
Configuring 192.168.0.2...
  ✓ Successfully configured 192.168.0.2 with shared key
Configuring 192.168.3.2...
  ✓ Successfully configured 192.168.3.2 with shared key

Shared SSH setup complete!
All nodes can now SSH to each other using the shared key (id_ed25519_shared).
```

> [!NOTE]
> If you encounter any errors, follow Option 2 below to manually configure SSH and debug the issue.

### Option 2: Manually discover and configure SSH

Find the IP addresses for the CX7 interfaces that are up. On all nodes, run the following commands and take note of the addresses for the next step.

```bash
ip addr show enp1s0f0np0
ip addr show enp1s0f1np1
```

Example output:

```
## In this example, we are using interface enp1s0f1np1.
nvidia@node-1:~$ ip addr show enp1s0f1np1
    4: enp1s0f1np1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
        link/ether 3c:6d:66:cc:b3:b7 brd ff:ff:ff:ff:ff:ff
        inet **192.168.1.1**/24 brd 192.168.1.255 scope link noprefixroute enp1s0f1np1
          valid_lft forever preferred_lft forever
        inet6 fe80::3e6d:66ff:fecc:b3b7/64 scope link
          valid_lft forever preferred_lft forever
```

In this example, the IP address for Node 1 is **192.168.1.1**. Repeat the process for the other nodes.

On all nodes, run the following commands to enable passwordless SSH:

```bash
## Copy your SSH public key to all nodes. Replace the IP addresses with the ones you found in the previous step.
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 1>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 2>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 3>
```

## Step 5. Verify multi-node communication

Test basic multi-node functionality:

```bash
## Test hostname resolution across nodes
ssh <IP for Node 1> hostname
ssh <IP for Node 2> hostname
ssh <IP for Node 3> hostname
```

## Step 6. Run NCCL tests

Your cluster is set up to run distributed workloads across three nodes. Try running the NCCL bandwidth test.

Use the steps below to run the script which will run the NCCL test on the cluster:

```bash
## Clone the repository
git clone https://github.com/NVIDIA/dgx-spark-playbooks

## Enter the script directory
cd dgx-spark-playbooks/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup

## Check the README.md in the script directory for steps to run the NCCL tests with "--run-nccl-test" option
```

## Step 7. Cleanup and rollback

> [!WARNING]
> These steps reset network configuration on the node where you run them.

```bash
## Rollback network configuration
sudo rm /etc/netplan/40-cx7.yaml
sudo netplan apply
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Network unreachable" errors | Network interfaces not configured | Verify netplan config and run `sudo netplan apply` |
| SSH authentication failures | SSH keys not properly distributed | Re-run `./discover-sparks` and enter passwords when prompted |
| Nodes not visible in cluster | Network connectivity issue | Verify QSFP cable connection; check IP configuration on all four CX7 interfaces with `ibdev2netdev` and `ip addr show` |
| Discovery script fails writing keys | `~/.ssh` directory missing | Run `mkdir -p ~/.ssh && chmod 700 ~/.ssh` on all three nodes, then retry |
| Discovery script exits with `avahi-browse not found` | `avahi-utils` not installed | Install `avahi-utils` on all three nodes, then re-run the script |
| No interfaces show as `Up` | Cable or port issue | Reseat the QSFP cables, confirm the ring wiring (Port0 to Port1 on the next node), reboot, and re-check `ibdev2netdev` |
| "APT update" errors (for example, `E: The list of sources could not be read.`) | APT sources errors, conflicting sources, or signing keys | Check APT and Ubuntu documentation to fix the APT sources or keys conflicts |

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
