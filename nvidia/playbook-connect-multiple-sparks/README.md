# Connect Multiple DGX Sparks for Distributed Workloads

> Stacked, ring, or switch topologies for high-speed multi-node jobs

## Table of Contents

- [Overview](#overview)
- [Two Sparks Stacked Topology](#two-sparks-stacked-topology)
- [Three Sparks Ring Topology](#three-sparks-ring-topology)
  - [Option 1: Automatically configure SSH](#option-1-automatically-configure-ssh)
  - [Option 2: Manually discover and configure SSH](#option-2-manually-discover-and-configure-ssh)
- [Multiple Sparks Through Switch Topology](#multiple-sparks-through-switch-topology)
  - [Step 3.1. Verify negotiated Link speed](#step-31-verify-negotiated-link-speed)
  - [4.1 Script for Cluster networking configuration](#41-script-for-cluster-networking-configuration)
  - [4.2 Manual Cluster networking configuration](#42-manual-cluster-networking-configuration)
  - [Option 1: Automatically configure SSH](#option-1-automatically-configure-ssh)
  - [Option 2: Manually discover and configure SSH](#option-2-manually-discover-and-configure-ssh)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Configure multiple nodes of multi-node capable hardware for high-speed inter-node
communication using 200GbE QSFP connections. Choose a **stacked** (two nodes),
**ring** (three nodes), or **switch** (two or more nodes) topology. This setup
enables distributed workloads by establishing network connectivity and configuring
SSH authentication.

> [!TIP]
> **Recommended setup:** Use [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) for supported DGX Spark clusters: two or three directly connected systems, or two to four systems connected through a switch. After the QSFP cabling is in place, Cluster Assistant configures the ConnectX-7 network, validates the links, and sets up inter-device SSH.
>
> If Cluster Assistant reports that setup completed successfully, do not repeat the manual network or SSH configuration in this playbook. Continue to your workload playbook, such as [NCCL](https://build.nvidia.com/playbooks/nccl). Use the topology tabs here only when you need a manual setup path or troubleshooting. For compatibility with downstream helper scripts, choose to standardize user information when Cluster Assistant prompts you.

## What you'll accomplish

You'll physically connect multi-node capable hardware platforms with QSFP cables,
configure network interfaces for cluster communication, and establish passwordless
SSH between nodes to create a functional distributed computing environment.

Pick the tab that matches your topology:

- **Two Sparks Stacked Topology** — direct QSFP link between two nodes
- **Three Sparks Ring Topology** — three-node ring over QSFP without a switch
- **Multiple Sparks Through Switch Topology** — expandable cluster through a managed QSFP switch

## What to know before starting

**Required:**

- Basic understanding of distributed computing concepts
- Working with network interface configuration and netplan
- Experience with SSH key management

**Optional (switch topology only):**

- Experience configuring a managed QSFP network switch (ports, bridging, link speed, MTU). Refer to your switch manuals to:
  - Connect to the switch for port and feature management
  - Enable or disable QSFP ports and create a software bridge
  - Configure link speed manually and disable auto-negotiation if needed
  - Configure MTU on the switch ports

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Stacked, ring, or switch QSFP (200GbE); netplan + passwordless SSH | ✅ (QSFP) |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Multiple multi-node capable hardware platforms (two for stacked; three for ring; two or more for switch)
- One QSFP cable per node for 200GbE connection. Use a [recommended QSFP cable](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/qsfp-cable-0-4m-for-dgx-spark/) or similar
- The same username on all systems
- Switch topology only: a managed QSFP56-DD / QSFP56 switch that can provide 200Gbps to each node

**Software requirements**

- SSH access available to all systems
- Root or sudo access on all systems: `sudo whoami`
- All systems updated to the latest OS and firmware (see Resources)
- Network tooling for interface discovery: `ibdev2netdev`
- `avahi-utils` on all systems, for the automatic SSH setup option: `avahi-browse --version`

## Ancillary files

All required assets are in [this playbook's assets folder](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets).

- [`discover-sparks`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets/discover-sparks) — automatic node discovery and SSH key distribution
- [`spark_cluster_setup`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets/spark_cluster_setup) — automatic network configuration, validation, and NCCL sanity test
- [`performance_benchmarking_guide.md`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets/performance_benchmarking_guide.md) — performance benchmarking guidance

## Time & risk

- **Estimated time:** 2 HOURS including validation
- **Risk level:** Medium
  - Involves network reconfiguration on every node
  - Switch topology also depends on correct switch port and bridge configuration
- **Rollback:** Network changes can be reversed by removing netplan configs or IP assignments (see Cleanup in each topology tab)
- **Last Updated:** 08/12/2026
  - Added NVIDIA Sync Cluster Assistant as the recommended setup path and clarified when to skip manual configuration

## Two Sparks Stacked Topology

> [!TIP]
> **Use NVIDIA Sync Cluster Assistant (recommended).** For a two-Spark direct cluster, follow the [Cluster Assistant guide](https://docs.nvidia.com/sync/latest/cluster-assistant.html) instead of the manual network and SSH setup below. Choose to standardize user information when prompted so downstream helper scripts can use the same username on both systems.
>
> If Cluster Assistant reports that setup completed successfully, skip this manual tab and continue to the [NCCL playbook](https://build.nvidia.com/playbooks/nccl) or your workload playbook. Use the steps below only for manual setup or troubleshooting.

## Step 1. Ensure the same username on both systems

On both systems check the username and make sure it's the same:

```bash
## Check current username
whoami
```

If usernames don't match, create a new user (e.g., nvidia) on both systems and login in with the new user:

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

Connect the QSFP cable between both DGX Spark systems using any QSFP interface
on each device. Make sure to use the same physical port on each device to prevent issues with NCCL tests.
This establishes the 200GbE direct connection required for high-speed inter-node communication.
Upon connection between the two nodes, you will see an output like the one below: in this example
the interfaces showing as 'Up' are **enp1s0f1np1** / **enP2p1s0f1np1** (each physical port has two logical interface).

Example output:
```bash
## Check QSFP interface availability on both nodes
nvidia@dxg-spark-1:~$ ibdev2netdev
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
```

> [!NOTE] 
> If none of the interfaces are showing as 'Up', please check the QSFP cable connection, reboot the systems and try again.
> The interfaces showing as 'Up' depend on which port you are using to connect the two nodes. Each physical port has two logical interfaces, for example, enp1s0f1np1 and enP2p1s0f1np1 refer to the same physical port.

## Step 3. Network interface configuration

Choose one option to setup the network interfaces. Option 1 and 2 are mutually exclusive.

> [!NOTE] 
> Full bandwidth can be achieved with just one QSFP cable.
> When two QSFP cables are connected, all four interfaces must be assigned IP addresses to obtain full bandwidth.

**Option 1: Manual IP Assignment with the netplan configure file**

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


**Option 2: Manual IP Assignment with command line**

> [!NOTE]
> Using this option, the IPs assigned to the interfaces will change if you reboot the system.

Use the "(Up)" interfaces. In this example, we'll use **enp1s0f1np1** and **enP2p1s0f1np1**.

On Node 1:
```bash
## Assign static IP and bring up interface.
sudo ip addr add 192.168.100.10/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up

sudo ip addr add 192.168.101.10/24 dev enP2p1s0f1np1
sudo ip link set enP2p1s0f1np1 up
```

Repeat the same process for Node 2, but using IP **192.168.100.11/24** and **192.168.101.11/24**. Ensure to use the correct interface name using `ibdev2netdev` command.
```bash
## Assign static IP and bring up interface.
sudo ip addr add 192.168.100.11/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up

sudo ip addr add 192.168.101.11/24 dev enP2p1s0f1np1
sudo ip link set enP2p1s0f1np1 up
```

You can verify the IP assignment on both nodes by running the following command on each node:
```bash
## Check the interfaces showing as "(Up)" in your output, eg. enp1s0f1np1 and enP2p1s0f1np1
ip addr show enp1s0f1np1
ip addr show enP2p1s0f1np1
```

## Step 4. Set up passwordless SSH authentication

#### Option 1: Automatically configure SSH

Run the DGX Spark [**discover-sparks.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets/discover-sparks) script from one of the nodes to automatically discover and configure SSH:

```bash
bash ./discover-sparks
```

Expected output similar to the below, with different IPs and node names. The first time you run the script, you'll be prompted for your password for each node.
```
Found: 192.168.100.10 (dgx-spark-1.local)
Found: 192.168.100.11 (dgx-spark-2.local)

Setting up bidirectional SSH access (local <-> remote nodes)...
You may be prompted for your password for each node.

SSH setup complete! Both local and remote nodes can now SSH to each other without passwords.
```

> [!NOTE]
> If you encounter any errors, please follow Option 2 below to manually configure SSH and debug the issue.

#### Option 2: Manually discover and configure SSH

You will need to find the IP addresses for the CX-7 interfaces that are up. On both nodes, run the following command to find the IP addresses and take note of them for the next step. You only need the IP address of one of the interfaces for configuring the SSH keys eg. **enp1s0f1np1**
```bash
  ip addr show enp1s0f1np1
```

Example output:
```
## In this example, we are using interface enp1s0f1np1.
nvidia@dgx-spark-1:~$ ip addr show enp1s0f1np1
    4: enp1s0f1np1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
        link/ether 3c:6d:66:cc:b3:b7 brd ff:ff:ff:ff:ff:ff
        inet **192.168.100.10**/24 brd 192.168.100.255 scope global noprefixroute enp1s0f1np1
          valid_lft forever preferred_lft forever
        inet6 fe80::3e6d:66ff:fecc:b3b7/64 scope link
          valid_lft forever preferred_lft forever
```

In this example, the IP address for Node 1 is **192.168.100.10**. Repeat the process for Node 2.

On both nodes, run the following commands to enable passwordless SSH:
```bash
## Copy your SSH public key to both nodes. Please replace the IP addresses with the ones you found in the previous step.
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 1>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 2>
```

## Step 5. Verify Multi-Node Communication

Test basic multi-node functionality:

```bash
## Test hostname resolution across nodes
ssh <IP for Node 1> hostname
ssh <IP for Node 2> hostname
```

## Step 6. Cleanup and Rollback

> [!WARNING]
> These steps will reset network configuration.

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

## Three Sparks Ring Topology

> [!TIP]
> **Use NVIDIA Sync Cluster Assistant (recommended).** For a three-Spark direct ring, follow the [Cluster Assistant guide](https://docs.nvidia.com/sync/latest/cluster-assistant.html) instead of the manual network and SSH setup below. Choose to standardize user information when prompted so downstream helper scripts can use the same username on every system.
>
> If Cluster Assistant reports that setup completed successfully, skip this manual tab and continue to the [NCCL playbook](https://build.nvidia.com/playbooks/nccl) or your workload playbook. Use the steps below only for manual setup or troubleshooting.

## Step 1. Ensure the same username on all systems

On all systems check the username and make sure it's the same:

```bash
## Check current username
whoami
```

If usernames don't match, create a new user (e.g., nvidia) on all systems and log in with the new user:

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

Connect the QSFP cables between the three DGX Spark systems in a ring topology.
Here, Port0 is the CX7 port next to the Ethernet port and Port1 is the CX7 port further away from it.
1. Node1 (Port0) to Node2 (Port1)
2. Node2 (Port0) to Node3 (Port1)
3. Node3 (Port0) to Node1 (Port1)

> [!NOTE]
> Double check that the connections are correct otherwise the network configuration might fail.

This establishes the 200GbE direct connection required for high-speed inter-node communication.
Upon connection between the three nodes, you will see an output like the one below on all nodes: in this example the interface showing as 'Up' is **enp1s0f0np0** / **enP2p1s0f0np0** and **enp1s0f1np1** / **enP2p1s0f1np1** (each physical port has two logical interfaces).

Example output:
```bash
## Check QSFP interface availability on all nodes
nvidia@dgx-spark-1:~$ ibdev2netdev
rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
```

> [!NOTE] 
> If all of the interfaces are not showing as 'Up', please check the QSFP cable connection, reboot the systems and try again.

## Step 3. Network interface configuration

Choose one option to set up the network interfaces. The options are mutually exclusive. Option 1 is recommended to avoid complexity of network setup.

> [!NOTE] 
> Each CX7 port provides full 200GbE bandwidth.
> In a three node ring topology all four interfaces on each node must be assigned an IP address to form a symmetric cluster.

**Option 1: Automatic IP Assignment with script**

We have created a script [here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets/spark_cluster_setup) which automates the following:
1. Interface network configuration for all DGX Sparks
2. Set up passwordless authentication between the DGX Sparks
3. Verify multi-node communication
4. Run NCCL Bandwidth tests

> [!NOTE]
> If you use the script steps below, you can skip rest of the setup instructions in this playbook.

Use the steps below to run the script:

```bash
## Clone the repository
git clone https://github.com/NVIDIA/dgx-spark-playbooks

## Enter the script directory
cd client-hardware-playbooks/nvidia/playbook-connect-multiple-sparks/assets/spark_cluster_setup

## Check the README.md for steps to run the script and configure the cluster networking
```

**Option 2: Manual IP Assignment with the netplan configuration file**

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

Run the DGX Spark [**discover-sparks.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets/discover-sparks) script from one of the nodes to automatically discover and configure SSH:

```bash
curl -O https://raw.githubusercontent.com/NVIDIA/client-hardware-playbooks/refs/heads/main/nvidia/playbook-connect-multiple-sparks/assets/discover-sparks
bash ./discover-sparks
```

Expected output similar to the below, with different IPs and node names. You may see more than one IP for each node as four interfaces (**enp1s0f0np0**, **enP2p1s0f0np0**, **enp1s0f1np1** and **enP2p1s0f1np1**) have IP addresses assigned. This is expected and does not cause any issues. The first time you run the script, you'll be prompted for your password for each node.
```
Found: 192.168.0.1 (dgx-spark-1.local)
Found: 192.168.0.2 (dgx-spark-2.local)
Found: 192.168.3.2 (dgx-spark-3.local)

Setting up bidirectional SSH access (local <-> remote nodes)...
You may be prompted for your password for each node.

SSH setup complete! All nodes can now SSH to each other without passwords.
```

> [!NOTE]
> If you encounter any errors, please follow Option 2 below to manually configure SSH and debug the issue.

### Option 2: Manually discover and configure SSH

You will need to find the IP addresses for the CX-7 interfaces that are up. On all nodes, run the following command to find the IP addresses and take note of them for the next step.
```bash
  ip addr show enp1s0f0np0
  ip addr show enp1s0f1np1
```

Example output:
```
## In this example, we are using interface enp1s0f1np1.
nvidia@dgx-spark-1:~$ ip addr show enp1s0f1np1
    4: enp1s0f1np1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
        link/ether 3c:6d:66:cc:b3:b7 brd ff:ff:ff:ff:ff:ff
        inet **192.168.1.1**/24 brd 192.168.1.255 scope link noprefixroute enp1s0f1np1
          valid_lft forever preferred_lft forever
        inet6 fe80::3e6d:66ff:fecc:b3b7/64 scope link
          valid_lft forever preferred_lft forever
```

In this example, the IP address for Node 1 is **192.168.1.1**. Repeat the process for other nodes.

On all nodes, run the following commands to enable passwordless SSH:
```bash
## Copy your SSH public key to all nodes. Please replace the IP addresses with the ones you found in the previous step.
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 1>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 2>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 3>
```

## Step 5. Verify Multi-Node Communication

Test basic multi-node functionality:

```bash
## Test hostname resolution across nodes
ssh <IP for Node 1> hostname
ssh <IP for Node 2> hostname
ssh <IP for Node 3> hostname
```

## Step 6. Run NCCL tests

Now your cluster is set up to run distributed workloads across three nodes. Try running the NCCL bandwidth test.

Use the steps below to run the script which will run the NCCL test on the cluster:

```bash
## Clone the repository
git clone https://github.com/NVIDIA/dgx-spark-playbooks

## Enter the script directory
cd client-hardware-playbooks/nvidia/playbook-connect-multiple-sparks/assets/spark_cluster_setup

## Check the README.md in the script directory for steps to run the NCCL tests with "--run-nccl-test" option
```

## Step 7. Cleanup and Rollback

> [!WARNING]
> These steps will reset network configuration.

```bash
## Rollback network configuration
sudo rm /etc/netplan/40-cx7.yaml
sudo netplan apply
```

## Multiple Sparks Through Switch Topology

> [!TIP]
> **Use NVIDIA Sync Cluster Assistant (recommended for two to four Sparks).** For a supported switch-connected cluster, follow the [Cluster Assistant guide](https://docs.nvidia.com/sync/latest/cluster-assistant.html) instead of the manual network and SSH setup below. Configure the switch and cabling as described in that guide, and choose to standardize user information when prompted so downstream helper scripts can use the same username on every system.
>
> If Cluster Assistant reports that setup completed successfully, skip this manual tab and continue to the [NCCL playbook](https://build.nvidia.com/playbooks/nccl) or your workload playbook. Cluster Assistant supports a maximum of four systems; use the manual steps below for larger clusters or for troubleshooting.

## Step 1. Ensure the same username on all systems

On all systems check and make sure the usernames are the same:

```bash
## Check current username
whoami
```

If usernames don't match, create a new user (e.g., nvidia) on all systems and login in with the new user:

```bash
## Create nvidia user and add to sudo group
sudo useradd -m nvidia
sudo usermod -aG sudo nvidia

## Set password for nvidia user
sudo passwd nvidia

## Switch to nvidia user
su - nvidia
```

## Step 2. Switch management

Most QSFP switches offer some form of management interface, either through CLI or UI. Refer to the documentation and connect to the management interface. Make sure that the ports on the switch are enabled. You will need to ensure that the switch is configured to provide 200Gbps connection to each DGX Spark. If not done already, refer to the [Overview](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/overview) of this playbook for the prior knowledge and pre-requisites required for this playbook.

## Step 3. Physical hardware connection

Connect the QSFP cables between DGX Spark systems and the switch(QSFP56-DD/QSFP56 ports) using one CX7 port on each Spark system. It is recommended to use the same CX7 port on all Spark systems for easier network configuration and avoiding NCCL test failures. In this playbook the second port (the one further from the ethernet port) is used. This should establish the 200Gbps connection required for high-speed inter-node communication. You will see an output like the one below on all sparks. In this example the interfaces showing as 'Up' are **enp1s0f1np1** and **enP2p1s0f1np1** (each physical port has two logical interfaces).

Example output:
```bash
## Check QSFP interface availability on all nodes
nvidia@dxg-spark-1:~$ ibdev2netdev
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
```

> [!NOTE]
> If none of the interfaces are showing as 'Up', please check the QSFP cable connection, reboot the systems and try again.
> The interfaces showing as 'Up' depend on which port you are using to connect the nodes to the switch. Each physical port has two logical interfaces, for example, Port 1 has two interfaces - enp1s0f1np1 and enP2p1s0f1np1. Please disregard enp1s0f0np0 and enP2p1s0f0np0, and use enp1s0f1np1 and enP2p1s0f1np1 only.

### Step 3.1. Verify negotiated Link speed

The link speed might not default to 200Gbps with auto-negotiation. To confirm, run the command below on all sparks and check that the speed is shown as 200000Mb/s. If it shows lesser than that value, then the link speed needs to be set to 200Gbps manually in the switch port configuration and auto-negotiation should be disabled. Refer to the switch's manual/documentation to disable auto-negotiation and set the link speed manually to 200Gbps (eg. 200G-baseCR4)

Example output:
```bash
nvidia@dxg-spark-1:~$ sudo ethtool enp1s0f1np1 | grep Speed
	Speed: 100000Mb/s

nvidia@dxg-spark-1:~$ sudo ethtool enP2p1s0f1np1 | grep Speed
	Speed: 100000Mb/s
```

After setting the correct speed on the switch ports. Verify the link speed on all the DGX Sparks again.

Example output:
```bash
nvidia@dxg-spark-1:~$ sudo ethtool enp1s0f1np1 | grep Speed
	Speed: 200000Mb/s

nvidia@dxg-spark-1:~$ sudo ethtool enP2p1s0f1np1 | grep Speed
	Speed: 200000Mb/s
```

## Step 4. Network Interface Configuration

> [!NOTE]
> Full bandwidth can be achieved with just one QSFP cable.

For a clustered setup, all DGX sparks:
1. Should be accessible for management (eg. SSH and run commands)
2. Should be able to access internet (eg. to download models/utilities)
3. Should be able to talk to each other using TCP/IP over CX7. The steps below help configure that.

It is recommended to use the Ethernet/WiFi network for management and internet traffic and keep it separate from the CX7 network to avoid CX7 bandwidth from being used for non-workload traffic.

The supported way to configure a cluster with switch requires configuring a bridge (or using the default bridge) on the switch and adding all the ports of interest (ports connected to DGX sparks) to it through the switch management interface.
1. This way, all ports are part of a single layer-2 domain which is required for cluster networking configuration
2. Some switches have restriction that Hardware offloading can only be enabled on one bridge, so keeping all ports in a single bridge is required

Once you are done creating/adding ports to the bridge, you should be ready to configure networking on the DGX Spark side.

### 4.1 Script for Cluster networking configuration

We have created a script [here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets/spark_cluster_setup) which automates the following:
1. Interface network IP configuration for all DGX Sparks
2. Set up password-less authentication between the DGX Sparks
3. Verify multi-node communication
4. Run NCCL Bandwidth tests

> [!NOTE]
> You can use the script or continue with the manual configurations in the following sections. If you use the script, you can skip the rest of the setup sections in this playbook.

Use the steps below to run the script:

```bash
## Clone the repository
git clone https://github.com/NVIDIA/dgx-spark-playbooks

## Enter the script directory
cd client-hardware-playbooks/nvidia/playbook-connect-multiple-sparks/assets/spark_cluster_setup

## Check the README.md in the script directory for steps to run the script and configure the cluster networking with "--run-setup" argument
```

### 4.2 Manual Cluster networking configuration

In this case, you can choose one of the options to assign the IPs to the CX7 logical interfaces. Options 1 and 2 are mutually exclusive.
1. DHCP server on the switch (recommended, if it is supported)
2. Manual IP addressing (netplan will be different on each node but provides more control and deterministic IPs)

#### Option 1: Configure DHCP server on the switch

1. Configure the DHCP server on the switch with a subnet large enough to assign IPs to all sparks. A /24 subnet should work well for configuration and any future expansion.
2. Configure the 'UP' CX7 interfaces in the DGX sparks to acquire IP using DHCP. For eg. if the logical interfaces **enp1s0f1np1** / **enP2p1s0f1np1** are 'UP' then create a netplan like below on all sparks.

```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      dhcp4: true
    enP2p1s0f1np1:
      dhcp4: true
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

3. Confirm that the interfaces get IPs assigned

```bash
## In this example, we are using interface enp1s0f1np1. Similarly check enP2p1s0f1np1.
nvidia@dgx-spark-1:~$ ip addr show enp1s0f1np1 | grep -w inet
    inet 100.100.100.4/24 brd 100.100.100.255 scope global noprefixroute enp1s0f1np1
```

#### Option 2: Manual IP Assignment with the netplan configuration file

> [!NOTE]
> `enp1s0f1np1` and `enP2p1s0f1np1` are assigned to **different subnets** (`192.168.100.x/24` and `192.168.101.x/24` respectively). This is required — assigning two distinct network interfaces to the same subnet causes networking and software conflicts (e.g., routing ambiguity and NCCL communication failures).

On node 1:
```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.1/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.1/24
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
        - 192.168.100.2/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.2/24
      dhcp4: no
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
    enp1s0f1np1:
      addresses:
        - 192.168.100.3/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.3/24
      dhcp4: no
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

On node 4:
```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.4/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.4/24
      dhcp4: no
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

## Step 5. Set up passwordless SSH authentication

### Option 1: Automatically configure SSH

Run the DGX Spark [**discover-sparks.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets/discover-sparks) script from one of the nodes to automatically discover and configure SSH:

```bash
curl -O https://raw.githubusercontent.com/NVIDIA/client-hardware-playbooks/refs/heads/main/nvidia/playbook-connect-multiple-sparks/assets/discover-sparks
bash ./discover-sparks
```

Expected output similar to the below, with different IPs and node names. You may see up to two IPs for each node as two interfaces (eg. **enp1s0f1np1** and **enP2p1s0f1np1**) have IP addresses assigned. This is expected and does not cause any issues. The first time you run the script, you'll be prompted for your password for each node.
```
Found: 192.168.100.1 (dgx-spark-1.local)
Found: 192.168.100.2 (dgx-spark-2.local)
Found: 192.168.100.3 (dgx-spark-3.local)
Found: 192.168.100.4 (dgx-spark-4.local)

Setting up bidirectional SSH access (local <-> remote nodes)...
You may be prompted for your password for each node.

SSH setup complete! All local and remote nodes can now SSH to each other without passwords.
```

> [!NOTE]
> If you encounter any errors, please follow Option 2 below to manually configure SSH and debug the issue.

### Option 2: Manually discover and configure SSH

You will need to find the IP addresses for the CX-7 interfaces that are up. On all nodes, run the following command to find the IP addresses and take note of them for the next step.
```bash
  ip addr show enp1s0f1np1
```

Example output:
```
## In this example, we are using interface enp1s0f1np1.
nvidia@dgx-spark-1:~$ ip addr show enp1s0f1np1
    4: enp1s0f1np1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
        link/ether 3c:6d:66:cc:b3:b7 brd ff:ff:ff:ff:ff:ff
        inet **192.168.100.1**/24 brd 192.168.100.255 scope global noprefixroute enp1s0f1np1
          valid_lft forever preferred_lft forever
        inet6 fe80::3e6d:66ff:fecc:b3b7/64 scope link
          valid_lft forever preferred_lft forever
```

In this example, the IP address for Node 1 is **192.168.100.1**. Repeat the process for other nodes.

On all nodes, run the following commands to enable passwordless SSH:
```bash
## Copy your SSH public key to all nodes. Replace the IP addresses with the ones you found in the previous step.
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 1>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 2>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 3>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 4>
```

## Step 6. Verify Multi-Node Communication

Test basic multi-node functionality from the head node:

```bash
## Test hostname resolution across nodes
ssh <IP for Node 1> hostname
ssh <IP for Node 2> hostname
ssh <IP for Node 3> hostname
ssh <IP for Node 4> hostname
```

## Step 7. Running Tests and Workloads

Now your cluster is set up to run distributed workloads across the nodes. Try running the [NCCL playbook](https://build.nvidia.com/spark/nccl/stacked-sparks).

> [!NOTE]
> Wherever the playbook asks to run a command on **two nodes**, just run it on **all nodes**.
> Make sure to adapt the *mpirun* NCCL command which you run on the **head node** to accommodate **all nodes**. This example shows four nodes.

Example mpirun command for NCCL:
```bash
## Set network interface environment variables (use your Up interface from the previous step)
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export OMPI_MCA_btl_tcp_if_include=enp1s0f1np1

## Run the all_gather performance test across all nodes (eg. 4 nodes) (replace the IP addresses with the ones you found in the previous step)
mpirun -np 4 -H <IP for Node 1>:1,<IP for Node 2>:1,<IP for Node 3>:1,<IP for Node 4>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf
```

## Step 8. Cleanup and Rollback

> [!WARNING]
> These steps will reset network configuration.

```bash
## Rollback network configuration
sudo rm /etc/netplan/40-cx7.yaml
sudo netplan apply
```

> [!NOTE]
> If disconnecting the switch, then make sure to do the following
> 1. Re-enable auto-negotiation to avoid issues later if the switch is used for different purposes.
> 2. Remove the DHCP server configuration on the switch if you used that to assign IPs to Sparks.
> 3. If you created a new bridge, move the ports back to the default bridge and delete the new bridge.

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every supported platform.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| "Network unreachable" errors | All hardware platforms | Network interfaces not configured | Verify netplan config and run `sudo netplan apply` |
| SSH authentication failures | All hardware platforms | SSH keys not properly distributed | Re-run `./discover-sparks` and enter passwords when prompted |
| Nodes not visible in cluster | All hardware platforms | Network connectivity issue | Verify QSFP cable connection; check IP configuration with `ibdev2netdev` and `ip addr show` |
| Discovery script fails writing keys | All hardware platforms | `~/.ssh` directory missing | Run `mkdir -p ~/.ssh && chmod 700 ~/.ssh` on all nodes, then retry |
| Discovery script exits with `avahi-browse not found` | All hardware platforms | `avahi-utils` not installed | Install `avahi-utils` on all nodes, then re-run the script |
| No interfaces show as `Up` | All hardware platforms | Cable or port issue | Reseat the QSFP cables, confirm topology wiring, reboot, and re-check `ibdev2netdev` |
| "APT update" errors (for example, `E: The list of sources could not be read.`) | All hardware platforms | APT sources errors, conflicting sources, or signing keys | Check APT and Ubuntu documentation to fix the APT sources or keys conflicts |
| NCCL test failures (for example, `libnccl.so.2: cannot open shared object file`) | All hardware platforms | NCCL not configured on all nodes | Follow the NCCL playbook on **all** nodes before running the NCCL test |

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
