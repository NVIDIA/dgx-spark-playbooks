# Connect Three DGX Spark in a Ring Topology

> Connect and set up three DGX Spark devices in a ring topology

## Table of Contents

- [Overview](#overview)
- [Run on Three Sparks](#run-on-three-sparks)
  - [Option 1: Automatically configure SSH](#option-1-automatically-configure-ssh)
  - [Option 2: Manually discover and configure SSH](#option-2-manually-discover-and-configure-ssh)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Configure three DGX Spark systems in a ring topology for high-speed inter-node communication
using 200GbE direct QSFP connections. This setup enables distributed workloads across three
DGX Spark nodes by establishing network connectivity and configuring SSH authentication.

## What you'll accomplish

You will physically connect three DGX Spark devices with QSFP cables, configure network
interfaces for cluster communication, and establish passwordless SSH between nodes to create
a functional distributed computing environment.

## What to know before starting

- Basic understanding of distributed computing concepts
- Working with network interface configuration and netplan
- Experience with SSH key management

## Prerequisites

- Three DGX Spark systems
- Three QSFP cables for direct 200GbE connection between the devices in a ring topology. Use [recommended cable](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/qsfp-cable-0-4m-for-dgx-spark/) or similar.
- SSH access available to all systems
- Root or sudo access on all systems: `sudo whoami`
- The same username on all systems
- Update all systems to the latest OS and Firmware. Refer to the DGX Spark documentation https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html

## Ancillary files

This playbook's files can be found [here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-three-sparks/)

- [**discover-sparks.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) script for automatic node discovery and SSH key distribution
- [**Cluster setup scripts**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup) for automatic network configuration, validation and running NCCL sanity test

## Time & risk

- **Duration:** 1 hour including validation

- **Risk level:** Medium - involves network reconfiguration

- **Rollback:** Network changes can be reversed by removing netplan configs or IP assignments

- **Last Updated:** 3/19/2026
  * First publication

## Run on Three Sparks

## Step 1. Ensure Same Username on all Systems

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

## Step 2. Physical Hardware Connection

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

## Step 3. Network Interface Configuration

Choose one option to set up the network interfaces. The options are mutually exclusive. Option 1 is recommended to avoid complexity of network setup.

> [!NOTE] 
> Each CX7 port provides full 200GbE bandwidth.
> In a three node ring topology all four interfaces on each node must be assigned an IP address to form a symmetric cluster.

**Option 1: Automatic IP Assignment with script**

We have created a script [here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup) which automates the following:
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
cd dgx-spark-playbooks/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup

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
        - 192.168.0.2/24
    enp1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.1.1/24
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
        - 192.168.2.1/24
    enP2p1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.2.2/24
    enp1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.0.3/24
    enP2p1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.0.4/24
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
        - 192.168.1.3/24
    enP2p1s0f0np0:
      dhcp4: false
      addresses:
        - 192.168.1.4/24
    enp1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.2.3/24
    enP2p1s0f1np1:
      dhcp4: false
      addresses:
        - 192.168.2.4/24
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

## Step 4. Set up passwordless SSH authentication

### Option 1: Automatically configure SSH

Run the DGX Spark [**discover-sparks.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) script from one of the nodes to automatically discover and configure SSH:

```bash
curl -O https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks
bash ./discover-sparks
```

Expected output similar to the below, with different IPs and node names. You may see more than one IP for each node as four interfaces (**enp1s0f0np0**, **enP2p1s0f0np0**, **enp1s0f1np1** and **enP2p1s0f1np1**) have IP addresses assigned. This is expected and does not cause any issues. The first time you run the script, you'll be prompted for your password for each node.
```
Found: 192.168.0.1 (dgx-spark-1.local)
Found: 192.168.0.3 (dgx-spark-2.local)
Found: 192.168.1.3 (dgx-spark-3.local)

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
cd dgx-spark-playbooks/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup

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

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Network unreachable" errors | Network interfaces not configured | Verify netplan config and `sudo netplan apply` |
| SSH authentication failures | SSH keys not properly distributed | Re-run `./discover-sparks` and enter passwords |
| Nodes not visible in cluster | Network connectivity issue | Verify QSFP cable connection, check IP configuration |
| "APT update" errors (eg. E: The list of sources could not be read.) | APT sources errors, conflicting sources or signing keys | Check APT and Ubuntu documentation to fix the APT sources or keys conflicts |
