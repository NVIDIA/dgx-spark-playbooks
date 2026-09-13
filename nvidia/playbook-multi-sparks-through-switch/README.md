# Connect Multiple DGX Sparks Through a Switch

> Expandable clusters for larger multi-node jobs

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Step 3.1. Verify negotiated link speed](#step-31-verify-negotiated-link-speed)
  - [4.1 Script for cluster networking configuration](#41-script-for-cluster-networking-configuration)
  - [4.2 Manual cluster networking configuration](#42-manual-cluster-networking-configuration)
  - [Option 1: Automatically configure SSH](#option-1-automatically-configure-ssh)
  - [Option 2: Manually discover and configure SSH](#option-2-manually-discover-and-configure-ssh)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Configure four or more multi-node capable hardware platforms for high-speed inter-node communication using 200Gbps QSFP connections through a QSFP switch. This setup enables distributed workloads across multiple nodes by establishing network connectivity and configuring SSH authentication.

> [!TIP]
> **Recommended for two to four DGX Spark systems:** Use [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) to configure and validate the switch-connected cluster. If Cluster Assistant reports success, do not repeat the manual network or SSH steps in this playbook; continue to your workload playbook. For more than four systems, use the manual setup below.

## What you'll accomplish

You'll physically connect four or more hardware platforms with QSFP cables and a QSFP switch, configure network interfaces for cluster communication, and establish passwordless SSH between nodes to create a functional distributed computing environment. The same setup can be scaled to additional nodes connected through the same switch.

## What to know before starting

**Required:**

- Basic understanding of distributed computing concepts
- Working with network interface configuration and netplan
- Experience with SSH key management
- Basic understanding and experience configuring the managed QSFP network switch you plan to use. Refer to the switch instruction manuals to:
  - Connect to the switch for management of ports and features
  - Enable or disable QSFP ports and create a software bridge on the switch
  - Configure the link speed manually on the port and disable auto-negotiation if needed

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | QSFP switch + ConnectX (CX7) cluster networking; passwordless SSH | ✅ (QSFP switch) |

## Prerequisites

**Hardware requirements**

- Supported multi-node capable hardware — see Supported hardware platforms matrix above
- Four hardware platforms (these instructions work for any number of nodes connected through a switch)
- QSFP switch with at least 4 QSFP56-DD ports (at least 200Gbps each)
- QSFP cables for 200Gbps connection from the switch to each node. Use a [recommended cable](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/qsfp-cable-0-4m-for-dgx-spark/) or similar.
  - One cable per node
  - If the switch has 400Gbps ports, you can also use breakout cables to split them into two 200Gbps ports
- Same username on all nodes
- Hardware platforms powered on, updated to the latest OS and firmware, and reachable for SSH — see Resources for platform update guidance

**Software requirements**

- SSH access available to all nodes
- Root or sudo access on all nodes: `sudo whoami`
- Netplan available for interface configuration: `netplan --version`

## Ancillary files

All required assets can be found [in this playbook's repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-multi-sparks-through-switch/).

- [`spark_cluster_setup`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-multi-sparks-through-switch/assets/spark_cluster_setup) — automatic network configuration, validation, and NCCL sanity test
- [`discover-sparks`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) — automatic node discovery and SSH key distribution (shared with the two-node connection playbook)

## Time & risk

- **Estimated time:** 2 HOURS (including validation)
- **Risk level:** Medium
  - Involves network reconfiguration on each node and on the switch
  - Incorrect link-speed or bridge settings can leave interconnect ports down
- **Rollback:** Remove netplan configs or IP assignments and re-apply netplan; restore switch auto-negotiation and bridge settings if you changed them
- **Last Updated:** 08/12/2026
  - Added NVIDIA Sync Cluster Assistant for supported two- to four-system clusters and retained manual setup for larger clusters

## Instructions

> [!TIP]
> If [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) successfully configured this switch-connected DGX Spark cluster, skip this manual setup and continue to your workload playbook. Cluster Assistant supports up to four systems; use the steps below for larger clusters.

## Step 1. Ensure the same username on all nodes

On all four (or more) nodes, check that the usernames match:

```bash
## Check current username
whoami
```

If usernames don't match, create a shared user (for example, `nvidia`) on all nodes and log in with that user:

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

Most QSFP switches offer a management interface through CLI or UI. Refer to the switch documentation and connect to the management interface. Make sure the ports on the switch are enabled. For connecting four nodes, ensure the switch is configured to provide a 200Gbps connection to each hardware platform. If not done already, refer to the **Overview** tab for the prior knowledge and prerequisites required for this playbook.

## Step 3. Physical hardware connection

Connect the QSFP cables between each hardware platform and the switch (QSFP56-DD/QSFP56 ports) using one ConnectX (CX7) port on each node. Use the same CX7 port on all nodes for easier network configuration and to avoid NCCL test failures. In this playbook the second port (the one further from the ethernet port) is used. This should establish the 200Gbps connection required for high-speed inter-node communication. You will see output like the example below on all nodes. In this example the interfaces showing as `Up` are **enp1s0f1np1** and **enP2p1s0f1np1** (each physical port has two logical interfaces).

Example output:

```bash
## Check QSFP interface availability on all nodes
nvidia@node-1:~$ ibdev2netdev
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
```

> [!NOTE]
> If none of the interfaces are showing as `Up`, check the QSFP cable connection, reboot the systems, and try again.
> The interfaces showing as `Up` depend on which port you use to connect the nodes to the switch. Each physical port has two logical interfaces — for example, Port 1 has `enp1s0f1np1` and `enP2p1s0f1np1`. Disregard `enp1s0f0np0` and `enP2p1s0f0np0`, and use `enp1s0f1np1` and `enP2p1s0f1np1` only when following this playbook's port choice.

### Step 3.1. Verify negotiated link speed

The link speed might not default to 200Gbps with auto-negotiation. To confirm, run the command below on all nodes and check that the speed is shown as `200000Mb/s`. If it shows less than that value, set the link speed to 200Gbps manually in the switch port configuration and disable auto-negotiation. Refer to the switch manual to disable auto-negotiation and set the link speed manually to 200Gbps (for example, `200G-baseCR4`).

Example output:

```bash
nvidia@node-1:~$ sudo ethtool enp1s0f1np1 | grep Speed
	Speed: 100000Mb/s

nvidia@node-1:~$ sudo ethtool enP2p1s0f1np1 | grep Speed
	Speed: 100000Mb/s
```

After setting the correct speed on the switch ports, verify the link speed on all nodes again.

Example output:

```bash
nvidia@node-1:~$ sudo ethtool enp1s0f1np1 | grep Speed
	Speed: 200000Mb/s

nvidia@node-1:~$ sudo ethtool enP2p1s0f1np1 | grep Speed
	Speed: 200000Mb/s
```

## Step 4. Network interface configuration

> [!NOTE]
> Full bandwidth can be achieved with just one QSFP cable.

For a clustered setup, all nodes:

1. Should be accessible for management (for example, SSH and run commands)
2. Should be able to access the internet (for example, to download models or utilities)
3. Should be able to talk to each other using TCP/IP over the ConnectX interconnect. The steps below help configure that.

Use the Ethernet/Wi-Fi network for management and internet traffic and keep it separate from the ConnectX network so interconnect bandwidth is reserved for workloads.

The supported way to configure a cluster with a switch requires configuring a bridge (or using the default bridge) on the switch and adding all ports of interest (ports connected to the nodes) to it through the switch management interface.

1. This way, all ports are part of a single layer-2 domain, which is required for cluster networking configuration
2. Some switches restrict hardware offloading to one bridge, so keeping all ports in a single bridge is required

Once you have created or added ports to the bridge, you are ready to configure networking on each hardware platform.

### 4.1 Script for cluster networking configuration

A script is available [in this playbook's repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-multi-sparks-through-switch/assets/spark_cluster_setup) which automates the following:

1. Interface network IP configuration for all nodes
2. Set up passwordless authentication between the nodes
3. Verify multi-node communication
4. Run NCCL bandwidth tests

> [!NOTE]
> You can use the script or continue with the manual configurations in the following sections. If you use the script, you can skip the rest of the setup sections in this playbook.

Use the steps below to run the script:

```bash
## Clone the repository
git clone https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-multi-sparks-through-switch
cd nvidia/playbook-multi-sparks-through-switch/assets/spark_cluster_setup

## Check the README.md in the script directory for steps to run the script
## and configure the cluster networking with the "--run-setup" argument
```

### 4.2 Manual cluster networking configuration

You can choose one of the options to assign IPs to the ConnectX logical interfaces. Options 1 and 2 are mutually exclusive.

1. DHCP server on the switch (recommended, if supported)
2. Manual IP addressing (netplan will differ on each node but provides more control and deterministic IPs)

#### Option 1: Configure DHCP server on the switch

1. Configure the DHCP server on the switch with a subnet large enough to assign IPs to all nodes. A `/24` subnet should work well for configuration and future expansion.
2. Configure the `Up` ConnectX interfaces on each node to acquire an IP using DHCP. For example, if the logical interfaces **enp1s0f1np1** / **enP2p1s0f1np1** are `Up`, create a netplan like the one below on all nodes.

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

3. Confirm that the interfaces get IPs assigned:

```bash
## In this example, we are using interface enp1s0f1np1. Similarly check enP2p1s0f1np1.
nvidia@node-1:~$ ip addr show enp1s0f1np1 | grep -w inet
    inet 100.100.100.4/24 brd 100.100.100.255 scope global noprefixroute enp1s0f1np1
```

#### Option 2: Manual IP assignment with the netplan configuration file

> [!NOTE]
> `enp1s0f1np1` and `enP2p1s0f1np1` are assigned to **different subnets** (`192.168.100.x/24` and `192.168.101.x/24` respectively). This is required — assigning two distinct network interfaces to the same subnet causes networking and software conflicts (for example, routing ambiguity and NCCL communication failures).

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

Run the [`discover-sparks`](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks) script from one of the nodes to automatically discover and configure SSH:

```bash
curl -O https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks
bash ./discover-sparks
```

Expected output similar to the below, with different IPs and node names. You may see up to two IPs for each node as two interfaces (for example, **enp1s0f1np1** and **enP2p1s0f1np1**) have IP addresses assigned. This is expected and does not cause issues. The first time you run the script, you'll be prompted for your password for each node.

```
Found: 192.168.100.1 (node-1.local)
Found: 192.168.100.2 (node-2.local)
Found: 192.168.100.3 (node-3.local)
Found: 192.168.100.4 (node-4.local)

Setting up bidirectional SSH access (local <-> remote nodes)...
You may be prompted for your password for each node.

SSH setup complete! All local and remote nodes can now SSH to each other without passwords.
```

> [!NOTE]
> If you encounter any errors, follow Option 2 below to manually configure SSH and debug the issue.

### Option 2: Manually discover and configure SSH

Find the IP addresses for the ConnectX interfaces that are up. On all nodes, run the following command and take note of the addresses for the next step:

```bash
ip addr show enp1s0f1np1
```

Example output:

```
## In this example, we are using interface enp1s0f1np1.
nvidia@node-1:~$ ip addr show enp1s0f1np1
    4: enp1s0f1np1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
        link/ether 3c:6d:66:cc:b3:b7 brd ff:ff:ff:ff:ff:ff
        inet **192.168.100.1**/24 brd 192.168.100.255 scope global noprefixroute enp1s0f1np1
          valid_lft forever preferred_lft forever
        inet6 fe80::3e6d:66ff:fecc:b3b7/64 scope link
          valid_lft forever preferred_lft forever
```

In this example, the IP address for Node 1 is **192.168.100.1**. Repeat the process for the other nodes.

On all nodes, run the following commands to enable passwordless SSH:

```bash
## Copy your SSH public key to all nodes. Replace the IP addresses with the ones you found in the previous step.
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 1>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 2>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 3>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 4>
```

## Step 6. Verify multi-node communication

Test basic multi-node functionality from the head node:

```bash
## Test hostname resolution across nodes
ssh <IP for Node 1> hostname
ssh <IP for Node 2> hostname
ssh <IP for Node 3> hostname
ssh <IP for Node 4> hostname
```

## Step 7. Running tests and workloads

Your cluster is now set up to run distributed workloads across four or more nodes. Next, use the NCCL multi-node GPU communication playbook to install and benchmark interconnect bandwidth. Adapt any two-node examples to all nodes in your cluster.

> [!NOTE]
> Wherever a related playbook asks to run a command on **two nodes**, run it on **all** nodes in your switch-connected cluster.
> Adapt the *mpirun* NCCL command on the **head node** to include every node.

Example mpirun command for NCCL:

```bash
## Set network interface environment variables (use your Up interface from the previous step)
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export OMPI_MCA_btl_tcp_if_include=enp1s0f1np1

## Run the all_gather performance test across four nodes (replace the IP addresses with the ones you found earlier)
mpirun -np 4 -H <IP for Node 1>:1,<IP for Node 2>:1,<IP for Node 3>:1,<IP for Node 4>:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf
```

## Step 8. Cleanup

> [!WARNING]
> These steps will reset network configuration.

```bash
## Rollback network configuration
sudo rm /etc/netplan/40-cx7.yaml
sudo netplan apply
```

> [!NOTE]
> If disconnecting the switch, make sure to do the following:
> 1. Re-enable auto-negotiation to avoid issues later if the switch is used for different purposes.
> 2. Remove the DHCP server configuration on the switch if you used that to assign IPs to the nodes.
> 3. If you created a new bridge, move the ports back to the default bridge and delete the new bridge.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Network unreachable" errors | Network interfaces not configured | Verify netplan config and `sudo netplan apply` |
| SSH authentication failures | SSH keys not properly distributed | Re-run `./discover-sparks` and enter passwords |
| Nodes not visible in cluster | Network connectivity issue | Verify QSFP cable connection, check IP configuration |
| "APT update" errors (for example, E: The list of sources could not be read.) | APT sources errors, conflicting sources, or signing keys | Check APT and Ubuntu documentation to fix the APT sources or keys conflicts |
| NCCL test failures (for example, libnccl.so.2: cannot open shared object file) | NCCL configuration not done on all nodes | Follow the NCCL multi-node GPU communication playbook to configure **all** nodes before running the NCCL test |

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
