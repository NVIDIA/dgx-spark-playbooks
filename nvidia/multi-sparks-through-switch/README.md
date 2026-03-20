# Connect Multiple DGX Spark through a Switch

> Set up a cluster of DGX Spark devices that are connected through Switch

## Table of Contents

- [Overview](#overview)
- [Run on Four Sparks](#run-on-four-sparks)
  - [Step 3.1. Verify negotiated Link speed](#step-31-verify-negotiated-link-speed)
  - [4.1 Script for Cluster networking configuration](#41-script-for-cluster-networking-configuration)
  - [4.2 Manual Cluster networking configuration](#42-manual-cluster-networking-configuration)
  - [Option 1: Automatically configure SSH](#option-1-automatically-configure-ssh)
  - [Option 2: Manually discover and configure SSH](#option-2-manually-discover-and-configure-ssh)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Configure four DGX Spark systems for high-speed inter-node communication using 200Gbps QSFP connections through a QSFP switch. This setup enables distributed workloads across multiple DGX Spark nodes by establishing network connectivity and configuring SSH authentication.

## What you will accomplish

In this playbook, you will physically connect four DGX Spark devices with QSFP cables and a QSFP switch, configure network interfaces for cluster communication, and establish passwordless SSH between nodes to create a functional distributed computing environment. The same set up can be scaled up to more DGX Spark devices connected via the same switch.

## What to know before starting

- Basic understanding of distributed computing concepts
- Working with network interface configuration and netplan
- Experience with SSH key management
- Basic understanding and experience in configuring the managed QSFP network switch which you plan to use. Refer to the instruction manuals to:
  - Know how to connect to the switch for management of ports and features
  - Know how to enable/disable QSFP ports and create a software bridge on the switch
  - Know how to configure the link speed manually on the port and disable auto-negotiation if needed

## Prerequisites

- Four DGX Spark systems (these instructions will work for any number of DGX Spark devices connected with a switch)
- QSFP switch with at least 4 QSFP56-DD ports (at least 200Gbps each)
- QSFP cables for 200Gbps connection from the switch to the devices. Use [recommended cable](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/qsfp-cable-0-4m-for-dgx-spark/) or similar.
  - One cable per spark
  - If the switch has 400Gbps ports then you can also use breakout cables to split them into two 200Gbps ports
- SSH access available to all systems
- Root or sudo access on all systems: `sudo whoami`
- The same username on all systems
- Update all systems to the latest OS and Firmware. Refer to the DGX Spark documentation https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html

## Ancillary files

All required files for this playbook can be found [here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/multi-sparks-through-switch/)

- [**discover-sparks.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) script for automatic node discovery and SSH key distribution
- [**Cluster setup script**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup) for automatic network configuration, validation and running NCCL sanity test

## Time & risk

- **Duration:** 2 hours including validation

- **Risk level:** Medium - involves network reconfiguration

- **Rollback:** Network changes can be reversed by removing netplan configs or IP assignments

- **Last Updated:** 3/19/2026
  * First publication

## Run on Four Sparks

## Step 1. Ensure Same Username on all four Systems

On all four systems check and make sure the usernames are the same:

```bash
## Check current username
whoami
```

If usernames don't match, create a new user (e.g., nvidia) on all four systems and login in with the new user:

```bash
## Create nvidia user and add to sudo group
sudo useradd -m nvidia
sudo usermod -aG sudo nvidia

## Set password for nvidia user
sudo passwd nvidia

## Switch to nvidia user
su - nvidia
```

## Step 2. Switch Management

Most QSFP switches offer some form of management interface, either through CLI or UI. Refer to the documentation and connect to the management interface. Make sure that the ports on the switch are enabled. For connecting four sparks, you will need to ensure that the switch is configured to provide 200Gbps connection to each DGX Spark. If not done already, refer to the [Overview](https://build.nvidia.com/spark/multi-sparks-through-switch/overview) of this playbook for the prior knowledge and pre-requisites required for this playbook.

## Step 3. Physical Hardware Connection

Connect the QSFP cables between DGX Spark systems and the switch(QSFP56-DD/QSFP56 ports) using one CX7 port on each Spark system. It is recommended to use the same CX7 port on all Spark systems for easier network configuration and avoiding NCCL test failures. In this playbook the second port (the one further from the ethernet port) is used. This should establish the 200Gbps connection required for high-speed inter-node communication. You will see an output like the one below on all four sparks. In this example the interfaces showing as 'Up' are **enp1s0f1np1** and **enP2p1s0f1np1** (each physical port has two logical interfaces).

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

We have created a script [here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup) which automates the following:
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
cd dgx-spark-playbooks/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup

## Check the README.md in the script directory for steps to run the script and configure the cluster networking with "--run-setup" argument
```

### 4.2 Manual Cluster networking configuration

In this case, you can choose one of the options to assign the IPs to the CX7 logical interfaces. Options 1, 2 and 3 are mutually exclusive.
1. DHCP server on the switch (recommended, if it is supported)
2. Link local IP addressing (netplan is the same across all nodes)
3. Manual IP addressing (netplan will be different on each node but provides more control and deterministic IPs)

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

#### Option 2: Automatic Link local IP Assignment

Configure network interfaces using netplan on all DGX Spark nodes for automatic link-local addressing:

```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      link-local: [ ipv4 ]
    enP2p1s0f1np1:
      link-local: [ ipv4 ]
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

#### Option 3: Manual IP Assignment with the netplan configuration file

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
        - 192.168.100.11/24
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
        - 192.168.100.12/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.100.13/24
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
        - 192.168.100.14/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.100.15/24
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
        - 192.168.100.16/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.100.17/24
      dhcp4: no
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

## Step 5. Set up passwordless SSH authentication

### Option 1: Automatically configure SSH

Run the DGX Spark [**discover-sparks.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) script from one of the nodes to automatically discover and configure SSH:

```bash
curl -O https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks
bash ./discover-sparks
```

Expected output similar to the below, with different IPs and node names. You may see up to two IPs for each node as two interfaces (eg. **enp1s0f1np1** and **enP2p1s0f1np1**) have IP addresses assigned. This is expected and does not cause any issues. The first time you run the script, you'll be prompted for your password for each node.
```
Found: 169.254.35.62 (dgx-spark-1.local)
Found: 169.254.35.63 (dgx-spark-2.local)
Found: 169.254.35.64 (dgx-spark-3.local)
Found: 169.254.35.65 (dgx-spark-4.local)

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
        inet **169.254.35.62**/16 brd 169.254.255.255 scope link noprefixroute enp1s0f1np1
          valid_lft forever preferred_lft forever
        inet6 fe80::3e6d:66ff:fecc:b3b7/64 scope link
          valid_lft forever preferred_lft forever
```

In this example, the IP address for Node 1 is **169.254.35.62**. Repeat the process for other nodes.

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

Now your cluster is set up to run distributed workloads across four nodes. Try running the [NCCL playbook](https://build.nvidia.com/spark/nccl/stacked-sparks).

> [!NOTE]
> Wherever the playbook asks to run a command on **two nodes**, just run it on **all four nodes**.
> Make sure to adapt the *mpirun* NCCL command which you run on the **head node** to accommodate **four nodes**

Example mpirun command for NCCL:
```bash
## Set network interface environment variables (use your Up interface from the previous step)
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export OMPI_MCA_btl_tcp_if_include=enp1s0f1np1

## Run the all_gather performance test across four nodes (replace the IP addresses with the ones you found in the previous step)
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

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Network unreachable" errors | Network interfaces not configured | Verify netplan config and `sudo netplan apply` |
| SSH authentication failures | SSH keys not properly distributed | Re-run `./discover-sparks` and enter passwords |
| Nodes not visible in cluster | Network connectivity issue | Verify QSFP cable connection, check IP configuration |
| "APT update" errors (eg. E: The list of sources could not be read.) | APT sources errors, conflicting sources or signing keys | Check APT and Ubuntu documentation to fix the APT sources or keys conflicts |
| NCCL test failures (eg. libnccl.so.2: cannot open shared object file) | NCCL configuration not done on all nodes | Make sure to follow the NCCL playbook to configure **all** nodes before running the NCCL test|
