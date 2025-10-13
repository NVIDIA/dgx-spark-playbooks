# Connect Two Sparks

> Connect two Spark devices and setup them up for inference and fine-tuning

## Table of Contents

- [Overview](#overview)
- [Run on Two Sparks](#run-on-two-sparks)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Configure two DGX Spark systems for high-speed inter-node communication using 200GbE direct
QSFP connections. This setup enables distributed workloads across multiple DGX Spark nodes
by establishing network connectivity and configuring SSH authentication.

## What you'll accomplish

You will physically connect two DGX Spark devices with a QSFP cable, configure network
interfaces for cluster communication, and establish passwordless SSH between nodes to create
a functional distributed computing environment.

## What to know before starting

- Basic understanding of distributed computing concepts
- Working with network interface configuration and netplan
- Experience with SSH key management

## Prerequisites

- Two DGX Spark systems
- One QSFP cable for direct 200GbE connection between two devices
- SSH access available to both systems
- Root or sudo access on both systems: `sudo whoami`
- The same username on both systems

## Ancillary files

All required files for this playbook can be found [here on GitHub](https://gitlab.com/nvidia/dgx-spark/temp-external-playbook-assets/dgx-spark-playbook-assets/-/blob/main/nvidia/connect-two-sparks/)

- [**discover-sparks.sh**](https://gitlab.com/nvidia/dgx-spark/temp-external-playbook-assets/dgx-spark-playbook-assets/-/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) script for automatic node discovery and SSH key distribution

## Time & risk

- **Duration:** 1 hour including validation

- **Risk level:** Medium - involves network reconfiguration

- **Rollback:** Network changes can be reversed by removing netplan configs or IP assignments

## Run on Two Sparks

## Step 1. Ensure Same Username on Both Systems

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

## Step 2. Physical Hardware Connection

Connect the QSFP cable between both DGX Spark systems using any QSFP interface
on each device. This establishes the 200GbE direct connection required for high-speed
inter-node communication. Upon connection between the two nodes, you will see the an output like the one below: in this example the interface showing as 'Up' is **enp1s0f1np1** / **enP2p1s0f1np1** (each physical port has two names).

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
> The interface showing as 'Up' depends on which port you are using to connect the two nodes. Each physical port has two names, for example, enp1s0f1np1 and enP2p1s0f1np1 refer to the same physical port. Please disregard enP2p1s0f0np0 and enP2p1s0f1np1, and use enp1s0f0np0 and enp1s0f1np1 only.

## Step 3. Network Interface Configuration

Choose one option to setup the network interfaces. Option 1 and 2 are mutually exclusive.

**Option 1: Automatic IP Assignment (Recommended)**

Configure network interfaces using netplan on both DGX Spark nodes for automatic
link-local addressing:

```bash
## Create the netplan configuration file
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      link-local: [ ipv4 ]
    enp1s0f1np1:
      link-local: [ ipv4 ]
EOF

## Set appropriate permissions
sudo chmod 600 /etc/netplan/40-cx7.yaml

## Apply the configuration
sudo netplan apply
```

> [!NOTE]
> Using this option, the IPs assigned to the interfaces will change if you reboot the system.

**Option 2: Manual IP Assignment (Advanced)**

First, identify which network ports are available and up:

```bash
## Check network port status
ibdev2netdev
```

Example output:
```
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
```

Use an interface that shows as "(Up)" in your output. In this example, we'll use **enp1s0f1np1**. You can disregard interfaces starting with the prefix`enP2p<...>` and only use interfaces starting with `enp1<...>` instead.

On Node 1:
```bash
## Assign static IP and bring up interface.
sudo ip addr add 192.168.100.10/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
```

Repeat the same process for Node 2, but using IP **192.168.100.11/24**. Ensure to use the correct interface name using `ibdev2netdev` command.
```bash
## Assign static IP and bring up interface.
sudo ip addr add 192.168.100.11/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
```

You can verify the IP assignment on both nodes by running the following command on each node:
```bash
## Replace enp1s0f1np1 with the interface showing as "(Up)" in your output, either enp1s0f0np0 or enp1s0f1np1
ip addr show enp1s0f1np1
```

## Step 3. Set up passwordless SSH authentication

#### Option 1: Automatically configure SSH

Run the DGX Spark [**discover-sparks.sh**](https://gitlab.com/nvidia/dgx-spark/temp-external-playbook-assets/dgx-spark-playbook-assets/-/blob/main/nvidia/connect-two-sparks/assets/discover-sparks) script from one of the nodes to automatically discover and configure SSH:

```bash
bash ./discover-sparks
```

Expected output similar to the below, with different IPs and node names. The first time you run the script, you'll be prompted for your password for each node.
```
Found: 169.254.35.62 (dgx-spark-1.local)
Found: 169.254.35.63 (dgx-spark-2.local)

Setting up bidirectional SSH access (local <-> remote nodes)...
You may be prompted for your password for each node.

SSH setup complete! Both local and remote nodes can now SSH to each other without passwords.
```

> [!NOTE]
> If you encounter any errors, please follow Option 2 below to manually configure SSH and debug the issue.

#### Option 2: Manually discover and configure SSH

You will need to find the IP addresses for the CX-7 interfaces that are up. On both nodes, run the following command to find the IP addresses and take note of them for the next step.
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
        inet **169.254.35.62**/16 brd 169.254.255.255 scope link noprefixroute enp1s0f1np1
          valid_lft forever preferred_lft forever
        inet6 fe80::3e6d:66ff:fecc:b3b7/64 scope link
          valid_lft forever preferred_lft forever
```

In this example, the IP address for Node 1 is **169.254.35.62**. Repeat the process for Node 2.

On both nodes, run the following commands to enable passwordless SSH:
```bash
## Copy your SSH public key to both nodes. Please replace the IP addresses with the ones you found in the previous step.
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 1>
ssh-copy-id -i ~/.ssh/id_rsa.pub <username>@<IP for Node 2>
```

## Step 4. Verify Multi-Node Communication

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
## Rollback network configuration (if using Option 1)
sudo rm /etc/netplan/40-cx7.yaml
sudo netplan apply

## Rollback network configuration (if using Option 2)
sudo ip addr del 192.168.100.10/24 dev enp1s0f0np0  # Adjust the interface name to the one you used in step 3.
sudo ip addr del 192.168.100.11/24 dev enp1s0f0np0  # Adjust the interface name to the one you used in step 3.
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Network unreachable" errors | Network interfaces not configured | Verify netplan config and `sudo netplan apply` |
| SSH authentication failures | SSH keys not properly distributed | Re-run `./discover-sparks` and enter passwords |
| Node 2 not visible in cluster | Network connectivity issue | Verify QSFP cable connection, check IP configuration |
