# Connect Multiple DGX Sparks for Distributed Workloads

> Use NVIDIA Sync clustering so larger models and training can span nodes.

## Table of Contents

- [Overview](#overview)
- [Cable the Devices](#cable-the-devices)
  - [Two-device direct link](#two-device-direct-link)
  - [Three-device direct ring](#three-device-direct-ring)
  - [Four-device direct ring](#four-device-direct-ring)
  - [Switch-connected devices](#switch-connected-devices)
- [Configure with NVIDIA Sync](#configure-with-nvidia-sync)
- [Configure Manually](#configure-manually)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Some workloads need more compute and memory than a single DGX Spark can provide, but you can still run those workloads if you "cluster" two or more Sparks.
Clustering devices is the process of physically connecting them with high-speed cables (and potentially a switch) and then configuring a particular network across those cables.
You can't really do it over a wireless connection because the data will not move between devices quickly enough.

The primary component enabling a high-speed cluster across Sparks is the [ConnectX-7](https://resources.nvidia.com/en-us-accelerated-networking-resource-library/connectx-7-datasheet) network adapter that each Spark comes with.
The ConnectX-7 adapter, often called a "NIC", is a software and hardware layer that enables data transfer directly between memory across devices through [RDMA](https://blogs.nvidia.com/blog/what-is-rdma/).

If you physically cable two or more Sparks with [QSFP](https://en.wikipedia.org/wiki/Small_Form-factor_Pluggable) cables and properly configure the ConnectX-7 network, you can have high-speed super computing cluster on your desktop.

This playbook shows you two paths to configure the ConnectX-7 network across two to four DGX Spark devices.

- **Beginner:** Follow the Configure with NVIDIA Sync tab to set up and test the ConnectX-7 network ([see demo video](https://www.youtube.com/watch?v=MehBUQtb9qM)).
- **Advanced:** Experienced users can do things manually with the NVIDIA provided scripts in the Configure Manually tab.

Both paths start with following the Cable the Devices tab to make sure the devices are appropriately connected.

## What you'll accomplish

- You will physically connect the devices directly with QSFP cables, or a switch if needed. **Beginner - Advanced**
- You will use [NVIDIA Sync](https://docs.nvidia.com/sync/latest/cluster-assistant.html) to configure the ConnectX-7 network across the devices. **Beginner**
- Or, you will use commands and scripts to manually set up the ConnectX-7 network across the devices. **Advanced**

## What to know before starting

- How to [plug a QSFP cable into a DGX Spark](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html#plugging-in-a-qsfp-cable) and configure switch settings if required. **Beginner - Advanced**
- How to [set up a DGX Spark](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html) on a LAN. **Beginner - Advanced**
- How to create and edit `json` files. **Advanced** 
- How to interpret and run `bash` scripts. **Advanced**
- A basic grasp of network configuration and [Netplan files](https://netplan.readthedocs.io/en/stable/) and IP . **Advanced**
- A basic grasp of [ConnectX-7 networking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html). **Advanced**

## Supported hardware platforms

Check the table below to see if this playbook is for your hardware.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 64GB or 128GB Unified Memory | Direct or switch QSFP links | ✅ (200GbE QSFP) |

## Prerequisites

**Hardware requirements**

- Two to four DGX Sparks.
- The number of QSFP cables for your layout in Cable the Devices.
- (if using a switch) A switch with one 200 Gbit/s Ethernet link for each device. 

**Software requirements**

- Each Spark must be on the same LAN, and you must know its management IP address or mDNS name.
- You must have a user name and password with `sudo` privileges on each device.
- Each Spark is updated to the [April 2026 DGX OS release](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html#april-2026-release) or later.

## Ancillary files

These files are only needed for the manual path.
You can find them in [this playbook's assets folder](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets).

- [`spark_cluster_setup`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets/spark_cluster_setup) — network and SSH setup for the manual path

## Time & risk

- **Estimated time:** 10 minutes with NVIDIA Sync; 1 hour for manual setup
- **Risk level:** Low with NVIDIA Sync; medium with manual setup
- **Rollback:** Delete the cluster in NVIDIA Sync. For manual setup, follow the rollback steps in **Configure Manually**.
- **Last Updated:** 09/24/2026
  - Changes file paths and tightened focus on NVIDIA Sync.

## Cable the Devices

## Creating a cluster requires physical cables 

You can connect the devices directly with the QSFP cables, or you can use a switch and connect each device to the switch with one QSFP cable per device.

In either case, you will plug QSFP cables into the appropriate ports on the devices.

If you are using a switch, **do not** connect some devices directly and others through a switch.

## Step 1. Pick a cluster layout

The procedure changes based on the number of devices and whether you use a switch.

| Devices | Layout | Cables |
| --- | --- | --- |
| Two | Direct | One cable between the devices |
| Three | Direct ring | Three cables; each device links to the other two |
| Four | Direct ring | Four cables; each device links to the next device, and the last device links back to the first |
| Two or more | Switch | One cable and one 200 Gbit/s link from each device to the switch |

NVIDIA Sync and the provided helper support up to four devices through a switch. Larger switch clusters need manual configuration or your own automation.

## Step 2. Set up the devices and cables

1. Turn on each DGX Spark.
2. Make sure each device is on the same LAN ([see here](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html)).
3. Then use the [DGX Dashboard](https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html#using-dgx-dashboard-for-updates) to update each device to the current DGX Spark system software.
4. Use a supported QSFP112 DAC cable in Ethernet mode. See [QSFP ports and cables](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html#the-qsfp-ports-and-cables) for the approved cable list.
5. Place the devices within reach of the cables.

## Step 3. Connect your layout

See [Plugging in a QSFP Cable](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html#plugging-in-a-qsfp-cable) for an image.

Viewed from the back of a Spark, **port 0** is closest to the ordinary Ethernet port and **port 1** is farther away. Use the wiring table for your layout below.

**Use the following steps each time you plug in a QSFP cable.**

1. Turn each DGX Spark so that the back faces you.
2. Select the QSFP port specified by your layout's wiring table.
3. Hold the cable with its pull tab facing up.
4. Push the cable into the port until it is fully seated.

> [!WARNING]
> Do not force a cable into a port. If it does not slide in, stop and check the pull tab and port alignment.

### Two-device direct link

Connect one device to the other with a **single** QSFP cable.
Using two cables will **not** increase performance.

Choose **one** of the port correspondence options below to meet the NVIDIA provided helper's interface requirements used in the Configure Manually tab.

| Port Correspondence | Spark 1 | Spark 2 |
| --- | --- | --- |
| Port 0 on both | Port 0 | Port 0 |
| OR Port 1 on both | Port 1 | Port 1 |

### Three-device direct ring

Use three QSFP cables to make a ring. 
Each Spark will have **two** QSFP cables plugged into it.

Use the following cabling pattern to meet the NVIDIA provided helper's interface requirements used in the Configure Manually tab.

| Cable | From | To |
| --- | --- | --- |
| 1 | Spark 1, port 0 | Spark 2, port 1 |
| 2 | Spark 2, port 0 | Spark 3, port 1 |
| 3 | Spark 3, port 0 | Spark 1, port 1 |

### Four-device direct ring

Use four QSFP cables to make a ring.
Each Spark will have **two** QSFP cables plugged into it.
Connect port 0 of each Spark to port 1 of the next Spark.
The last cable connects Spark 4 back to Spark 1 and closes the ring.

Use this layout only with the Configure Manually tab.

| Cable | From | To |
| --- | --- | --- |
| 1 | Spark 1, port 0 | Spark 2, port 1 |
| 2 | Spark 2, port 0 | Spark 3, port 1 |
| 3 | Spark 3, port 0 | Spark 4, port 1 |
| 4 | Spark 4, port 0 | Spark 1, port 1 |

Spark 1 and Spark 3 are not connected directly. Spark 2 and Spark 4 are not connected directly.

### Switch-connected devices

Set up the switch before you configure the cluster:

1. Check that the switch, ports, and cables support 200 Gbit/s Ethernet links.
2. Set each Spark-facing port for 200 Gbit/s. A 400 Gbit/s port may need to be split into two 200 Gbit/s ports.
3. Put all Spark-facing ports on the same Layer 2 network. Depending on the switch, this may be a bridge or VLAN.
4. If the switch reports hardware-offload status, confirm that the Spark-facing ports use the switch chip instead of the switch CPU.
5. Keep the default MTU for switch connections.
6. Apply or save the switch configuration.

> [!NOTE]
> Changing how a high-speed port is split may restart other links that use the same port group. Set the port mode before you rely on those links.

Connect one cable from each Spark to a separate 200 Gbit/s switch connection. Choose port 0 on every Spark or port 1 on every Spark. The table uses port 1; replace it with port 0 in every row if you choose port 0.

| Device | Spark QSFP port | Switch connection |
| --- | --- | --- |
| Spark 1 | Port 1 | First Spark-facing connection |
| Spark 2 | Port 1 | Second Spark-facing connection |
| Spark 3, if present | Port 1 | Third Spark-facing connection |
| Spark 4, if present | Port 1 | Fourth Spark-facing connection |
| Each additional Spark | Port 1 | Another available 200 Gbit/s connection |

The switch connection names are examples; use the actual port or breakout connection names shown by your switch.

Check the connected devices:

1. Check the switch interface for each Spark.
2. Confirm that each link is up at 200 Gbit/s.
3. If a link is down or runs at the wrong speed, check the cable type, port mode, auto-negotiation, and FEC.

Switch settings and port names differ by maker and model. Follow the switch maker's guide for the exact steps. Do not use commands written for a different switch model.

For MikroTik CRS804 and CRS812 switches, see [MikroTik wired interface compatibility](https://help.mikrotik.com/docs/spaces/ROS/pages/220233794/MikroTik%2Bwired%2Binterface%2Bcompatibility) and [MikroTik bridge configuration](https://help.mikrotik.com/docs/spaces/ROS/pages/328068/Bridging%2Band%2BSwitching).

## Step 4. Move on to configure the network

Continue with tab Configure with NVIDIA Sync for a click-through, managed clustering process. 

Or, go under the hood with the Configure Manually tab to use NVIDIA provided helper scripts to set things up.

## Configure with NVIDIA Sync

## This is the recommended path for most users

If you are working with up to four devices, this is the recommended path.

You can see how it works in [this demo video](https://www.youtube.com/watch?v=MehBUQtb9qM).

If you want to cluster more than four devices, then follow and adapt the Configure Manually tab.

## Step 1. Make sure the devices are properly connected

**Follow the instructions in the Cable the Devices tab.**

## Step 2. Install NVIDIA Sync on your laptop

**Install NVIDIA Sync on your Windows, macOS, or Ubuntu laptop.**

::spark-download

- **Windows:** Open the `.exe` file and follow the setup steps.
- **macOS:** Open `nvidia-sync.dmg`, move NVIDIA Sync to the Applications folder, and open it.
- **Ubuntu:** Follow the [NVIDIA Sync install guide](https://docs.nvidia.com/sync/latest/getting-started.html#installation-and-onboarding).

## Step 3. Add each DGX Spark to NVIDIA Sync

**Make sure your laptop can reach each DGX Spark on the LAN.**

For each device:

1. Open NVIDIA Sync and select **Add New**.
2. Pick the device if its mDNS name appears. If it does not, select **Add device manually**.
3. Enter the device name or IP address, user name, and password.
4. Select **Add**.

**Success**: Each DGX Spark should now appear in NVIDIA Sync.

## Step 4. Start the NVIDIA Sync Cluster Assistant

1. Open the NVIDIA Sync **Settings** and select **Cluster Assistant**.
2. Then select **Add New Cluster** and give it a name.
3. Next, select the devices that you physically connected.

## Step 5. Resolve any issues NVIDIA Sync finds in the device and link checks

After you select the devices, NVIDIA Sync will check the devices and physical connections are appropriately set up. 

These checks are divided into stages.

- SSH access to each device
- The hardware and system software
- `sudo` privileges and password requirement
- User names and IDs across devices
- Link configuration across devices
- Existing compliant network plan on each device

**If a check fails, fix the issue and retry.**

1. NVIDIA Sync should already have SSH access because you added the devices
2. If one of the selected devices isn't a Spark, you must remove it
3. If any device needs a DGX OS update, you must update it and retry
4. If `sudo` requires a password on a device, you must enter it for temporary privileges
5. (optional) If the user name, ID and group ID is different across the devices, you may want to make them the same - not necessary but can simplify downstream workloads
6. If a cable is not properly connected, fix and retry
7. (optional) If the negotiated speed of each physical link is too slow, fix it and retry

**Success**: All checks pass and you are asked to confirm the network plan

## Step 6. NVIDIA Sync sets up inter-device SSH

Once the network plan is confirmed, NVIDIA Sync will set up key-based SSH with an alias between the devices on the ConnectX-7 network for process management across devices. 

This SSH **does not** use the LAN that you use for SSH connections to the devices. 
It goes across the physical connections.

This step can take a few minutes. 
If one device times out after five minutes, try the step again.

**Success**: You see the success screen with steps to move on to configuring an actual workload on the cluster.
The ConnectX-7 network and inter-device SSH are now ready.

## Step 7. Save the cluster details to a text file

**Save the cluster details to a text file when NVIDIA Sync shows the success page.**

1. Copy the network details shown on the success page. If your version of NVIDIA Sync shows a **Copy** button, select it.
2. Save the details in a file for later use.
3. To continue, use the example workloads in **Next steps** below. If your version of NVIDIA Sync shows **See Example Workloads**, you can select it instead.

## Next steps

- [Learn how to use the NVIDIA Sync cluster assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html#nvidia-sync-cluster-assistant)
- [Set up NCCL on the cluster](https://build.nvidia.com/playbooks/nccl)
- [Run a fine-tuning workload with PyTorch](https://build.nvidia.com/spark/multi-sparks-distributed-finetuning)
- [Set up vLLM on the cluster](https://build.nvidia.com/spark/vllm)

## Configure Manually

## Configure a DGX Spark cluster manually

You will use the `setup-cluster.sh` helper script from the [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) repository to configure the ConnectX-7 network and SSH access between DGX Sparks.

The same script covers every layout in Cable the Devices: two devices with a direct link, three devices in a ring, four devices in a ring, and devices through a switch.

The full reference for the script is [Automated cluster setup](https://github.com/eugr/spark-vllm-docker/blob/main/docs/NETWORKING.md#automated-cluster-setup) in that repository.

> [!NOTE]
> The script changes network, SSH, and Docker group settings on each device. If NVIDIA Sync has already configured your cluster successfully, continue to your workload playbook.

## Step 1. Physically connect the devices

**Follow the port wiring table for your layout in Cable the Devices.**

The cables (and switch) establish the physical links.
In the following steps, you will use the script to configure the network so the devices are reachable by other software and workloads over those links.

The cabling tab includes the port-to-port connections and switch preparation.
Complete those before you run the script.
The script does not configure the switch.
## Step 2. Check the requirements on every device

The script connects to every device over SSH on your management LAN. Each device must meet these requirements:

| Requirement | Details |
| --- | --- |
| Management IP address | Each device has an IPv4 address on your LAN that you can reach with SSH |
| User account | The same username exists on every device, and that account has `sudo` access |
| Software | Ubuntu Netplan with its Python `yaml` module, Python 3.10 or later, OpenSSH client and server, `ip`, `ping`, `ssh-keygen`, and `runuser` |
| ConnectX-7 interfaces | The interfaces use the standard DGX Spark names, and they are not renamed, bonded, bridged, or attached to a VLAN |

You do not need passwordless SSH before you start.
The script asks for the account password when it first needs it.
The script keeps the password in memory only. It does not write the password to a file.

The script does not install packages. If a package is missing, install it before you continue.

## Step 3. Pick a device in the cluster and download the script to it

You do **not** run the script from your laptop.
You **must** run it from one of the devices (or "nodes") that you connected in Step 1.

**First, choose one Spark to be the head node and open a terminal on it.**

The script sets up all of the nodes from the head node.
The head node is configured as part of the cluster too, so at the end of the process all of the nodes are ready to use.
For a ring, use Spark 1 from the wiring table as the head node.

Run the script as your normal login user. Do not run it with `sudo`.

**Then clone the repository to the head node and open its folder.**

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

Only the head node needs the repository. The script sends its node helper to the other devices over SSH.

| Files | Role |
| --- | --- |
| `setup-cluster.sh` | Entry point. You run it with the management IP addresses and options in Steps 4 and 5 |
| `setup-cluster.py` | Plans the network, sets up SSH, applies the configuration, and verifies it across the devices |
| `setup_cluster_node.py` | Runs on each device to inspect interfaces, write Netplan files, and save backups |

## Step 4. Preview the setup with a dry run

**Find the management IP address of each device.**

List the addresses with the head node first.
For a ring, list the devices in cable order: Spark 1, Spark 2, Spark 3, and then Spark 4 if present.
In a ring, port 0 of each device connects to port 1 of the next device. This matches the wiring table in Cable the Devices.

**Run the command for your layout with `--dry-run`.**

Replace the example addresses with your management IP addresses.

| Layout | Command |
| --- | --- |
| Two Sparks, direct | `./setup-cluster.sh 192.168.1.11 192.168.1.12 --topology direct --mtu 9000 --dry-run` |
| Three Sparks, ring | `./setup-cluster.sh 192.168.1.11 192.168.1.12 192.168.1.13 --topology mesh --mtu 1500 --dry-run` |
| Four Sparks, ring | `./setup-cluster.sh 192.168.1.11 192.168.1.12 192.168.1.13 192.168.1.14 --topology ring --mtu 1500 --dry-run` |
| Sparks through a switch | `./setup-cluster.sh 192.168.1.11 192.168.1.12 192.168.1.13 192.168.1.14 --topology switch --mtu 1500 --dry-run` |

Use `--mtu 9000` only for two Sparks with a direct link. Use `--mtu 1500` for all other layouts. These values match the MTU that NVIDIA Sync Cluster Assistant sets. If you leave out `--mtu`, the script uses 9000 for every layout.

The script uses the name `mesh` for the three-device ring, because three devices in a ring are also a full mesh.
The script uses the name `ring` for four devices in a ring.

If you list the ring devices in the wrong order, the verification step fails. If LLDP is already running on the devices, the script compares your order with the LLDP neighbors and rejects an order that does not match.

If you leave out `--topology`, the script tries to detect the layout from the active ports. With one active port on each device, the script selects `direct` for two devices and `switch` for three or more. The script cannot tell a two-device switch from a direct link, so give `--topology` for every layout.

The dry run checks SSH and `sudo` access, inspects the ConnectX-7 interfaces, and validates the planned Netplan configuration in a temporary folder. The dry run does not change the network or SSH settings.

**If a check fails, use the error information to fix the issue before you run the script again.**

**Success:** The output shows the planned interface addresses for each device and ends with `Dry run complete; no persistent configuration was changed.`

## Step 5. Run the setup

**Run the same command without `--dry-run`.**

For example, for two Sparks with a direct link:

```bash
./setup-cluster.sh 192.168.1.11 192.168.1.12 --topology direct --mtu 9000
```

The script prints the plan again and asks `Apply this cluster setup plan? [y/N]`. Review the plan, then type `y`.

If the script shows an unknown host key fingerprint, verify the fingerprint before you accept it.

`netplan apply` can briefly interrupt networking. Run the setup when the cluster is idle.

**The script makes these changes on every device:**

```text
Back up files → Create SSH keys → Add Docker group membership → Apply Netplan → Verify links and SSH → Save .env on the head node
```

| Change | Location |
| --- | --- |
| ConnectX-7 network settings, with static IPv4 addresses and the MTU you selected | `/etc/netplan/98-spark-vllm-docker.yaml` |
| A separate Ed25519 SSH key for each device | `~/.ssh/spark-vllm-cluster/` |
| SSH settings that select that key for the cluster addresses | A block at the top of `~/.ssh/config` |
| Public keys of all devices | `~/.ssh/authorized_keys` |
| Membership in the `docker` group | The current user on every device |
| Backups for the restore step | `/var/lib/spark-vllm-docker/setup-cluster/` on each device |
| Restore manifest | `~/.local/state/spark-vllm-docker/cluster-setup.json` on the head node |
| Launch configuration for spark-vllm-docker | `.env` in the repository folder on the head node |

The script also moves existing ConnectX-7 entries out of other files in `/etc/netplan` into its own file, so Netplan does not combine old addresses with the new ones. The script backs up those files first.

If setup fails or you press Ctrl-C, the script tries to roll back the changes on every device.

The script assigns `/24` subnets from `10.20.0.0/16` and gives each device the host numbers `.11`, `.12`, and so on, in the order you listed the devices.
We recommend `/24` subnets for the ConnectX-7 network. NCCL uses `/24` as the default subnet size for subnet-aware routing. If you plan the network yourself, also use `/24` subnets.
Each QSFP port has two Ethernet interfaces, one for each PCIe path, so each link uses two subnets.

| Layout | Subnet pattern |
| --- | --- |
| Two Sparks, direct | Two subnets across the single cable |
| Three Sparks, ring | Two subnets per cable; six across the ring |
| Four Sparks, ring | Two subnets per cable; eight across the ring |
| Sparks through a switch | Two shared subnets across the cluster; each Spark gets one address in each subnet |

See [Understand the ConnectX-7 Network](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html#understand-the-connectx-7-network) for the hardware explanation, subnet design, and addressing details.

For other options, such as another subnet range or port selection, see [Automated cluster setup](https://github.com/eugr/spark-vllm-docker/blob/main/docs/NETWORKING.md#automated-cluster-setup).

**Success:** The script prints `Cluster networking and mutual passwordless SSH are configured and verified.`

Before it prints this message, the script checks ping on both interface paths at the full MTU, in both directions. The script also checks passwordless SSH from every device to every other device.

## Step 6. Log in again and verify the result

**Open a new login session on each device.**

The `docker` group change applies only to new login sessions. Existing SSH sessions keep their old groups.

**Run the doctor check from the head node.**

```bash
./setup-cluster.sh --doctor --dry-run
```

The doctor check confirms the saved files, ConnectX-7 addresses, routes, MTU, links, passwordless SSH, the launch configuration, and Docker access. The command returns a nonzero exit status if any issue remains.

**Success:** The output ends with `Doctor: all saved configuration, CX7 links, SSH, and Docker checks passed.`

To repair the issues that the doctor check finds, run `./setup-cluster.sh --doctor` without `--dry-run`. The script lists the repairs and asks before it applies them.

## Next steps

Open the distributed workload playbook you want to use. It provides the application configuration and performance tests for the network you just established.

## Roll back the setup

**Run the restore command on the same head node, as the same user.**

```bash
./setup-cluster.sh --restore
```

Restore uses the manifest on the head node and the backups on each device. Keep these files until restore is complete. Every device must be reachable over the management LAN.

Restore puts back the original Netplan, SSH, and `.env` files with their original permissions. Restore also removes the SSH keys and files that the script created, and reapplies the previous network configuration. Restore removes the `docker` group membership only if the script added it.

If you edited a file that the script manages after setup, restore stops and names that file. Reconcile your edits, then run restore again.

To change the layout, the head node, or the addresses, run restore first. Then run the setup again.

For a switch, also restore any port, bridge or VLAN, DHCP, link speed, or MTU settings you changed on the switch.

## Troubleshooting

## Connection and NVIDIA Sync issues

| Symptom | Cause | Fix |
| --- | --- | --- |
| A device does not appear in NVIDIA Sync | The device is off, on another network, or cannot accept SSH connections | Turn on the device. Make sure it and the computer that runs NVIDIA Sync are on the same local network. Test direct SSH access. |
| The GB10 check fails | The selected system is not a DGX Spark or GB10 device | Remove the unsupported system from the cluster. Cluster Assistant supports only DGX Spark and GB10 devices. |
| The software check fails | A device has an old system release | Update each device. Cluster Assistant requires the April 2026 system release or later. |
| The password check fails | NVIDIA Sync cannot use `sudo` on a device | Select **Fix Now** and enter the device password. Check that the user has `sudo` access. |
| Cluster Assistant finds the wrong layout | A cable is loose, the layout is not supported, or the Spark-facing switch ports are not on the same Layer 2 network | Reseat each cable and compare the links with **Connect the Devices**. Do not mix direct and switch links. For a switch, check that every Spark-facing port is up and on the same Layer 2 network. Then run the layout check again. |
| A switch link is down or is not 200 Gbit/s | The cable is not supported, the switch port uses the wrong mode, or the two ends do not agree on link settings | Check the cable and port mode. A 400 Gbit/s port may need to be split into two 200 Gbit/s ports. If the link stays down, check auto-negotiation and FEC in the switch maker's guide. |
| The cluster connects but data transfer is slow | A link can report 200 Gbit/s while the switch sends traffic through its CPU or the link records errors | Check hardware-offload status, switch CPU use, and port error counters. Test traffic in both directions. If the issue remains, contact the switch maker or NVIDIA support. |
| SSH setup times out | One device took more than five minutes | Retry the SSH step in Cluster Assistant. |

For more Cluster Assistant help, see the [NVIDIA Sync troubleshooting guide](https://docs.nvidia.com/sync/latest/cluster-assistant.html#troubleshooting).

## Manual setup issues

| Symptom | Cause | Fix |
| --- | --- | --- |
| The manual pre-check fails | A management IP, SSH login, password, or `sudo` setting is wrong | Fix the value in `config/my-cluster.json`, test SSH to each management IP, and run `--pre-validate-only` again. |
| No ConnectX-7 interface is up | A cable or port is not active | Reseat the cables, confirm the layout, reboot the devices, and run `ibdev2netdev` again. |
