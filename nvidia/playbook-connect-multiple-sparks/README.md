# Connect Multiple DGX Sparks for Distributed Workloads

> Use NVIDIA Sync clustering so larger models and training can span nodes.

## Table of Contents

- [Overview](#overview)
- [Cable the Devices](#cable-the-devices)
  - [Two-device direct link](#two-device-direct-link)
  - [Three-device direct ring](#three-device-direct-ring)
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

- **Beginner:** Follow the [Configure with NVIDIA Sync](beginner.md) tab to set up and test the ConnectX-7 network ([see demo video](https://www.youtube.com/watch?v=MehBUQtb9qM)).
- **Advanced:** Experienced users can do things manually with the NVIDIA provided scripts in the [Configure Manually](advanced.md) tab.

Both paths start with following the [Cable the Devices](cable-devices.md) tab to make sure the devices are appropriately connected.

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
| **DGX Spark** | DGX OS (Linux) | 64 GB or 128 GB Unified Memory | Direct or switch QSFP links | ✅ (200GbE QSFP) |

## Prerequisites

**Hardware requirements**

- Two to four DGX Sparks.
- The number of QSFP cables for your layout in [Cable the Devices](cable-devices.md).
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

Choose **one** of the port correspondence options below to meet the NVIDIA provided helper's interface requirements used in the [Configure Manually](advanced.md) tab.

| Port Correspondence | Spark 1 | Spark 2 |
| --- | --- | --- |
| Port 0 on both | Port 0 | Port 0 |
| OR Port 1 on both | Port 1 | Port 1 |

### Three-device direct ring

Use three QSFP cables to make a ring. 
Each Spark will have **two** QSFP cables plugged into it.

Use the following cabling pattern to meet the NVIDIA provided helper's interface requirements used in the [Configure Manually](advanced.md) tab.

| Cable | From | To |
| --- | --- | --- |
| 1 | Spark 1, port 0 | Spark 2, port 1 |
| 2 | Spark 2, port 0 | Spark 3, port 1 |
| 3 | Spark 3, port 0 | Spark 1, port 1 |


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

Continue with tab [Configure with NVIDIA Sync](beginner.md) for a click-through, managed clustering process. 

Or, go under the hood with the [Configure Manually](advanced.md) tab to use NVIDIA provided helper scripts to set things up.

## Configure with NVIDIA Sync

## This is the recommended path for most users

If you are working with up to four devices, this is the recommended path.

You can see how it works in [this demo video](https://www.youtube.com/watch?v=MehBUQtb9qM).

If you want to cluster more than four devices, then follow and adapt the [Configure Manually](manual.md) tab.

## Step 1. Make sure the devices are properly connected

**Follow the instructions in the [Connect the Devices](connect-devices.md) tab.**

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

1. Select **Copy** to copy the network details.
2. Save the details in a file for later use.
3. Select **See Example Workloads**.

## Next steps

- [Learn how to use the NVIDIA Sync cluster assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html#nvidia-sync-cluster-assistant)
- [Set up NCCL on the cluster](https://build.nvidia.com/playbooks/nccl)
- [Run a fine-tuning workload with PyTorch](https://build.nvidia.com/spark/multi-sparks-distributed-finetuning)
- [Set up vLLM on the cluster](https://build.nvidia.com/spark/vllm)

## Configure Manually

## Configure a DGX Spark cluster manually

You will use the NVIDIA provided cluster helper scripts to configure the ConnectX-7 network and SSH access between DGX Sparks.

You will follow thethe basic procedure and the choices that depend on your layout, so you can generalize the process to a larger cluster.

The helper supports two to four devices. For more than four devices through a switch, use the same procedure with manual configuration or your own automation.

> [!NOTE]
> The helper changes network and SSH settings on each device. If NVIDIA Sync has already configured your cluster successfully, continue to your workload playbook.

## Step 1. Physically connect the devices

**Follow the port wiring table for your layout in [Cable the Devices](cable-devices.md).**

The cables (and switch) establish the physical links. 
In the following steps, you will use the helper scripts to configure the network so the devices are reachable by other software and workloads over those links.

The cabling tab includes the port-to-port connections and switch preparation. 
Complete those before checking the interfaces in Step 4.

## Step 2. Pick a device in the cluster and download the helper scripts to it

You do **not** run the helper script from your laptop.
You **must** run it from one of the devices (or "nodes") that you connected above in Step 1. 

**First, choose one Spark from which to set up the cluster and open a terminal on it.**

The helper script will do it's work no matter which node you pick, and it will get all of the nodes into the correct network configuration per the   

is built to work from one of the devices in the cluster nodes in the cluster to set things up. get all of the devices (or "nodes") into the same shape relative to the other nodes. 

No matter which node you pick, the helper script will also configure it to be part of the cluster, and at the end of the process all of the nodes will be

**Then clone the NVIDIA playbook repository to that device, open the helper folder and inspect the files.**

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-connect-multiple-sparks/assets/spark_cluster_setup
```

If you inspect the individual files, you can piece together the overall process of how the ConnectX-7 network can be configured in more general situations than those shown here.

| Files | Role | Behavior |
| --- | --- | --- |
| `spark_cluster_setup.sh` | Entry point for running checks or setup | You run it with a configuration file and a check or setup option in Steps 4 and 5 |
| `spark_cluster_setup.py` | Coordinates checks, network setup, and SSH access across the devices | Invoked by `spark_cluster_setup.sh` |
| `node_scripts/detect_and_configure_cluster_networking.py` | Discovers connections and configures network addresses on each device | Copied to each device and invoked by `spark_cluster_setup.py` during setup |
| `config/*` | JSON files specifying device SSH information on the LAN | Used by `spark_cluster_setup.py` for `sudo` privileges and setting up inter-device SSH |

## Step 3. Edit the appropriate JSON file to enable inter-device SSH 

The helper script inspects all devices to verify the cabled layout and coordinate the intterfaces, IP address and network plans for implementation on each device. 
In addition, inter-device communications need to be set up with key based SSH access to avoid requiring a password.
Finally, implementing a network configuration requires `sudo` permissions on each device. 

So, you need to give the script the IP addresses, usernames and related passwords.   

**First, copy the appropriate JSON template for your device count.**

1. Selet the appropriate JSON template from the table below, `<cluster-count.json>`
2. Copy the template to a new file, `config/my-cluster.json`
3. Change the permissions on the file so it's visible only to the user on the head node: chmod 600 config/my-cluster.json

| Number of nodes | Sample file |
| --- | --- |
| Two devices, direct or switch | `config/spark_config_b2b.json` |
| Three devices, ring or switch | `config/spark_config_ring.json` |
| Four devices through a switch | `config/spark_config_switch.json` |

**Then, edit the copy to include every device to be clustered, including the head node.**

1. Open the file with a file editor like `vim`
2. Edit the individual entries with the appropriate information from the table below
3. Make sure to have correct entries for each device, including the head node
4. Finally, save the file

| Field | Value to supply |
| --- | --- |
| `ip_address` | The device's existing management IP address on your LAN |
| `port` | Its SSH port, normally `22` |
| `user` | A login account with sudo access |
| `password` | The password for that account |

> [!WARNING]
> The JSON file stores passwords as plain text. Delete it when setup completes.

## Step 4. Run the setup script to inspect the ConnectX-7 interfaces, links and link speeds. 

The helper uses the login details in your JSON file to check each device before changing the network:

```text
Check LAN SSH and sudo access → Check active ConnectX-7 interfaces → Check link speeds
```

`spark_cluster_setup.py` coordinates these checks from the node where you run the helper. Pre-validation does not assign cluster addresses or change SSH settings.

**Run the setup script with the pre-validation flag.** 

```bash
bash spark_cluster_setup.sh -c config/my-cluster.json --pre-validate-only
```

**If a check fails, use the error information to troubleshoot the issue before re-running the script.**

On failure, the script will output an error with relevant information for you to troubleshoot.

**Success:** The output will print `Pre-setup validations completed successfully.`

## Step 5. Back up the existing network implementation and create a new one to include the ConnectX-7 interfaces


The helper repeats the checks from Step 4, then runs `detect_and_configure_cluster_networking.py` on each device. That script discovers the connected peers and turns the chosen addresses into a saved network configuration:

```text
Discover connections → Choose addresses → Build Netplan YAML → Save the file → Apply it
```

Netplan is how Linux reads and applies the saved interface and address settings. After the network scripts finish, the coordinator checks cluster connectivity and sets up key-based SSH between devices. It uses the LAN connections to coordinate the process throughout.

**Back up existing network configuration on every device before applying changes.**

The network script writes `/etc/netplan/40-cx7.yaml` on each device and replaces that file if it already exists. Preserve the existing file and inspect other Netplan files configuring the same interfaces, since Netplan combines their settings. See [Inspect and Verify a ConnectX-7 Cluster Network Plan](https://docs.nvidia.com/sync/latest/cluster-network-inspection.html) for how to inspect the saved configuration and compare it with the active network.


**Run the helper from its own folder:**

```bash
bash spark_cluster_setup.sh -c config/my-cluster.json --run-setup
```

**The subnet plan follows the layout.**

For these configurations, each connected QSFP port needs two subnets, one for each Ethernet interface representing a PCIe path. Corresponding interfaces on connected peers must share the appropriate subnet.

| Layout | Subnet pattern | What changes as you adapt the setup |
| --- | --- | --- |
| Two Sparks, direct | Two subnets across the single cable | Match each subnet between the corresponding interfaces on both Sparks |
| Three Sparks, ring | Two subnets per cable; six across the ring | Match assignments to the actual cable endpoints; each Spark participates in four subnets |
| Sparks through a switch | Two shared subnets across the cluster | Give each additional Spark a unique address in each subnet |

See [Understand the ConnectX-7 Network](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html#understand-the-connectx-7-network) for the hardware explanation, subnet design, and addressing details.

**For a larger switch cluster, extend the address plan across all nodes.**

Adding a Spark requires another address in each of the two shared subnets, rather than another subnet pair. Choose subnets with enough available addresses for the planned cluster and room for expansion. Keep them separate from your management LAN and other existing networks.

Apply the matching interface and address settings on every node using Netplan or your own automation. Confirm the switch has enough suitable ports and capacity. Then establish passwordless SSH between the accounts that will run your workloads. A common username simplifies configuration; otherwise specify the destination user explicitly.

For helper setup, the scripts create or reuse `~/.ssh/id_ed25519_shared`, distribute that shared key, and update SSH configuration. For your own setup, you can instead authorize each node's public key on its peers with `ssh-copy-id`. Configure access in every required direction.

**Verify the resulting network on every node.**

Compare the Netplan configuration with the active interfaces and addresses, then test the peers expected for your topology.

| Check | Command | Expected result |
| --- | --- | --- |
| Active links | `ip -br link` | Connected interfaces show `LOWER_UP`; unused ports may show `NO-CARRIER` |
| Assigned addresses | `ip -br addr` | Both interfaces of each connected port have addresses in their respective planned subnets |
| Route to a peer | `ip route get <peer-cluster-IP>` | The route selects the intended ConnectX-7 interface |
| Communication on each path | `ping -I <interface> -c 3 <peer-cluster-IP>` | The corresponding peer responds through that interface |
| Passwordless SSH | `ssh -o BatchMode=yes <user>@<peer-cluster-IP> hostname` | The expected hostname prints without a password prompt |

Verify and accept each peer's host key during an initial interactive SSH connection before using `BatchMode=yes`.

Use cluster addresses for these checks. In a ring, choose the peer address on the subnet shared by the source and that peer; not every address on a peer is reachable through every link.

See [Inspect and Verify a ConnectX-7 Cluster Network Plan](https://docs.nvidia.com/sync/latest/cluster-network-inspection.html) for how to compare configured and active settings and identify the expected peers. For helper setup, inspect `40-cx7.yaml` rather than the NVIDIA Sync filename used in that guide.

**Success:** The helper prints `Spark cluster setup completed successfully.`, and the checks confirm the subnet plan, communication over both interface paths, and passwordless SSH between the required accounts.

## Step 6. Remove the password file

**After setup works, delete the JSON file containing the passwords:**

```bash
rm config/my-cluster.json
```

If you used your own automation, remove any temporary credential files it created.

## Next steps

Open the distributed workload playbook you want to use. It provides the application configuration and performance tests for the network you just established.

## Roll back the manual setup

**Inspect the active configuration before restoring the previous settings.**

Follow [Inspect and Verify a ConnectX-7 Cluster Network Plan](https://docs.nvidia.com/sync/latest/cluster-network-inspection.html#remove-the-connectx-7-netplan-configuration-on-an-individual-node), using the filename your setup created. The helper uses `/etc/netplan/40-cx7.yaml`; NVIDIA Sync uses `/etc/netplan/99-nvidia-sync-cluster.yaml`.

Restore a saved configuration if setup replaced an existing file. If the cluster file was newly created, move it out of `/etc/netplan`, then regenerate and try the restored configuration as described in the documentation. Verify management access and confirm the cluster addresses have been removed.

Network rollback does not remove SSH changes. Remove only the key authorizations and SSH settings added for this cluster. For a switch, also restore any port, bridge or VLAN, DHCP, link speed, or MTU settings you changed.

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
