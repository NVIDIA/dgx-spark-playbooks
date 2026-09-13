# Connect Multiple DGX Sparks for Distributed Workloads

> Stacked, ring, or switch topologies for high-speed multi-node jobs

## Table of Contents

- [Overview](#overview)
- [Connect the Devices](#connect-the-devices)
  - [Two-device direct link](#two-device-direct-link)
  - [Three-device direct ring](#three-device-direct-ring)
  - [Two-to-four-device switch](#two-to-four-device-switch)
- [Configure with NVIDIA Sync](#configure-with-nvidia-sync)
- [Configure Manually](#configure-manually)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

You can connect two or more DGX Spark devices into a high-speed cluster to run workloads that won't fit on a single device. Configuring the cluster takes more than just plugging in cables. Setting up the DGX Spark ConnectX-7 network by hand has many steps and can be confusing.

This playbook shows you how to create a cluster of two to four DGX Spark devices with the NVIDIA Sync Cluster Assistant ([see demo video](https://www.youtube.com/watch?v=MehBUQtb9qM)).
NVIDIA Sync streamlines the software and network configuration so you can get to a functioning cluster without configuring each device from a terminal.

When you finish, choose a workload playbook to set up on your cluster.


## What you'll accomplish

- You will physically connect your devices directly with QSFP cables or through a switch and QSFP cables.
- You will use [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) to set up and test the ConnectX-7 network.

## What to know before starting

**Required:**

- How to [plug a QSFP cable into a DGX Spark or GB10 device](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html#plugging-in-a-qsfp-cable)
- Access to the switch settings and the switch maker's setup guide, if you use a switch
- How to [set up a DGX Spark or GB10 device](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html) on the same network as the computer that runs NVIDIA Sync

**Suggested:**

- A basic grasp of [ConnectX-7 networking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)

## Supported hardware platforms

Check the table below to see if this playbook is for your hardware.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Direct or switch QSFP links | ✅ (200GbE QSFP) |

## Prerequisites

**Hardware requirements**

- Two to four DGX Spark or GB10 devices
- The QSFP cables listed for your layout in **Connect the Devices**
- A switch with one 200 Gbit/s Ethernet link for each device, if you use a switch. Some 400 Gbit/s ports must be split into 200 Gbit/s ports before you set up the cluster.

**Software requirements**

- Each device must be on the same local network as your laptop, and you must know its IP address or mDNS name
- You must have a user name and password with `sudo` privileges on each device
- Each DGX Spark or GB10 device is updated to the [April 2026 DGX OS release](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html#april-2026-release)

## Ancillary files

These files are **only** needed if you follow the manual instructions.
You can find them in [this playbook's assets folder](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets).

- [`discover-sparks`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets/discover-sparks) — node discovery and SSH key setup for the manual path
- [`spark_cluster_setup`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/assets/spark_cluster_setup) — network setup, SSH setup, and an NCCL test for the manual path

## Time & risk

- **Estimated time:** 10 minutes using NVIDIA Sync
- **Risk level:** Low with NVIDIA Sync; medium with manual setup
- **Rollback:** Delete the cluster in NVIDIA Sync. For manual setup, follow the rollback steps in **Configure Manually**.
- **Last Updated:** 09/09/2026
  - Made NVIDIA Sync the main path and moved cabling into its own tab.

## Connect the Devices

## Step 1. Pick a cluster layout

Pick one layout before you connect the cables.

| Devices | Layout | Cables |
| --- | --- | --- |
| Two | Direct | One cable between the devices |
| Three | Direct ring | Three cables; each device links to the other two |
| Two, three, or four | Switch | One cable and one 200 Gbit/s link from each device to the switch |

Do not mix direct and switch links. Use only one cable for each link. Four devices require a switch.

## Step 2. Check the devices and cables

1. Turn on each DGX Spark.
2. Make sure each device is on the same local network as the computer that runs NVIDIA Sync.
3. Update each device to the current DGX Spark system software.
4. Use a supported QSFP112 DAC cable in Ethernet mode. See [QSFP ports and cables](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html#the-qsfp-ports-and-cables) for the approved cable list.
5. Place the devices within reach of the cables.

## Step 3. Connect your layout

Use these steps each time you plug in a cable:

1. Turn each DGX Spark so that the back faces you.
2. Pick either QSFP port. The ports work the same with NVIDIA Sync.
3. Hold the cable with its pull tab facing up.
4. Push the cable into the port until it is fully seated.

> [!WARNING]
> Do not force a cable into a port. If it does not slide in, stop and check the pull tab and port alignment.

See [Plugging in a QSFP Cable](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html#plugging-in-a-qsfp-cable) for a port image and more help.

Choose only one of the layouts below.

### Two-device direct link

Connect one device to the other with one QSFP cable.

### Three-device direct ring

Use three QSFP cables to make a ring:

1. Connect device 1 to device 2.
2. Connect device 2 to device 3.
3. Connect device 3 to device 1.

Each device in the ring should have one cable in each QSFP port.

### Two-to-four-device switch

Set up the switch before you run NVIDIA Sync:

1. Check that the switch, ports, and cables support 200 Gbit/s Ethernet links.
2. Set each Spark-facing port for 200 Gbit/s. A 400 Gbit/s port may need to be split into two 200 Gbit/s ports.
3. Put all Spark-facing ports on the same Layer 2 network. Depending on the switch, this may be a bridge or VLAN.
4. If the switch reports hardware-offload status, confirm that the Spark-facing ports use the switch chip instead of the switch CPU.
5. Keep the default MTU for switch connections.
6. Apply or save the switch configuration.

> [!NOTE]
> Changing how a high-speed port is split may restart other links that use the same port group. Set the port mode before you rely on those links.

Connect and check the devices:

1. Connect one QSFP cable from each DGX Spark to the switch.
2. Check the switch interface for each Spark.
3. Confirm that each link is up at 200 Gbit/s.
4. If a link is down or runs at the wrong speed, check the cable type, port mode, auto-negotiation, and FEC.

Switch settings and port names differ by maker and model. Follow the switch maker's guide for the exact steps. Do not use commands written for a different switch model.

For MikroTik CRS804 and CRS812 switches, see [MikroTik wired interface compatibility](https://help.mikrotik.com/docs/spaces/ROS/pages/220233794/MikroTik%2Bwired%2Binterface%2Bcompatibility) and [MikroTik bridge configuration](https://help.mikrotik.com/docs/spaces/ROS/pages/328068/Bridging%2Band%2BSwitching).

## Step 4. Continue with NVIDIA Sync

Follow **Configure with NVIDIA Sync**. NVIDIA Sync will check the devices and detected layout before it sets up the cluster network.

## Configure with NVIDIA Sync

## Step 1. Make sure the devices are properly connected

Follow the instructions in the **Connect the Devices** tab.

## Step 2. Install NVIDIA Sync on your laptop

Install NVIDIA Sync on your Windows, macOS, or Ubuntu laptop.

::spark-download

- **Windows:** Open the `.exe` file and follow the setup steps.
- **macOS:** Open `nvidia-sync.dmg`, move NVIDIA Sync to the Applications folder, and open it.
- **Ubuntu:** Follow the [NVIDIA Sync install guide](https://docs.nvidia.com/sync/latest/getting-started.html#installation-and-onboarding).

## Step 3. Add each DGX Spark to NVIDIA Sync

Make sure your laptop can reach each DGX Spark on the local network.

For each device:

1. Open NVIDIA Sync and select **Add New**.
2. Pick the device if its mDNS name appears. If it does not, select **Add device manually**.
3. Enter the device name or IP address, user name, and password.
4. Select **Add**.

Each DGX Spark should now appear in NVIDIA Sync.

## Step 4. Start the NVIDIA Sync Cluster Assistant

1. Open **Settings**.
2. Select **Cluster Assistant**.
3. Select **Add New Cluster**.
4. Name the cluster.
5. Pick the devices that you connected.

## Step 5. NVIDIA Sync checks the devices

After you select the devices, NVIDIA Sync checks:

- SSH access
- The hardware and system software
- `sudo` access

If a required check fails, fix the issue and try again.

If `sudo` requires a password, enter the password for that device. NVIDIA Sync uses the password for setup and does not save or log it.

NVIDIA Sync also compares the user name, user ID, and group ID on each device. Matching values are optional, but they can make later work easier. Choose whether to make them match, then go on.

## Step 6. NVIDIA Sync checks the physical connections and network plan

Once the device checks are complete, NVIDIA Sync checks:

- The detected ConnectX-7 interfaces and cables
- The negotiated speed of each link
- The current network setup and any changes it must make

If NVIDIA Sync finds the wrong layout, check the cables in **Connect the Devices** and try again.

Review the network plan. If NVIDIA Sync will change the network, or if a link is not set to 200 Gbit/s, it will tell you.

Select **Confirm Network Configuration** to apply the plan.

## Step 7. NVIDIA Sync tests each link

NVIDIA Sync runs a speed test on each link. A link turns green when it meets the 184 Gbit/s lower bound.

If a link does not pass, fix the cable or switch setting and select **Run Test Again**. You can go on after a warning, but the cluster may run below its best speed.

## Step 8. NVIDIA Sync sets up inter-device SSH

Let NVIDIA Sync set up key-based SSH between the devices. It adds an SSH alias for each device.

This step can take a few minutes. If one device times out after five minutes, try the step again.

## Step 9. Save the cluster details to a text file

When NVIDIA Sync shows the success page:

1. Select **Copy** to copy the network details.
2. Save the details in a file for later use.
3. Select **See Example Workloads**.

The ConnectX-7 network and inter-device SSH are now ready.

## Next steps

Set up a workload on the cluster:

- [NCCL](https://build.nvidia.com/playbooks/nccl)
- [Fine-tune with PyTorch](https://build.nvidia.com/spark/multi-sparks-distributed-finetuning)
- [vLLM](https://build.nvidia.com/spark/vllm)

Cluster Assistant sets up the network. It does not install or run the workload.

## Delete the cluster with NVIDIA Sync

Delete the cluster before you change its devices or cable layout:

1. Open **Settings** in NVIDIA Sync.
2. Select **Clusters**.
3. Pick the cluster.
4. Open the overflow menu (**...**).
5. Select **Delete**.

This removes the node-to-node SSH setup and the cluster from NVIDIA Sync.

## Configure Manually

> [!NOTE]
> Use this tab to configure the cluster without NVIDIA Sync. The helper changes the network and SSH settings on each device. It also installs the tools needed for an NCCL test and runs that test.

## Step 1. Connect the devices

Follow **Connect the Devices** for your direct, ring, or switch layout.

If you use a switch, set it up before you run the helper. Put all device ports in one Layer 2 bridge and make sure each link can run at 200 Gbit/s.

## Step 2. Get the cluster setup files

On one DGX Spark, clone this repo and open the helper folder:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd client-hardware-playbooks/nvidia/playbook-connect-multiple-sparks/assets/spark_cluster_setup
```

## Step 3. Add the device login details

Pick the sample file that matches the number of devices:

- `config/spark_config_b2b.json` for two devices
- `config/spark_config_ring.json` for three devices
- `config/spark_config_switch.json` for four devices

Copy the sample to a new file. This example uses two devices:

```bash
cp config/spark_config_b2b.json config/my-cluster.json
```

Edit `config/my-cluster.json`. For each device, add its management IP address, SSH port, user name, and password.

> [!WARNING]
> The JSON file stores passwords as plain text. Keep the file private and delete it when setup is done.

## Step 4. Check the cluster

Run the checks before you change the network:

```bash
bash spark_cluster_setup.sh -c config/my-cluster.json --pre-validate-only
```

The check should end with `Pre-setup validations completed successfully.` Fix any error before you go on.

## Step 5. Set up the cluster

Run the helper from its own folder:

```bash
bash spark_cluster_setup.sh -c config/my-cluster.json --run-setup
```

The helper will:

1. Check the devices and cable layout.
2. Set IP addresses on the ConnectX-7 network.
3. Set up key-based SSH between the devices.
4. Check the links between the devices.
5. Install the tools needed for the NCCL test and run it.

The setup should print `Spark cluster setup completed successfully.` The test should then print `NCCL test completed.`

## Step 6. Remove the password file

After setup works, delete the JSON file that holds the passwords:

```bash
rm config/my-cluster.json
```

## Next steps

Open the workload playbook you want to use. The [NCCL playbook](https://build.nvidia.com/playbooks/nccl) can run a fuller network test.

## Roll back the manual setup

Follow [Inspect and Verify a ConnectX-7 Cluster Network Plan](https://docs.nvidia.com/sync/latest/cluster-network-inspection.html) to inspect the network before you remove its Netplan file.

If you used a switch, also undo any port, bridge, DHCP, link speed, or MTU changes in the switch maker's tool.

## Troubleshooting

## Common issues

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
| The manual pre-check fails | A management IP, SSH login, password, or `sudo` setting is wrong | Fix the value in `config/my-cluster.json`, test SSH to each management IP, and run `--pre-validate-only` again. |
| No ConnectX-7 interface is up | A cable or port is not active | Reseat the cables, confirm the layout, reboot the devices, and run `ibdev2netdev` again. |
| The manual helper reports an APT error | A package source or signing key is broken | Fix the APT source or key error on that device, then run the helper again. |
| The NCCL test cannot load `libnccl.so.2` | NCCL is not ready on every device | Follow the [NCCL playbook](https://build.nvidia.com/playbooks/nccl) on every device, then run the test again. |

For more Cluster Assistant help, see the [NVIDIA Sync troubleshooting guide](https://docs.nvidia.com/sync/latest/cluster-assistant.html#troubleshooting).
