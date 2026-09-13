# Run cuPyNumeric Across Two DGX Sparks

> Distributed NumPy-compatible workloads beyond single-GPU memory

## Table of Contents

- [Overview](#overview)
- [Multi-node](#multi-node)
  - [Prerequisites (multi-node)](#prerequisites-multi-node)
  - [8.1 Create MPI hostfile](#81-create-mpi-hostfile)
  - [8.2 Verify MPI communication](#82-verify-mpi-communication)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

cuPyNumeric is a scalable NumPy-compatible library for multi-GPU and multi-node numerical computing. You write familiar Python NumPy-style code while the runtime distributes large arrays and linear-algebra workloads across nodes of multi-node capable hardware.

This playbook walks you through a two-node setup: shared storage, a conda environment with cuPyNumeric, MPI launch configuration, and an end-to-end matrix-multiplication check that uses both nodes.

## What you'll accomplish

You'll run distributed cuPyNumeric workloads across two nodes of multi-node capable hardware. Your setup will support large-scale matrix operations that exceed single-GPU memory limits while keeping NumPy-like Python syntax. As an end-to-end check, you'll run a 20k × 20k matrix multiplication benchmark across both nodes.

## What to know before starting

**Required:**

- Comfortable with Python and NumPy programming
- Experience working in a Linux terminal
- Basic concept of conda environments
- Familiarity with SSH and shared filesystems (NFS)

**Optional:**

- Basic understanding of linear algebra (high-school level math is sufficient)
- Familiarity with MPI launchers and high-speed interconnect networking on multi-node capable hardware

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Shared Miniforge + `cupynumeric` conda env over NFS; Legate/MPI across two nodes | ✅ (direct high-speed interconnect) |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Two nodes of multi-node capable hardware
- Multi-node networking and SSH configured with [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) (recommended) or [Connect two nodes for distributed workloads](https://build.nvidia.com/playbooks/connect-two-sparks)
- The same username on both systems (recommended; simplifies NFS and SSH configuration)

**Software requirements**

- NVIDIA driver and CUDA toolkit available: `nvidia-smi` and `nvcc --version`
- SSH access available to both systems
- Administrative privileges on both systems: `sudo whoami` returns `root`
- Internet connectivity on both systems for package downloads: `ping nvidia.com`

## Ancillary files

This playbook does not ship local helper scripts. You install packages with conda and clone cuPyNumeric examples from the upstream repository during the **Multi-node** tab.

- Upstream examples — [nv-legate/cupynumeric](https://github.com/nv-legate/cupynumeric) (clone at the release tag that matches your installed package)

## Time & risk

- **Estimated time:** 60 MIN (including software installation and validation)
- **Risk level:** Medium
  - Involves NFS and firewall changes on multi-node capable hardware
  - Incorrect interconnect or SSH setup will block MPI launches until fixed
- **Rollback:** Remove the NFS export/mount and conda environment (`conda remove --name cupynumeric --all`). Manage or remove the interconnect through the same path used to create the cluster; the cable can be unplugged to return nodes to a standalone state.
- **Last Updated:** 08/12/2026
  - Added NVIDIA Sync Cluster Assistant as the recommended cluster prerequisite while retaining manual setup as a fallback

## Multi-node

## Multi-node

Use this tab to run cuPyNumeric across two nodes of **multi-node capable hardware**.

> [!TIP]
> If [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) successfully configured your two-Spark cluster, do not repeat the manual connection playbook. Continue below using the interconnect details from your completed cluster setup. Otherwise, complete [Connect two nodes for distributed workloads](https://build.nvidia.com/playbooks/connect-two-sparks) first.

> [!NOTE]
> **Multi-node in this playbook applies to DGX Spark only.** Use the interconnect IPs and interface names from your completed cluster setup wherever the steps below refer to `$SERVER_IP`, `$CLIENT_IP`, `$IB_SUBNET`, or firewall interfaces.

### Prerequisites (multi-node)

- Two nodes of multi-node capable hardware with a working high-speed interconnect
- Passwordless SSH between nodes (including each node to itself over the interconnect IPs)
- The same username on both systems (recommended)

---

## Step 1. Set environment variables

Set up environment variables for consistent interconnect IP references. Match these values to the IPs assigned during cluster setup. Run on **both nodes**:

```bash
## Define network configuration - run on BOTH nodes
export SERVER_IP="192.168.100.11"    # Server node interconnect IP
export CLIENT_IP="192.168.100.10"    # Client node interconnect IP
export IB_SUBNET="192.168.100.0/24"  # Interconnect subnet

## Make these persistent across sessions
echo 'export SERVER_IP="192.168.100.11"' >> ~/.bashrc
echo 'export CLIENT_IP="192.168.100.10"' >> ~/.bashrc
echo 'export IB_SUBNET="192.168.100.0/24"' >> ~/.bashrc
```

> [!NOTE]
> If your cluster setup uses different IPs, update the values above before continuing. In this layout, the **server** node hosts NFS and coordinates distributed runs; the **client** mounts the share and participates in MPI ranks.

## Step 2. Verify hardware and pre-flight state

**Run on BOTH nodes:**

```bash
## Check GPU availability
nvidia-smi

## Verify CUDA installation
nvcc --version

## Confirm interconnect interfaces (names vary by port used)
ip link show | grep -E 'enp1|enP2'
ibdev2netdev
```

Example interconnect interfaces after the cable is connected (names depend on which port you use):

```text
6: enP2p1s0f0np0: <BROADCAST,MULTICAST,UP,LOWER_UP> ... state UP ...
7: enP2p1s0f1np1: <NO-CARRIER,BROADCAST,MULTICAST,UP> ... state DOWN ...
```

Inspect existing configuration that could conflict with later steps:

```bash
## Check for existing IPs on interconnect interfaces
ip addr show | grep -A 3 -E 'enp1|enP2'

## List existing netplan configs
ls /etc/netplan/

## Check for existing SSH config that may conflict with cluster SSH keys
cat ~/.ssh/config 2>/dev/null || echo "No existing SSH config"

## Check if NFS server is already running
systemctl is-active nfs-server 2>/dev/null || echo "NFS server not running"
```

> [!NOTE]
> Review the output before proceeding. If `~/.ssh/config` has `IdentityFile` entries, they may interfere with passwordless SSH used by MPI.

## Step 3. Configure firewall for MPI

Allow MPI traffic on the interconnect interfaces. Replace interface names if they differ from your cluster setup.

**Run on BOTH nodes:**

```bash
sudo ufw allow in on enp1s0f0np0
sudo ufw allow in on enp1s0f1np1
sudo ufw allow in on enP2p1s0f0np0
sudo ufw allow in on enP2p1s0f1np1
```

**Verify** the firewall rules were applied:

```bash
sudo ufw status | grep -E 'enp1|enP2'
```

## Step 4. Install Network File System (NFS)

NFS shares a common directory so both nodes can use the same conda environment and hostfile without duplicating installs.

> [!IMPORTANT]
> Before configuring NFS, confirm your username and home directory:
> - Find your username: `echo $USER`
> - Find your home directory: `echo $HOME`
>
> For example, if your username is `nvidia`, your home directory is `/home/nvidia`, and the shared folder path should be `/home/nvidia/shared`.

**On the server node** (at `$SERVER_IP`):

```bash
## 1. Create the shared folder
mkdir -p $HOME/shared

## 2. Install NFS server
sudo apt update
sudo apt install nfs-kernel-server -y

## 3. Configure exported directory
echo "/home/$USER/shared $CLIENT_IP(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports

## 4. Start and enable the NFS service
sudo systemctl enable nfs-server
sudo systemctl start nfs-server

## 5. Reload exports (required — systemctl start is a no-op if already running)
sudo exportfs -ra

## 6. Verify the export is active
sudo exportfs -v
## Expected output should include a line like:
## /home/<your-username>/shared  192.168.100.10(rw,sync,...)

## 7. Configure firewall to allow client access
sudo ufw allow from $CLIENT_IP to any port 2049 proto tcp

## 8. Verify server listening
ss -tulpn | grep 2049
## Expected:
## tcp   LISTEN 0  64  0.0.0.0:2049  0.0.0.0:*
## tcp   LISTEN 0  64     [::]:2049     [::]:*
```

**On the client node** (at `$CLIENT_IP`):

```bash
## 1. Install NFS client tools
sudo apt install nfs-common -y

## 2. Create mount folder
mkdir -p $HOME/shared

## 3. Mount NFS
## NFSv4 without fsid=0 requires the full server-side path.
## If both machines share the same username, $USER expands correctly.
## If the server has a different username, replace $USER with the server's username.
sudo mount -t nfs -o vers=4 $SERVER_IP:/home/$USER/shared $HOME/shared

## 4. Verify mount
mount | grep nfs
nfsstat -m

## 5. Enable automatic mount at boot (optional)
echo "$SERVER_IP:/home/$USER/shared $HOME/shared nfs4 defaults 0 0" | sudo tee -a /etc/fstab
```

## Step 5. Install conda into the shared folder

Install Miniforge into the NFS share so both nodes use the same conda prefix.

**Run on the server node** (at `$SERVER_IP`):

```bash
## Create directory and download Miniforge for ARM64
mkdir -p ~/shared/miniforge3
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O ~/shared/miniforge3/miniforge.sh

## Install Miniforge
bash ~/shared/miniforge3/miniforge.sh -b -u -p ~/shared/miniforge3

## Clean up installer
rm -rf ~/shared/miniforge3/miniforge.sh

## Initialize conda for bash and zsh
~/shared/miniforge3/bin/conda init bash
~/shared/miniforge3/bin/conda init zsh

## Reload shell configuration
source ~/.bashrc
```

**Run on the client node** (at `$CLIENT_IP`):

```bash
## Initialize conda for bash and zsh (Miniforge is already installed via NFS)
~/shared/miniforge3/bin/conda init bash
~/shared/miniforge3/bin/conda init zsh

## Reload shell configuration
source ~/.bashrc
```

**Verify conda installation (run on BOTH nodes):**

```bash
## Check which conda is being used
which conda

## Expected: ~/shared/miniforge3/bin/conda or /home/$USER/shared/miniforge3/bin/conda
## If you see a different path, another conda install may be taking priority.

## Verify conda environment path
conda info --base

## Expected: ~/shared/miniforge3 or /home/$USER/shared/miniforge3
```

> [!NOTE]
> If `which conda` shows a different installation, move the `~/shared/miniforge3` conda init block to the end of `~/.bashrc`, or remove other conda installations from PATH.

## Step 6. Create the cuPyNumeric conda environment

**Run on the server node** (the environment is created on the NFS share):

```bash
## Create new conda environment with cuPyNumeric installed
conda create -n cupynumeric -c legate cupynumeric

## Verify the environment was created in the shared location
conda env list
## The cupynumeric environment should be listed under ~/shared/miniforge3/envs/
```

> [!NOTE]
> If you accidentally installed cuPyNumeric under a different conda prefix:
> 1. Check all environments: `conda env list`.
> 2. Remove the incorrect environment with `/path/to/wrong/conda/bin/conda env remove -n cupynumeric`.
> 3. Recreate with the shared Miniforge at `~/shared/miniforge3`.

## Step 7. Validate the installation on each node

**Run on BOTH nodes separately.**

Create a test script:

```bash
cat > test_cupynumeric.py << 'EOF'
import numpy as np
try:
    import cupynumeric as cn
    print("cuPyNumeric imported successfully")

#    # Create test array
    x = cn.ones((1000, 1000))
    result = cn.sum(x)
    assert result == 1000000, f"wrong result: {result}"
    print("cuPyNumeric basic test passed")
except ImportError as e:
    print(f"Failed to import cuPyNumeric: {e}")
except Exception as e:
    print(f"Error during test: {e}")
EOF
```

Run the test:

```bash
## Activate the environment
conda activate cupynumeric

## Run the test
python test_cupynumeric.py
```

## Step 8. Configure and verify MPI communication

### 8.1 Create MPI hostfile

**Run on either node (server preferred)** — the hostfile is visible on both via NFS:

```bash
## Create hostfile with both nodes
echo -e "$CLIENT_IP slots=1\n$SERVER_IP slots=1" > $HOME/shared/hostfile
```

> [!NOTE]
> The hostfile path must be absolute when used in MPI commands. `$HOME` expands to an absolute path (for example `/home/nvidia/shared/hostfile`).

### 8.2 Verify MPI communication

**Run on either node (server preferred):**

```bash
## Test MPI across both nodes
mpirun -n 2 -npernode 1 --hostfile $HOME/shared/hostfile hostname
```

This command prints the hostname of each node. If it hangs, verify firewall rules, the hostfile, passwordless SSH (including SSH to self over interconnect IPs), and the interconnect configuration from your cluster setup.

## Step 9. Test single- and multi-node computations

Use a matrix multiplication example to see how cuPyNumeric scales across nodes.

**Run on either node (server preferred)** — commands execute across nodes via MPI:

```bash
## Check the installed cuPyNumeric version to match the examples repo tag
conda activate cupynumeric
conda list cupynumeric | grep cupynumeric
## Note the version number (e.g., 25.01.00)

## Clone the examples repo at the matching release tag (NOT main).
## Replace <version-tag> with the tag matching your installed version
## (e.g., v25.01.00 — check tags at https://github.com/nv-legate/cupynumeric/tags)
git clone --branch <version-tag> https://github.com/nv-legate/cupynumeric.git
cd cupynumeric

## Run a 20k x 20k matrix multiplication on 1 node and on two nodes.

## Run on one node
legate --launcher mpirun --launcher-extra="--hostfile $HOME/shared/hostfile --mca oob_tcp_if_include $IB_SUBNET --mca pml ucx" --nodes 1 --gpus 1 --fbmem 40960 --cpus 4 ./examples/gemm.py -n 20000 -i 2

## Run on two nodes
legate --launcher mpirun --launcher-extra="--hostfile $HOME/shared/hostfile --mca oob_tcp_if_include $IB_SUBNET --mca pml ucx" --nodes 2 --gpus 1 --fbmem 40960 --cpus 4 ./examples/gemm.py -n 20000 -i 2
```

> [!IMPORTANT]
> The `--hostfile` parameter requires an absolute path. Prefer `$HOME/shared/hostfile` (expanded) over `~/shared/hostfile` when passing through `--launcher-extra`.

Open MPI configuration notes:

- `--mca oob_tcp_if_include $IB_SUBNET` — restricts Open MPI out-of-band TCP to the interconnect subnet.
- `--mca pml ucx` — uses the UCX point-to-point messaging layer for high-performance communication over the interconnect.

For more details, see the [cuPyNumeric tutorial](https://docs.nvidia.com/cupynumeric/latest/user/tutorial.html).

## Step 10. Cleanup

> [!WARNING]
> These commands remove configuration changes made during this playbook. Manage or remove the interconnect through the same path you used to create the cluster: Cluster Assistant or the manual connection playbook.

**On the client node:**

```bash
## Unmount the NFS share (remove the fstab line first if you added one)
sudo umount $HOME/shared
```

**On the server node:**

```bash
## Remove the conda environment installed on the NFS share
conda deactivate
conda remove --name cupynumeric --all -y

## Optionally remove the shared folder contents after clients have unmounted
## rm -rf $HOME/shared
```

Finally, disconnect the interconnect cable if you no longer need the two-node link.

---

If you encounter any issues, see the **Troubleshooting** tab for common symptoms and fixes.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ping` fails between interconnect IPs | Interconnect interface not configured or cable down | If you used [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html), verify or retry the cluster there. Otherwise, re-check physical cabling and IP assignment in [Connect two nodes for distributed workloads](https://build.nvidia.com/playbooks/connect-two-sparks). Confirm with `ip addr show` and `ibdev2netdev`. |
| `ip link show` / `ibdev2netdev` shows no active interconnect | Cable not connected or interface naming differs | Check the physical cable; note the interface marked `Up` and use that name in firewall rules |
| `mpirun` hangs or times out | SSH not configured for interconnect IPs, firewall blocking MPI, or stacked IPs on the interconnect interface | Verify `ssh -o BatchMode=yes $SERVER_IP hostname` and `ssh -o BatchMode=yes $CLIENT_IP hostname` from both nodes (including SSH to self). Confirm firewall allows the interconnect interfaces. Run `ip addr show <INTERFACE>` and confirm only one IP is assigned |
| Passwordless SSH falls back to password prompt | Existing `~/.ssh/config` has an `IdentityFile` directive that overrides the cluster SSH key | Inspect with `cat ~/.ssh/config` and remove or comment out conflicting entries |
| NFS mount fails with "access denied by server" or "No such file or directory" | New `/etc/exports` entries were not reloaded, or NFSv4 mount path is incomplete | Run `sudo exportfs -ra && sudo exportfs -v` on the server. On the client, mount with the full server path: `$SERVER_IP:/home/$USER/shared` |
| Interconnect stops working after installing extra driver stacks | Unsigned DKMS modules rejected by Secure Boot | Prefer the inbox interconnect drivers shipped with the OS. Remove conflicting packages if installed, then reboot |
| `which conda` shows a non-shared installation | Multiple conda installations on PATH; `~/.bashrc` initializes the wrong one last | Move the `~/shared/miniforge3` conda init block to the end of `~/.bashrc`, or remove other conda installations from PATH |
| cuPyNumeric examples fail with `ImportError` for missing symbols | Cloned `main` instead of the release tag matching the installed package | Re-clone with `git clone --branch <version-tag> https://github.com/nv-legate/cupynumeric.git`. Find the version with `conda list cupynumeric` |
| `conda` install fails | Network connectivity issue | Verify internet access with `ping nvidia.com` |

> [!NOTE]
> Some hardware platforms use Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. If you hit memory pressure even when within rated capacity, manually flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
