# Connect Two Nodes for Distributed Workloads

> Combined memory and compute over a direct high-speed link

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Overall flow](#overall-flow)
  - [Command style by control host](#command-style-by-control-host)
  - [Required preparation before the GDR variant](#required-preparation-before-the-gdr-variant)
- [Troubleshooting](#troubleshooting)
  - [Stop conditions](#stop-conditions)

---

## Overview

## Basic idea

Two multi-node capable hardware platforms can be connected directly by two **ConnectX-8 (CX8)** QSFP cables — one cable per rail — with no switch and no rail bonding. Each rail is an independent 400 Gb/s RoCEv2 path: rail 0 maps to `mlx5_0` (QSFP0-to-QSFP0) and rail 1 maps to `mlx5_1` (QSFP1-to-QSFP1). Cross-node traffic uses CX8 RoCEv2; there is no inter-node NVLink.

This playbook walks through the **manual fabric bring-up** from a separate **control host** (Linux, macOS, or Windows). The control host SSHs into both nodes and runs numbered setup scripts that configure rail IPs, MTU 9000, RoCEv2, and GPUDirect runtime settings, then validate the two-rail fabric before you move on to NCCL, Ray, vLLM, or other distributed workloads.

## What you'll accomplish

You'll connect two nodes of multi-node capable hardware with two CX8 QSFP rails and prove the fabric in two levels:

- Confirm OS SSH access and prerequisite tools on both nodes.
- Physically cable QSFP0-to-QSFP0 and QSFP1-to-QSFP1 and verify 400 Gb/s link on both rails.
- Assign private rail IPs and MTU 9000 on each CX8 netdev.
- Apply RoCEv2 runtime settings.
- Validate jumbo ping, RDMA port state, and route selection on both rails.
- Run a host-memory RDMA bandwidth smoke test (`ib_write_bw`) per rail.
- For GPU-performance work, verify the GDR-capable `perftest` flags and ACS/Data Direct topology before starting a GDR test.
- After those prerequisites pass, validate the GDR path on both rails with `./08_run_perftest_pair.sh --rail <0-or-1> --gdr`.

After `./07_validate_setup.sh` passes, the basic two-rail RoCE fabric is up. After the host-memory `./08_run_perftest_pair.sh` tests pass, basic RDMA bandwidth is proven. Before using the pair for GPU-performance workloads such as NCCL, Ray, vLLM, or other distributed applications, require the perftest-capability and ACS/Data Direct checks to pass on both DUTs, then use the GDR variant on both rails to prove the CUDA DMA-BUF/Data Direct GPU-memory RDMA path.

## What to know before starting

**Required:**

- Basic Linux command line and SSH usage
- Familiarity with network concepts: IP addresses, MTU, point-to-point links
- Understanding that this playbook configures the **fabric layer only** — container RDMA exposure, NCCL environment, and distributed launchers are covered as next steps, not in the core setup

**Optional:**

- Two people or a clear cable-labeling plan helps avoid crisscrossing QSFP rails
- Familiarity with InfiniBand/RoCE tooling such as `ibdev2netdev` and `ib_write_bw`

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Station** | DGX OS (Linux) | Large HBM + Grace DRAM | Two CX8 QSFP rails (`mlx5_0` / `mlx5_1`); control-host setup scripts | ✅ (ConnectX-8 QSFP rails) |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Two nodes of multi-node capable hardware (DUT A and DUT B), each with CX8 adapters exposing `mlx5_0` and `mlx5_1`
- Two 400G-capable QSFP cables (one per rail) for full-speed validation
- A separate **control host** with a shell, `ssh`, and `tar` — Linux, macOS, or Windows

**Software requirements**

On both nodes:

- OS SSH access for user `nvidia` (or your configured `CX8_OS_USER`)
- `sudo` for privileged network and module commands on the DUTs
- RDMA stack: `ibdev2netdev`, `ibv_devinfo`, `show_gids`, `ib_write_bw`
- RoCE tuning tools: `mlnx_qos`, `cma_roce_tos` (where supported by the installed OFED/DOCA image)
- Both systems are expected to already have firmware, OS, NVIDIA driver, CUDA, DOCA/OFED, and RDMA/perftest tools installed. The setup scripts check for required commands; missing core tools indicate an incomplete software image. MFT utilities `flint` and `mlxconfig` are optional. `nmcli` is required only when NetworkManager is active; when NetworkManager is inactive or unavailable, the supported `ip`-based configuration path does not require `nmcli`.
- For GDR validation, `ib_write_bw` must advertise both `--use_cuda_dmabuf` and `--use_data_direct`, `rdma_topo` must be installed, and the Script 11 ACS/Data Direct inspection must pass on both DUTs before Script 08 starts a GDR test.

On the control host:

- `ssh` and `tar` (OpenSSH client on Windows if using PowerShell)
- Bash shell scripts on Linux/macOS, **or** Python 3 with OpenSSH in `PATH` for the cross-platform `cx8_setup.py` wrapper

## Ancillary files

All required assets can be found [in the playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-two-stations/assets/). Run the setup from the playbook `assets/` directory.

- `00_env.local.example` — Template for the local configuration file; copy to `00_env.local` and edit hostnames/IPs before running the setup scripts
- `00_env.local` — Your lab-specific configuration (git-ignored; the only file operators should create or edit)
- `01_probe_access.sh` through `11_configure_acs_grub.sh` — Numbered control-host scripts for setup, validation, and required GPUDirect/Data Direct readiness checks
- `99_cleanup_runtime.sh` — Optional runtime cleanup
- `cx8_setup.py` — Cross-platform Python wrapper (Windows, macOS, Linux) that runs the same steps as the shell scripts
- `dut-assets/` — Linux DUT helper scripts copied to both nodes by `./02_push_assets.sh`

## Time & risk

- **Estimated time:** 60 MIN for core bring-up; longer if optional GDR perftest build or ACS/Data Direct reboot is required
- **Risk level:** Medium
  - Incorrect QSFP cabling (crisscrossed rails) causes hard-to-debug RDMA/NCCL issues
  - Cable thermal or module errors (`Cable error`, `High Temperature`) require physical intervention before software setup can continue
  - Runtime rail configuration is temporary unless `--persist` is used; a reboot clears non-persistent settings
- **Rollback:** Run `99_cleanup_runtime.sh` to remove rail IPs and reset MTU; add `--remove-persist` if netplan was written; add `--down` to bring interfaces down
- **Last Updated:** 09/04/2026
  - Hardened perftest server cleanup, positive timeout validation, and `ib_write_bw` selection (NVBug 6725953)
  - Replaced authored numeric step references with action/script names and corrected `ib_write_bw` capability detection
  - Consolidated two-node CX8 QSFP fabric bring-up with rail IP/MTU, RoCEv2, RDMA validation, and optional GPUDirect/Data Direct checks

## Key terms

| Term | Meaning in this setup |
|---|---|
| DUT | Device under test — the two hardware platforms being connected |
| Control host | The separate system where you run this playbook; requires `ssh` and `tar` |
| CX8 | ConnectX-8, the NVIDIA network adapter used for the direct two-node connection |
| Rail | One independent network path between the two nodes (`mlx5_0`/QSFP0 and `mlx5_1`/QSFP1) |
| Rail IP | Private IP assigned to one CX8 rail (for example `192.168.100.1` on DUT A, `192.168.100.2` on DUT B) |
| MTU | Maximum Transmission Unit; this setup uses MTU 9000 for jumbo frames |
| Jumbo ping | Large-packet ping with do-not-fragment set; proves MTU 9000 works end to end |
| RDMA | Remote Direct Memory Access — low-CPU-overhead data movement between systems |
| RoCE | RDMA over Converged Ethernet; this setup uses RoCEv2 over the CX8 rails |
| GPUDirect RDMA | NVIDIA technology for RDMA to access GPU memory directly |
| CUDA DMA-BUF/Data Direct | GPUDirect path used by the optional `--gdr` perftest when supported by `ib_write_bw` |
| ACS/Data Direct GRUB configuration | Boot-time PCIe/ACS configuration written by `rdma_topo` when the GPU and CX8 Data Direct path are not in the required topology for GPUDirect/Data Direct |
| `nvidia_peermem` | Optional legacy/module-based GPUDirect path on some stacks; default OS images may not load it, and this playbook does not require it when CUDA DMA-BUF/Data Direct works |
| MFT | NVIDIA/Mellanox Firmware Tools (`flint`, `mlxconfig`); optional for this basic setup |

## Instructions

> [!TIP]
> Run all commands from a **control host** — a separate Linux, macOS, or Windows system with `ssh` and `tar` access to both nodes. Do not run the control-host scripts directly on the DUTs.
> Applying ACS/Data Direct GRUB settings with `./11_configure_acs_grub.sh --apply` does not reboot either DUT. A separate reboot is required afterward, so do not use a DUT being configured as the control host.

> [!NOTE]
> Before starting, the only file you should create or edit is `00_env.local`. Do not edit the tracked scripts for lab-specific hostnames, IP addresses, or user overrides.

## Step 0. Get the setup package and create local configuration

Clone the playbook repository on your control host and create the local environment file.

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks client-hardware-playbooks
cd client-hardware-playbooks/nvidia/playbook-connect-two-stations/assets
cp 00_env.local.example 00_env.local
```

Edit `00_env.local` before continuing. Set the Node A/B names and OS hosts for your pair. The OS user and rail CIDRs have defaults; change them only if your setup differs.

| Variable | Meaning | Edit needed? | Default/example value |
|---|---|---:|---|
| `CX8_A_NAME` | Friendly name for Node A | Yes | `station-a` |
| `CX8_A_HOST` | Node A OS hostname or IP | Yes | `<station-a-os-host-or-ip>` |
| `CX8_B_NAME` | Friendly name for Node B | Yes | `station-b` |
| `CX8_B_HOST` | Node B OS hostname or IP | Yes | `<station-b-os-host-or-ip>` |
| `CX8_OS_USER` | OS SSH user used for both DUTs | Only if not `nvidia` | `nvidia` |
| `CX8_SSH_STRICT_HOST_KEY_CHECKING` | SSH host-key policy | Only for controlled lab automation | `ask` |
| `CX8_GPU_BDF` | GPU PCI bus ID for the `./08_run_perftest_pair.sh` GDR test when the DUT has more than one CUDA device | Only if CUDA device 0 is not the intended GPU | Example: `0009:06:00.0` |
| `CX8_TRY_NVIDIA_PEERMEM` | Opt-in legacy peermem load attempt during RoCE/GPUDirect runtime configuration | Only if you specifically need to test `nvidia_peermem` | `0` |
| `CX8_PERFTEST_REPO` | Perftest git repository used by the optional `./10_install_perftest_gdr.sh` source build | Only if a mirror or fork is required | `https://github.com/linux-rdma/perftest.git` |
| `CX8_PERFTEST_REF` | Perftest git ref used by the optional `./10_install_perftest_gdr.sh` source build | Only if a different tested ref is required | `26.04.17` |
| `CX8_PERFTEST_PREFIX` | Install prefix used by the optional `./10_install_perftest_gdr.sh` source build | Only if `/usr/local` is not desired | `/usr/local` |
| `CX8_PERFTEST_SERVER_READY_TIMEOUT` | Positive integer wait time, in seconds, for the remote `ib_write_bw` server started by `./08_run_perftest_pair.sh` | Only if GPU-memory server startup is unusually slow | `30` |
| `STATION_A_RAIL0_CIDR` / `STATION_B_RAIL0_CIDR` | Rail 0 private IPs assigned by `./05_configure_rails_runtime.sh` | Only if subnet conflicts | `192.168.100.1/24` / `192.168.100.2/24` |
| `STATION_A_RAIL1_CIDR` / `STATION_B_RAIL1_CIDR` | Rail 1 private IPs assigned by `./05_configure_rails_runtime.sh` | Only if subnet conflicts | `192.168.101.1/24` / `192.168.101.2/24` |

Passwords are intentionally not stored in this directory. SSH and `sudo` will prompt when needed. `00_env.local` is ignored by git.

If the default private rail subnets conflict with your lab, change the four `STATION_*_RAIL*_CIDR` values before running `./05_configure_rails_runtime.sh`. Keep rail 0 and rail 1 on different subnets.

### Overall flow

Run the probe, asset-copy, prerequisite, cable, rail, RoCE, validation, and host-memory perftest actions in the order shown as the core bring-up path. That proves both rails are cabled, configured, routable, jumbo-ping clean, and RDMA-capable with host memory.

GPU-memory RDMA/Data Direct validation has an additional mandatory readiness gate. Before the first `./08_run_perftest_pair.sh --gdr` command, confirm that `ib_write_bw` has both required GDR flags and that `./11_configure_acs_grub.sh` passes on both DUTs. If the ACS check does not pass, apply its generated GRUB configuration and reboot before attempting GDR. Script 08 repeats these read-only checks and will not launch a GDR server or client unless both DUTs are ready.

### Command style by control host

The DUTs are Linux systems, so the copied `dut-assets/*.sh` helpers are Bash scripts on the DUTs. The control host can be Windows, macOS, or Linux:

| Control host | Recommended command style |
|---|---|
| Linux or macOS with Bash | Use the numbered shell scripts, for example `./01_probe_access.sh` |
| Windows PowerShell or any host without Bash | Use the cross-platform Python wrapper, for example `py -3 .\cx8_setup.py 01` |
| Linux/macOS with Python preferred | Use `python3 ./cx8_setup.py 01` |

The Python wrapper requires Python 3 and an OpenSSH client in `PATH`. On Windows, enable or install the Windows OpenSSH client first if `ssh` is not available.

The Python wrapper uses the same `00_env.local` file as the shell scripts. Numeric aliases match the setup scripts:

| Action | Shell command | Python command |
|---|---|---|
| Probe OS SSH access | `./01_probe_access.sh` | `python3 ./cx8_setup.py 01` |
| Copy setup scripts | `./02_push_assets.sh` | `python3 ./cx8_setup.py 02` |
| Check prerequisites/current state | `./03_prereq_check.sh` | `python3 ./cx8_setup.py 03` |
| Check cable/link presence | `./04_check_cable_presence.sh` | `python3 ./cx8_setup.py 04` |
| Configure rail IP/MTU temporarily | `./05_configure_rails_runtime.sh` | `python3 ./cx8_setup.py 05` |
| Configure rail IP/MTU persistently | `./05_configure_rails_runtime.sh --persist` | `python3 ./cx8_setup.py 05 --persist` |
| Configure RoCE/GPUDirect runtime | `./06_configure_roce_gdr_runtime.sh` | `python3 ./cx8_setup.py 06` |
| Validate the basic setup | `./07_validate_setup.sh` | `python3 ./cx8_setup.py 07` |
| Test rail 0 RDMA bandwidth | `./08_run_perftest_pair.sh --rail 0` | `python3 ./cx8_setup.py 08 --rail 0` |
| Test rail 1 RDMA bandwidth | `./08_run_perftest_pair.sh --rail 1` | `python3 ./cx8_setup.py 08 --rail 1` |
| Collect logs | `./09_collect_logs.sh` | `python3 ./cx8_setup.py 09` |
| Install or upgrade GDR-capable perftest | `./10_install_perftest_gdr.sh` | `python3 ./cx8_setup.py 10` |
| Verify the GDR ACS/Data Direct prerequisite | `./11_configure_acs_grub.sh` | `python3 ./cx8_setup.py 11` |
| Apply ACS/Data Direct GRUB settings | `./11_configure_acs_grub.sh --apply` | `python3 ./cx8_setup.py 11 --apply` |
| Clean up runtime configuration | `./99_cleanup_runtime.sh` | `python3 ./cx8_setup.py 99` |

For unattended lab loops, configure SSH keys and passwordless `sudo` outside this playbook. The playbook supports that setup but does not create it or store credentials.

## Step 1. Probe OS SSH access

Confirm that both nodes are reachable over OS SSH and that the expected user can run basic shell commands. This step also lists whether the tools needed for CX8 setup are already installed.

```bash
./01_probe_access.sh
```

If SSH reports an unknown host key, verify the DUT identity and answer the OpenSSH prompt. This is intentional: the default SSH policy is `StrictHostKeyChecking=ask`, not automatic trust-on-first-use. In controlled lab automation only, `CX8_SSH_STRICT_HOST_KEY_CHECKING="accept-new"` can be set in `00_env.local`.

Missing `flint` or `mlxconfig` is allowed. Missing `nmcli` is also allowed when NetworkManager is inactive or unavailable; when NetworkManager is active, `nmcli` is required. If any other required command is missing, stop and contact NVIDIA support — the software image is missing a required tool.

`flint` and `mlxconfig` are optional for this basic setup. They are provided by NVIDIA MFT (Mellanox Firmware Tools), commonly through `mft` or `mstflint` packages depending on the image, and are used only for CX8 firmware/config visibility. Missing MFT tools do not block the two-rail network setup or validation.

## Step 2. Copy setup scripts to both nodes

Copy the reusable setup helpers from your local checkout to both nodes. The script streams helper files through one SSH session per DUT, so each DUT should require only one OS password prompt. The script normalizes helper scripts to Linux LF line endings and uses a clean `ustar` stream so macOS extended attributes are not sent to the DUT.

```bash
./02_push_assets.sh
```

Run `./01_probe_access.sh` before `./02_push_assets.sh` so each DUT host key is confirmed before the tar-over-SSH copy.

This copies the generic setup scripts to:

```text
${CX8_REMOTE_BASE}/assets/
```

Later steps run those copied scripts from the same remote directory on each system, so both nodes execute the same logic.

## Step 3. Run prerequisite and current-state checks

Collect the current CX8/GPU/RDMA state before changing the rail configuration. This step confirms CX8 devices exist, maps `mlx5_0` and `mlx5_1` to Linux network interfaces, shows link state, checks RoCE GIDs, checks firmware/tool visibility, and records recent CX/RDMA/GPU/PCIe error messages.

```bash
./03_prereq_check.sh
```

The console output is intentionally short. The script saves the full raw log locally and prints a summary for each DUT: required missing commands, optional missing commands, CX8 rail mapping, port state, optional `nvidia_peermem` state, and RDMA-topology visibility. The terminal summary colors `PASS` green and `FAIL` red; the saved `summary.md` remains plain text. The default check is non-privileged, so it should ask only for the OS SSH password once per DUT.

Expected before cabling:

- `mlx5_0` and `mlx5_1` exist
- Both ports report Ethernet link layer
- Required tools are present, or missing tools are clearly listed

Expected after cabling:

- Both CX8 ports are active
- Each rail can eventually train at 400 Gb/s

## Step 4. Connect QSFP cables and check link presence

Physically connect the two CX8 ports as two matching rails. Rail 0 stays on QSFP0-to-QSFP0 and rail 1 stays on QSFP1-to-QSFP1. For full-speed validation, use two 400G-capable QSFP cables. Keeping the rails symmetric makes route, RDMA, and NCCL debugging much easier.

```text
Node A QSFP0  <---->  Node B QSFP0
Node A QSFP1  <---->  Node B QSFP1
```

Do not crisscross the rails.

After connecting both cables, run the software-visible link check:

```bash
./04_check_cable_presence.sh
```

This does not configure IP addresses. It checks whether the OS can see each CX8 rail as physically linked by reading `ibv_devinfo`, `ethtool`, carrier, and operstate. It also checks whether each linked rail reports the expected 400G speed for this setup and reports recent CX8 port-module events such as `Cable error` or `High Temperature`.

A green `PASS` means both rails show link at 400G on both DUTs. If `./04_check_cable_presence.sh` reports link down, `Cable error`, or `High Temperature`, stop the software setup path. Let the QSFP cable/module cool, confirm CX8/QSFP airflow, reseat both cable ends on the failed rail, and rerun the cable/link presence check.

If `./04_check_cable_presence.sh` reports `speed_ok=no`, `Speed: 200000Mb/s`, or `200G`, the cable is linked but degraded for this 400G-per-rail / 800G-total setup. Treat that as a cable capability, cable orientation, port, or negotiation issue. You may continue through rail configuration, RoCE setup, basic validation, and perftest for limited-speed functional validation, but do not claim the 800G setup passed until the cable/link presence check reports 400G on both rails.

## Step 5. Configure rail IP and MTU

Bring up the two CX8 Linux network interfaces as independent point-to-point RoCE rails. The script maps `mlx5_0` and `mlx5_1` to their netdevs, sets MTU 9000, brings the interfaces up, and assigns one `/24` subnet per rail. When NetworkManager manages a rail, the script activates an in-memory manual-address profile so an autoconnecting DHCP profile cannot remove the address. On systems where NetworkManager does not manage the rail, it uses the equivalent `ip` commands.

Recommended: use the temporary runtime setup for normal bring-up and validation. This changes the CX8 rail IPs/MTU in the current running OS only. If the system reboots, those settings disappear.

```bash
./05_configure_rails_runtime.sh
```

You can rerun this command safely. The temporary NetworkManager profiles are not written to disk and disappear when NetworkManager restarts or the system reboots. If `./07_validate_setup.sh` later reports that a rail is missing its configured `/24` address after either event, rerun `./05_configure_rails_runtime.sh` and then rerun `./07_validate_setup.sh`.

This step runs privileged commands on each DUT. Expect up to two prompts per DUT: one for OS SSH login and one for remote `sudo`. With SSH keys and passwordless sudo configured, those prompts can be reduced, but this playbook does not require storing passwords.

> [!NOTE]
> Enter the OS SSH password at the first prompt, for example `nvidia@<dut-ip>'s password:`. If a second `[sudo on ...] password for nvidia:` prompt appears, enter the same DUT OS password again. Password input is hidden, so the terminal will not show typed characters.

This assigns:

| Rail | A address | B address |
|---|---:|---:|
| rail0 / `mlx5_0` | `192.168.100.1/24` | `192.168.100.2/24` |
| rail1 / `mlx5_1` | `192.168.101.1/24` | `192.168.101.2/24` |

To use a different private address range, set these in `00_env.local` before running `./05_configure_rails_runtime.sh`:

```bash
STATION_A_RAIL0_CIDR="192.168.100.1/24"
STATION_B_RAIL0_CIDR="192.168.100.2/24"
STATION_A_RAIL1_CIDR="192.168.101.1/24"
STATION_B_RAIL1_CIDR="192.168.101.2/24"
```

Only the CIDR values need to change. The scripts pass the configured values to the remote DUT helpers during rail configuration, basic validation, and RDMA bandwidth testing.

Use `--persist` only when this pair should keep the same rail IP/MTU settings after reboot:

```bash
./05_configure_rails_runtime.sh --persist
```

`--persist` is not required for the normal temporary bring-up flow and should not be treated as a bug workaround. It is the right choice when a longer GDR/Data Direct validation run should keep the rail settings across later reboots, or when this pair is expected to keep the same private CX8 rail addresses for repeated performance runs.

To remove persistent netplan configuration later:

```bash
source ./00_env.sh
ssh "${remote_a}" 'sudo rm -f /etc/netplan/60-cx8-fabric.yaml && sudo netplan apply'
ssh "${remote_b}" 'sudo rm -f /etc/netplan/60-cx8-fabric.yaml && sudo netplan apply'
```

## Step 6. Configure RoCEv2 and GPUDirect runtime settings

Apply the runtime settings needed for RoCE traffic and GPU RDMA validation. The script configures DSCP/ToS handling where the installed tools support it and verifies whether the installed `ib_write_bw` supports the CUDA DMA-BUF/Data Direct GPUDirect path used by the `./08_run_perftest_pair.sh` GDR test.

`nvidia_peermem` is optional on the default path. Some OS images do not load it, and that is not by itself a setup failure. If you specifically need to test the legacy/module-based peermem path, set `CX8_TRY_NVIDIA_PEERMEM=1` in `00_env.local` before running this step.

```bash
./06_configure_roce_gdr_runtime.sh
```

This step also runs privileged commands on each DUT, so expect SSH and remote `sudo` prompts.

> [!NOTE]
> Enter the OS SSH password at the first prompt. If a second `[sudo on ...] password for nvidia:` prompt appears, enter the same DUT OS password again. Password input is hidden.

If the script prints a yellow `WARN` for `cma_roce_tos`, continue with `./07_validate_setup.sh` for basic RoCE validation. If `CX8_TRY_NVIDIA_PEERMEM=1` was set and `nvidia_peermem` fails with `Invalid argument`, that means only the peermem-based GPUDirect path is unavailable; it does not block basic rail/RoCE validation. CUDA DMA-BUF/Data Direct readiness still requires the Script 06 perftest-flag pass and the Script 11 ACS/Data Direct pass before any GDR test.

If `./06_configure_roce_gdr_runtime.sh` says `ib_write_bw` does not advertise `--use_cuda_dmabuf` and `--use_data_direct`, the installed `perftest` package is not ready for the GDR test. Continue with `./07_validate_setup.sh` and the non-`--gdr` host-memory `./08_run_perftest_pair.sh` test first. If GPU-memory RDMA validation is required, run `./10_install_perftest_gdr.sh` to install or build a GDR-capable perftest, then rerun `./06_configure_roce_gdr_runtime.sh`.

When `./10_install_perftest_gdr.sh` installs `ib_write_bw` under `/usr/local`, `./06_configure_roce_gdr_runtime.sh` checks that prefix before the OS default path. This avoids a false warning from `sudo` using an older `/usr/bin/ib_write_bw`.

## Step 7. Validate basic setup

Prove that the two-rail fabric is usable before running NCCL, Ray, vLLM, or application-level tests. This step checks each rail independently so a single-rail cable, MTU, route, or RDMA issue is caught before higher-level software hides the root cause.

```bash
./07_validate_setup.sh
```

This validates:

- Expected IP and MTU on both rails
- Route lookup uses the correct rail
- Jumbo ping succeeds on both rails
- Both ports report 400 G where the driver exposes speed
- RDMA verbs report Ethernet link layer and `PORT_ACTIVE`
- optional `nvidia_peermem` status is reported
- RDMA error counters are visible

`ibv_devinfo` may not report an `active_speed` field on this Ethernet/RoCE path. In that case, `./07_validate_setup.sh` uses `ethtool` 400G link speed as the speed authority and keeps validating the rail.

Only continue to `./08_run_perftest_pair.sh` when `./07_validate_setup.sh` prints `PASS: basic CX8 two-station setup validated` for both DUTs. If basic validation prints any red `ERROR` or `FAIL`, do not run perftest; fix the specific failed rail first. If basic validation prints a yellow warning that one or more rails are below 400G, perftest may be used for limited-speed functional validation, but not as an 800G setup pass.

If `./07_validate_setup.sh` reports `Link detected: no`, `Speed: Unknown`, or `link is not detected`, rerun the cable/link presence check and inspect the reported cable/port before retrying basic validation:

```bash
./04_check_cable_presence.sh
```

If `./07_validate_setup.sh` reports `Link detected: no (Overheat)`, `High Temperature`, or `Cable error`, stop software setup. This is a QSFP/CX8 thermal or cable condition. Let the module cool, verify CX8/QSFP airflow or fan operation, reseat the cable, and swap/replace the failed rail cable if the condition returns.

## Step 8. Optional RDMA bandwidth smoke test

Run a simple RDMA bandwidth test after basic validation passes. The wrapper starts `ib_write_bw` server-side on Node A, runs the matching client on Node B, then saves both logs locally. Use this to confirm that the fabric is not only linked but also moving RDMA traffic at expected bandwidth.

Prerequisite: `./07_validate_setup.sh` must pass on both DUTs. Do not use `./08_run_perftest_pair.sh` to bypass a basic-validation red `ERROR` or `FAIL`. If basic validation passed with yellow speed warnings, perftest is allowed only as a limited-speed functional smoke test; expect lower bandwidth and keep the result labeled as degraded, not 800G accepted.

This step opens SSH sessions to both DUTs. On Linux/macOS, the wrapper uses temporary SSH connection reuse during one script run, so expect one password prompt for Node A and one for Node B in the normal case. On Windows Git Bash/MSYS/Cygwin, SSH multiplexing is disabled automatically because OpenSSH control-master sockets can reset there; extra password prompts are expected. SSH multiplexing affects only SSH connection reuse and does not change RDMA, Data Direct, or perftest behavior. See the troubleshooting guide only if the control host reports an SSH control-socket error.

The wrapper waits until the remote server is ready before starting the client. The default wait is 30 seconds; change `CX8_PERFTEST_SERVER_READY_TIMEOUT` only if the GDR server needs more time on your image, and set it to a positive integer. If readiness fails, the wrapper terminates the detached server before returning an error so a retry does not inherit a stale process.

The non-`--gdr` host-memory test is a functional RDMA smoke test. It uses a simple single-QP configuration and may report less than 400 Gb/s on a 400G rail.

Rail 0:

```bash
./08_run_perftest_pair.sh --rail 0
```

Rail 1:

```bash
./08_run_perftest_pair.sh --rail 1
```

### Required preparation before the GDR variant

Do not add `--gdr` to Script 08 until all of the following are true on both DUTs:

- Script 07 passes the basic setup.
- Script 06 prints `PASS: ib_write_bw supports the CUDA DMA-BUF and Data Direct flags required by the GDR bandwidth-test action`. If it does not, run `./10_install_perftest_gdr.sh`, then rerun Script 06.
- `./11_configure_acs_grub.sh` passes its ACS/Data Direct inspection on both DUTs.

Run the required ACS/Data Direct inspection before the first GDR test:

```bash
./11_configure_acs_grub.sh
```

Only continue when it prints `PASS: both DUTs satisfy the ACS/Data Direct prerequisite for GDR.` If either DUT is not ready, do not start a GDR test. Apply the generated configuration:

```bash
./11_configure_acs_grub.sh --apply
```

Script 11 does not reboot either DUT. After it completes successfully, confirm both DUTs are idle and obtain workload-owner approval before rebooting both systems through the approved lab procedure. Then restore and revalidate the runtime state, including a new ACS inspection:

```bash
./05_configure_rails_runtime.sh
./06_configure_roce_gdr_runtime.sh
./07_validate_setup.sh
./11_configure_acs_grub.sh
```

The temporary Script 05 configuration is sufficient for these tests. Use `--persist` only when the pair must retain the private rail configuration across later reboots or repeated performance runs.

Script 08 repeats the read-only perftest-capability and ACS checks on both DUTs whenever `--gdr` is specified. It starts neither the GDR server nor client unless every prerequisite passes. After the readiness gate passes, run the GPUDirect/Data Direct variant:

The ACS check requires privileged PCIe configuration readback, so the GDR readiness gate may prompt once for remote `sudo` on each DUT before it starts either endpoint.

```bash
./08_run_perftest_pair.sh --rail 0 --gdr
./08_run_perftest_pair.sh --rail 1 --gdr
```

The `--gdr` variant uses CUDA DMA-BUF/Data Direct flags in `ib_write_bw`. It does not require `nvidia_peermem` to be loaded on stacks where DMA-BUF/Data Direct is the supported GPUDirect path. Do not reinstall the OS or driver only because `nvidia_peermem` is absent.

If the DUT has more than one CUDA device, set `CX8_GPU_BDF` in `00_env.local` or pass `--gpu-bdf <PCI_BUS_ID>` so the GDR smoke test uses the intended GPU instead of the default CUDA device 0.

## Step 9. Optional collect logs

Archive setup logs from both DUTs and the control host for support or post-run review.

```bash
./09_collect_logs.sh
```

By default, each script writes logs under `./logs/` in the current playbook `assets` directory on the control host. Remote logs are written under `${CX8_REMOTE_BASE}/logs/` on each DUT.

## Step 10. Optional install or upgrade perftest for GPUDirect/Data Direct

Use this before the first GDR test when Script 06 does not confirm that `ib_write_bw` supports both `--use_cuda_dmabuf` and `--use_data_direct`. Do not attempt GDR with an unqualified perftest binary. This action first tries the OS package path, then falls back to building `linux-rdma/perftest` from source with CUDA header support and installs it under `/usr/local` by default. The fallback source build defaults to pinned upstream tag `26.04.17` rather than a moving branch.

This is not a read-only smoke test. It may install build dependencies or upgrade related OS packages through the DUT package manager. Use `--check-only` first if you only want to inspect the current `ib_write_bw` capability.

```bash
./10_install_perftest_gdr.sh
```

After it passes, rerun Script 06 and complete the Script 11 ACS/Data Direct readiness gate before either GDR rail test:

```bash
./06_configure_roce_gdr_runtime.sh
./11_configure_acs_grub.sh
./08_run_perftest_pair.sh --rail 0 --gdr
./08_run_perftest_pair.sh --rail 1 --gdr
```

For a dry check that makes no package or source changes:

```bash
./10_install_perftest_gdr.sh --check-only
```

If your validation requires a different perftest ref, set `CX8_PERFTEST_REF` in `00_env.local` or pass `--ref <git-ref>`.

## Step 11. Verify and, when required, configure ACS/Data Direct GRUB settings

Run this inspection on both DUTs before the first `./08_run_perftest_pair.sh --gdr` test. It uses `rdma_topo` to verify that the GPU and ConnectX-8 Data Direct function satisfy the required ACS/IOMMU topology. Only the `--apply` form writes the generated ACS/Data Direct GRUB configuration.

The ACS/Data Direct action applies a selective `config_acs=` policy generated by `rdma_topo`; it is not a blanket "disable ACS everywhere" change. See the ACS/Data Direct appendix for an example of the default-vs-expected values it fixes.

Run `./11_configure_acs_grub.sh` from the separate control host. If a DUT is used as the control host, rebooting that DUT will interrupt the playbook and can leave the two systems in different states.

First inspect without changing either system:

```bash
./11_configure_acs_grub.sh
```

Only proceed to GDR when this inspection passes on both DUTs. If either output shows `NOT READY` or an ACS/IOMMU failure and you need GPUDirect/Data Direct validation, apply the GRUB configuration before attempting GDR:

```bash
./11_configure_acs_grub.sh --apply
```

Script 11 does not reboot either DUT. Before rebooting, confirm both DUTs are idle, no shared workload is active, and the workload owner has approved the downtime. Reboot both DUTs after `--apply`, then rerun:

During `--apply`, a yellow `CHANGE REQUIRED` line describes the pre-change ACS/IOMMU state that the command is correcting. A red `FAIL` or `ERROR` still means the apply did not complete and must be investigated before rebooting.

```bash
./05_configure_rails_runtime.sh
./06_configure_roce_gdr_runtime.sh
./07_validate_setup.sh
./11_configure_acs_grub.sh
./08_run_perftest_pair.sh --rail 0 --gdr
./08_run_perftest_pair.sh --rail 1 --gdr
```

The prerequisite/current-state check is optional after reboot unless the OS/packages changed. The cable/link presence check is optional unless the cable/link state is uncertain. Use `./05_configure_rails_runtime.sh --persist` instead only when the pair must keep the same private rail settings across later reboots or repeated performance runs.

## Step 99. Optional cleanup

Use this when you want to return the two DUTs to a clean runtime state before rerunning setup or handing the systems back. Cleanup does not reinstall the OS, update firmware, remove drivers, or reboot the DUTs.

Default cleanup removes the configured rail IP addresses, deletes any in-memory NetworkManager profiles created by the playbook, and resets both CX8 rail MTUs to 1500:

```bash
./99_cleanup_runtime.sh
```

If `./05_configure_rails_runtime.sh` was run with `--persist`, also remove the generated netplan file:

```bash
./99_cleanup_runtime.sh --remove-persist
```

To also bring the two CX8 rail interfaces down:

```bash
./99_cleanup_runtime.sh --down
```

Options can be combined:

```bash
./99_cleanup_runtime.sh --remove-persist --down
```

After cleanup, rerun `./03_prereq_check.sh` for a full state re-check, or resume with `./05_configure_rails_runtime.sh` if the prerequisites and cable/link state are still known-good. If you changed the rail CIDRs in `00_env.local`, cleanup removes those configured CIDRs rather than the default example addresses.

## Troubleshooting quick reference

| Symptom | Meaning | Action |
|---|---|---|
| Basic validation or perftest says a rail IP is missing | The temporary runtime rail IP/MTU setup is not present, commonly after reboot or NetworkManager restart | Rerun `./05_configure_rails_runtime.sh`, then `./06_configure_roce_gdr_runtime.sh` and `./07_validate_setup.sh`. Use `./05_configure_rails_runtime.sh --persist` when the pair must keep rail IPs after reboot or across repeated performance runs. |
| Host-memory `./08_run_perftest_pair.sh` bandwidth is around 220-230 Gb/s on a 400G rail | Single-QP host-memory smoke test is below line-rate; the rail can still be functional | Use the result as basic RDMA proof. For GPU performance acceptance, run the GDR variant on each rail. |
| The `./08_run_perftest_pair.sh --gdr` prerequisite gate stops before starting a server | Perftest GDR flags or ACS/Data Direct topology did not pass on one or both DUTs | Inspect the saved `*_gdr_preflight.log`. Complete Script 10 only if the flags are missing; complete Script 11 and its reboot flow if ACS is not ready. Rerun Scripts 05 through 07 and Script 11 before retrying GDR. |
| The `./08_run_perftest_pair.sh` GDR server exits after the prerequisite gate passed | A runtime CUDA/GDR failure occurred after both static readiness checks, or rail state changed after validation | Inspect the saved server log and current kernel messages. Do not repeat blindly; reconfirm Scripts 05 through 07 and Script 11, then investigate the first new runtime error. |
| A killed or interrupted GDR run leaves CUDA/GPU state stuck | CUDA context cleanup may not have completed after interruption | Stop stale `ib_write_bw` processes. Only if the GPU is idle, no shared workload is active, and policy/workload-owner approval allows it, reset the GPU with `sudo nvidia-smi --gpu-reset -i <gpu-index>`, then rerun the `./08_run_perftest_pair.sh` GDR test. |
| `./06_configure_roce_gdr_runtime.sh` says `nvidia_peermem` is not loaded | Default path can use CUDA DMA-BUF/Data Direct instead of legacy peermem | Continue unless you explicitly need peermem. Before GDR, require the Script 06 perftest-flag pass and Script 11 ACS/Data Direct pass. |

## Next steps

Use the validation level that matches the next workload:

- A `./07_validate_setup.sh` pass means the two nodes have basic two-rail RoCE connectivity: correct rail IPs, MTU, route selection, link state, jumbo ping, and RDMA port visibility.
- A `./08_run_perftest_pair.sh` pass without `--gdr` means host-memory RDMA bandwidth works on each rail. It is not the line-rate acceptance bar for GPU workloads.
- A `./08_run_perftest_pair.sh` pass with `--gdr` means the CUDA DMA-BUF/Data Direct GPU-memory RDMA path works. Use this bar before accepting the setup for GPU-performance work such as NCCL, vLLM, or other distributed AI workloads.

Full NCCL/application validation still requires container RDMA device exposure, NCCL environment selection, and a distributed launcher configuration.

```text
QSFP cables
  -> CX8 link up at 400G x 2
  -> Linux netdevs with rail IPs and MTU 9000
  -> RoCEv2 / libibverbs / RDMA devices mlx5_0 and mlx5_1
  -> Optional ./08_run_perftest_pair.sh --gdr CUDA DMA-BUF/Data Direct validation
  -> NCCL NET/IB transport
  -> PyTorch, vLLM, Ray, MPI, or another distributed application
```

| Layer | What still needs to be set up | How to tell it is working |
|---|---|---|
| Container/runtime | Expose RDMA devices to the application environment. For Docker this usually means `--network host`, `--ipc host`, `--device /dev/infiniband`, and enough locked-memory limit. | Inside the container, `/dev/infiniband` exists and RDMA/NCCL commands can open `mlx5_0` and `mlx5_1`. |
| NCCL | Install NCCL and select the CX8 rails, for example `NCCL_IB_HCA=mlx5_0,mlx5_1`. Use `NCCL_DEBUG=INFO` for first validation. | NCCL log shows `NET/IB` and both `mlx5_0` / `mlx5_1`. |
| Distributed launcher | Configure MPI, PyTorch distributed, Ray, vLLM, or another launcher so both systems join the same job. | The launcher reports both nodes/ranks as alive and the workload uses both GPUs/systems. |
| GPUDirect RDMA | Before GPU-memory testing, verify the required `ib_write_bw` flags and require `./11_configure_acs_grub.sh` to pass on both DUTs. Use Script 10 only when the flags are missing, and Script 11 `--apply` plus reboot only when the ACS inspection is not ready. | The readiness gates pass first; then `./08_run_perftest_pair.sh --rail <0-or-1> --gdr` passes at expected per-rail bandwidth and NCCL can move GPU buffers without host-memory staging. |

The key NCCL validation signal is the transport line in the NCCL log:

```text
Good: NCCL log shows NET/IB and mlx5_0 / mlx5_1
Bad:  NCCL log shows only NET/Socket, which means TCP fallback
```

If the `./08_run_perftest_pair.sh` host-memory RDMA test passes but NCCL falls back to `NET/Socket`, the CX8 fabric is likely usable and the issue is usually in the container, NCCL environment, RDMA device visibility, or launcher configuration. If GPU-memory RDMA is required, validate the supported GPUDirect path explicitly with the GDR variant and NCCL `NET/IB` logs before accepting the setup for performance work.

Inspect ACS/Data Direct readiness with `./11_configure_acs_grub.sh` before the first GDR test. Use `--apply` only after the basic two-rail setup is proven, the inspection is not ready, and GPUDirect/Data Direct validation requires the change.

## Appendix: ACS/Data Direct reference

The platform default ACS policy may be conservative for normal PCIe isolation but not suitable for GPU + CX8 Data Direct GPU-memory RDMA. In that failure mode, `rdma_topo check` reports ACS bit mismatches and the CX8 Data Direct DMA function and GPU are in different IOMMU groups. `./11_configure_acs_grub.sh --apply` uses `rdma_topo write-grub-acs` to write the selective `config_acs=` policy expected by this topology.

Example failure before applying the ACS/Data Direct GRUB configuration:

```text
FAIL ACS for grace_rp 0009:00:00.0 0011101 != xx111x0, 0x1d != 0x1c
FAIL ACS for cx_switch 0009:02:00.0 0011101 != xx110x1, 0x1d != 0x19
FAIL ACS for cx_switch 0009:02:02.0 0011101 != xx101x1, 0x1d != 0x15
FAIL Kernel iommu_group for DMA 0009:03:00.0 and GPU 0009:06:00.0 are not equal
```

Expected success after applying the ACS/Data Direct GRUB configuration and rebooting:

```text
OK ACS for grace_rp 0009:00:00.0 has correct values 0011100 = xx111x0
OK ACS for cx_switch 0009:02:00.0 has correct values 0011001 = xx110x1
OK ACS for cx_switch 0009:02:02.0 has correct values 0010101 = xx101x1
OK Kernel iommu_group for DMA 0009:03:00.0 and GPU 0009:06:00.0 are equal
```

## Tools and commands reference

| Command | Required for basic setup? | Usually provided by | Used in | Purpose |
|---|---:|---|---|---|
| `bash` | Yes | Base OS | All scripts | Runs the setup scripts |
| `ssh` | Yes, on control host | OpenSSH client | All control-host scripts | Connects from control host to each DUT |
| `python3` / `py -3` | Optional control-host alternative | Python 3 | Cross-platform wrapper | Runs `cx8_setup.py` on Windows, macOS, or Linux |
| `sudo` | Yes | Base OS | Rail configuration, RoCE configuration, and cleanup | Privileged network, module, and cleanup commands on DUTs |
| `lspci` | Yes | `pciutils` | Access probe and prerequisite check | Lists PCIe devices; confirms CX8/NVIDIA visibility |
| `nvidia-smi` | Yes | NVIDIA driver | Access probe and prerequisite check | GPU driver visibility, identity, PCI bus IDs |
| `ibdev2netdev` | Yes | DOCA/OFED or `rdma-core` | Access probe through RDMA bandwidth test | Maps `mlx5_0` and `mlx5_1` to Linux netdevs |
| `ibv_devinfo` | Yes | `rdma-core` / OFED | Access probe, prerequisite, cable, and basic validation checks | CX8 port state, link layer, MTU, active speed |
| `ip` | Yes | `iproute2` | Access probe, prerequisite check, rail configuration, and basic validation | Rail IP, link state, MTU, route selection |
| `ethtool` | Yes | `ethtool` package | Access probe, prerequisite/cable checks, rail configuration, and basic validation | Ethernet link status and speed per CX8 netdev |
| `show_gids` | Yes for RoCE validation | RDMA/OFED tools | Access probe and prerequisite check | RoCE GIDs and IPv4 RoCEv2 GID visibility |
| `rdma_topo` | Yes for GPUDirect/Data Direct | DOCA Host / DOCA-OFED | Access probe, prerequisite check, and ACS/Data Direct configuration | GPU/NIC PCIe topology and ACS/Data Direct prerequisites |
| `mlnx_qos` | Yes for RoCE tuning | Mellanox/NVIDIA OFED | Access probe and RoCE configuration | DSCP trust and PFC settings for RoCE |
| `cma_roce_tos` | Yes for RoCE ToS setup | Mellanox/NVIDIA OFED | Access probe and RoCE configuration | RoCE CM ToS/DSCP value for RDMA connection setup |
| `ib_write_bw` | Yes for RDMA bandwidth test | `perftest` package | Access probe and RDMA bandwidth test | RDMA write bandwidth test between DUTs |
| `ping` | Yes for basic rail validation | `iputils-ping` | Basic validation | Jumbo-frame ICMP probes per rail |
| `lsmod` | Yes for GPUDirect status | `kmod` | Prerequisite check and basic validation | Reports whether `nvidia_peermem` is loaded |
| `modprobe` | Optional for peermem setup attempt | `kmod` | RoCE configuration only when `CX8_TRY_NVIDIA_PEERMEM=1` | Attempts to load `nvidia_peermem`; the `./08_run_perftest_pair.sh` GDR test can still use CUDA DMA-BUF/Data Direct when supported |
| `dmesg` | Useful for debug | `util-linux` | Prerequisite check | Recent CX/RDMA/GPU/PCIe error highlights |
| `netplan` | Optional | Ubuntu netplan | Persistent rail configuration | Applies persistent rail IP/MTU configuration |
| `flint` | Optional | NVIDIA MFT | Access probe and prerequisite visibility only | CX8 firmware version and product info |
| `mlxconfig` | Optional | NVIDIA MFT | Access probe visibility only | NIC firmware configuration queries |
| `tar` | Yes on control host and DUTs | Base OS | Asset copy and log packaging | Streams setup helpers to DUTs |
| `sha256sum` or `shasum` | Optional | GNU coreutils or macOS `shasum` | Log packaging | Checksum for the archived log bundle |

If `flint` and `mlxconfig` are missing, that is acceptable for the basic rail setup path because this playbook is not changing NIC firmware configuration.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `mlx5_0` or `mlx5_1` is missing in `./03_prereq_check.sh` output | CX8 driver not loaded, PCIe issue, or incomplete software image | Run `lspci \| grep -i mellanox` and `ibdev2netdev` on the DUT. Confirm DOCA/OFED and the NVIDIA driver are installed. Contact NVIDIA support if the HCA is not visible. |
| Required command missing in `./01_probe_access.sh` output (other than `flint` / `mlxconfig`) | Software image missing a core RDMA or network tool | Stop setup. Install or restore the missing package (`rdma-core`, `perftest`, `ethtool`, etc.) or contact NVIDIA support. |
| `./04_check_cable_presence.sh` reports link down on one or both rails | Cable not seated, wrong QSFP port, or crisscrossed rails | Confirm QSFP0-to-QSFP0 and QSFP1-to-QSFP1 with no crisscross. Reseat both ends on the failed rail and rerun the cable/link presence check. |
| `./04_check_cable_presence.sh` reports `Cable error` or `High Temperature` | QSFP module thermal or cable fault | Stop software setup. Let the module cool, verify CX8/QSFP airflow, and reseat the cable. Swap the cable with the other rail or replace it if the condition returns. |
| `./04_check_cable_presence.sh` reports `speed_ok=no` or `200G` instead of `400G` | Cable not 400G-capable, degraded negotiation, or port/module issue | Replace with 400G-capable QSFP cables. Limited-speed rail/RoCE/perftest validation is possible, but do not claim 800G pass until both rails report 400G. |
| Rail or RoCE runtime configuration fails with permission errors | Missing `sudo` access for the SSH user | Ensure the `nvidia` user (or `CX8_OS_USER`) can run privileged network commands. Enter the DUT OS password at both the SSH and `[sudo]` prompts. |
| `./07_validate_setup.sh` reports missing rail IP or wrong MTU | `./05_configure_rails_runtime.sh` did not complete, or NetworkManager/system restart removed the temporary in-memory rail profile | Rerun `./05_configure_rails_runtime.sh` and then `./07_validate_setup.sh`. Do not use a raw `ip addr` assignment on a NetworkManager-managed rail because its DHCP profile can remove that address. |
| `./07_validate_setup.sh` jumbo ping fails on one rail | MTU mismatch, wrong route, or cable/link issue on that rail | Check `ip addr` and `ip route` on both DUTs for the failed rail subnet. Rerun `./04_check_cable_presence.sh` for link health, then `./05_configure_rails_runtime.sh` for IP/MTU. |
| `./07_validate_setup.sh` reports `Link detected: no` or `Speed: Unknown` | Physical link not up despite earlier actions | Rerun `./04_check_cable_presence.sh`. Inspect cable, port, and module before retrying `./07_validate_setup.sh`. |
| `./07_validate_setup.sh` reports `Link detected: no (Overheat)`, `High Temperature`, or `Cable error` | QSFP/CX8 thermal or cable condition | Stop software setup. Let the module cool, verify airflow, reseat the cable, and swap/replace the failed rail cable if the condition returns. |
| `./07_validate_setup.sh` warns that rails are below 400G | Degraded link speed from cable, port, or negotiation | Continue only for limited-speed functional validation. Do not treat perftest results as an 800G pass until `./04_check_cable_presence.sh` reports 400G on both rails. |
| `./06_configure_roce_gdr_runtime.sh` prints a yellow `WARN` for `cma_roce_tos` | RoCE ToS tool not available or not applicable on this image | Continue with `./07_validate_setup.sh` for basic RoCE validation. ToS tuning is optional for the basic fabric path. |
| `nvidia_peermem` is not loaded | Normal on some OS images when CUDA DMA-BUF/Data Direct is the GPUDirect path | Continue. Basic rail/RoCE validation is not blocked. Before GDR, require the Script 06 perftest-flag pass and Script 11 ACS/Data Direct pass. |
| `nvidia_peermem` fails to load (`Invalid argument`) after `CX8_TRY_NVIDIA_PEERMEM=1` | The optional peermem-based GPUDirect path is unavailable on this stack | Do not treat this as a basic setup failure. Use the CUDA DMA-BUF/Data Direct path only after its Script 06 and Script 11 readiness gates pass. |
| Script 06 or the Script 08 GDR prerequisite gate says `ib_write_bw` does not support `--use_cuda_dmabuf` or `--use_data_direct` | Installed `perftest` is too old or was built without CUDA DMA-BUF/Data Direct support | Run host-memory Script 08 first for basic RDMA. If GPU-memory RDMA is required, run Script 10, rerun Script 06, and complete the Script 11 ACS readiness gate before either GDR rail test. |
| The Script 08 GDR prerequisite gate stops before starting a server | Perftest GDR flags or ACS/Data Direct topology did not pass on one or both DUTs | Inspect `*_gdr_preflight.log`. Use Script 10 only for missing flags. Use Script 11 `--apply` plus reboot only for an ACS-not-ready result; then rerun Scripts 05 through 07 and Script 11 before retrying GDR. |
| A GDR test reaches `mlx5dv_reg_dmabuf_mr` and fails with error 524 despite the prerequisite gate | Prerequisite enforcement was bypassed, topology changed after the check, or a new kernel/driver defect exists | Stop and collect the GDR logs plus current `rdma_topo check` and kernel messages. Do not treat this as the normal setup path or repeatedly rerun GDR. |
| `./08_run_perftest_pair.sh` fails or shows very low bandwidth | Wrong rail selected, firewall, or degraded link speed | Confirm `./07_validate_setup.sh` passed. Run perftest per rail separately. Check `./04_check_cable_presence.sh` speed reports. Verify rail IPs and routes match the selected `--rail` index. |
| NCCL shows `NET/Socket` instead of `NET/IB` after fabric passes | Container missing RDMA devices, wrong `NCCL_IB_HCA`, or launcher config | Expose `/dev/infiniband` to the runtime. Set `NCCL_IB_HCA=mlx5_0,mlx5_1`. Use `NCCL_DEBUG=INFO` and confirm both HCAs appear in the log. |
| `rdma_topo check` reports ACS/Data Direct prerequisites not met | PCIe topology or ACS configuration not set for GPUDirect/Data Direct | Complete basic two-rail setup first. Use Script 11 `--apply` only when GDR validation is required, then reboot. Restore runtime setup with Scripts 05 through 07 and require Script 11 to pass before either GDR rail test. |
| SSH control-socket reset or repeated connection-reuse failure | SSH multiplexing is incompatible with the control-host environment | Retry with `CX8_DISABLE_SSH_MUX=1 ./08_run_perftest_pair.sh --rail <0-or-1>`. This affects only SSH connection reuse, not RDMA, Data Direct, or perftest behavior. Windows Git Bash/MSYS/Cygwin disables multiplexing automatically. |
| Rail settings lost after reboot | `./05_configure_rails_runtime.sh` was run without `--persist` | Expected for runtime setup. Rerun the command after reboot, or use `--persist` if settings must survive reboot. |
| Persistent netplan needs removal | `./05_configure_rails_runtime.sh --persist` was used and you want to undo it | Run `./99_cleanup_runtime.sh --remove-persist` or manually remove `/etc/netplan/60-cx8-fabric.yaml` on both DUTs and `sudo netplan apply`. |

### Stop conditions

Stop and inspect before changing anything else if:

- `mlx5_0` or `mlx5_1` is missing
- A rail remains down after the cable is connected
- Jumbo ping fails on either rail
- RDMA tools are missing
- CUDA DMA-BUF/Data Direct perftest flags are unavailable
- `rdma_topo check` says ACS/Data Direct prerequisites are not met

Inspect ACS/Data Direct readiness before the first GDR test. Add the generated GRUB configuration with Script 11 `--apply` only after the basic two-rail setup is proven, the inspection is not ready, and GPUDirect/Data Direct validation requires it.
