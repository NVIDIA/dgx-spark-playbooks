# Connect Remotely to Your AI Compute

> Reach your machine remotely via NVIDIA Sync

## Table of Contents

- [Overview](#overview)
- [Connect with NVIDIA Sync](#connect-with-nvidia-sync)
- [Connect with Manual SSH](#connect-with-manual-ssh)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Your hardware platform can be used as a local desktop (keyboard, mouse, and monitor) or as a remote device over a network.

This playbook shows two paths to connect to your hardware platform over SSH on a local network:

* **With NVIDIA Sync:** A desktop app that configures SSH and launches remote applications through a click-through interface
* **Manual SSH:** Terminal commands for direct SSH access and port forwarding

NVIDIA Sync gives you a reusable connection you can return to later. Manual SSH uses commands that you may need to repeat each time you connect.

## What you'll accomplish

You'll establish secure SSH access to your hardware platform and then open the DGX Dashboard as an example of launching a remote web application.

## What to know before starting

**Required:**

- With NVIDIA Sync: How to install a desktop application; the basics of NVIDIA Sync ([documentation](https://docs.nvidia.com/sync/latest/direct-connections.html))
- Manual SSH: Terminal/command usage and the basics of SSH configuration, including port forwarding

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | NVIDIA Sync for remote SSH; Manual SSH as an alternative | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Hardware platform powered on, networked, and reachable from your laptop on the same network
- A user account on the hardware platform (username and password)
- The hardware platform's mDNS hostname or its IP address on the network

**Software requirements**

- With NVIDIA Sync path: NVIDIA Sync installed on your laptop (download steps are in the **Connect with NVIDIA Sync** tab)
- Manual SSH path: An SSH client on your laptop (`ssh -V`)
- Web browser access to forwarded ports (for example, DGX Dashboard on port `11000`)

## Time & risk

- **Estimated time:** 5–10 MIN
- **Risk level:** Low
  - SSH setup configures credentials and key-based access without system-level changes to the hardware platform
- **Rollback:** Remove SSH keys by editing `~/.ssh/authorized_keys` on the hardware platform; disconnect or remove the device in NVIDIA Sync
- **Last Updated:** 08/03/2026
  - Remote SSH access with NVIDIA Sync or Manual SSH, including verification and rollback guidance

## Connect with NVIDIA Sync

## Step 1. Install NVIDIA Sync on your laptop

NVIDIA Sync is a desktop app that connects your laptop to remote devices over a local network.
It replaces running manual commands in a terminal with a configured, click-through interface.
You can use it to manage SSH access and launch development tools on your hardware platform.

::spark-download

**For Windows:** After download, double-click the `.exe` installer and follow the instructions.

**For macOS:** After download, open `nvidia-sync.dmg`, drag and drop it into the Applications folder, then launch it from Applications.

**For Debian/Ubuntu:** Install from the NVIDIA APT repository.

* First, configure the package repository:

  ```bash
  curl -fsSL  https://workbench.download.nvidia.com/stable/linux/gpgkey  |  sudo tee -a /etc/apt/trusted.gpg.d/ai-workbench-desktop-key.asc
  echo "deb https://workbench.download.nvidia.com/stable/linux/debian default proprietary" | sudo tee -a /etc/apt/sources.list
  ```
* Then, update package lists:

  ```bash
  sudo apt update
  ```
* Finally, install NVIDIA Sync:

  ```bash
  sudo apt install nvidia-sync
  ```

**Success:** A "Let's Get Started" modal opens and asks you to read and agree to the EULA.

## Step 2. Complete onboarding by agreeing to the EULA and selecting applications to launch

Click the link to the EULA, read the EULA, and then select **Agree** in the "Let's Get Started" modal.

NVIDIA Sync will then prompt you to choose local developer applications for it to launch.
You can always add more applications later in the Settings window.

Select **Next** to proceed.

## Step 3. Add your hardware platform to NVIDIA Sync

> [!NOTE]
> Your hardware platform must be on the same network as your laptop, and you must know its mDNS hostname or its IP address.
> See the documentation linked under **Resources** for first-boot and networking guidance for your hardware platform.

Once onboarding completes, NVIDIA Sync shows a modal while it searches for mDNS devices.
If your network allows mDNS broadcasting, NVIDIA Sync should detect your hardware platform (for example, `spark-abcd.local`) and prompt you to select it.

Otherwise, the modal will transition to a form requesting specific fields to connect:

- **Name:** A descriptive name you will remember (for example, "My Home Lab")
- **Hostname or IP:** The mDNS hostname (for example, `spark-abcd.local`) or IP address of your hardware platform
- **Username:** Your hardware platform user account name
- **Password:** Your hardware platform user account password

Fill out the fields and select **Add**.

**Success:** The form will transition to a modal prompting you to get started.

> [!NOTE]
> The password is used to configure SSH key-based authentication only when you add the device. It is not persisted or logged.

## Step 4. Connect to your hardware platform and launch the DGX Dashboard

Select **Get Started** in the modal to connect to your hardware platform.

The device window will open near the task or menu bar and then expand and populate with apps that you can launch on the hardware platform.

The DGX Dashboard is a pre-installed web application that helps you monitor and manage the system remotely.

To launch it, select the DGX Dashboard icon in the device window.

When it opens, you will be prompted to log in using your username and password for the hardware platform.

**Success:** The DGX Dashboard web app opens in your browser and you see the main screen.

## Step 5. Next steps

- Learn more about NVIDIA Sync:
  - [NVIDIA Sync Tailscale integration](https://docs.nvidia.com/sync/latest/tailscale.html#nvidia-sync-tailscale)
  - [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html)
- Try related workflows that use NVIDIA Sync, such as remote development tools or browser-based apps launched through the device window.

## Connect with Manual SSH

## Step 1. Verify SSH client availability

Confirm that you have an SSH client installed on your system. Most modern operating systems
include SSH by default. Run the following in your terminal:

```bash
## Check SSH client version
ssh -V
```

Expected output should show OpenSSH version information.

## Step 2. Gather connection information

Collect the required connection details for your hardware platform:

- **Username:** Your hardware platform user account name
- **Password:** Your hardware platform account password
- **Hostname:** Your device's mDNS hostname (for example, `spark-abcd.local`)
- **IP Address:** An alternative only needed if mDNS does not work on your network as described below

In some network configurations, such as complex corporate environments, mDNS will not work as expected
and you will have to use your device's IP address directly to connect. You will know you are in this situation when
you try to SSH and the command hangs indefinitely or you get an error like:

```
ssh: Could not resolve hostname spark-abcd.local: Name or service not known
```

**Testing mDNS resolution**

To test if mDNS is working, use the `ping` utility:

```bash
ping spark-abcd.local
```

If mDNS is working and you can SSH using the hostname, you should see something like this:

```
$ ping -c 3 spark-abcd.local
PING spark-abcd.local (10.9.1.9): 56 data bytes
64 bytes from 10.9.1.9: icmp_seq=0 ttl=64 time=6.902 ms
64 bytes from 10.9.1.9: icmp_seq=1 ttl=64 time=116.335 ms
64 bytes from 10.9.1.9: icmp_seq=2 ttl=64 time=33.301 ms
```

If mDNS is **not** working, indicating you will have to use your IP directly, you will see something like this:

```
$ ping -c 3 spark-abcd.local
ping: cannot resolve spark-abcd.local: Unknown host
```

If none of these work, you'll need to:

- Log into your router's admin panel to find the IP address
- Connect a display, keyboard, and mouse to check from the Ubuntu desktop

## Step 3. Test initial connection

Connect to your hardware platform for the first time to verify basic connectivity:

```bash
## Connect using mDNS hostname (preferred)
ssh <YOUR_USERNAME>@<DEVICE_HOSTNAME>.local
```

or

```bash
## Alternative: Connect using IP address
ssh <YOUR_USERNAME>@<DEVICE_IP_ADDRESS>
```

Replace placeholders with your actual values:

- `<YOUR_USERNAME>`: Your hardware platform account name
- `<DEVICE_HOSTNAME>`: Device hostname without the `.local` suffix
- `<DEVICE_IP_ADDRESS>`: Your device's IP address

On first connection, you'll see a host fingerprint warning. Type `yes` and press Enter,
then enter your password when prompted.

## Step 4. Verify remote connection

Once connected, confirm you're on the hardware platform:

```bash
## Check hostname
hostname
## Check system information
uname -a
## Exit the session
exit
```

## Step 5. Use SSH tunneling for web applications

To access web applications running on your hardware platform, use SSH port
forwarding. In this example you'll access the DGX Dashboard web application.

> [!NOTE]
> DGX Dashboard runs on localhost, port 11000.

Open the tunnel:

```bash
## local port 11000 → remote port 11000
ssh -L 11000:localhost:11000 <YOUR_USERNAME>@<DEVICE_HOSTNAME>.local
```

After establishing the tunnel, access the forwarded web app in your browser: [http://localhost:11000](http://localhost:11000)

## Step 6. Next steps

With SSH access configured, you can:

- Open persistent terminal sessions: `ssh <YOUR_USERNAME>@<DEVICE_HOSTNAME>.local`
- Forward web application ports: `ssh -L <local_port>:localhost:<remote_port> <YOUR_USERNAME>@<DEVICE_HOSTNAME>.local`

## Troubleshooting

## Possible issues connecting via NVIDIA Sync

| Symptom | Cause | Fix |
|---------|--------|-----|
| Device name doesn't resolve | mDNS blocked on network | Use IP address instead of hostname.local |
| Connection refused/timeout | Hardware platform not booted or SSH not ready | Wait for device boot completion; SSH available after updates finish |
| Authentication failed | SSH key setup incomplete | Re-run device setup in NVIDIA Sync; check credentials |

## Possible issues connecting via Manual SSH

| Symptom | Cause | Fix |
|---------|--------|-----|
| Device name doesn't resolve | mDNS blocked on network | Use IP address instead of hostname.local |
| Connection refused/timeout | Hardware platform not booted or SSH not ready | Wait for device boot completion; SSH available after updates finish |
| Port forwarding fails | Service not running or port conflict | Verify remote service is active; try a different local port |

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
