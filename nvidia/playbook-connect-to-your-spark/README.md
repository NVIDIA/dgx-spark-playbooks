# Use NVIDIA Sync to Connect Remotely to Your AI Compute

> Reach your machine remotely via NVIDIA Sync

## Table of Contents

- [Overview](#overview)
- [Connect with NVIDIA Sync](#connect-with-nvidia-sync)
- [Enable Tailscale](#enable-tailscale)
- [Connect with Manual SSH](#connect-with-manual-ssh)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

[NVIDIA Sync](https://docs.nvidia.com/sync/latest/index.html) is a free desktop app that simplifies using a remote device over a local network or Tailscale.
It replaces running commands in a terminal with a configured, click-through interface that lets you connect to remote devices, launch applications and services on the remote, and then access them from your laptop.

This playbook shows how to install NVIDIA Sync on your laptop and use it to connect to a DGX Spark on your home network. 
The instructions, except for the DGX Dashboard, generalize to any remote device with a Debian based Linux operating system, including a DGX Station.

## What you'll accomplish

- **Use NVIDIA Sync**: Connect to your DGX Spark and launch remote applications from your laptop
  - One time: Install NVIDIA Sync and add your Spark with its SSH credentials
  - Optional one time: Set up the [Tailscale integration](tailscale-via-sync.md) to reach your Spark from another network
  - Repeat use: Connect in NVIDIA Sync and launch a terminal, the [DGX Dashboard](https://docs.nvidia.com/dgx/dgx-spark/dgx-dashboard.html#spark-dgx-dashboard), or the [Resource Monitor](https://docs.nvidia.com/sync/latest/resource-monitor.html)

- **Advanced path:** Use SSH commands to connect to your Spark and forward the DGX Dashboard to your laptop.

## What to know before starting

- Required: You must have your username and password to access the DGX Spark
- Required for initial setup: Your Spark must be on the same network as your laptop ([see here for DGX Spark](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html))
- Required: You must have the Spark's IP address on the network or know its mDNS broadcast name

## Supported hardware platforms

- Local device: You can install the NVIDIA Sync desktop app on Windows 11, macOS, or Ubuntu 24.04 or later
- Remote target device: Any Debian-based device accessible over SSH, including DGX Spark and DGX Station.

## Prerequisites

- Required: Your laptop should be Windows 11, macOS or Ubuntu 24.04 or higher
- Required for initial setup: Your laptop and Spark should be on the same network. After you enable Tailscale, they can be on different networks.
- Required: You know the Spark's mDNS hostname or its IP address on the network
- Suggested: You have signed up for a free [Tailscale account](https://login.tailscale.com/start) using a non-corporate email address

## Time & risk

- **Estimated time:** 5 MIN
- **Risk level:** Low
- **Rollback:** Remove SSH keys by editing `~/.ssh/authorized_keys` on the remote device; disconnect or remove the device in NVIDIA Sync
- **Last Updated:** 09/24/2026
  - Updating NVIDIA Sync instructions and added Tailscale instructions.

## Connect with NVIDIA Sync

## Step 1. Install NVIDIA Sync on your laptop (one time)

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

## Step 2. Agree to the EULA and select applications to launch (one time)

1. Click the link to read the EULA, and then select **Agree** in the "Let's Get Started" modal.
2. Then, NVIDIA Sync will show you any IDEs you have installed locally to select for launch on a remote device.
3. Select **Next** to proceed.

## Step 3. Find your DGX Spark on the network and add it to NVIDIA Sync (one time)

DGX Spark devices broadcast their hostname through mDNS, and NVIDIA Sync will open a modal while it searches the network for broadcasting devices.

- If your DGX Spark is on a home network, NVIDIA Sync should discover the device name (for example, `spark-abcd.local`) and prompt you to select it.
- If your Spark doesn't appear, select "Add a device manually" and enter the information below to access the device.

- **Name:** A descriptive name you will remember (for example, "My Home Lab")
- **Hostname or IP:** The device's mDNS hostname or IP address
- **Username:** The user account name
- **Password:** The associated account password

Then, select **Add** to proceed.

**Success:** The form will transition and prompt you to get started.

> [!NOTE]
> Your password is only temporarily used for SSH authentication and configuring key-based authentication when you add the device. It is not persisted or logged.

## Step 4. Connect to your remote device and launch a terminal and the DGX Dashboard (repeat use)

1. Select **Get Started** in the addition confirmation modal to initiate the connection.
2. The task bar utility will open and show that it is connecting to the device.
3. When the connection succeeds, the utility will show apps to can launch on the remote.
4. Select the Terminal app to launch a terminal on the remote.
5. Select the DGX Dashboard app to launch it on the remote. It will open in your browser. 
6. Select the Resource Monitor app to launch it. It will open in a new application window.

**Success:** The applications start and open.

## Next steps

- Use NVIDIA Sync to [add Tailscale](tailscale-via-sync.md) so you can connect to your device from anywhere
- Use NVIDIA Sync to [cluster two or more DGX Spark devices](https://build.nvidia.com/playbooks/connect-multiple-sparks/connect-devices)
- Use NVIDIA Sync to [launch a vLLM container](https://build.nvidia.com/playbooks/vllm)

## Enable Tailscale

## Step 1. Decide whether you need Tailscale

Tailscale is a free service that creates secure tunnels between devices on different networks.
It lets you connect devices that aren't exposed to the internet, for example a DGX Spark on your home network, to
other devices that are on a different network, for example your laptop when you are working from a coffee shop.

NVIDIA Sync has a Tailscale integration that uses those tunnels to reach your device when a direct local connection is unavailable.
You do not need a separate Tailscale app on your laptop.

## Step 2. Create a Tailscale account

Tailscale requires you to sign up through a third party auth provider like Google or GitHub.

[Go here](https://login.tailscale.com/start) to create a Tailscale account.

## Step 3. Enable Tailscale in NVIDIA Sync

1. Open NVIDIA Sync **Settings** and select **Tailscale**.
2. Select **Enable Tailscale**. It can take a while for Tailscale to pick up the request.
3. In the browser window, sign in to Tailscale and select **Connect**.
4. Wait for **Login Successful**, then return to NVIDIA Sync.

## Step 4. Add your device to Tailscale

1. In NVIDIA Sync **Settings → Tailscale**, select **Add a Device**.
2. Select the device you already added to NVIDIA Sync.
3. Use the link in the dialog to create a Tailscale authentication key with the default settings.
4. Paste the key into NVIDIA Sync and select **Add Device**.
5. A terminal opens on the device. Follow its prompts to install and authenticate Tailscale on the device.

**Success:** The device appears in the Tailscale device list in NVIDIA Sync.

## Step 5. Connect when you are away

1. When your laptop is away from the local network, connect to the device in NVIDIA Sync as usual.
2. Check the connection indicator. NVIDIA Sync selects Tailscale automatically when a direct connection is unavailable.

For more help, see [Tailscale Connections in the NVIDIA Sync User Guide](https://docs.nvidia.com/sync/latest/tailscale.html#nvidia-sync-tailscale).

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
- **IP Address:** Use this if your device does not advertise an mDNS hostname, as with DGX Station, or if mDNS does not work on your network

In some network configurations, such as complex corporate environments, mDNS will not work as expected
and you will have to use your device's IP address directly to connect. A name-resolution error looks like this:

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
- If the device has a desktop, connect a display, keyboard, and mouse to check its IP address

## Step 3. Test initial connection

Connect to your hardware platform for the first time to verify basic connectivity:

```bash
## Connect using an mDNS hostname, if your device advertises one
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

To access web applications running on your remote device, use SSH port
forwarding. This DGX OS example uses the DGX Dashboard. For another Debian-based device, use the port of an application running on that device.

> [!NOTE]
> On DGX OS, DGX Dashboard runs on localhost, port 11000.

Open the tunnel:

```bash
## local port 11000 → remote port 11000
ssh -L 11000:localhost:11000 <YOUR_USERNAME>@<DEVICE_HOSTNAME>.local
```

If you use the device's IP address instead of its mDNS hostname, open the tunnel with:

```bash
ssh -L 11000:localhost:11000 <YOUR_USERNAME>@<DEVICE_IP_ADDRESS>
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
