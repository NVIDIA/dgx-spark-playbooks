# Set Up Local Network Access

> NVIDIA Sync helps set up and configure SSH access

## Table of Contents

- [Overview](#overview)
- [Connect with NVIDIA Sync](#connect-with-nvidia-sync)
- [Connect with Manual SSH](#connect-with-manual-ssh)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

DGX Spark lets you access it as a desktop (i.e. connected to keyboard, mouse and monitor)
or as a remote device over a network.

The playbook shows you two different paths to connect to your DGX Spark via SSH over a network:

* **With NVIDIA Sync**: Use this path for a user-friendly interface that handles SSH and remote applications on your Spark
* **Manual SSH**: Use this path to get under the hood and work in a terminal

The first path gives you a click through experience that you can use to connect to the Spark anytime. 
The second path uses manual commands, some of which will need to be repeated every time you connect to the Spark. 

## What you'll accomplish

You will establish secure SSH access to your DGX Spark and then connect to the DGX Dashboard as an example of launching a remote application.

## What to know before starting

- With NVIDIA Sync: How to install a desktop application; the basics of the Sync application ([see documentation here](https://docs.nvidia.com/sync/latest/direct-connections.html))
- Manual SSH: Terminal/command usage and the basics of SSH configuration, including port forwarding

## Prerequisites

- Your DGX Spark [is set up](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html)
- You have a user account on the Spark, i.e. username and password
- Your laptop and the Spark are on the same network
- You have the Spark's default [mDNS hostname](https://en.wikipedia.org/wiki/Multicast_DNS), or its IP address on the network 

## Time & risk

- **Time estimate:** 5-10 minutes
- **Risk level:** Low - SSH setup involves credential configuration but no system-level changes to the DGX Spark device
- **Rollback:** SSH key removal can be done by editing `~/.ssh/authorized_keys` on your DGX Spark.
- **Last Updated:** 10/28/2025
  * Minor copyedits

## Connect with NVIDIA Sync

## Step 1. Install NVIDIA Sync on your laptop

NVIDIA Sync is a desktop app that connects your laptop to remote devices over a local network. 
It replaces running manual commands in a terminal with a configured, click through interface.
You can use it as a simple interface to manage SSH access and launch development tools on your DGX Spark. 

::spark-download

**For Windows**: After download, double click the `.exe` installer and follow instructions.

**For macOS**: After download, open `nvidia-sync.dmg`, drag and drop it into the Applications folder, then launch it from Applications. 

**For Debian/Ubuntu**: Install from the NVIDIA APT repository. 

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

**Success**:  A "Let's Get Started" modal opens and asks you to read and agree to the EULA.

## Step 2. Complete onboarding by agreeing to the EULA and selecting applications to launch

Click the link to the EULA, read the EULA, and then select "Agree" in the "Let's Get Started" modal.

NVIDIA Sync will then prompt you to choose local developer applications for it to launch. 
You can always add more applications later in the Settings window.

Select "Next" to proceed. 


## Step 3. Add your DGX Spark to NVIDIA Sync

> [!NOTE]
> Your DGX Spark must be on the same network as your laptop, and you must know its default mDNS hostname or its IP address.
> [Learn more about DGX Spark networking in the documentation](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html). 

Once onboarding completes, NVIDIA Sync shows a modal while it searches for mDNS devices.
If your network allows mDNS broadcasting, NVIDIA Sync should detect your Spark (e.g. `spark-abcd.local`) and prompt you to select it.

Otherwise, the modal will transition to a form requesting specific fields to connect to your Spark:  
- **Name**: A descriptive name you will remember (e.g., "My Home Spark")
- **Hostname or IP**: The mDNS hostname (e.g. `spark-abcd.local`) or IP address of your Spark
- **Username**: Your DGX Spark user account name
- **Password**: Your DGX Spark user account password

Fill out the fields and select "Add".

**Success**: The form will transition to a modal prompting you to get started.

> [!NOTE]
> The password is used to configure SSH key-based authentication only when you add the device. It is not persisted or logged.

## Step 4. Connect to your DGX Spark and launch the DGX Dashboard 

Select "Get Started" in the modal to connect to your Spark.

The device window will open near the task or menu bar and then expand and populate with apps that you can launch on your Spark.

The DGX Dashboard is a pre-installed web application that helps you monitor and manage the system remotely. 

To launch it, select the DGX Dashboard icon in the device window.

When it opens, you will be prompted to log in using your username and password for the Spark.

**Success**: The DGX Dashboard web app opens in your browser and you see main screen.

## Step 5. Next steps

- Try related playbooks that use the NVIDIA Sync application: 
  - [Explore the DGX Dashboard](https://build.nvidia.com/spark/dgx-dashboard/instructions)
  - [Open WebUI on your Spark](https://build.nvidia.com/spark/open-webui/overview)
  - [NVIDIA AI Workbench on your Spark](https://build.nvidia.com/spark/rag-ai-workbench)

- Learn more about NVIDIA Sync:
  - [See how to use the NVIDIA Sync Tailscale integration](https://docs.nvidia.com/sync/latest/tailscale.html#nvidia-sync-tailscale)
  - [See how to use the NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html)

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

Collect the required connection details for your DGX Spark:

- **Username**: Your DGX Spark user account name
- **Password**: Your DGX Spark account password
- **Hostname**: Your device's mDNS hostname (from the Quick Start Guide, e.g., `spark-abcd.local`)
- **IP Address**: An alternative only needed if mDNS doesn't work on your network as described below

In some network configurations, like complex corporate environments, mDNS won't work as expected 
and you'll have to use your device's IP address directly to connect. You'll know you are in this situation when
you try to SSH and the command hangs indefinitely or you get an error like:

```
ssh: Could not resolve hostname spark-abcd.local: Name or service not known
```

**Testing mDNS Resolution**

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
- Log into your router's admin panel to find the IP Address
- Connect a display, keyboard, and mouse to check from the Ubuntu desktop

## Step 3. Test initial connection

Connect to your DGX Spark for the first time to verify basic connectivity:

```bash
## Connect using mDNS hostname (preferred)
ssh <YOUR_USERNAME>@<SPARK_HOSTNAME>.local
```

or

```bash
## Alternative: Connect using IP address
ssh <YOUR_USERNAME>@<DEVICE_IP_ADDRESS>
```

Replace placeholders with your actual values:
- `<YOUR_USERNAME>`: Your DGX Spark account name
- `<SPARK_HOSTNAME>`: Device hostname without `.local` suffix
- `<DEVICE_IP_ADDRESS>`: Your device's IP address

On first connection, you'll see a host fingerprint warning. Type `yes` and press Enter,
then enter your password when prompted.

## Step 4. Verify remote connection

Once connected, confirm you're on the DGX Spark device:

```bash
## Check hostname
hostname
## Check system information
uname -a
## Exit the session
exit
```

## Step 5. Use SSH tunneling for web applications

To access web applications running on your DGX Spark, use SSH port
forwarding. In this example we'll access the DGX Dashboard web application.

> [!NOTE]
> DGX Dashboard runs on localhost, port 11000.

Open the tunnel:

```bash
## local port 11000 → remote port 11000
ssh -L 11000:localhost:11000 <YOUR_USERNAME>@<SPARK_HOSTNAME>.local
```

After establishing the tunnel, access the forwarded web app in your browser: [http://localhost:11000](http://localhost:11000)

## Step 6. Next steps

With SSH access configured, you can:
- Open persistent terminal sessions: `ssh <YOUR_USERNAME>@<SPARK_HOSTNAME>.local`.
- Forward web application ports: `ssh -L <local_port>:localhost:<remote_port> <YOUR_USERNAME>@<SPARK_HOSTNAME>.local`.

## Troubleshooting

## Possible issues connecting via NVIDIA Sync

| Symptom | Cause | Fix |
|---------|--------|-----|
| Device name doesn't resolve | mDNS blocked on network | Use IP address instead of hostname.local |
| Connection refused/timeout | DGX Spark not booted or SSH not ready | Wait for device boot completion; SSH available after updates finish |
| Authentication failed | SSH key setup incomplete | Re-run device setup in NVIDIA Sync; check credentials |

## Possible issues connecting via manual SSH

| Symptom | Cause | Fix |
|---------|--------|-----|
| Device name doesn't resolve | mDNS blocked on network | Use IP address instead of hostname.local |
| Connection refused/timeout | DGX Spark not booted or SSH not ready | Wait for device boot completion; SSH available after updates finish |
| Port forwarding fails | Service not running or port conflict | Verify remote service is active; try different local port |
