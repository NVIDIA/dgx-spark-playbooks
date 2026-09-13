# Set Up Tailscale for Remote Access

> Encrypted SSH from any network without port forwarding or firewall changes

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Tailscale creates an encrypted peer-to-peer mesh network that lets you reach your hardware platform from anywhere without complex firewall configuration or port forwarding. Install Tailscale on your hardware platform and on your client devices to join a private "tailnet," where each device gets a stable private IP address and hostname. You can then use SSH over that mesh whether you are at home, at work, or on another network.

## What you'll accomplish

You'll install and authenticate Tailscale on your **hardware platform** and on one or more client devices, then SSH into the hardware platform from a different network using a Tailscale hostname or IP. Traffic is encrypted end to end, and NAT traversal is handled for you.

## What to know before starting

**Required:**

- Experience with the Linux command line
- Basic SSH concepts and usage
- Installing packages with `apt` on Ubuntu-based systems
- A user account with sudo privileges on your hardware platform

**Optional:**

- Familiarity with systemd service management
- Existing SSH key pairs on your client device

## Supported hardware platforms

Use the matrix below to confirm your hardware platform and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Tailscale apt package (Ubuntu noble) | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Hardware platform powered on, networked, and reachable for local or SSH terminal access
- Client device (macOS, Windows, or Linux) for remote access
- Client device and hardware platform on different networks when testing remote connectivity
- Internet connectivity on both the hardware platform and client devices

**Software requirements**

- Valid email account for Tailscale authentication (Google, GitHub, Microsoft, or another supported provider)
- SSH server available on the hardware platform: `systemctl status ssh`
- Working package manager: `sudo apt update`
- User account with sudo privileges on the hardware platform

## Time & risk

- **Estimated time:** 30 MIN (about 15–30 minutes for the first device; roughly 5 minutes per additional client)
- **Risk level:** Medium
  - SSH service configuration conflicts are possible during setup
  - Network connectivity issues can interrupt authentication or first join
  - Authentication depends on your chosen identity provider being available
- **Rollback:** Remove Tailscale with `sudo apt remove --purge tailscale`; network routing reverts to the default configuration
- **Last Updated:** 07/31/2026
  - Tailscale install, authentication, and SSH remote access for supported hardware platforms

## Instructions

## Step 1. Verify system requirements

Confirm that your hardware platform is running a supported Ubuntu-based OS and has internet connectivity. Run these commands on the hardware platform.

```bash
## Check Ubuntu version (should be 20.04 or newer)
lsb_release -a

## Test internet connectivity
ping -c 3 google.com

## Verify you have sudo access
sudo whoami
```

Expected output should show Ubuntu 20.04 or newer, successful pings, and `root` from `sudo whoami`.

## Step 2. Install SSH server (if needed)

Tailscale provides mesh connectivity; SSH is what you use for interactive remote access. Run these commands on the hardware platform.

```bash
## Check if SSH is running
systemctl status ssh --no-pager
```

**If SSH is not installed or running:**

```bash
## Install OpenSSH server
sudo apt update
sudo apt install -y openssh-server

## Enable and start SSH service
sudo systemctl enable ssh --now --no-pager

## Verify SSH is running
systemctl status ssh --no-pager
```

Expected output should show the SSH service as `active (running)`.

## Step 3. Install Tailscale on the hardware platform

Add the official Tailscale Ubuntu repository and install the client on your hardware platform.

```bash
## Update package list
sudo apt update

## Install required tools for adding external repositories
sudo apt install -y curl gnupg

## Add Tailscale signing key
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg | \
  sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg > /dev/null

## Add Tailscale repository
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list | \
  sudo tee /etc/apt/sources.list.d/tailscale.list

## Update package list with new repository
sudo apt update

## Install Tailscale
sudo apt install -y tailscale
```

## Step 4. Verify Tailscale installation

Confirm Tailscale installed correctly on the hardware platform before authenticating.

```bash
## Check Tailscale version
tailscale version

## Check Tailscale service status
sudo systemctl status tailscaled --no-pager
```

Expected output should show a Tailscale version string and `tailscaled` as `active (running)`.

## Step 5. Connect the hardware platform to your Tailscale network

Authenticate the hardware platform with Tailscale using your identity provider. This joins (or creates) your private tailnet and assigns a stable IP address.

```bash
## Start Tailscale and begin authentication
sudo tailscale up

## Follow the URL displayed to complete login in your browser
## Choose from: Google, GitHub, Microsoft, or other supported providers
```

## Step 6. Install Tailscale on client devices

Install Tailscale on the devices you will use to reach your hardware platform remotely.

**On macOS:**

- Option 1: Install from the Mac App Store by searching for "Tailscale", then Get → Install
- Option 2: Download the `.pkg` installer from the [Tailscale download page](https://tailscale.com/download)

**On Windows:**

- Download the installer from the [Tailscale download page](https://tailscale.com/download)
- Run the `.msi` file and follow the installation prompts
- Launch Tailscale from the Start Menu or system tray

**On Linux:**

Use the same apt repository install steps as on the hardware platform:

```bash
## Update package list
sudo apt update

## Install required tools for adding external repositories
sudo apt install -y curl gnupg

## Add Tailscale signing key
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg | \
  sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg > /dev/null

## Add Tailscale repository
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list | \
  sudo tee /etc/apt/sources.list.d/tailscale.list

## Update package list with new repository
sudo apt update

## Install Tailscale
sudo apt install -y tailscale
```

## Step 7. Connect client devices to the tailnet

Log in to Tailscale on each client with the **same** identity provider account you used on the hardware platform.

**On macOS / Windows (GUI):**

- Launch the Tailscale app
- Click **Log in**
- Sign in with the same account used on the hardware platform

**On Linux (CLI):**

```bash
## Start Tailscale on the client
sudo tailscale up

## Complete authentication in the browser using the same account
```

## Step 8. Verify network connectivity

Confirm devices can reach each other on the Tailscale network before attempting SSH.

```bash
## On any device, check tailnet status
tailscale status

## Test ping to the hardware platform (use hostname or IP from status output)
tailscale ping <HARDWARE_HOSTNAME>
```

Expected output should show successful pings to the hardware platform.

## Step 9. Configure SSH authentication

Set up SSH key authentication for secure access. Generate the key on the client; install the public key on the hardware platform.

**Generate an SSH key on the client (if you do not already have one):**

```bash
## Generate new SSH key pair
ssh-keygen -t ed25519 -f ~/.ssh/tailscale_remote

## Display public key to copy
cat ~/.ssh/tailscale_remote.pub
```

**Add the public key on the hardware platform:**

```bash
## On the hardware platform, add the client's public key
echo "<YOUR_PUBLIC_KEY>" >> ~/.ssh/authorized_keys

## Set correct permissions
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

## Step 10. Test SSH connection

Connect to your hardware platform over Tailscale to verify the full path works.

```bash
## Connect using Tailscale hostname (preferred)
ssh -i ~/.ssh/tailscale_remote <USERNAME>@<HARDWARE_HOSTNAME>

## Or connect using Tailscale IP address
ssh -i ~/.ssh/tailscale_remote <USERNAME>@<TAILSCALE_IP>
```

## Step 11. Validate installation

Confirm Tailscale status and that remote SSH actions succeed.

```bash
## From the client device, check connection status
tailscale status

## Create a test file on the client device
echo "test file for remote access" > test.txt

## Test file transfer over SSH
scp -i ~/.ssh/tailscale_remote test.txt <USERNAME>@<HARDWARE_HOSTNAME>:~/

## Verify you can run commands remotely
ssh -i ~/.ssh/tailscale_remote <USERNAME>@<HARDWARE_HOSTNAME> 'nvidia-smi'
```

Expected output:

- Tailscale status listing both devices as active
- Successful file transfer
- Remote command execution working

## Step 12. Cleanup

Remove Tailscale if you need to roll back. This disconnects the device from the tailnet and removes package configuration.

> [!WARNING]
> This permanently removes the device from your Tailscale network and requires re-authentication to rejoin.

```bash
## Stop Tailscale service
sudo tailscale down

## Remove Tailscale package
sudo apt remove --purge tailscale

## Remove repository and keys (optional)
sudo rm /etc/apt/sources.list.d/tailscale.list
sudo rm /usr/share/keyrings/tailscale-archive-keyring.gpg

## Update package list
sudo apt update
```

To restore access, re-run installation and authentication (Steps 3–5 on the hardware platform; Steps 6–7 on clients).

## Step 13. Next steps

Your Tailscale setup is complete. You can now:

1. Reach your hardware platform from any network with `ssh <USERNAME>@<HARDWARE_HOSTNAME>`
2. Transfer files securely with `scp file.txt <USERNAME>@<HARDWARE_HOSTNAME>:~/`
3. Port-forward a local service over SSH when you need browser access to a tool running on the hardware platform — for example:
   `ssh -L 8888:localhost:<LOCAL_PORT> <USERNAME>@<HARDWARE_HOSTNAME>`

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `tailscale up` authentication fails | Network or login endpoint unreachable | Check internet connectivity; try `curl -I login.tailscale.com` |
| SSH connection refused | SSH server not running on the hardware platform | Run `sudo systemctl start ssh --no-pager` on the hardware platform |
| SSH authentication failure | Wrong or missing SSH keys | Confirm the public key is in `~/.ssh/authorized_keys` on the hardware platform |
| Cannot ping Tailscale hostname | MagicDNS / DNS resolution issue | Use the Tailscale IP from `tailscale status` instead of the hostname |
| Devices missing from `tailscale status` | Different Tailscale accounts | Sign in with the same identity provider account on every device |

For product documentation, see [Tailscale documentation](https://tailscale.com/kb). For platform documentation, see the links under **Resources**.
