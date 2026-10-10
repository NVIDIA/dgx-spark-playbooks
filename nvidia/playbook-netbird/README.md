# Set Up NetBird for Remote Access

> Encrypted SSH to your hardware platform from any network, without port forwarding

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NetBird is an open-source, WireGuard®-based overlay network that lets you reach your hardware platform from anywhere without complex firewall configuration or port forwarding. Install NetBird on your hardware platform and on your client devices to join one private network, where each device gets a stable private IP address and DNS name. You can then use SSH over that network whether you are at home, at work, or on another network.

You can use the managed NetBird service or self-host NetBird. If you self-host, add your management URL when you connect.

## What you'll accomplish

You'll install and authenticate NetBird on your **hardware platform** and on one or more client devices, then SSH into the hardware platform from a different network using its NetBird DNS name or IP. Traffic is encrypted end to end, and NAT traversal is handled for you. You'll also replace the default allow-all access policy with one that only allows SSH from your own devices.

## What to know before starting

**Required:**

- Experience with the Linux command line
- Basic SSH concepts and usage
- Installing packages with `apt` on Ubuntu-based systems
- A user account with sudo privileges on your hardware platform

**Optional:**

- Familiarity with systemd service management
- Existing SSH key pairs on your client device
- A self-hosted NetBird deployment, if you do not want to use the managed service

## Supported hardware platforms

Use the matrix below to confirm your hardware platform and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | NetBird apt package, CLI (arm64) | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform (see the Supported hardware platforms matrix above)
- Hardware platform powered on, networked, and reachable for local or SSH terminal access
- Client device (macOS, Windows, or Linux) for remote access
- Client device and hardware platform on different networks when testing remote connectivity
- Internet connectivity on both the hardware platform and client devices

**Software requirements**

- A NetBird account at [app.netbird.io](https://app.netbird.io) (sign in with Google, GitHub, Microsoft, or email), or a self-hosted NetBird deployment
- SSH server available on the hardware platform: `ss -tln | grep ':22 '`
- Working package manager: `sudo apt update`
- User account with sudo privileges on the hardware platform

## Time & risk

- **Estimated time:** 30 MIN (about 15–30 minutes for the first device; roughly 5 minutes per additional client)
- **Risk level:** Medium
  - SSH service configuration conflicts are possible during setup
  - Network connectivity issues can interrupt authentication or first join
  - Authentication depends on your chosen identity provider being available
- **Rollback:** Remove NetBird with `sudo apt remove --purge -y netbird` and delete `/var/lib/netbird`; network routing reverts to the default configuration
- **Last Updated:** 10/10/2026
  - NetBird install, authentication, access policy, and SSH remote access for supported hardware platforms

## Instructions

## Step 1. Verify system requirements

Confirm that your hardware platform is running a supported Ubuntu-based OS and has internet connectivity. Run these commands on the hardware platform.

```bash
## Check Ubuntu version (should be 20.04 or newer)
lsb_release -a

## Check the CPU architecture (DGX Spark reports aarch64)
uname -m

## Test internet connectivity
ping -c 3 google.com

## Verify you have sudo access
sudo whoami
```

Expected output should show Ubuntu 20.04 or newer, `aarch64`, successful pings, and `root` from `sudo whoami`.

## Step 2. Install SSH server (if needed)

NetBird provides the private network; SSH is what you use for interactive remote access. Run these commands on the hardware platform.

```bash
## Check that the SSH server is listening on port 22
ss -tln | grep ':22 '
```

On recent Ubuntu releases SSH can be socket-activated, so `systemctl status ssh` may report `inactive` even when SSH works. Checking port 22 avoids that.

**If SSH is not installed or running:**

```bash
## Install OpenSSH server
sudo apt update
sudo apt install -y openssh-server

## Enable and start SSH service
sudo systemctl enable ssh --now

## Verify SSH is listening
ss -tln | grep ':22 '
```

Expected output should show a `LISTEN` line for port `22`.

## Step 3. Install NetBird on the hardware platform

Add the official NetBird apt repository and install the NetBird client on your hardware platform.

```bash
## Update package list
sudo apt update

## Install required tools for adding external repositories
sudo apt install -y ca-certificates curl gnupg

## Add NetBird signing key
curl -fsSL https://pkgs.netbird.io/debian/public.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/netbird-archive-keyring.gpg

## Add NetBird repository
echo 'deb [signed-by=/usr/share/keyrings/netbird-archive-keyring.gpg] https://pkgs.netbird.io/debian stable main' | \
  sudo tee /etc/apt/sources.list.d/netbird.list

## Update package list with new repository
sudo apt update

## Install NetBird
sudo apt install -y netbird
```

On DGX Spark, use the `netbird` command line client. The Linux desktop app (`netbird-ui`) is built for x86_64 only, so it is not available for arm64. macOS and Windows client devices can use the desktop app.

## Step 4. Verify NetBird installation

Confirm NetBird installed correctly on the hardware platform before authenticating.

```bash
## Check NetBird version
netbird version

## Check NetBird service status
sudo systemctl status netbird --no-pager

## Check NetBird connection status
sudo netbird status
```

Expected output should show a NetBird version string, the `netbird` service as `active (running)`, and `Daemon status: NeedsLogin`.

## Step 5. Connect the hardware platform to your NetBird network

Join the hardware platform to your NetBird network. This assigns it a stable IP address and DNS name. Use a setup key (Option A) or a browser login (Option B).

**Option A: Setup key (recommended for the hardware platform)**

Sign in to the NetBird dashboard at [app.netbird.io](https://app.netbird.io). Your NetBird network is created the first time you sign in. Go to **Peers**, select **Add Peer → Server**, and select **Generate Key**. The dashboard shows the `netbird up` command with the key filled in. NetBird is already installed, so run only that command on the hardware platform:

```bash
## Join the NetBird network with the setup key
sudo netbird up --setup-key <SETUP_KEY>
```

This key works once and expires after 24 hours. To enroll several hardware platforms, create a reusable key under **Settings → Setup Keys** instead, and set **Auto-assigned groups** to `dgx-spark` so each one joins the group used in Step 12.

**Option B: Browser login**

```bash
## Start NetBird and begin authentication
sudo netbird up

## If no browser opens on the hardware platform, open the URL it prints
## on any device and sign in. Choose Google, GitHub, Microsoft, or email.
```

> [!NOTE]
> A device that joins through a browser login drops off the network when its session expires (24 hours by default on new accounts) and shows `Login required` in the dashboard. For a hardware platform you reach remotely, use a setup key, or turn off **Session Expiration** for it under **Peers**.

**Self-hosted NetBird:** add your management URL to either option.

```bash
## Join a self-hosted NetBird network with the setup key
sudo netbird up --management-url https://<YOUR_NETBIRD_DOMAIN> --setup-key <SETUP_KEY>

## Join a self-hosted NetBird network with a browser login
sudo netbird up --management-url https://<YOUR_NETBIRD_DOMAIN>
```

Expected output should show `Connected`. Run `sudo netbird status` to see the hardware platform's NetBird IP and DNS name.

## Step 6. Install NetBird on client devices

Install NetBird on the devices you will use to reach your hardware platform remotely.

**On macOS:**

- Option 1: Download the installer for your processor: [Apple silicon](https://pkgs.netbird.io/macos/arm64) or [Intel](https://pkgs.netbird.io/macos/amd64)
- Option 2: Install with Homebrew: `brew install --cask netbirdio/tap/netbird-ui`

**On Windows:**

- Download the [EXE installer](https://pkgs.netbird.io/windows/x64) or the [MSI installer](https://pkgs.netbird.io/windows/msi/x64)
- Run the `.exe` or `.msi` file and follow the installation prompts
- Launch NetBird from the Start Menu or system tray

**On Linux:**

Use the same apt repository install steps as on the hardware platform:

```bash
## Update package list
sudo apt update

## Install required tools for adding external repositories
sudo apt install -y ca-certificates curl gnupg

## Add NetBird signing key
curl -fsSL https://pkgs.netbird.io/debian/public.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/netbird-archive-keyring.gpg

## Add NetBird repository
echo 'deb [signed-by=/usr/share/keyrings/netbird-archive-keyring.gpg] https://pkgs.netbird.io/debian stable main' | \
  sudo tee /etc/apt/sources.list.d/netbird.list

## Update package list with new repository
sudo apt update

## Install NetBird
sudo apt install -y netbird
```

## Step 7. Connect client devices to your NetBird network

Log in to NetBird on each client with the **same NetBird account** the hardware platform joined (the account where you created the setup key).

**On macOS / Windows (desktop app):**

- Launch the NetBird app
- Click **Connect**
- Sign in to the same NetBird account

**On Linux (CLI):**

```bash
## Start NetBird on the client
sudo netbird up

## Complete authentication in the browser with the same NetBird account
```

If you self-host NetBird, use the same management URL on every client. On Linux, run `sudo netbird up --management-url https://<YOUR_NETBIRD_DOMAIN>`. The desktop app asks for the URL on first launch, when you choose between NetBird Cloud and a self-hosted deployment.

## Step 8. Verify network connectivity

Confirm devices can reach each other on the NetBird network before attempting SSH.

```bash
## Test ping to the hardware platform from a client (use the DNS name or IP from `sudo netbird status` on the hardware platform)
ping -c 5 <HARDWARE_HOSTNAME>.netbird.cloud

## List the peers this device can reach
netbird status -d
```

Expected output should show successful pings and list the hardware platform as `Connected`. A peer can show `Idle` before the first traffic, and some of the first pings can be lost while the connection comes up. If that happens, run the ping again.

With the managed service, DNS names end in `.netbird.cloud`. A self-hosted deployment uses the DNS domain set in your account settings (`.netbird.selfhosted` by default). The ping works because a new account's **Default** access policy allows all traffic between devices; Step 12 replaces it.

## Step 9. Configure SSH authentication

Set up SSH key authentication for secure access. Generate the key on the client; install the public key on the hardware platform.

**Generate an SSH key on the client (if you do not already have one):**

```bash
## Generate new SSH key pair
ssh-keygen -t ed25519 -f ~/.ssh/netbird_remote

## Display public key to copy
cat ~/.ssh/netbird_remote.pub
```

**Add the public key on the hardware platform:**

```bash
## On the hardware platform, create the SSH directory if it does not exist
mkdir -p ~/.ssh

## Add the client's public key
echo "<YOUR_PUBLIC_KEY>" >> ~/.ssh/authorized_keys

## Set correct permissions
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

## Step 10. Test SSH connection

Connect to your hardware platform over NetBird to verify the full path works.

```bash
## Connect using the NetBird DNS name (preferred)
ssh -i ~/.ssh/netbird_remote <USERNAME>@<HARDWARE_HOSTNAME>.netbird.cloud

## Or connect using the NetBird IP address
ssh -i ~/.ssh/netbird_remote <USERNAME>@<NETBIRD_IP>
```

On first connection, SSH asks you to confirm the hardware platform's host key. Type `yes` and press Enter.

## Step 11. Validate installation

Confirm NetBird status and that remote SSH actions succeed.

```bash
## From the client device, check connection status
netbird status -d

## Create a test file on the client device
echo "test file for remote access" > test.txt

## Test file transfer over SSH
scp -i ~/.ssh/netbird_remote test.txt <USERNAME>@<HARDWARE_HOSTNAME>.netbird.cloud:~/

## Verify you can run commands remotely
ssh -i ~/.ssh/netbird_remote <USERNAME>@<HARDWARE_HOSTNAME>.netbird.cloud 'nvidia-smi'
```

Expected output:

- NetBird status listing the hardware platform as `Connected`
- Successful file transfer
- Remote command execution working

## Step 12. Restrict access to the hardware platform (recommended)

The Default policy lets every device in your network reach every other device on any port. Replace it with a policy that only lets your client devices reach the hardware platform over SSH. Make these changes in the NetBird dashboard.

> [!WARNING]
> Create and test the new policy before you disable the Default policy. Without any policy, NetBird devices cannot reach each other.

1. Go to **Peers**, select the hardware platform, add it to a new group named `dgx-spark`, and select **Save Changes** (skip this if your setup key already assigns the group)
2. Add each client device to a new group named `spark-clients`
3. Go to **Access Control → Policies** and select **Add Policy**
4. Set **Source** to `spark-clients`, **Destination** to `dgx-spark`, **Protocol** to `TCP`, and **Ports** to `22`
5. Name the policy (for example, `SSH to DGX Spark`) and save it
6. Disable the **Default** policy with the switch in the **Active** column

Re-run Step 10 to confirm SSH still works.

All other traffic between the hardware platform and your devices is now blocked, unless another policy allows it. That includes the `ping` from Step 8. Add a policy with protocol `ICMP` if you want ping to work. To reach a web service on the hardware platform, add its port to the policy or use SSH port forwarding (see Step 14). Policies only apply to traffic over NetBird, so a service that listens on all interfaces is still reachable from the local network.

## Step 13. Cleanup

Remove NetBird if you need to roll back. This disconnects the device from your NetBird network and removes the NetBird package, service, and local configuration.

> [!WARNING]
> Deleting `/var/lib/netbird` removes the device's NetBird identity. To rejoin, you must authenticate again, and the device gets a new IP address.

```bash
## Disconnect from the NetBird network
sudo netbird down

## Remove NetBird package (this also stops and removes the NetBird service)
sudo apt remove --purge -y netbird

## Remove the device identity and logs, which the package leaves behind
sudo rm -rf /var/lib/netbird /var/log/netbird

## Remove repository and keys (optional)
sudo rm /etc/apt/sources.list.d/netbird.list
sudo rm /usr/share/keyrings/netbird-archive-keyring.gpg

## Update package list
sudo apt update
```

Also delete the device under **Peers** in the dashboard. Otherwise a reinstalled device registers as a new peer with a suffixed DNS name (for example, `<HARDWARE_HOSTNAME>-169-66`). Unless your setup key assigns `dgx-spark`, the new peer is also not in that group, so the Step 12 policy does not apply to it.

To restore access, re-run installation and authentication (Steps 3–5 on the hardware platform; Steps 6–7 on clients). If your setup key does not assign the `dgx-spark` group, add the hardware platform to it again.

## Step 14. Next steps

Your NetBird setup is complete. Add `-i ~/.ssh/netbird_remote` to these commands if you use the key from Step 9. You can now:

1. Reach your hardware platform from any network with `ssh <USERNAME>@<HARDWARE_HOSTNAME>.netbird.cloud`
2. Transfer files securely with `scp file.txt <USERNAME>@<HARDWARE_HOSTNAME>.netbird.cloud:~/`
3. Port-forward a service over SSH when you need browser access to a tool running on the hardware platform. Use the same port on both ends so local tools keep their usual address, for example:
   `ssh -L <SERVICE_PORT>:localhost:<SERVICE_PORT> <USERNAME>@<HARDWARE_HOSTNAME>.netbird.cloud`
4. Open the [DGX Dashboard](../playbook-dgx-dashboard/) from anywhere with `ssh -L 11000:localhost:11000 <USERNAME>@<HARDWARE_HOSTNAME>.netbird.cloud`, then browse to `http://localhost:11000`
5. Add more hardware platforms with a reusable setup key (Step 5), and give teammates access by adding their devices to the `spark-clients` group (Step 12)

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `netbird up` authentication fails | Network or management endpoint unreachable | Check internet connectivity; run `curl -I https://api.netbird.io` (or your self-hosted management URL). Any HTTP response, even `404`, means it is reachable |
| `netbird up` waits and no browser opens | No browser on the hardware platform | Open the URL printed in the terminal on any device, or use a setup key (Step 5) |
| Hardware platform drops off after a day and shows `Login required` in the dashboard | Session expired after a browser login | Run `sudo netbird up` on the hardware platform, then rejoin with a setup key or turn off session expiration for it (Step 5) |
| SSH connection refused | SSH server not running on the hardware platform | Run `sudo systemctl start ssh` on the hardware platform |
| SSH authentication failure | Wrong or missing SSH keys | Confirm the public key is in `~/.ssh/authorized_keys` on the hardware platform |
| Peer shows `Connected` but SSH or ping times out | No access policy allows the traffic | Check **Access Control → Policies** in the dashboard; the Default policy or the Step 12 policy must include both devices |
| Cannot resolve the NetBird DNS name | DNS resolution issue on the client | Use the NetBird IP from `netbird status -d` instead of the DNS name |
| Devices missing from `netbird status -d` | Different NetBird accounts or management URLs | Sign in with the same account (and the same management URL, if self-hosted) on every device |

For product documentation, see [NetBird documentation](https://docs.netbird.io). For platform documentation, see the links under **Resources**.
