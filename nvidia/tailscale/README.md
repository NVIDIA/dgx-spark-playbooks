# Set up Tailscale on Your Spark

> Use Tailscale to connect to your Spark on your home network no matter where you are


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Step 1. Verify system requirements](#step-1-verify-system-requirements)
  - [Step 2. Install SSH server (if needed)](#step-2-install-ssh-server-if-needed)
  - [Step 3. Install Tailscale on NVIDIA DGX Spark](#step-3-install-tailscale-on-nvidia-dgx-spark)
  - [Step 4. Verify Tailscale installation](#step-4-verify-tailscale-installation)
  - [Step 5. Connect your DGX Spark to Tailscale network](#step-5-connect-your-dgx-spark-to-tailscale-network)
  - [Step 6. Install Tailscale on client devices](#step-6-install-tailscale-on-client-devices)
  - [Step 7. Connect client devices to tailnet](#step-7-connect-client-devices-to-tailnet)
  - [Step 8. Verify network connectivity](#step-8-verify-network-connectivity)
  - [Step 9. Configure SSH authentication](#step-9-configure-ssh-authentication)
  - [Step 10. Test SSH connection](#step-10-test-ssh-connection)
  - [Step 11. Validate installation](#step-11-validate-installation)
  - [Step 12. Access DGX Dashboard over Tailnet](#step-12-access-dgx-dashboard-over-tailnet)
  - [Step 13. Next steps](#step-13-next-steps)
  - [Step 14. Cleanup and rollback](#step-14-cleanup-and-rollback)
  
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Tailscale creates an encrypted peer-to-peer mesh network that allows secure access
to your NVIDIA DGX Spark device from anywhere without complex firewall configurations
or port forwarding. By installing Tailscale on both your DGX Spark and client devices,
you establish a private "tailnet" where each device gets a stable private IP
address and hostname, enabling seamless SSH access whether you're at home, work,
or a coffee shop.

## What you'll accomplish

You will set up Tailscale on your DGX Spark device and client machines to
create secure remote access. After completion, you'll be able to SSH into your
DGX Spark from anywhere using simple commands like `ssh user@spark-hostname`, with
all traffic automatically encrypted and NAT traversal handled transparently.

## What to know before starting

- Working with terminal/command line interfaces
- Basic SSH concepts and usage
- Installing packages using `apt` on Ubuntu
- Understanding of user accounts and authentication
- Familiarity with systemd service management

## Prerequisites

**Hardware Requirements:**
-  NVIDIA Grace Blackwell GB10 Superchip System

**Software Requirements:**
- NVIDIA DGX OS
- Client device (Mac, Windows, or Linux) for remote access
- Client device and DGX Spark not on the same network when testing connectivity
- Internet connectivity on both devices
- Valid email account for Tailscale authentication (Google, GitHub, Microsoft)
- SSH server availability check: `systemctl status ssh`
- Package manager working: `sudo apt update`
- User account with sudo privileges on your DGX Spark device

## Time & risk

* **Duration**: 15-30 minutes for initial setup, 5 minutes per additional device
* **Risks**: Medium
  * Potential SSH service configuration conflicts
  * Network connectivity issues during initial setup
  * Authentication provider service dependencies
* **Rollback**: Tailscale can be completely removed with `sudo apt remove tailscale` and all network routing automatically reverts to default settings.
* **Last Updated:** 11/07/2025
  * Minor copyedits

## Instructions

### Step 1. Verify system requirements

Check that your NVIDIA DGX Spark device is running a supported Ubuntu version and
has internet connectivity. This step runs on the DGX Spark device to confirm
prerequisites.

```bash
## Check Ubuntu version (should be 20.04 or newer)
lsb_release -a

## Test internet connectivity
ping -c 3 google.com

## Verify you have sudo access
sudo whoami
```

### Step 2. Install SSH server (if needed)

Ensure SSH server is running on your DGX Spark device since Tailscale provides
network connectivity but requires SSH for remote access. This step runs on
the DGX Spark device.

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

### Step 3. Install Tailscale on NVIDIA DGX Spark

Install Tailscale on your DGX Spark using the official Ubuntu
repository. This step adds the Tailscale package repository and installs
the client.

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

### Step 4. Verify Tailscale installation

Confirm Tailscale installed correctly on your DGX Spark device before proceeding
with authentication.

```bash
## Check Tailscale version
tailscale version

## Check Tailscale service status
sudo systemctl status tailscaled --no-pager
```

### Step 5. Connect your DGX Spark to Tailscale network

Authenticate your DGX Spark device with Tailscale using your chosen identity
provider. This creates your private tailnet and assigns a stable IP address.

```bash
## Start Tailscale and begin authentication
sudo tailscale up

## Follow the URL displayed to complete login in your browser
## Choose from: Google, GitHub, Microsoft, or other supported providers
```

### Step 6. Install Tailscale on client devices

Install Tailscale on the devices you'll use to connect to your DGX Spark remotely.

Choose the appropriate method for your client operating system:

**On macOS:**
- Option 1: Install from Mac App Store by searching for "Tailscale" and then clicking Get → Install
- Option 2: Download the .pkg installer from the [Tailscale website](https://tailscale.com/download)


**On Windows:**
- Download installer from the [Tailscale website](https://tailscale.com/download)
- Run the .msi file and follow installation prompts
- Launch Tailscale from Start Menu or system tray


**On Linux:**

Follow the same instructions used for the DGX Spark installation.

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

### Step 7. Connect client devices to tailnet

Log in to Tailscale on each client device using the same identity provider
account you used for your DGX Spark.

**On macOS/Windows (GUI):**
- Launch Tailscale app
- Click "Log in" button
- Sign in with same account used on DGX Spark

**On Linux (CLI):**

```bash
## Start Tailscale on client
sudo tailscale up

## Complete authentication in browser using same account
```

### Step 8. Verify network connectivity

Test that devices can communicate through the Tailscale network before
attempting SSH connections.

```bash
## On any device, check tailnet status
tailscale status

## Test ping to Spark device (use hostname or IP from status output)
tailscale ping <SPARK_HOSTNAME>

## Example output should show successful pings
```

### Step 9. Configure SSH authentication

Set up SSH key authentication for secure access to your DGX Spark. This
step runs on your client device and DGX Spark device.

**Generate SSH key on client (if not already done):**

```bash
## Generate new SSH key pair
ssh-keygen -t ed25519 -f ~/.ssh/tailscale_spark

## Display public key to copy
cat ~/.ssh/tailscale_spark.pub
```

**Add public key to DGX Spark:**

```bash
## On Spark device, add client's public key
echo "<YOUR_PUBLIC_KEY>" >> ~/.ssh/authorized_keys

## Set correct permissions
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

### Step 10. Test SSH connection

Connect to your DGX Spark using SSH over the Tailscale network to verify
the complete setup works.

```bash
## Connect using Tailscale hostname (preferred)
ssh -i ~/.ssh/tailscale_spark <USERNAME>@<SPARK_HOSTNAME>

## Or connect using Tailscale IP address
ssh -i ~/.ssh/tailscale_spark <USERNAME>@<TAILSCALE_IP>

## Example:
## ssh -i ~/.ssh/tailscale_spark nvidia@my-spark-device
```

### Step 11. Validate installation

Verify that Tailscale is working correctly and your SSH connection is stable.

```bash
## From client device, check connection status
tailscale status

## Create a test file on the client device
echo "test file for the spark" > test.txt

## Test file transfer over SSH
scp -i ~/.ssh/tailscale_spark test.txt <USERNAME>@<SPARK_HOSTNAME>:~/

## Verify you can run commands remotely
ssh -i ~/.ssh/tailscale_spark <USERNAME>@<SPARK_HOSTNAME> 'nvidia-smi'
```

Expected output:
- Tailscale status displaying both devices as "active"
- Successful file transfers
- Remote command execution working

### Step 12. Access DGX Dashboard over Tailnet

The DGX Dashboard is locked to localhost:11000 for security. This means you can only access it over localhost thorugh the ssh tunnel. Instead of manually creating an SSH tunnel every time, use Tailscale Serve to proxy the traffic so you can access it via your Tailscale IP/URL from any device.

## On your DGX Spark machine, run:
```bash
## Proxy incoming Tailnet traffic to the local dashboard
## The --bg flag ensures this keeps running after you close your terminal
sudo tailscale serve --bg --http=11000 localhost:11000
```

## Verify proxy is active:
```bash
tailscale serve status
```

You can access the dashboard using the Tailscale IP address:

`http://<TAILSCALE_IP>:11000`

You can find your Tailscale IP by running `tailscale ip -4` on the DGX Spark device.

Alternatively, if you set up tailsale with Magic DNS, you can use your tailscale URL with:

`http://SPARK_HOST_NAME.XXXXX-YYYYYY.ts.net:11000`

Where XXXXX an YYYYYY are part of the custom domain name to your tailnet.

You can now bookmark this URL and access it anywhere on your tailnet.

**Option: Enable HTTPS (recommended for security)**

For secure HTTPS access with SSL certificates, enable MagicDNS and HTTPS Certificates in your Tailscale Admin Console:

1. Go to your Tailscale Admin Console
2. Under DNS, ensure MagicDNS is enabled
3. Scroll down to HTTPS Certificates and click Enable

Then, on your DGX Spark machine, reset the HTTP proxy and start the HTTPS proxy:

```bash
# First, reset the old HTTP proxy
sudo tailscale serve --http=11000 off

# Now, start the HTTPS proxy
sudo tailscale serve --bg --https=11000 localhost:11000
```

Access the dashboard securely via: `https://SPARK_HOST_NAME.XXXXX-YYYYYY.ts.net:11000`
 > **Note:** It may take a little longer on first load to set the SSL certificate. This is normal.

### Step 13. Next steps

Your Tailscale setup is complete. You can now:

- Access your DGX Spark device from any network with: `ssh <USERNAME>@<SPARK_HOSTNAME>`
- Transfer files securely: `scp file.txt <USERNAME>@<SPARK_HOSTNAME>:~/`
- Open the DGX Dashboard and start JupyterLab, then connect with:
  `ssh -L 8888:localhost:1102 <USERNAME>@<SPARK_HOSTNAME>`

  > **Note:** Alternatively, see Step 12 for accessing the DGX Dashboard over Tailnet without manual SSH tunneling.


### Step 14. Cleanup and rollback

Remove Tailscale completely if needed. This will disconnect devices from the
tailnet and remove all network configurations.

**Option A: Remove only DGX Dashboard access**

If you want to keep Tailscale installed but stop serving the DGX Dashboard:

```bash
## Remove DGX Dashboard access from tailnet (from Step 12)
sudo tailscale serve --http=11000 off
sudo tailscale serve --https=11000 off
```

> [!WARNING]
> This will permanently remove the device from your Tailscale network and require re-authentication to rejoin.

**Option B: Full Tailscale removal**

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


To restore: Re-run installation steps 3-5.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `tailscale up` auth fails | Network issues | Check internet, try `curl -I login.tailscale.com` |
| SSH connection refused | SSH not running | Run `sudo systemctl start ssh --no-pager` on Spark |
| SSH auth failure | Wrong SSH keys | Check public key in `~/.ssh/authorized_keys` |
| Cannot ping hostname | DNS issues | Use IP from `tailscale status` instead |
| Devices missing | Different accounts | Use same identity provider for all devices |


For latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
