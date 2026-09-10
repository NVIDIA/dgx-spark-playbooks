# Set Up VS Code for Local and Remote Development

> Editing, debugging, and terminal access in one development workspace

## Table of Contents

- [Overview](#overview)
- [Desktop Use](#desktop-use)
- [Remote Use](#remote-use)
  - [Optional: connect with Remote - SSH](#optional-connect-with-remote-ssh)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Visual Studio Code (VS Code) is a source-code editor with an integrated terminal, debugging tools, Git support, and a broad extension ecosystem.

This walkthrough covers two development workflows: running VS Code directly on your hardware platform, or connecting remotely through NVIDIA Sync from VS Code on another computer.

## What you'll accomplish

You'll configure VS Code for local or remote development on your hardware platform and verify that you can edit and run code.

## What to know before starting

**Required:**

- Basic familiarity with VS Code
- Basic experience with Linux commands

**Optional:**

- Familiarity with SSH and NVIDIA Sync for the remote workflow

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Linux ARM64 `.deb` package for local use; NVIDIA Sync for remote use | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- At least 200 MB of available storage for a local VS Code installation
- Keyboard, monitor, and mouse for local desktop use
- A network connection between your computer and hardware platform for remote use

**Software requirements**

- Administrative (`sudo`) access for local installation
- Internet access to download VS Code and extensions
- For remote use, [VS Code](https://code.visualstudio.com/download) installed on your computer
- For remote use, NVIDIA Sync installed and connected to your hardware platform — see the [NVIDIA Sync documentation](https://docs.nvidia.com/sync/latest/direct-connections.html#nvidia-sync-direct-connections)

## Time & risk

- **Estimated time:** 10 MIN (a single path is often closer to 5 MIN)
- **Risk level:** Low
  - Local installation adds the official VS Code package and its dependencies
  - Remote setup through NVIDIA Sync configures secure SSH access and launches VS Code against your hardware platform
- **Rollback:** Uninstall the local package or disconnect/remove the device in NVIDIA Sync; optional settings removal is documented in the instructions
- **Last Updated:** 07/31/2026
  - Added local and remote VS Code setup paths with verification and rollback guidance

## Desktop Use

## Step 1. Verify the system architecture

Open a terminal on your hardware platform and verify that the operating system reports the ARM64 architecture:

```bash
dpkg --print-architecture
```

Expected output:

```text
arm64
```

## Step 2. Download VS Code

Download the current stable Linux ARM64 `.deb` package from Microsoft to your home directory:

```bash
wget 'https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-arm64' -O ~/vscode-arm64.deb
```

## Step 3. Install VS Code

Install the package and its dependencies:

```bash
sudo apt install ~/vscode-arm64.deb
```

Verify the installation:

```bash
code --version
```

The command should print the installed VS Code version and architecture.

## Step 4. Open a workspace

Create a project directory and open it in VS Code:

```bash
mkdir -p ~/vscode-workspace
cd ~/vscode-workspace
code .
```

VS Code should open the directory in a new window. Use **Terminal > New Terminal** to open the integrated terminal.

## Step 5. Validate the setup

In the integrated terminal, create and run a small Python program:

```bash
printf 'print("Hello from VS Code!")\n' > test.py
python3 test.py
```

Expected output:

```text
Hello from VS Code!
```

Open `test.py` in the editor and confirm that syntax highlighting works.

## Step 6. Install extensions

Open the **Extensions** view from the Activity Bar and install the extensions needed for your development workflow. Confirm that each extension supports Linux ARM64 before installing it.

## Step 7. Cleanup

Cleanup is optional. Remove the VS Code package when you no longer need the local installation:

```bash
sudo apt remove code
rm -f ~/vscode-arm64.deb
```

> [!WARNING]
> The following commands permanently remove your user settings, cached data, and installed extensions. Run them only if you want a complete reset.

```bash
rm -rf ~/.config/Code
rm -rf ~/.vscode
```

## Remote Use

> These steps assume NVIDIA Sync is installed and connected to your hardware platform. If you still need to set that up, complete NVIDIA Sync remote-access setup for your hardware platform first. See also the [NVIDIA Sync documentation](https://docs.nvidia.com/sync/latest/direct-connections.html#nvidia-sync-direct-connections).

## Step 1. Install and configure NVIDIA Sync

Follow the [NVIDIA Sync documentation](https://docs.nvidia.com/sync/latest/direct-connections.html#nvidia-sync-direct-connections) to:

- Install NVIDIA Sync for your operating system
- Configure which development tools you want to use (VS Code, Cursor, Terminal, and others)
- Add your hardware platform by providing its hostname or IP address and credentials

NVIDIA Sync configures SSH key-based authentication for secure, password-free access.

## Step 2. Launch VS Code through NVIDIA Sync

1. Click the NVIDIA Sync icon in your system tray or taskbar.
2. Ensure your hardware platform is connected (click **Connect** if needed).
3. Click **VS Code** to launch it with an automatic SSH connection to your hardware platform.
4. Wait for the remote connection to be established. Your computer may ask for a password or to authorize the connection.
5. You may be prompted to trust the authors of the files in this folder when you first land in the home directory after a successful connection.

## Step 3. Validate the remote workspace

- Verify that you can access your hardware platform's filesystem with VS Code as a text editor.
- Open the integrated terminal in VS Code (**Terminal > New Terminal**) and run:

```bash
hostnamectl
whoami
```

The output should identify the remote hardware platform and your remote user.

- Navigate to a project directory and start editing files.
- Install VS Code extensions for your development workflow when VS Code offers to install them in the remote environment.
- Clone repositories from your version control system as needed.

## Step 4. Cleanup

Cleanup is optional. When you finish your session:

1. Close the remote VS Code window.
2. In NVIDIA Sync, disconnect from the hardware platform if you no longer need the connection.

To remove the hardware platform from NVIDIA Sync, open the device settings in NVIDIA Sync and delete the saved device entry.

---

### Optional: connect with Remote - SSH

If you prefer not to use NVIDIA Sync, you can connect from VS Code on your computer with the Microsoft Remote - SSH extension:

1. Install [VS Code](https://code.visualstudio.com/download) and the **Remote - SSH** extension on your computer.
2. Confirm you can reach the hardware platform with `ssh <USER>@<HARDWARE_HOST>`.
3. In VS Code, run **Remote-SSH: Connect to Host**, select the host, and open a remote folder when the connection succeeds.

## Troubleshooting

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `dpkg --print-architecture` does not return `arm64` | The downloaded package does not match the system architecture | Stop the local installation and download the VS Code package that matches the reported architecture |
| Package installation reports unmet dependencies | The package index is stale or required packages are unavailable | Run `sudo apt update`, then retry `sudo apt install ~/vscode-arm64.deb` |
| `code` does not open a window | The local session does not have a graphical display | Confirm that you are signed into the desktop session and run `echo "$DISPLAY"`; use the Remote Use tab if the hardware platform is headless |
| An extension cannot be installed locally | The extension does not provide a Linux ARM64 build or network access is unavailable | Check the extension's platform support and verify internet access |
| NVIDIA Sync cannot connect to the hardware platform | The hostname, credentials, or network route is incorrect | Confirm the device entry in NVIDIA Sync, verify network reachability, and retry **Connect** |
| VS Code does not launch from NVIDIA Sync | VS Code is not selected as a development tool, or the remote session failed to start | In NVIDIA Sync device settings, enable VS Code as a development tool, reconnect, and launch again |
| Remote extensions are unavailable after Sync launch | Extensions were installed only on the local computer | Install the extensions in the remote environment when VS Code prompts you |

For product documentation, see the [VS Code documentation](https://code.visualstudio.com/docs) and the [NVIDIA Sync documentation](https://docs.nvidia.com/sync/latest/direct-connections.html#nvidia-sync-direct-connections).
