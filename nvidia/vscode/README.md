# Install VS Code

> Install and use VS Code locally or remotely

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Access with NVIDIA Sync](#access-with-nvidia-sync)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea
This walkthrough establishes a local Visual Studio Code development environment directly on DGX Spark devices. By installing VS Code natively on the ARM64-based Spark system, you gain access to a full-featured IDE with extensions, integrated terminal, and Git integration while leveraging the specialized hardware for development and testing.

## What you'll accomplish
You will have Visual Studio Code running natively on your DGX Spark device with access to the system's ARM64 architecture and GPU resources. This setup enables direct code development, debugging, and execution on the target hardware without remote development overhead.

## What to know before starting

• Basic experience working with Visual Studio Code interface and features

• Familiarity with package management on Linux systems

• Understanding of file permissions and authentication on Linux

## Prerequisites

• DGX Spark device with administrative privileges

• Active internet connection for downloading the VS Code installer

• Verify ARM64 architecture:
  ```bash
  uname -m
#  # Expected output: aarch64
  ```
• Verify GUI desktop environment available:
  ```bash
  echo $DISPLAY
#  # Should return display information like :0 or :10.0
  ```


## Time & risk

* **Duration:** 10-15 minutes
* **Risk level:** Low - installation uses official packages with standard rollback
* **Rollback:** Standard package removal via system package manager

## Instructions

## Step 1. Verify system requirements

Before installing VS Code, confirm your DGX Spark system meets the requirements and has GUI support.

```bash
## Verify ARM64 architecture
uname -m

## Check available disk space (VS Code requires ~200MB)
df -h /

## Verify desktop environment is running
ps aux | grep -E "(gnome|kde|xfce)"
```

## Step 2. Download VS Code ARM64 installer

Navigate to the VS Code [download](https://code.visualstudio.com/download) page and download the appropriate ARM64 `.deb` package for your system. 

Alternatively, you can download the installer with this command:

```bash
wget https://code.visualstudio.com/sha/download?build=stable\&os=linux-deb-arm64 -O vscode-arm64.deb
```

## Step 3. Install VS Code package

Install the downloaded package using the system package manager. 

You can click on the installer file directly or use the command line. 

```bash
## Install the downloaded .deb package
sudo dpkg -i vscode-arm64.deb

## Fix any dependency issues if they occur
sudo apt-get install -f
```

## Step 4. Verify installation

Confirm the VS Code app is installed successfully and can launch. 

You can open the app directly from the list of applications or use the command line. 

```bash
## Check if VS Code is installed
which code

## Verify version
code --version

## Test launch (will open VS Code GUI)
code &
```

VS Code should launch and display the welcome screen.

## Step 5. Configure for Spark development

Set up VS Code for development on the DGX Spark platform.

```bash
## Launch VS Code if not already running
code

## Or create a new project directory and open it
mkdir ~/spark-dev-workspace
cd ~/spark-dev-workspace
code .
```

From within VS Code:

* Open **File** > **Preferences** > **Settings**
* Search for "terminal integrated shell" to configure default terminal
* Install recommended extensions via **Extensions** tab (left sidebar)

## Step 6. Validate setup and test functionality

Test core VS Code functionality to ensure proper operation on ARM64.

Create a test file:
```bash
## Create test directory and file
mkdir ~/vscode-test
cd ~/vscode-test
echo 'print("Hello from DGX Spark!")' > test.py
code test.py
```

Within VS Code:
* Verify syntax highlighting works
* Open integrated terminal (**Terminal** > **New Terminal**)
* Run the test script: `python3 test.py`
* Test Git integration by running `git status` in the terminal

## Step 8. Uninstalling VS Code

> [!WARNING]
> Uninstalling VS Code will remove all user settings and extensions.

To remove VS Code if needed:
```bash
## Remove VS Code package
sudo apt-get remove code

## Remove configuration files (optional)
rm -rf ~/.config/Code
rm -rf ~/.vscode
```

## Access with NVIDIA Sync

## Step 1. Install and configure NVIDIA Sync

Follow the [NVIDIA Sync setup guide](/spark/connect-to-your-spark/sync) to:
- Install NVIDIA Sync for your operating system
- Configure which development tools you want to use (VS Code, Cursor, Terminal, etc.)
- Add your DGX Spark device by providing its hostname/IP and credentials

NVIDIA Sync will automatically configure SSH key-based authentication for secure, password-free access.

## Step 2. Launch VS Code through NVIDIA Sync

- Click the NVIDIA Sync icon in your system tray/taskbar
- Ensure your device is connected (click "Connect" if needed)
- Click on "VS Code" to launch it with an automatic SSH connection to your Spark
- Wait for the remote connection to be established (may ask your local machine for a password or to authorize the connection)
- It may prompt you to "trust the authors of the files in this folder" when you first land in the home directory after a successful SSH connection

## Step 3. Validation and follow-ups

- Verify that you can access your Spark's filesystem with VS Code as a text editor
- Open the integrated terminal in VS Code and run test commands like `hostnamectl` and `whoami` to ensure you are remotely accessing your Spark
- Navigate to a specific file path or directory and start editing/writing files
- Install VS Code extensions for your development workflow (Python, Docker, GitLens, etc.)
- Clone repositories from GitHub or other version control systems
- Configure and locally host an LLM code assistant if desired

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `dpkg: dependency problems` during install | Missing dependencies | Run `sudo apt-get install -f` |
| VS Code won't launch with GUI error | No display server/X11 | Verify GUI desktop is running: `echo $DISPLAY` |
| Extensions fail to install | Network connectivity or ARM64 compatibility | Check internet connection, verify extension ARM64 support |
