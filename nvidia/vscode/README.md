# VS Code

> Install and use VS Code locally or remotely

## Table of Contents

- [Overview](#overview)
- [Desktop Use](#desktop-use)
- [Remote Use](#remote-use)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea
This walkthrough will help you install and use Visual Studio Code on your DGX Spark.  There are two different approaches, depending on how you are accessing your Spark:

 * **Desktop Use**: Use this path if you use your Spark as a local device, i.e. you work on it directly through a connected keyboard, monitor and mouse.

 * **Remote Use**: Use this path if you use your Spark as a remote device, i.e. you work on it by connecting to it through SSH over a network.

## What you'll accomplish
You will set up VS Code on your DGX Spark device for code development, debugging, and execution

## What to know before starting
You should have basic experience with VS Code, as well as experience relevant to the desktop or remote path:

* **Desktop Use**:
  * Minimal experience with installing Linux packages

* **Remote Use**:
  * Minimal experience with SSH

## Prerequisites

* A DGX Spark set up with an active internet connection [see NVIDIA docs here](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html)
* `sudo` privileges to the Spark
* At least 200MB of disk space for the VS Code application
* (Remote Use only) VS Code installed on your laptop, downloaded from https://code.visualstudio.com/download.

## Time & risk

* **Duration:** 5 minutes
* **Risk level:** Low - installation uses official packages with standard rollback
* **Rollback:** Standard package removal via system package manager
* **Last Updated:** 07/23/2026
  * Clarify options and minor copyedits

## Desktop Use

## Step 1. Download VS Code ARM64 installer to your DGX Spark

Go to the VS Code [download](https://code.visualstudio.com/download) page and download the appropriate ARM64 `.deb` package for your system.

Alternatively, download the installer with this command:

```bash
wget https://code.visualstudio.com/sha/download?build=stable\&os=linux-deb-arm64 -O vscode-arm64.deb
```

## Step 2. Install VS Code

Install the downloaded package using the system package manager. 

You can click on the installer file directly or use the command line. 

```bash
## Install the downloaded .deb package
sudo dpkg -i vscode-arm64.deb

## Fix any dependency issues if they occur
sudo apt-get install -f
```

## Step 3. Open VS Code

You can open the app directly from the list of applications or from the command line.

```bash
## Test launch (will open VS Code GUI)
code &
```

VS Code should launch and display the welcome screen.

## Step 4. Configure for Spark development

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

## Step 5. Validate setup and test functionality

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
* Test Git integration by running `git init` and then `git status` in the terminal

## Step 6. Uninstalling VS Code

> [!WARNING]
> Removing the hidden folders will remove all user settings and extensions.

To remove VS Code if needed:
```bash
## Remove VS Code package
sudo apt-get remove code

## Remove configuration files (optional)
rm -rf ~/.config/Code
rm -rf ~/.vscode
```

## Remote Use

## Step 1. Install and configure NVIDIA Sync

Follow the [NVIDIA Sync setup guide](https://build.nvidia.com/spark/connect-to-your-spark/sync) to:
- Install NVIDIA Sync for your operating system
- Configure which development tools you want to use (VS Code, Cursor, Terminal, etc.)
- Add your DGX Spark device by providing its hostname/IP and credentials

NVIDIA Sync will automatically configure SSH key-based authentication for secure, password-free access.

## Step 2. Launch VS Code through NVIDIA Sync

- Click the NVIDIA Sync icon in your system tray/taskbar
- Ensure your device is connected (click "Connect" if needed)
- Click on "VS Code" to launch it with an automatic SSH connection to your DGX Spark
- Wait for the remote connection to be established (your local machine may ask for a password or to authorize the connection)
- You may be prompted to "trust the authors of the files in this folder" when you first land in the home directory after a successful SSH connection

## Step 3. Validation and follow-ups

- Verify that you can access your DGX Spark's filesystem with VS Code as a text editor
- Open the integrated terminal in VS Code and run test commands like `hostnamectl` and `whoami` to ensure you are remotely accessing your DGX Spark
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


For latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
