# Register DGX Spark to Brev

> Link your DGX Spark to Brev for remote access and shared environments

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NVIDIA Brev is an AI development platform that makes GPU environments remotely accessible, shareable, and easy to standardize using preconfigured setups called Launchables. 

This walkthrough will help you connect your NVIDIA DGX Spark to Brev so it shows up as a managed GPU environment in Brev. After a one-time registration, your Spark becomes remotely accessible and shareable.

## What you'll accomplish

You’ll register your DGX Spark with Brev and it will be visible as a healthy node in the Brev web UI and CLI, ready to share access and accept workloads whenever needed.

## What to know before starting

While Brev automates the complex configuration, understanding a few key concepts when establishing the initial connection will be useful:

* **Terminal Basics**:
  * Familiarity with the command line to run a few simple setup commands

## Prerequisites

Your DGX Spark [device is set up](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html). You will also need the following:

* **Brev Account**:
  * Have an NVIDIA Brev account. Create one [here](https://login.brev.nvidia.com/signin) if you don’t have one.

* **Permissions**:
  * You have administrative (root or sudo) access on the DGX Spark device to run the registration command.

## Time & risk

* **Estimated time:** 5-10 minutes
* **Risk level:** Low - Registration configures the Spark for secure remote access without altering your existing workloads
* **Rollback:** The Brev configuration can be removed through the UI and CLI

## Instructions

## Step 1. Log in to Brev

Go to the [Brev UI](https://brev.nvidia.com), log in, and confirm you’re in the correct org (by clicking the org button on the top right hand side of the page). Once logged in, go to the [Registered Compute](https://brev.nvidia.com/org/environments?tab=registered-compute) section under the "GPU" tab in the main navigation.

Click the “Register Compute” button and follow the instructions in the pop-up window.

## Step 2. Complete Popup Instructions

* Install the Brev CLI
* Configure your compute
    * Add a name for compute
    * To configure ssh, ensure the “Enable SSH access” toggle is on
* Run the registration command

## Step 3. Follow Registration Flow

In the CLI, you’ll be walked through registration. Go through the flow until registration is complete.

## Step 4. Confirm Spark in Brev UI

* Go to the [Brev UI](https://brev.nvidia.com)
* Navigate to the [Registered Compute](https://brev.nvidia.com/org/environments?tab=registered-compute)
* Confirm that the DGX Spark appears as a registered node with a **Connected** status 

## Step 5. Next Steps

Your Spark is now integrated into Brev as a secure, remotely accessible GPU environment.

Now that your hardware is connected, you can:

* **Share Access Anywhere:** Access your machine from anywhere and share access with others through the Brev UI by:
    * Adding the user to your [Team](https://brev.nvidia.com/org/team)
    * Navigating to your instance in the [Registered Compute](https://brev.nvidia.com/org/environments?tab=registered-compute) section
    * In **SSH Access** section of the instance, search for the user you wish to add and click **Modify Access** to enable access

## Step 6. Cleanup

If you ever decide to unregister your Spark with Brev, you can either do so through the Brev UI or the Brev CLI.

With the CLI simply run:

```bash
brev deregister
```

In the UI:
* Go to the [Brev UI](https://brev.nvidia.com)
* Navigate to the section listing “GPU Environments” and look under “Registered Compute”
* Click the “Remove” menu item on the Spark you wish to delete from Brev.
* Confirm your selection.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Your DGX Spark is showing up in the wrong org | You registered your DGX Spark to the wrong org | Run `brev set <my-org>` and then redo the registration |
| Unable to `brev shell <name>` | Need to refresh | `brev refresh` |

For the latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
