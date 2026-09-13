# Register AI Compute with Brev

> A shared GPU workspace your team can access from anywhere

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NVIDIA Brev is an AI development platform that makes GPU environments remotely accessible, shareable, and easy to standardize using preconfigured setups called Launchables.

This walkthrough helps you connect your hardware platform to Brev so it appears as a managed GPU environment. After a one-time registration, your hardware becomes remotely accessible and shareable.

## What you'll accomplish

You'll register your hardware platform with Brev. It will appear as a healthy node in the Brev web UI and CLI, ready to share access and accept workloads.

## What to know before starting

**Required:**

- Familiarity with the Linux command line to run a few setup commands

**Optional:**

- Basic understanding of SSH access and team/org membership in cloud developer portals

## Supported hardware platforms

Use the matrix below to confirm your hardware platform. The same registration workflow applies across supported hardware platforms.

| Hardware platform | OS | Memory  | Multi-node capable hardware |
| :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS | 128 GB Unified Memory | — |
| **DGX Station** | DGX OS | Large HBM + Grace DRAM | — |

> [!NOTE]
> Only platforms listed in the Supported hardware platforms table above are covered by this playbook.

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Hardware platform is powered on, networked, and reachable for local or SSH terminal access

**Software requirements**

- An NVIDIA Brev account — [create an account](https://login.brev.nvidia.com/signin) if you do not have one
- Administrative (root or sudo) access on the hardware platform to run the registration command
- Network access from the hardware platform to Brev services

## Time & risk

- **Estimated time:** 5–10 MIN
- **Risk level:** Low
  - Registration configures the hardware platform for secure remote access without altering existing workloads
- **Rollback:** Remove the Brev registration through the Brev UI or CLI (see Cleanup in the **Instructions** tab)
- **Last Updated:** 07/31/2026
  - Registration workflow for connecting supported hardware platforms to NVIDIA Brev

## Instructions

## Step 1. Log in to Brev

Go to the [Brev UI](https://brev.nvidia.com), log in, and confirm you are in the correct org (click the org control in the upper-right of the page). Once logged in, open [Registered Compute](https://brev.nvidia.com/org/environments?tab=registered-compute) under the **GPU** tab in the main navigation.

Click **Register Compute** and follow the instructions in the pop-up window.

## Step 2. Complete the pop-up instructions

In the Register Compute flow:

- Install the Brev CLI
- Configure your compute
  - Add a name for the compute
  - To configure SSH, ensure the **Enable SSH access** toggle is on
- Run the registration command on your hardware platform (requires administrative privileges)

## Step 3. Follow the registration flow

In the CLI on your hardware platform, complete the interactive registration flow until registration finishes successfully.

## Step 4. Confirm registration in the Brev UI

1. Go to the [Brev UI](https://brev.nvidia.com)
2. Open [Registered Compute](https://brev.nvidia.com/org/environments?tab=registered-compute)
3. Confirm that your hardware platform appears as a registered node with a **Connected** status

## Step 5. Next steps

Your hardware platform is now integrated into Brev as a secure, remotely accessible GPU environment.

You can share access through the Brev UI by:

1. Adding the user to your [Team](https://brev.nvidia.com/org/team)
2. Opening your instance under [Registered Compute](https://brev.nvidia.com/org/environments?tab=registered-compute)
3. In the **SSH Access** section for the instance, search for the user and click **Modify Access** to enable access

## Step 6. Cleanup

To unregister your hardware platform from Brev, use either the Brev CLI or the Brev UI.

**CLI:**

```bash
brev deregister
```

**UI:**

1. Go to the [Brev UI](https://brev.nvidia.com)
2. Open **Registered Compute** under GPU Environments
3. Choose **Remove** for the registered compute you want to delete from Brev
4. Confirm your selection

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every supported platform.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| Registered compute appears in the wrong org | All hardware platforms | Registration was completed while signed into a different org | Run `brev set <my-org>`, then redo the registration flow |
| Unable to run `brev shell <name>` | All hardware platforms | Local CLI state is stale | Run `brev refresh` |

For product documentation, see [NVIDIA Brev documentation](https://docs.nvidia.com/brev/latest).
