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

This walkthrough helps you connect your hardware platform to Brev so it appears as a managed GPU environment. After registration and SSH setup, your hardware becomes remotely accessible and shareable.

## What you'll accomplish

You'll register your hardware platform with Brev and configure SSH access. It will appear as a healthy node in the Brev web UI and CLI, ready to share access and accept workloads.

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
- Owner or Admin access to your Brev org
- Administrative (root or sudo) access on the hardware platform to run the registration command
- Network access from the hardware platform to Brev services

## Time & risk

- **Estimated time:** 5–10 MIN
- **Risk level:** Low
  - Registration configures the hardware platform for secure remote access without altering existing workloads
- **Rollback:** Run `brev deregister` on the hardware platform (see Cleanup in the **Instructions** tab)
- **Last Updated:** 09/21/2026
  - Registration workflow for connecting supported hardware platforms to NVIDIA Brev

## Instructions

## Step 1. Log in to Brev

Go to the [Brev UI](https://brev.nvidia.com), log in, and confirm you are in the correct org. Once logged in, open [Compute](https://brev.nvidia.com/org/environments).

Click **Connect a local device** (or **Register Compute**) and follow the instructions in the pop-up window.

## Step 2. Complete the pop-up instructions

In the Brev Connect flow:

- Add a name for the compute
- Create or enter an unexpired **Read & Write Personal API key** for your org
- Run the generated command on your hardware platform to install the Brev CLI and register the compute (requires administrative privileges)

Save your API key and click **Done** after registration. Do not share the key or command.

## Step 3. Enable SSH access

On your hardware platform, enable SSH and grant yourself access using the same API key:

```bash
export PATH="$HOME/.local/bin:$PATH"
brev enable-ssh --api-key "<your-api-key>"
brev grant-ssh --api-key "<your-api-key>"
```

Grant your Brev user access using the same Linux user and SSH port (default: `22`) selected during SSH setup.

## Step 4. Confirm registration in the Brev UI

1. Go to the [Brev UI](https://brev.nvidia.com)
2. Open [Compute](https://brev.nvidia.com/org/environments)
3. Confirm that your hardware platform appears as a registered node with a **Connected** status

## Step 5. Next steps

Your hardware platform is now integrated into Brev as a secure, remotely accessible GPU environment.

You can share access through the Brev UI by:

1. Adding the user to your [Team](https://brev.nvidia.com/org/team)
2. Opening your instance under [Compute](https://brev.nvidia.com/org/environments)
3. In the **SSH Access** section for the instance, search for the user, click **Modify Access**, enable access for the Linux user selected in Step 3, and click **Save**

On your local machine, [install the Brev CLI](https://docs.nvidia.com/brev/cli/getting-started), run `brev login`, select the same org with `brev org set <org>`, and connect with `brev shell <name>`.

## Step 6. Cleanup

To unregister your hardware platform and complete local cleanup, run the following on that platform.

**CLI:**

```bash
brev deregister --api-key "<your-api-key>"
```

**UI:**

UI removal revokes access. Run `brev deregister` on the hardware platform to complete local cleanup.

1. Go to the [Brev UI](https://brev.nvidia.com)
2. Open [Compute](https://brev.nvidia.com/org/environments)
3. Choose **Remove** from the compute's menu
4. Enter the compute name and click **Remove Node**

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every supported platform.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| Registered compute appears in the wrong org | All hardware platforms | The API key belongs to a different org | Run `brev deregister --api-key "<original-org-key>"` on the hardware platform, then register with a key for the correct org |
| SSH access is denied | All hardware platforms | SSH is not enabled or access has not been granted | Complete Steps 3 and 5 in the **Instructions** tab |
| Unable to run `brev shell <name>` | All hardware platforms | Local CLI state is stale | Run `brev refresh` |

For product documentation, see [NVIDIA Brev documentation](https://docs.nvidia.com/brev/latest).
