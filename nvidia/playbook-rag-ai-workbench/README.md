# Build a RAG Application with AI Workbench

> An agentic retrieval flow with query routing and hallucination checks

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This walkthrough shows how to set up and run an agentic retrieval-augmented generation (RAG) project using NVIDIA AI Workbench. You'll use AI Workbench to clone and run a pre-built agentic RAG application that routes queries, evaluates responses for relevancy and hallucination, and iterates through evaluation and generation cycles. The project uses a Gradio web interface and can work with NVIDIA-hosted API endpoints or self-hosted models.

## What you'll accomplish

You'll have a fully functional agentic RAG application running in NVIDIA AI Workbench with a web interface where you can submit queries and receive intelligent responses. The system demonstrates advanced RAG capabilities including query routing, response evaluation, and iterative refinement, giving you hands-on experience with both AI Workbench's development environment and sophisticated RAG architectures.

## What to know before starting

**Required:**

- Basic familiarity with retrieval-augmented generation (RAG) concepts
- Understanding of API keys and how to generate them
- Comfort working with web applications and browser interfaces

**Optional:**

- Basic understanding of containerized development environments

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | NVIDIA AI Workbench + [workbench-example-agentic-rag](https://github.com/NVIDIA/workbench-example-agentic-rag) (Gradio chat) | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Hardware platform powered on, networked, and reachable for local desktop access

**Software requirements**

- NVIDIA AI Workbench installed or ready to install on the hardware platform
- Free NVIDIA API key: generate at [NGC API Keys](https://org.ngc.nvidia.com/setup/api-keys) (include **Public API Endpoints** permissions)
- Free Tavily API key: generate at [Tavily](https://tavily.com/)
- Internet connection for cloning repositories and accessing APIs
- Web browser for accessing the Gradio interface

## Time & risk

- **Estimated time:** 30–45 MIN (including AI Workbench installation if needed)
- **Risk level:** Low
  - Uses pre-built containers and established APIs
  - API authentication errors if keys are missing, expired, or lack permissions
- **Rollback:** Delete the cloned project from AI Workbench to remove all components. No system changes are made outside the AI Workbench environment.
- **Last Updated:** 08/03/2026
  - Clone and run the agentic RAG example in NVIDIA AI Workbench with query routing, hallucination checks, and a Gradio chat interface on supported hardware platforms

## Instructions

## Step 1. Install NVIDIA AI Workbench

Install AI Workbench on your hardware platform and complete the initial setup wizard.

On your hardware platform, open the **NVIDIA AI Workbench** application and click **Begin Installation**.

1. The installation wizard will prompt for authentication
2. Wait for the automated install to complete (several minutes)
3. Click **Let's Get Started** when installation finishes

> [!NOTE]
> If you encounter the following error message, reboot your hardware platform and then reopen NVIDIA AI Workbench:
> "An error occurred ... container tool failed to reach ready state. try again: docker is not running"

## Step 2. Verify API key requirements

Ensure you have both required API keys before proceeding with the project setup. Keep these keys safe.

- Tavily API Key: https://tavily.com/
- NVIDIA API Key: https://org.ngc.nvidia.com/setup/api-keys
- Ensure this key has **Public API Endpoints** permissions

Keep both keys available for the next step.

## Step 3. Clone the agentic RAG project

Clone the pre-built agentic RAG project from GitHub into your AI Workbench environment.

From the AI Workbench landing page, select the **Local** location, if not done so already, then click **Clone Project** from the top right corner.

Paste this Git repository URL in the clone dialog: https://github.com/NVIDIA/workbench-example-agentic-rag

Click **Clone** to begin the clone and build process.

## Step 4. Configure project secrets

Configure the API keys required for the agentic RAG application to function properly.

While the project builds, configure the API keys using the yellow warning banner that appears:

1. Click **Configure** in the yellow banner
2. Enter your `NVIDIA_API_KEY`
3. Enter your `TAVILY_API_KEY`
4. Save the configuration

Wait for the project build to complete before proceeding.

## Step 5. Launch the chat application

Start the web-based chat interface where you can interact with the agentic RAG system.

Navigate to **Environment** > **Project Container** > **Apps** > **Chat** and start the web application.

A browser window will open automatically and load with the Gradio chat interface.

## Step 6. Test the basic functionality

Verify the agentic RAG system is working by submitting a sample query.

In the chat application, click on or type a sample query such as: `How do I add an integration in the CLI?`

Wait for the agentic system to process and respond. The response, while general, should demonstrate intelligent routing and evaluation.

## Step 7. Validate project

Confirm your setup is working correctly by testing the core features.

Verify the following components are functioning:

- Web application loads without errors
- Sample queries return responses
- No API authentication errors appear
- The agentic reasoning process is visible in the interface under **Monitor**

## Step 8. Complete optional quickstart

Evaluate advanced features by uploading data, retrieving context, and testing custom queries.

**Substep A: Upload sample dataset**

Complete the in-app quickstart instructions to upload the sample dataset and test improved RAG-based responses.

**Substep B: Test custom dataset (optional)**

Upload a custom dataset, adjust the Router prompt, and submit custom queries to test customization.

## Step 9. Cleanup

You can remove the project if needed. Cleanup is optional rollback — not required to finish the playbook.

> [!WARNING]
> This will permanently delete the project and all associated data.

To remove the project completely:

1. In AI Workbench, click on the three dots next to a project
2. Select **Delete Project**
3. Confirm deletion when prompted

> [!NOTE]
> All changes are contained within AI Workbench. No system-level modifications were made outside the AI Workbench environment.

## Step 10. Next steps

Explore further advanced features and development options with the agentic RAG system:

1. Modify component prompts in the project code
2. Upload different documents to test routing and customization
3. Experiment with different query types and complexity levels
4. Review the agentic reasoning logs in the **Monitor** tab to understand decision-making

Consider customizing the Gradio UI or integrating the agentic RAG components into your own projects.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Tavily API Error | Internet connection or DNS issues | Wait and retry the query |
| 401 Unauthorized | Wrong or malformed API key | Replace the key in Project Secrets and restart |
| 403 Unauthorized | API key lacks permissions | Generate a new key with **Public API Endpoints** access |
| Agentic loop timeout | Complex query exceeding time limit | Try a simpler query or retry |
| Container tool failed to reach ready state / docker is not running | Docker not ready after AI Workbench install | Reboot the hardware platform, then reopen NVIDIA AI Workbench |

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
