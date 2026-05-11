---
name: dgx-spark-nemoclaw
description: Install NemoClaw on DGX Spark with local Ollama inference and Telegram bot integration — on NVIDIA DGX Spark. Use when setting up nemoclaw on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/nemoclaw/README.md -->
# NemoClaw with Nemotron 3 Super and Telegram on DGX Spark

> Install NemoClaw on DGX Spark with local Ollama inference and Telegram bot integration

**NVIDIA NemoClaw** is an open-source reference stack that simplifies running OpenClaw always-on assistants more safely. It installs the **NVIDIA OpenShell** runtime -- an environment designed for executing agents with additional security -- and open-source models like NVIDIA Nemotron. A single installer command handles Node.js, OpenShell, and the NemoClaw CLI, then walks you through an onboard wizard to create a sandboxed agent on your DGX Spark using Ollama with Nemotron 3 Super.

By the end of this playbook you will have a working AI agent inside an OpenShell sandbox, accessible via a web dashboard and a Telegram bot, with inference routed to a local Nemotron 3 Super 120B model on your Spark -- all without exposing your host filesystem or network to the agent.

### What you'll accomplish

- Configure Docker and the NVIDIA container runtime for OpenShell on DGX Spark
- Install Ollama, pull Nemotron 3 Super 120B, and configure it for sandbox access

**Outcome**: - Configure Docker and the NVIDIA container runtime for OpenShell on DGX Spark
- Install Ollama, pull Nemotron 3 Super 120B, and configure it for sandbox access
- Install NemoClaw with a single command (handles Node.js, OpenShell, and the CLI)
- Run the onboard wizard to create a sandbox and configure local inference
- Chat with the agent via the CLI, TUI, and web UI
- Set up a Telegram bot that forwards messages to your sandboxed agent

### Notice and disclaimers

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/nemoclaw/README.md`
<!-- GENERATED:END -->
