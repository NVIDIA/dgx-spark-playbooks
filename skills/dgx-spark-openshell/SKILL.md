---
name: dgx-spark-openshell
description: Run OpenClaw with local models in an NVIDIA OpenShell sandbox on DGX Spark — on NVIDIA DGX Spark. Use when setting up openshell on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/openshell/README.md -->
# Secure Long Running AI Agents with OpenShell on DGX Spark

> Run OpenClaw with local models in an NVIDIA OpenShell sandbox on DGX Spark

OpenClaw is a local-first AI agent that runs on your machine, combining memory, file access, tool use, and community skills into a persistent assistant. Running it directly on your system means the agent can access your files, credentials, and network—creating real security risks.

**NVIDIA OpenShell** solves this problem. It is an open-source sandbox runtime that wraps the agent in kernel-level isolation with declarative YAML policies. OpenShell controls what the agent can read on disk, which network endpoints it can reach, and what privileges it has—without disabling the capabilities that make the agent useful.

By combining OpenClaw with OpenShell on DGX Spark, you get the full power of a local AI agent backed by 128GB of unified memory for large models, while enforcing explicit controls over filesystem access, network egress, and credential handling.

### Notice & Disclaimers
#### Quick Start Safety Check

**Outcome**: You will install the OpenShell CLI (`openshell`), deploy a gateway on your DGX Spark, and launch OpenClaw inside a sandboxed environment using the pre-built OpenClaw community sandbox. The sandbox enforces filesystem, network, and process isolation by default. You will also configure local inference routing so OpenClaw uses a model running on your Spark without needing external API keys.

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/openshell/README.md`
<!-- GENERATED:END -->
