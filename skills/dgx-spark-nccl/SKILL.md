---
name: dgx-spark-nccl
description: Install and test NCCL on two Sparks — on NVIDIA DGX Spark. Use when setting up nccl on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/nccl/README.md -->
# NCCL for Two Sparks

> Install and test NCCL on two Sparks

NCCL (NVIDIA Collective Communication Library) enables high-performance GPU-to-GPU communication
across multiple nodes. This walkthrough sets up NCCL for multi-node distributed training on
DGX Spark systems with Blackwell architecture. You'll configure networking, build NCCL from
source with Blackwell support, and validate communication between nodes.

**Outcome**: You'll have a working multi-node NCCL environment that enables high-bandwidth GPU communication
across DGX Spark systems for distributed training workloads, with validated network performance
and proper GPU topology detection.

Duration: 30 minutes for setup and validation · Risk: Medium - involves network configuration changes

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/nccl/README.md`
<!-- GENERATED:END -->
