---
name: dgx-spark-tailscale
description: Use Tailscale to connect to your Spark on your home network no matter where you are — on NVIDIA DGX Spark. Use when setting up tailscale on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/tailscale/README.md -->
# Set up Tailscale on Your Spark

> Use Tailscale to connect to your Spark on your home network no matter where you are

Tailscale creates an encrypted peer-to-peer mesh network that allows secure access
to your NVIDIA DGX Spark device from anywhere without complex firewall configurations
or port forwarding. By installing Tailscale on both your DGX Spark and client devices,
you establish a private "tailnet" where each device gets a stable private IP
address and hostname, enabling seamless SSH access whether you're at home, work,
or a coffee shop.

**Outcome**: You will set up Tailscale on your DGX Spark device and client machines to
create secure remote access. After completion, you'll be able to SSH into your
DGX Spark from anywhere using simple commands like `ssh user@spark-hostname`, with
all traffic automatically encrypted and NAT traversal handled transparently.

Duration: 15-30 minutes for initial setup, 5 minutes per additional device

**Full playbook**: `/Users/jkneen/Documents/GitHub/dgx-spark-playbooks/nvidia/tailscale/README.md`
<!-- GENERATED:END -->
