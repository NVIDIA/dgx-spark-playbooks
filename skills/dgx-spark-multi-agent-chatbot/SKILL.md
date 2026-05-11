---
name: dgx-spark-multi-agent-chatbot
description: Deploy a multi-agent chatbot system and chat with agents on your Spark — on NVIDIA DGX Spark. Use when setting up multi-agent-chatbot on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/multi-agent-chatbot/README.md -->
# Build and Deploy a Multi-Agent Chatbot

> Deploy a multi-agent chatbot system and chat with agents on your Spark

This playbook shows you how to use DGX Spark to prototype, build, and deploy a fully local multi-agent system. 
With 128GB of unified memory, DGX Spark can run multiple LLMs and VLMs in parallel — enabling interactions across agents.

At the core is a supervisor agent powered by gpt-oss-120B, orchestrating specialized downstream agents for coding, retrieval-augmented generation (RAG), and image understanding. 
Thanks to DGX Spark's out-of-the-box support for popular AI frameworks and libraries, development and prototyping are fast and frictionless. 
Together, these components demonstrate how complex, multimodal workflows can be executed efficiently on local, high-performance hardware.

**Outcome**: You will have a full-stack multi-agent chatbot system running on your DGX Spark, accessible through
your local web browser. 
The setup includes:
- LLM and VLM model serving using llama.cpp servers and TensorRT LLM servers
- GPU acceleration for both model inference and document retrieval
- Multi-agent system orchestration using a supervisor agent powered by gpt-oss-120B
- MCP (Model Context Protocol) servers as tools for the supervisor agent

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/multi-agent-chatbot/README.md`
<!-- GENERATED:END -->
