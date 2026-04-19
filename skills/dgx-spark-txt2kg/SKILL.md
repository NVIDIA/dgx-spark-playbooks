---
name: dgx-spark-txt2kg
description: Transform unstructured text into interactive knowledge graphs with LLM inference and graph visualization — on NVIDIA DGX Spark. Use when setting up txt2kg on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/txt2kg/README.md -->
# Text to Knowledge Graph on DGX Spark

> Transform unstructured text into interactive knowledge graphs with LLM inference and graph visualization

This playbook demonstrates how to build and deploy a comprehensive knowledge graph generation and visualization solution that serves as a reference for knowledge graph extraction.
The unified memory architecture enables running larger, more accurate models that produce higher-quality knowledge graphs and deliver superior downstream GraphRAG performance.

This txt2kg playbook transforms unstructured text documents into structured knowledge graphs using:
- **Knowledge Triple Extraction**: Using Ollama with GPU acceleration for local LLM inference to extract subject-predicate-object relationships
- **Graph Database Storage**: ArangoDB for storing and querying knowledge triples with relationship traversal
- **GPU-Accelerated Visualization**: Three.js WebGPU rendering for interactive 2D/3D graph exploration

**Outcome**: You will have a fully functional system capable of processing documents, generating and editing knowledge graphs, and providing querying, accessible through an interactive web interface.
The setup includes:
- **Local LLM Inference**: Ollama for GPU-accelerated LLM inference with no API keys required
- **Graph Database**: ArangoDB for storing and querying triples with relationship traversal
- **Interactive Visualization**: GPU-accelerated graph rendering with Three.js WebGPU
- **Modern Web Interface**: Next.js frontend with document management and query interface
- **Fully Containerized**: Reproducible deployment with Docker Compose and GPU support

Duration: - 2-3 minutes for initial setup and container deployment

**Full playbook**: `/Users/jkneen/Documents/GitHub/dgx-spark-playbooks/nvidia/txt2kg/README.md`
<!-- GENERATED:END -->
