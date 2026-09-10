# Build Knowledge Graphs with txt2kg

> Extract triples with Ollama or vLLM, store them in a graph database, and explore them in a GPU-accelerated web UI


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Defaults by stack](#defaults-by-stack)
  - [Ollama stacks](#ollama-stacks)
  - [vLLM stack](#vllm-stack)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Transform unstructured text into a structured knowledge graph you can explore and query. This playbook extracts subject–predicate–object triples with a local LLM, stores them in a graph database, and renders the graph in an interactive GPU-accelerated web UI.

The workflow covers:

- **Knowledge triple extraction** — local LLM inference (Ollama or vLLM) to extract relationships from documents
- **Graph database storage** — ArangoDB or Neo4j for storing and traversing triples
- **GPU-accelerated visualization** — Three.js WebGPU for interactive 2D/3D exploration
- **Web interface** — Next.js app for document upload, graph editing, and graph-based queries

## What you'll accomplish

A running, containerized system that:

- Processes uploaded documents (markdown, text, CSV)
- Generates and stores knowledge triples
- Lets you visualize and query the graph through a browser

## What to know before starting

**Required:**

- Basic Docker container usage
- Familiarity with command-line operations

**Optional:**

- Familiarity with knowledge graphs and graph databases

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, OS, memory, and whether multi-node applies. The same base workflow applies across supported hardware platforms; `./start.sh` always starts the default ArangoDB + Ollama stack.

| Hardware platform | OS | Memory  | Multi-node capable hardware |
| :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | — |
| **DGX Station** | DGX OS (Linux) | Large HBM + Grace DRAM | — |


#### Stack options by hardware platform

| Hardware platform | Default start command | Other stack options |
| ----------------- | --------------------- | ------------------- |
| **DGX Spark** | `./start.sh` → ArangoDB + Ollama | `./start.sh --neo4j` → Neo4j + Ollama; `./start.sh --vllm` → Neo4j + vLLM |
| **DGX Station** | `./start.sh` → ArangoDB + Ollama | `./start.sh --neo4j` → Neo4j + Ollama; `./start.sh --vllm` → Neo4j + vLLM |

> [!IMPORTANT]
> The 64 KB page-size issue is specific to DGX Station; DGX Spark is not affected. On affected DGX Station systems, prefer `./start.sh --neo4j`. Some upstream ArangoDB and Qdrant container images can abort at startup with `<jemalloc>: Unsupported system page size`; the Neo4j + Ollama stack preserves the fast local Ollama flow while avoiding ArangoDB.

> [!NOTE]
> Larger models generally produce higher-quality triples. Choose a model that fits the memory available on your hardware platform. See **Instructions → Step 3** for defaults and links to explore more models.

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient memory for your chosen LLM

**Software requirements**

- Docker installed and configured with the NVIDIA Container Toolkit
- Docker Compose
- Network access for container image and model downloads

**Ancillary files** (in `nvidia/playbook-txt2kg/assets` after Step 1):

- `start.sh` / `stop.sh` — launch and shut down services
- `deploy/compose/` — Docker Compose configurations

## Time & risk

- **Estimated time:** 30 MIN (longer on first run while models download; vLLM model load can take 30+ minutes)
- **Risk level:** Low
  - GPU memory needs depend on the chosen model
  - Document processing time scales with document size and complexity
- **Rollback:** Stop and remove containers; optionally delete downloaded models (see Instructions)
- **Last Updated:** 08/05/2026
  - Added explicit Neo4j + Ollama stack option; model defaults and explore links live in Instructions Step 3

## Instructions

## Step 1. Clone the repository

In a terminal, clone the playbook repository and navigate to the project assets directory.

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-txt2kg/assets
```

## Step 2. Start the txt2kg services

Use the provided start script. By default, `./start.sh` starts the ArangoDB + Ollama stack. Use one stack flag when you want a different graph database or LLM backend.

```bash
./start.sh
```

Common options:

```bash
## Stack flags
./start.sh --neo4j     # Neo4j + Ollama
./start.sh --vllm      # Neo4j + vLLM

## Optional flags
./start.sh --vector-search   # Add Qdrant + Sentence Transformers
./start.sh --help            # Full option list
```

DGX Spark is not affected by the 64 KB page-size issue. On affected DGX Station systems, start with `./start.sh --neo4j` to use Neo4j + Ollama.

The script will:

- Check for GPU availability
- Start Docker Compose services for the selected stack
- For the vLLM stack, start the backend container; model load can take 30+ minutes, and progress is available with `docker logs vllm-service -f`
- Print the web UI URL when ready

## Step 3. Choose and load a model

Triple extraction quality depends on the LLM behind Ollama or vLLM. Start with the stack default below, then explore other models to trade quality vs. speed and memory.

### Defaults by stack

| Stack | Default model | Notes |
| ----- | ------------- | ----- |
| **ArangoDB + Ollama** | `llama3.1:8b` | Pull with `docker exec ollama-compose ollama pull llama3.1:8b` |
| **Neo4j + Ollama** | `llama3.1:8b` | Pull with `docker exec ollama-compose ollama pull llama3.1:8b` |
| **Neo4j + vLLM** | `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8` | Loaded by the vLLM container; first start can take 30+ minutes |

### Ollama stacks

Pull a language model for knowledge extraction (skip if the default was already pulled on start):

```bash
docker exec ollama-compose ollama pull <model-name>
```

Examples:

```bash
docker exec ollama-compose ollama pull llama3.1:8b
```

Then select the model in the web UI.

**Explore more models:** browse the [Ollama model library](https://ollama.com/search). Prefer models that fit the memory available on your hardware platform.

### vLLM stack

The model is loaded automatically by the vLLM container from `VLLM_MODEL` in `deploy/compose/docker-compose.vllm.yml`.

To try a different Hugging Face checkpoint:

1. Set `VLLM_MODEL` to another handle (for example from [Hugging Face Models](https://huggingface.co/models) or [vLLM Recipes](https://recipes.vllm.ai/browse)).
2. Restart: `./stop.sh` then start again with your stack flags (for example `./start.sh --vllm`).
3. Confirm readiness: `docker logs vllm-service -f`

**Explore more models:** [vLLM Recipes — DGX Spark](https://recipes.vllm.ai/browse?panel=open&hw=dgx_spark_gb10) · [vLLM Recipes — DGX Station](https://recipes.vllm.ai/browse?panel=open&hw=dgx_station_gb300) · [Hugging Face](https://huggingface.co/models)

> [!NOTE]
> Larger models generally produce higher-quality triples but need more memory and load time. If you hit memory limits, choose a smaller or quantized model.

## Step 4. Access the web interface

Open your browser and navigate to:

```
http://localhost:3001
```

You can also access stack-specific services:

| Service | URL | Stack |
| ------- | --- | ----- |
| Web UI | http://localhost:3001 | All |
| ArangoDB Web Interface | http://localhost:8529 | ArangoDB + Ollama |
| Neo4j Browser | http://localhost:7474 | Neo4j + Ollama or Neo4j + vLLM |
| Ollama API | http://localhost:11434 | ArangoDB + Ollama or Neo4j + Ollama |
| vLLM API | http://localhost:8001 | Neo4j + vLLM |

## Step 5. Upload documents and build knowledge graphs

If the LLM backend is still loading, the UI may show an initializing banner until the backend is ready.

#### 5.1. Document upload

- Upload text documents (markdown, text, and CSV are supported)
- Documents are chunked and processed for triple extraction

#### 5.2. Knowledge graph generation

- The system extracts subject–predicate–object triples using the selected LLM (Ollama or vLLM)
- Triples are stored in the selected graph database: ArangoDB for the default stack, or Neo4j when started with `--neo4j` or `--vllm`

#### 5.3. Interactive visualization

- View the knowledge graph in 2D or 3D with GPU-accelerated rendering
- Explore nodes and relationships interactively

#### 5.4. Graph-based queries

- Ask questions about your documents in the query interface
- Graph traversal enriches context with entity relationships
- The LLM generates responses using the enriched graph context

## Step 6. Cleanup and rollback

Stop services with the same stack flags you used to start:

```bash
## Stop services (match the stack you started)
./stop.sh
## If you started with Neo4j + Ollama: ./stop.sh --neo4j
## If you started with Neo4j + vLLM: ./stop.sh --vllm

## Remove containers and volumes (optional)
## Ollama stack:
## docker compose -f deploy/compose/docker-compose.yml down -v
## Neo4j + Ollama stack:
## docker compose -f deploy/compose/docker-compose.neo4j.yml down -v
## Neo4j + vLLM stack:
## docker compose -f deploy/compose/docker-compose.vllm.yml down -v

## Remove downloaded Ollama models (Ollama stacks only, optional)
## docker exec ollama-compose ollama rm <model-name>
```

## Step 7. Next steps

- Experiment with different models (Step 3 links) for extraction quality vs. speed
- Customize triple extraction prompts for domain-specific knowledge
- Explore advanced graph querying and visualization features

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every supported platform.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| Ollama performance issues | All hardware platforms | Suboptimal Ollama settings | Set environment variables: `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KEEP_ALIVE=30m`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` |
| Memory pressure when switching Ollama models | DGX Spark | Unified memory buffer cache not released | Flush buffer cache (see UMA note below) |
| VRAM exhausted or memory pressure | DGX Station | GPU memory fragmentation | Clear GPU memory: `nvidia-smi --gpu-reset` or restart Docker containers |
| Slow triple extraction | All hardware platforms | Large model or large context window | Reduce document chunk size or use a faster model |
| ArangoDB connection refused | All hardware platforms (Ollama stack) | Service not fully started | Wait ~30s after `./start.sh`, then verify with `docker ps` |
| ArangoDB exits with `<jemalloc>: Unsupported system page size` | DGX Station systems with 64 KB page-size kernels | The upstream ArangoDB image may include jemalloc built for smaller pages | Use `./start.sh --neo4j` for Neo4j + Ollama |
| Qdrant exits or crash-loops with `<jemalloc>: Unsupported system page size` | DGX Station systems with 64 KB page-size kernels using `--vector-search` | The upstream Qdrant image may include jemalloc built for smaller pages | Leave vector search disabled, or retry after the Qdrant image is updated for 64 KB pages |
| Container fails to start with GPU error | All hardware platforms | NVIDIA Container Toolkit not configured | Run `nvidia-ctk runtime configure --runtime=docker` and restart Docker |
| Port already in use | All hardware platforms | Previous instance still running | Run `./stop.sh` (with the same stack flags) or `docker compose down` |
| Need another graph or LLM stack | All hardware platforms | Default stack is not the one you want | Use `./start.sh --neo4j` for Neo4j + Ollama or `./start.sh --vllm` for Neo4j + vLLM |
| vLLM takes long to become ready | All hardware platforms (vLLM stack) | Model load can take 30+ minutes | The UI may show an initializing banner while the model loads. Check progress: `docker logs vllm-service -f` |

The 64 KB page-size rows above apply to affected DGX Station systems only; DGX Spark is not affected.

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

> [!NOTE]
> **Model size vs. memory.** Larger models generally improve triple quality. If you hit memory limits, reduce context window size, use a quantized variant, or choose a smaller model for your hardware platform.
