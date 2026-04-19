---
name: dgx-spark
description: Catalog and router for NVIDIA DGX Spark playbooks — use when a user asks about setting up their DGX Spark, wants an overview of what they can run on Spark hardware, or needs help choosing between inference runtimes, fine-tuning frameworks, or networking setups. Lists all available dgx-spark-* skills and encodes the relationships between them (prerequisites, alternatives, composes-with, upgrade paths).
---

# DGX Spark Playbooks — Index

Use this catalog to route the user to the right specific `dgx-spark-*` skill. Each entry below names a leaf skill; invoke it when the user's intent matches.

## Categories

### Inference runtimes (serve models)
- `dgx-spark-ollama` — easiest, good default for most users
- `dgx-spark-vllm` — higher throughput, production-grade serving
- `dgx-spark-trt-llm` — maximum Blackwell performance, most complex setup
- `dgx-spark-sglang` — structured generation, batched inference
- `dgx-spark-llama-cpp` — lightweight, CPU/GPU flexibility
- `dgx-spark-lm-studio` — GUI-based model management
- `dgx-spark-nim-llm` — NVIDIA NIM microservices

### Chat & UI
- `dgx-spark-open-webui` — web chat UI, pairs with Ollama
- `dgx-spark-live-vlm-webui` — vision-language model interface
- `dgx-spark-dgx-dashboard` — GPU/system monitoring

### Fine-tuning
- `dgx-spark-pytorch-fine-tune` — baseline PyTorch fine-tuning
- `dgx-spark-nemo-fine-tune` — NVIDIA NeMo framework
- `dgx-spark-unsloth` — memory-efficient fine-tuning
- `dgx-spark-llama-factory` — multi-model fine-tuning framework
- `dgx-spark-flux-finetuning` — FLUX.1 Dreambooth LoRA (image models)

### Networking & multi-Spark
- `dgx-spark-connect-to-your-spark` — **foundational: local network access setup**
- `dgx-spark-tailscale` — VPN-based remote access
- `dgx-spark-connect-two-sparks` — link two Sparks
- `dgx-spark-connect-three-sparks` — ring topology
- `dgx-spark-multi-sparks-through-switch` — switched multi-Spark
- `dgx-spark-nccl` — collective communication across Sparks

### Dev environments & tooling
- `dgx-spark-vscode` — VS Code remote setup
- `dgx-spark-vibe-coding` — agentic coding in VS Code
- `dgx-spark-rag-ai-workbench` — RAG app in AI Workbench
- `dgx-spark-openshell` — secure long-running agents
- `dgx-spark-openclaw` — (advanced agent setup)
- `dgx-spark-nemoclaw` — Nemotron + Telegram agent

### Specialized workloads
- `dgx-spark-comfy-ui` — image generation UI
- `dgx-spark-isaac` — Isaac Sim / Isaac Lab (robotics)
- `dgx-spark-jax` — JAX on Spark
- `dgx-spark-cuda-x-data-science` — RAPIDS / data science
- `dgx-spark-multi-agent-chatbot` — multi-agent deployment
- `dgx-spark-multi-modal-inference` — multi-modal models
- `dgx-spark-nemotron` — Nemotron-3-Nano with llama.cpp
- `dgx-spark-nvfp4-quantization` — FP4 quantization workflows
- `dgx-spark-portfolio-optimization` — finance example
- `dgx-spark-single-cell` — single-cell RNA sequencing
- `dgx-spark-speculative-decoding` — speculative decoding inference
- `dgx-spark-spark-reachy-photo-booth` — Reachy robot demo
- `dgx-spark-txt2kg` — text-to-knowledge-graph
- `dgx-spark-vss` — video search & summarization agent

## Relationship graph

Edge types: `→prereq→` (must do first), `→pairs→` (composes naturally), `→alt→` (pick one, roughly equivalent choice), `→upgrade→` (next step when outgrowing this).

### Networking (almost everything depends on this)
- `connect-to-your-spark` →prereq→ **all remote-access playbooks**
- `tailscale` →alt→ `connect-to-your-spark`
- `connect-two-sparks` →prereq→ `nccl`
- `connect-two-sparks` →upgrade→ `connect-three-sparks` →upgrade→ `multi-sparks-through-switch`

### Inference stack
- `ollama` →pairs→ `open-webui` *(most common pairing — chat UI on top of Ollama)*
- `ollama` →alt→ `lm-studio` *(GUI vs CLI; roughly equivalent for local single-user use)*
- `ollama` →alt→ `llama-cpp` *(lower-level control)*
- `ollama` →upgrade→ `vllm` *(when throughput / OpenAI-compatible API matters)*
- `vllm` →alt→ `trt-llm` *(different use case — trt-llm for lowest latency with compiled engines; not strictly an upgrade)*
- `vllm` →composes→ `connect-two-sparks` + `nccl` *(multi-Spark serving for very large models)*
- `nemotron` →pairs→ `llama-cpp` *(playbook specifically uses llama.cpp runtime)*
- `nim-llm` →alt→ `vllm` *(NIM microservices vs. raw vLLM serving)*

### Fine-tuning pipelines
- `pytorch-fine-tune` →prereq→ `flux-finetuning` *(need baseline PyTorch setup first)*
- `nemo-fine-tune` →pairs→ `nim-llm` *(deploy tuned NeMo models via NIM)*
- `unsloth` →related→ `llama-factory` *(related but different specialties — unsloth for memory-efficient LoRA/QLoRA; llama-factory is a broader multi-technique framework)*

### Monitoring & observability
- `dgx-dashboard` →pairs→ **all inference/fine-tuning skills** *(GPU/system monitoring during workloads)*

### Performance tuning (compose with inference)
- `speculative-decoding` →composes→ `vllm`, `trt-llm` *(inference acceleration technique)*
- `nvfp4-quantization` →composes→ `vllm`, `trt-llm` *(quantize first, then serve)*

### Agent & automation stacks
- `nemoclaw` →pairs→ `nemotron` *(nemoclaw uses Nemotron internally)*
- `openclaw` →pairs→ `openshell` *(agent security pattern)*

### Dev env dependencies
- `vscode` →prereq→ `vibe-coding` *(vibe-coding builds on VS Code remote setup)*

## Suggestion rules

When the user's request is broad, narrow it with these questions before invoking a leaf:

| User says... | Ask / suggest |
|---|---|
| "chat with a model on Spark" | Default: `dgx-spark-ollama` + `dgx-spark-open-webui`. Ask: CLI-only or web UI? |
| "fastest inference" | `dgx-spark-trt-llm`, but warn it's the most complex. Ask if `vllm` would suffice. |
| "train" / "fine-tune a model" | Ask: from scratch (`nemo-fine-tune`) or adapt existing (`unsloth`, `llama-factory`)? Image model? → `flux-finetuning`. |
| "connect to my Spark" / "remote access" | `dgx-spark-connect-to-your-spark` first. Suggest `tailscale` as alternative for VPN use. |
| "multiple Sparks" | Ask: 2 (`connect-two-sparks`), 3 (`connect-three-sparks`), or more via switch? NCCL after physical link. |
| "I just got my Spark, what can I do" | List categories above. Suggest starting with `connect-to-your-spark` → `ollama`. |

## Curation notes

Edges above are a working starting point. Revise as real usage reveals which pairings matter most. In particular:
- `vllm →alt→ trt-llm` is inferred from the READMEs' positioning — confirm with users whether they see these as alternatives or a progression
- `speculative-decoding` and `nvfp4-quantization` compose with serving runtimes but the exact integration path may vary — check if users typically apply them standalone or as part of vllm/trt-llm setup
