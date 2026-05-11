---
description: Install Ollama on an NVIDIA DGX Spark and expose its API to a local laptop via NVIDIA Sync SSH tunnel. Use when a user wants to run LLM inference on DGX Spark hardware and call the API from their laptop on localhost:11434 without exposing ports on their network.
---

## When to use this skill
- User has an NVIDIA DGX Spark with NVIDIA Sync installed on their laptop
- Wants Ollama running on Spark, API accessible from their laptop
- Wants an easy-to-use inference runtime (vs. the complexity of vLLM or TRT-LLM)

## Key decisions to confirm before executing
- **Model choice** — default in the playbook is `qwen2.5:32b` (~18GB, optimized for Blackwell). Ask the user if they want a smaller model (`qwen2.5:7b`, `llama3.1:8b`, `phi3.5:3.8b`) for lower VRAM or faster download.
- **Check first** — run `ollama --version` on the Spark before installing; skip installation if already present.

## Non-obvious gotchas
- The SSH tunnel must be re-activated after NVIDIA Sync restarts — `localhost:11434` only works while the "Ollama Server" custom app is active in Sync.
- Uninstall is destructive: `sudo rm -rf /usr/share/ollama` removes all downloaded models (often tens of GB). Confirm with user before running cleanup.
- Streaming responses (`"stream": true`) behave differently than non-streaming — use `curl -N` to see them.

## Related skills
- **Prerequisite**: `dgx-spark-connect-to-your-spark` — NVIDIA Sync + local network access basics. If the user hasn't set this up yet, do it first.
- **Composes with**: `dgx-spark-open-webui` — web chat UI on top of Ollama. Most common follow-up.
- **Alternative**: `dgx-spark-lm-studio` — GUI-based model management instead of Ollama's CLI.
- **Alternative**: `dgx-spark-llama-cpp` — lower-level control over inference.
- **Upgrade path**: `dgx-spark-vllm` — when the user needs higher throughput or is serving multiple concurrent users.
