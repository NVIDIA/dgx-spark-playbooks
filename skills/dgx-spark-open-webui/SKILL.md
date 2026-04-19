---
name: dgx-spark-open-webui
description: Install Open WebUI on NVIDIA DGX Spark for a web-based chat interface to LLMs running on Spark GPU. Use when a user wants a browser UI for chatting with local models — most commonly paired with Ollama, either bundled inside Open WebUI or as a separate backend.
---

<!-- GENERATED:BEGIN from nvidia/open-webui/README.md -->
# Open WebUI with Ollama

> Install Open WebUI and use Ollama to chat with models on your Spark

Open WebUI is an extensible, self-hosted AI interface that operates entirely offline.
This playbook shows you how to deploy Open WebUI with an integrated Ollama server on your DGX Spark device that lets you access the web interface from your local browser while the models run on Spark's GPU.

**Outcome**: You will have a fully functional Open WebUI installation running on your DGX Spark. This will be accessible through your local web browser either via **NVIDIA Sync's managed SSH tunneling (recommended)** or via manual setup. The setup includes integrated Ollama for model management, persistent data storage, and GPU acceleration for model inference.

Duration: 15-20 minutes for initial setup, plus model download time (varies by model size)

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/open-webui/README.md`
<!-- GENERATED:END -->

## When to use this skill
- User has Spark SSH access (`dgx-spark-connect-to-your-spark`) and wants a web chat UI, not just CLI
- User already has Ollama and wants to chat through a browser
- User wants a self-hosted ChatGPT-like interface running entirely on their own hardware

## Key decisions
- **Bundled Ollama or separate?** — Open WebUI ships with an integrated Ollama option (single Docker container). Simpler for first-time users. If the user already ran `dgx-spark-ollama` separately, configure Open WebUI to connect to that existing Ollama instead of running two copies.
- **Sync-managed or manual Docker?** — NVIDIA Sync can manage the SSH tunnel + custom-port setup automatically. Manual Docker gives more control but requires the user to handle port forwarding themselves.

## Non-obvious gotchas
- User must be in the `docker` group on the Spark — `docker ps` without sudo must work. If not, add via `sudo usermod -aG docker $USER` and **log out/in to apply** (a new SSH session is not enough — the session must be fully re-established).
- The Open WebUI container stores user accounts and chat history in a named volume. Don't `docker rm -v` the container unless you intend to lose history.
- First-run creates an admin account from whoever signs up first — if the UI is port-forwarded somewhere other users can reach, sign up immediately before anyone else does.

## Related skills
- **Prerequisite**: `dgx-spark-connect-to-your-spark` — SSH + Sync setup
- **Pairs with**: `dgx-spark-ollama` — the most common backend. Open WebUI can bundle its own Ollama, but if `dgx-spark-ollama` was already set up, reuse it (saves disk, one set of models).
- **Alternative UIs**: `dgx-spark-lm-studio` (desktop GUI, not web) · `dgx-spark-live-vlm-webui` (vision-language models specifically)
