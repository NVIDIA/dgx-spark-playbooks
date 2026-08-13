# Nemotron Switchyard

> Co-locate Nemotron and Qwen on DGX Spark, then route each request to the appropriate model

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook builds on the [Sparky](https://github.com/hsrakri/sparky) dual-model recipe for a single NVIDIA DGX Spark. Sparky co-locates two OpenAI-compatible vLLM servers:

- `nvidia/Qwen3.6-35B-A3B-NVFP4` on port `8000` for coding work
- `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` on port `8001` for orchestration and general reasoning

Sparky originally wires these models into fixed orchestrator and delegated-coder roles. This playbook adds [NVIDIA NeMo Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) in front of both endpoints. Clients send every request to one model ID, `sparky`, and Switchyard classifies the request before selecting a backend:

- Implementation, debugging, tests, code review, and refactoring route to Qwen.
- Planning, decomposition, architecture, coordination, and general reasoning route to Nemotron.
- Invalid or unavailable classifier output falls back to Nemotron.

Switchyard selects the model but does not start, stop, or monitor the vLLM servers. Sparky remains responsible for the two-model deployment and its required startup order.

> [!NOTE]
> NeMo Switchyard 0.2.0 is pre-alpha software. Its APIs and configuration may change before a stable release. This playbook pins version `0.2.0`.

## What you'll accomplish

You will:

1. Start the Sparky Qwen and Nemotron servers sequentially on one DGX Spark.
2. Install the pinned Switchyard standalone server.
3. Validate a custom classifier route for the two local models.
4. Expose one OpenAI-compatible endpoint on port `4000` using the model ID `sparky`.
5. Verify which backend served coding and planning requests.

## What to know before starting

- Working with Docker and GPU-enabled containers
- Running terminal commands on Linux
- Using OpenAI-compatible chat-completion APIs
- Understanding that classifier routing adds one model call before the selected model call

## Prerequisites

- One NVIDIA DGX Spark or GB10 partner system
- Docker with NVIDIA Container Runtime
- Git, `curl`, and Rust with Cargo
- Access to GitHub, Hugging Face, and the vLLM container registry
- Enough disk space for two NVFP4 model checkpoints and the vLLM image
- Ports `8000`, `8001`, and `4000` available on the host

## Ancillary files

The Switchyard deployment is defined in [`assets/nemotron-switchyard-routes.toml`](assets/nemotron-switchyard-routes.toml).

## Time & risk

- **Estimated time:** About 20-40 minutes after the model weights and vLLM image are cached. First-time model downloads can take substantially longer.
- **Risks:** The two vLLM processes use most of DGX Spark's unified memory. Starting them concurrently can make the second server fail. Classifier routing adds latency and may occasionally choose an unexpected model.
- **Rollback:** Stop Switchyard, remove the two Docker containers, and remove downloaded model files or images if disk space must be recovered.
- **Last Updated:** 08/13/2026
  - Initial publication using Sparky and NeMo Switchyard 0.2.0
  - Added Rust and Cargo installation troubleshooting for Ubuntu 24.04
  - Reduced the dual-model memory baseline and added fail-fast startup checks

## Instructions

## Step 1. Validate the host

Verify the architecture, GPU, container runtime, required tools, free disk space, and ports:

```bash
uname -m
nvidia-smi
docker --version
nvidia-ctk --version
git --version
curl --version
rustc --version
cargo --version
openssl version
free -h
df -h .

ss -ltn '( sport = :4000 or sport = :8000 or sport = :8001 )'
```

Expected results:

- `uname -m` returns `aarch64`.
- Docker can access the GB10 GPU.
- Rust and Cargo are installed.
- `free -h` shows enough available unified memory to start both models.
- The port check returns no listeners.

If `rustc` and `cargo` are the only missing prerequisites on Ubuntu 24.04, install `rustup` and the stable Rust toolchain:

```bash
sudo apt update
sudo apt install -y rustup
rustup default stable

rustc --version
cargo --version
```

If the installation succeeds but either command is still not found, start a new shell or add Cargo's binary directory to the current shell, then verify again:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
hash -r
rustc --version
cargo --version
```

Do not continue until both version commands succeed. `cargo install` in Step 6 requires this toolchain.

Validate Docker GPU access:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
```

## Step 2. Clone the playbook and Sparky repositories

Clone both repositories into the same parent directory:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks.git
git clone https://github.com/hsrakri/sparky.git
git -C sparky checkout 6391f400cb2dc53c795f4ea8b1e4ba2a92e52604
```

If either repository already exists, reuse that checkout instead of cloning it again. The Sparky commit is pinned to the version reviewed for this playbook.

## Step 3. Set the local API key

Create one key that vLLM requires for both local endpoints and that Switchyard uses for upstream authentication:

```bash
export SPARKY_API_KEY="$(openssl rand -hex 32)"
```

Keep this value in the same host shell used to start Switchyard. Do not commit it to either repository.

## Step 4. Start the Qwen coder first

If you are restarting after a failed attempt, remove both playbook containers so the new memory settings take effect:

```bash
docker rm -f nemotron-vllm qwen36-vllm 2>/dev/null || true
```

Sparky requires the coder to become ready before Nemotron starts. Launch Qwen on port `8000`:

```bash
docker run -d --name qwen36-vllm --restart no \
  --gpus all --ipc=host -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:v0.26.0-aarch64 \
  nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --quantization modelopt \
    --attention-backend flashinfer \
    --moe-backend marlin \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --gpu-memory-utilization 0.35 \
    --max-model-len 32768 \
    --api-key "$SPARKY_API_KEY" \
    --host 0.0.0.0 --port 8000
```

Wait until the endpoint is ready:

```bash
until curl -fsS \
  -H "Authorization: Bearer $SPARKY_API_KEY" \
  http://127.0.0.1:8000/v1/models >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' qwen36-vllm 2>/dev/null)" != "true" ]; then
    docker logs --tail 100 qwen36-vllm
    break
  fi
  sleep 5
done

curl -fsS \
  -H "Authorization: Bearer $SPARKY_API_KEY" \
  http://127.0.0.1:8000/v1/models >/dev/null
```

Do not continue until the final `curl` command exits successfully.

## Step 5. Start the Nemotron orchestrator second

After Qwen is ready, launch Nemotron on port `8001`:

```bash
docker run -d --name nemotron-vllm --restart no \
  --gpus all --ipc=host -p 8001:8001 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:v0.26.0-aarch64 \
  nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --moe-backend marlin \
    --kv-cache-dtype fp8 \
    --mamba-backend flashinfer \
    --enforce-eager \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.35 \
    --max-model-len 32768 \
    --api-key "$SPARKY_API_KEY" \
    --host 0.0.0.0 --port 8001
```

Wait until both models are ready:

```bash
until curl -fsS \
  -H "Authorization: Bearer $SPARKY_API_KEY" \
  http://127.0.0.1:8001/v1/models >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' nemotron-vllm 2>/dev/null)" != "true" ]; then
    docker logs --tail 100 nemotron-vllm
    break
  fi
  sleep 5
done

curl -fsS \
  -H "Authorization: Bearer $SPARKY_API_KEY" \
  http://127.0.0.1:8001/v1/models >/dev/null

curl -fsS -H "Authorization: Bearer $SPARKY_API_KEY" \
  http://127.0.0.1:8000/v1/models
curl -fsS -H "Authorization: Bearer $SPARKY_API_KEY" \
  http://127.0.0.1:8001/v1/models
```

Cold starts can take several minutes per model. In another terminal, use `docker logs --follow qwen36-vllm` or `docker logs --follow nemotron-vllm` to watch progress. The readiness loops stop and print the last 100 log lines if a container exits.

## Step 6. Install NeMo Switchyard

Install the standalone server pinned to version `0.2.0`:

```bash
cargo install --locked switchyard-server --version 0.2.0
switchyard-server --version
```

The build installs the binary under `~/.cargo/bin` by default. If the command is not found after installation:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## Step 7. Validate and start the route

From the parent directory containing both cloned repositories, set the route path:

```bash
export NEMOTRON_SWITCHYARD_ROUTES="$PWD/dgx-spark-playbooks/nvidia/nemotron-switchyard/assets/nemotron-switchyard-routes.toml"
```

Validate the TOML configuration and environment without starting the server:

```bash
switchyard-server --config "$NEMOTRON_SWITCHYARD_ROUTES" --dry-run
```

Create the local state directory, then start Switchyard on the loopback interface:

```bash
mkdir -p "$HOME/.local/state"

RUST_LOG=switchyard_server=info,libsy=info \
switchyard-server \
  --config "$NEMOTRON_SWITCHYARD_ROUTES" \
  --host 127.0.0.1 \
  --port 4000 \
  --routing-log-file "$HOME/.local/state/nemotron-switchyard-routing.jsonl"
```

Keep this terminal open. Switchyard reads `SPARKY_API_KEY` when it starts.

## Step 8. Verify routing

Open a second terminal, then confirm Switchyard exposes the `sparky` route:

```bash
curl -fsS http://127.0.0.1:4000/health
curl -fsS http://127.0.0.1:4000/v1/models
```

Send a coding request:

```bash
curl -i http://127.0.0.1:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sparky",
    "messages": [
      {"role": "user", "content": "Implement a Python function that validates IPv4 addresses and include tests."}
    ]
  }'
```

Send a planning request:

```bash
curl -i http://127.0.0.1:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sparky",
    "messages": [
      {"role": "user", "content": "Create an architecture plan for migrating a monolith to services, including risks and sequencing."}
    ]
  }'
```

Inspect the `x-model-router-selected-model` and `x-model-router-rationale` response headers. Routing is model-dependent, so treat the example selections as expected behavior rather than a deterministic assertion. Use statistics and the routing log to review aggregate decisions:

```bash
curl -fsS http://127.0.0.1:4000/v1/stats
tail -n 20 "$HOME/.local/state/nemotron-switchyard-routing.jsonl"
```

Any OpenAI Chat Completions client can now use:

- Base URL: `http://127.0.0.1:4000/v1`
- Model: `sparky`
- API key: any non-empty placeholder if the client requires one; Switchyard does not require inbound authentication in this local configuration

## Step 9. Tune the routing policy

Edit `nemotron-switchyard-routes.toml` only after collecting representative routing decisions:

- Adjust the classifier prompt when requests repeatedly select the wrong specialization.
- Keep `default_target = "orchestrator"` so classifier failures use the broader model.
- Keep Qwen thinking disabled for classifier calls so Switchyard receives schema-valid JSON in normal assistant content.
- Increase `recent_turn_window` if follow-up requests need more conversation context.

Restart Switchyard after changing the route file, then run `--dry-run` again before serving traffic.

## Step 10. Cleanup and rollback

Stop Switchyard with `Ctrl+C`, then remove the model containers:

```bash
docker stop nemotron-vllm qwen36-vllm
docker rm nemotron-vllm qwen36-vllm
```

Optionally remove the Switchyard binary:

```bash
cargo uninstall switchyard-server
```

The Sparky repository includes [`scripts/vllm-stack-up.sh`](https://github.com/hsrakri/sparky/blob/main/scripts/vllm-stack-up.sh) for deterministic reboot ordering after the containers have been created. Do not enable independent Docker auto-restarts for these two containers because they can race for unified memory.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Nemotron exits with `CUDA error: out of memory` during `MemorySnapshot` | Qwen's allocation leaves too little unified-memory headroom for the second CUDA context | Remove both containers, confirm available memory with `free -h`, then recreate them sequentially with the documented `0.35 + 0.35` split and 32K context limits. |
| Second model reports no available cache-block memory | Both models started concurrently or their memory fractions are too high | Remove both containers, start Qwen first, wait for readiness, then start Nemotron. Reduce both memory fractions further if other host workloads consume substantial unified memory. |
| Nemotron hangs before loading weights | CUDA graph or compile deadlock under co-location | Keep `--enforce-eager` on the Nemotron command. |
| Nemotron reports `block_size (4176)` greater than `max_num_batched_tokens` | Incompatible Mamba cache alignment option | Do not add `--mamba-cache-mode align`. |
| `rustc` and `cargo` are not found during host validation | The Rust toolchain is not installed | On Ubuntu 24.04, install `rustup` with APT, run `rustup default stable`, and verify both commands before continuing. If they remain unavailable, start a new shell or add `$HOME/.cargo/bin` to `PATH`. |
| Switchyard dry-run reports a missing environment variable | `SPARKY_API_KEY` is not exported in the current shell | Export the same key used in both vLLM launch commands and rerun `--dry-run`. |
| Switchyard receives `401 Unauthorized` from a backend | The vLLM key and `SPARKY_API_KEY` differ | Restart the affected vLLM container and Switchyard with the same key. |
| Classifier always falls back to Nemotron | Qwen returned invalid or incomplete structured output | Confirm Qwen is healthy, keep classifier thinking disabled, and inspect Switchyard debug logs. |
| Coding or planning requests select the wrong model | The classification prompt does not match the workload | Review routing logs, refine the prompt with representative examples, and revalidate. |
| Client cannot connect to port 4000 | Switchyard is stopped, bound elsewhere, or the client is remote | Check `/health`. Keep loopback binding for local use; use an SSH tunnel rather than exposing the unauthenticated endpoint. |

## References

- [Sparky](https://github.com/hsrakri/sparky)
- [NeMo Switchyard](https://github.com/NVIDIA-NeMo/Switchyard)
- [Install Rust with rustup](https://rust-lang.github.io/rustup/installation/)
- [Switchyard 0.2.0 release](https://github.com/NVIDIA-NeMo/Switchyard/releases/tag/v0.2.0)
- [Switchyard classifier routing](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.2.0/docs/routing_algorithms/llm_classifier_routing.md)
- [Switchyard server configuration](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.2.0/docs/reference/toml_schema.md)

## License

Sparky is licensed under the MIT License. NeMo Switchyard is licensed under Apache License 2.0. Model weights and container images remain governed by their respective upstream licenses and terms.
