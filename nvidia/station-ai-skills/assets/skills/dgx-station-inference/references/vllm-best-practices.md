# vLLM best practices for DGX Station

These are decision rules, not universal flag values. Begin with
`system inspect`, search the pinned sources for the specific concern, and
resolve the exact user-named model. Only a validated recipe may provide
executable arguments.

If `compatibility.capabilities.recipe_execution` is false or no current
runnable recipe exists for the exact model:

- do not recommend a numeric flag value—including zero—from parameter count,
  nominal precision, or raw free-memory arithmetic;
- explain the qualitative tradeoff and the missing evidence instead; and
- cap recipe-fit decisions at the production 252 GiB GB300 limit even when
  development hardware reports more memory.

## Choose a container

“Latest available” and “safe to run for this model” are different states.

1. At use time, inspect the current official NGC vLLM tags, NVIDIA release
   notes, and framework support matrix. Record the observation date.
2. Compare the NVIDIA container's vLLM and CUDA versions, driver requirement,
   architecture support, model features, and known issues with the current
   upstream vLLM release and model documentation.
3. Treat both images as candidates:
   - NVIDIA NGC provides a monthly NVIDIA-integrated stack with documented
     CUDA and library versions, NVIDIA testing, and NVIDIA-specific known
     issues.
   - `vllm/vllm-openai` is the upstream project's image and follows upstream
     release behavior and packaging.
4. Select neither by reputation or tag recency alone. Use the immutable image
   digest attached to the current physical validation for the exact model and
   Station profile.

Never execute `latest`, another mutable tag, or a tag copied from a guide.
Never turn a newly discovered image into a runnable recipe without
qualification evidence.

Official sources:

- https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/vllm/-/tags
- https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/
- https://docs.nvidia.com/deeplearning/frameworks/support-matrix/
- https://docs.vllm.ai/en/latest/deployment/docker/

## Establish the workload contract

Record these before tuning:

- exact model and revision;
- input sequence length (ISL) distribution, not only its maximum;
- generated output-length distribution;
- target concurrency and arrival pattern;
- TTFT, inter-token latency, end-to-end latency, and throughput objectives;
- repeated-prefix rate and prefix sizes;
- tool use, structured output, and maximum supported context requirements;
- other GPU clients and the observed free HBM at startup.

Change one decision at a time and compare a warm, repeatable load sweep. Keep
the raw configuration, model/image digests, startup logs, metrics, and result.
Do not publish a percentage improvement without that evidence.

## `gpu_memory_utilization`

This is a per-instance model-executor memory budget that influences the
automatically sized KV cache. It is not a target utilization metric and it
does not reserve memory against another process.

- Start with the value in the validated recipe.
- Preserve headroom for weights, activations, CUDA graphs, compilation,
  allocator variation, and any co-tenant.
- Raising it can increase KV capacity and reduce recompute preemptions, but
  reduces safety headroom and can cause startup or runtime OOM.
- Lowering `max_num_seqs` or `max_num_batched_tokens` reduces cache pressure
  when preemption occurs.
- An explicit `kv_cache_memory_bytes` overrides the utilization-derived KV
  cache size and must be revalidated after any input changes.

Inspect startup memory/KV logs, OOMs, preemption counts, KV-cache use, and
latency under the real load. Do not blindly set the fraction to its maximum.

## HBM and offloading

Separate model-weight offload from KV-cache offload.

- CPU weight offload expands model capacity by keeping part of the weights in
  CPU memory. The offloaded part is accessed or transferred during every
  forward pass. UVA requires a fast CPU-GPU link; asynchronous prefetch may
  hide some transfer time but consumes GPU memory.
- KV-cache offload moves cache capacity rather than model weights and has its
  own backend and buffer controls.
- DGX Station's coherent CPU memory makes these mechanisms possible, but GPU
  access to CPU memory is not cached. The Development Guide still prefers
  GB300 HBM for performance-critical data.

Keep as much hot weight and KV state in HBM as the validated headroom allows.
Offload only the minimum needed to fit, then measure TTFT, inter-token
latency, throughput, and power/thermal state. Do not invent a performance
penalty; it varies with model, offloaded amount, access pattern, and backend.

## KV cache and concurrency

KV capacity remains after weights, activations, graphs, runtime overhead, and
headroom. At startup, vLLM reports total GPU KV-cache token capacity and an
estimated concurrency at `max_model_len`.

That estimate is not a service-level guarantee. Real demand depends on model
cache geometry, ISL plus generated tokens, scheduler behavior, and active
sequence count. A workload with shorter sequences can admit more concurrent
requests than a maximum-context estimate; long inputs and outputs consume
more cache and can trigger preemption.

Tune these together:

- `max_model_len`: the supported service limit, not an aspirational maximum;
- `max_num_seqs`: an admission/scheduler ceiling, not extra KV capacity;
- `max_num_batched_tokens`: the prefill/decode token budget;
- KV-cache allocation or dtype only when supported by the exact model and
  validation.

Run a concurrency sweep over representative ISL/output buckets. Stop
increasing load when SLOs fail, preemptions grow, OOM occurs, or the owned
service becomes degraded.

## Prefix caching

Automatic prefix caching reuses KV blocks only when requests share an
identical prefix. It can reduce repeated prefill work for long documents and
multi-turn histories. It does not accelerate decoding and provides little
benefit when outputs dominate or prefixes do not repeat.

Compare cache-hit/query metrics and prefill latency with the representative
prefix distribution. Account for cache capacity retained by reusable blocks;
do not enable it solely because the switch exists.

## Chunked prefill and batching

vLLM V1 enables chunked prefill whenever possible. It mixes compute-bound
prefill work with memory-bound decode work and prioritizes decodes.

- Smaller batch-token budgets can favor inter-token latency.
- Larger budgets can favor prompt processing, TTFT, or throughput.
- Upstream example values are experiment points, not DGX Station defaults.

Sweep the batch-token budget together with concurrency and representative
ISL/output lengths. Report TTFT, inter-token latency, throughput, preemption,
KV-cache use, errors, and memory headroom.

Pinned semantic source: upstream vLLM `v0.22.1`, commit
`0decac0d96c42b49572498019f0a0e3600f50398`, which matches the vLLM version
listed in NVIDIA container release 26.06. Recheck current releases before
answering a question about “latest.”
