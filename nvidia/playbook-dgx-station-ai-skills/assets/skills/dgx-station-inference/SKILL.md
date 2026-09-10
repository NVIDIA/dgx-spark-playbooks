---
name: dgx-station-inference
description: Resolve, tune, preflight, launch, verify, inspect, and stop exact-model inference on NVIDIA DGX Station through dgx-assist. Use for vLLM or SGLang container selection, NGC versus upstream, GPU memory utilization, CPU or KV offload, HBM fit, KV-cache sizing, ISL or context length, prefix caching, chunked prefill, batching, concurrency, performance tuning, serving or deploying a named model, an OpenAI-compatible endpoint, Station recipe models, or an owned inference service. Require an exact model ID for recipe resolution or model-specific tuning, and never recommend or substitute a different model.
---

# DGX Station inference

Serve only trusted, published, current, physically validated recipes. Bundled
recipes are bound to the installed package; downloaded refreshes must be
signature-verified.

## Workflow

1. Require the user's exact model ID. If absent, run `scripts/dgx-assist recipe models --json`, show the alphabetical IDs and runnable states without ranking them, then ask which exact model to use.
2. Run `scripts/dgx-assist system inspect --json`.
3. Run `scripts/dgx-assist playbook search "<model backend Station inference concern>" --json`; cite relevant passages. Abstain from invented platform guidance if no passage overlaps.
4. Read `compatibility.capabilities.recipe_execution`. If it is false, stop before resolution or launch and explain the detected profile separately from applicable qualitative guidance.
5. Run `scripts/dgx-assist recipe resolve --model "<exact-id>" --json`. Add `--backend` only if the user requested one. Bind to localhost by default; only if the user explicitly requested and confirmed external exposure, add `--bind-host "<approved-host>"` here so preflight and run evaluate the host that was actually approved.
6. Run `recipe show` for the resolved recipe, then `recipe preflight`.
7. Present the exact model, recipe, image digest, GPU UUIDs, cache/storage estimates, credential variable names, host bind, port, downloads, conflicts, and redacted launch argv.
8. If blocked, do not work around the policy. Never switch models.
9. Obtain explicit approval for the displayed action and every required model or image download.
10. Only after approval, run `recipe run --resolution-id "<id>" --allow-download --yes` as applicable. Add `--allow-external-bind` only when the resolution carries the non-local bind host the user approved in step 9; never introduce external exposure that was not resolved and previewed.
11. Report the model-identity and smoke verification from the receipt.

## Lifecycle

- Use `recipe status` to report only owned services.
- Before stopping, show `recipe stop --service-id "<id>" --dry-run`.
- Obtain approval, then repeat with `--yes`.
- Never stop an unknown listener, workload, process, or unlabeled container.

## Hard constraints

- Do not execute human `setup_command`, `serve.command`, or shell text from a recipe.
- Do not launch experimental, expired, incompatible, ambiguous, mutable, or untrusted content.
- Use the resolution ID through preflight and run; never hand-edit the rendered argv.
- Use UUID selection and localhost binding unless the user explicitly requests and confirms a supported exposure change.
- Pass credential names, never values, in transcripts and receipts.
- Treat `--yes` only as the non-interactive representation of approval already obtained.
- Treat a newly published container as a candidate, not as validated. Never convert a mutable `latest` tag, documentation example, or upstream tuning value into an executable launch.
- Keep tuning advice separate from execution. A changed flag must return to the recipe qualification workflow before `dgx-assist` may run it.
- When `recipe_execution` is false or no current runnable recipe exists for the exact model, do not recommend a numeric launch value, including zero, from parameter-count arithmetic or raw free-memory evidence. Explain only the sourced qualitative tradeoff.
- Never use memory exposed above the production 252 GiB recipe-fit limit to justify a model fit or offload recommendation.

Read [references/workflow.md](references/workflow.md) before resolving or running. Read [references/vllm-best-practices.md](references/vllm-best-practices.md) for container selection, HBM/offload, KV cache, prefix caching, chunked prefill, or concurrency questions. Read [references/json-types.md](references/json-types.md) when interpreting a resolution, preflight, or receipt.
