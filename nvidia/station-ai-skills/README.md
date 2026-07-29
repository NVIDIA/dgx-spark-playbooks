# DGX Station AI Skills and dgx-assist

> Inspect DGX Station software and route version-aware, CLI-backed workflows

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

DGX Station AI Skills teaches your AI coding agent how to operate a DGX Station
correctly. It installs four native Agent Skills plus a dependency-free
`dgx-assist` command-line tool into a project you choose, so that asking your
agent "serve this model" or "why is this GPU unavailable?" produces answers
grounded in your actual hardware instead of recalled generic advice.

The skills route the agent through a fixed workflow: inspect the real Station,
search a pinned snapshot of NVIDIA guidance, resolve an exact named model to a
qualified recipe, run preflight, ask you to approve, then act and verify. Every
mutation requires your confirmation, and `dgx-assist` only ever stops services
it started and labelled as its own.

The same CLI works without an agent. Add `--json` and it emits a stable
versioned envelope for scripts and CI; omit it and you get readable headings,
tables, and action previews for terminal use.

## What you'll accomplish

You'll install the four DGX Station skills and the `dgx-assist` CLI into a
project, then drive a qualified vLLM inference workload from a plain-language
request through preflight, approval, launch, and verification.

You'll also be able to:

- Inspect your Station's software profile and see exactly which actions its
  release qualifies it for.
- Search pinned NVIDIA Development Guide and Bring-Up Guide content with
  revision and source-digest provenance on every result.
- Plan MIG layouts and run read-only diagnostics that produce redacted support
  bundles.

## What to know before starting

- Experience with the Linux command line and running shell scripts
- Familiarity with an AI coding agent — Claude Code, Codex, Gemini CLI, or
  Cursor — and how it loads project-level context
- Basic understanding of Docker containers and GPU device selection
- Familiarity with vLLM or SGLang inference serving (helpful but not required)

Two safety rules matter more than the rest, and the skills enforce them for
you: select GPUs by UUID rather than by `nvidia-smi` index, which is not a CUDA
ordinal on this platform; and treat an unrecognized software build as unknown
rather than assuming it behaves like a qualified one.

## Prerequisites

**Hardware Requirements:**

- NVIDIA Grace Blackwell GB300 Ultra Superchip System (DGX Station)
- GB300 compute capability `10.3`, confirmed by `nvidia-smi --query-gpu=compute_cap --format=csv`
- At least 20GB available storage space for the qualified model weights and
  container image

**Software Requirements:**

- Python 3.11 or newer: `python3 --version`
- Docker with the NVIDIA Container Toolkit: `docker info | grep -i runtime`
- One supported AI coding agent installed: `claude --version`, `codex --version`,
  `gemini --version`, or Cursor
- Network access to download model weights and container images on first run
- No inbound port access is required; inference binds to localhost by default

## Ancillary files

All required assets can be found [in the DGX Station AI Skills playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/station-ai-skills/).

- `assets/install.sh` — Thin wrapper that runs the Python installer
- `assets/installer.py` — Installs, updates, migrates, and uninstalls the
  skills; writes `.dgx-station/install-manifest.json` and manages the delimited
  NVIDIA block in your agent context file
- `assets/dgx-assist.pyz` — The standalone zipapp CLI providing host
  inspection, guidance search, recipe resolution, preflight, service
  lifecycle, MIG planning, and diagnostics
- `assets/skills/dgx-station/` — Platform compatibility, coherency, GPU
  selection, container, CDI, and power guidance
- `assets/skills/dgx-station-inference/` — Exact-model recipe resolution,
  preflight, approved launch, verification, and owned service lifecycle
- `assets/skills/dgx-station-mig/` — Driver-discovered MIG inspection,
  planning, approved apply, and restoration evidence
- `assets/skills/dgx-station-diagnose/` — Read-only diagnostics, playbook
  correlation, redacted bundles, and one allowlisted fix at a time
- `assets/AGENTS.md` — The managed context block written into your agent's
  project context file

## Time & risk

* **Estimated time:** 15 minutes to install and verify, plus 20–40 minutes if
  you run the qualified inference recipe (most of that is the first model and
  container download)
* **Risk level:** Low
  * The installer refuses unmanaged file collisions and symlink destinations
    rather than overwriting anything it does not own
  * First-run model and container downloads require network bandwidth and may
    fail on a slow or interrupted connection
  * Only one model is qualified to launch in v1; larger bundled candidates are
    reported as non-runnable by design
  * MIG changes are disruptive and are blocked unless your release profile
    enables them and you explicitly approve the plan
* **Rollback:** Run `assets/install.sh uninstall --target /path/to/project` to
  remove every manifest-owned file and the delimited context block. The
  installer preserves any managed file you modified and leaves unrelated
  context untouched. A user-scope CLI installed with `install-cli` is removed
  by deleting `~/.local/bin/dgx-assist`. Cached state lives under
  `~/.cache/dgx-assist/` and `~/.local/state/dgx-assist/` and can be deleted.
* **Last Updated:** 07/28/2026
  * First Publication

## Instructions

## Step 1. Verify your environment

Confirm the Station has the hardware and software the skills expect. Python
3.11 or newer is required by the installer.

```bash
python3 --version
nvidia-smi --query-gpu=name,compute_cap,uuid --format=csv
docker info --format '{{.ServerVersion}}'
```

Expected output should show Python 3.11 or newer, a GB300 GPU reporting
compute capability `10.3`, and a running Docker daemon. Note the GPU UUIDs
rather than the row order — the `nvidia-smi` index is not a CUDA ordinal on
this platform, and the skills always select GPUs by UUID.

## Step 2. Clone the playbook

Clone the playbook repository so the installer and bundled skills are available
locally.

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/station-ai-skills
```

Everything the installer needs lives under `assets/`: the installer itself, the
`dgx-assist.pyz` CLI, and the four skill directories.

## Step 3. Preview the installation

Preview the exact changes before writing anything. The target is the project
that should receive the skills — not this playbook directory.

```bash
assets/install.sh install \
  --harness codex \
  --target /path/to/project \
  --dry-run
```

Expected output should show a `WOULD WRITE` line for each skill file, one for
`.dgx-station/bin/dgx-assist`, a diff of the managed NVIDIA block that will be
added to your context file, and a closing `DRY-RUN: no files changed`.

Choose the `--harness` value that matches your agent:

| Harness | Skill directory | Context file |
|---|---|---|
| `claude` | `.claude/skills/` | `CLAUDE.md` |
| `codex` | `.agents/skills/` | `AGENTS.md` |
| `gemini` | `.gemini/skills/` | `GEMINI.md` |
| `cursor` | `.cursor/skills/` | `AGENTS.md` |
| `all` | every supported harness | managed blocks as applicable |

Each harness receives complete native skill directories with their references,
scripts, and UI metadata — never a lossy transformed command or rule.

## Step 4. Install the skills and CLI

If the preview is correct, run the same command without `--dry-run`.

```bash
assets/install.sh install \
  --harness codex \
  --target /path/to/project
```

The installer backs up an existing context file before its first managed edit,
refuses unmanaged skill collisions and symlink destinations, and records every
installed hash in `.dgx-station/install-manifest.json`.

To install only the CLI for the current user, when a project-local
installation is not appropriate:

```bash
assets/install.sh install-cli --scope user
```

Restart your AI coding agent in the target project so it picks up the new
skills and context block.

## Step 5. Verify the installation

Confirm the CLI is present and the bundled content is intact.

```bash
cd /path/to/project
.dgx-station/bin/dgx-assist version
.dgx-station/bin/dgx-assist catalog status
.dgx-station/bin/dgx-assist playbook status
```

Expected output should show the CLI version, a valid bundled catalog, and a
healthy playbook search index. You can check the installation itself at any
time:

```bash
assets/install.sh status --target /path/to/project
```

## Step 6. Confirm your platform support profile

The skills refuse to guess what your Station can do. Inspect it and read the
resolved profile.

```bash
.dgx-station/bin/dgx-assist system inspect
```

Expected output should show an exact compatibility profile and per-feature
capabilities. The trusted release marker selects one of three profiles:

| Profile | Identity | Behavior |
|---|---|---|
| Software 1.0 | `7.4.1` or `7.4.1-GB300ws`; build `2026-02-20-05-22-42` | Guidance, diagnostics, read-only MIG inspection, and explicitly qualified recipes |
| Software 2.0 | `7.5.0`; build `2026-06-16-11-48-10` | Capability-scoped qualified workflows |
| Unknown | Any other exact identity | General read-only evidence only |

A recognized Software 2.0 profile requires this base identity and hardware
evidence, and the release marker must also pass ownership, file-type, symlink,
and mode checks:

| Field | Required value |
|---|---|
| `DGX_SWBUILD_VERSION` | `7.5.0` |
| `DGX_SWBUILD_DATE` | `2026-06-16-11-48-10` |
| `DGX_PRETTY_NAME` | `NVIDIA DGX GB300WS` |
| GB300 compute capability | `10.3` |

Software 1.0 does not inherit Software 2.0 CDMM, ordering-service, or `vsloshd`
expectations. It permits only recipes explicitly validated for its exact
profile; MIG mutation and platform fixes remain blocked. A different build is
reported as unknown rather than assumed compatible.

## Step 7. Ask your agent for a DGX Station task

This is the normal way to use the playbook. Open your agent in the target
project and make a plain-language request:

```text
Inspect this DGX Station and explain its compatibility profile and restrictions.
```

```text
Serve Qwen/Qwen2.5-Coder-1.5B-Instruct with vLLM. Show the preflight and wait
for my approval before starting anything.
```

The activated skill runs `dgx-assist --json`, interprets the evidence, carries
resolution, report, and plan IDs between commands, and presents the result and
approval boundary to you. You never need to read or copy raw JSON in this mode.

The remaining steps show the equivalent direct CLI commands, which are useful
for terminal work and for understanding what the agent is doing on your behalf.

## Step 8. Search the pinned NVIDIA guidance

Guidance comes from a bundled multi-source snapshot, not model memory.

```bash
.dgx-station/bin/dgx-assist playbook search "mixed coherency containers"
.dgx-station/bin/dgx-assist playbook search "CPU weight offload HBM forward pass"
.dgx-station/bin/dgx-assist playbook search "ISL KV cache maximum concurrency"
.dgx-station/bin/dgx-assist playbook show RESULT_ID
```

Every retrieved record carries its repository revision, source-file SHA-256,
heading, lines, role, and authority class. The snapshot pins the DGX Station
Development Guide at `76a1f6adf1a740699c2efff201377947d90f7fd8`, the GB300
Bring-Up Guide at `2f2d22b2fee4b6a2964045a97b786b86b366b65b`, upstream vLLM
`v0.22.1` at `0decac0d96c42b49572498019f0a0e3600f50398` matching NVIDIA vLLM
container 26.06, and the NVIDIA vLLM release notes.

Where sources conflict, the current Development Guide overrides older bring-up
statements about mixed GPU contexts and device indices; those passages and any
credential examples are excluded from retrieval. Retrieval abstains when query
terms do not overlap the pinned content — do not fill an abstention with
remembered platform commands.

## Step 9. Run the qualified inference recipe

Resolve an exact model ID to a recipe, inspect it, run preflight, then preview
the launch.

```bash
.dgx-station/bin/dgx-assist recipe models
.dgx-station/bin/dgx-assist recipe resolve --model Qwen/Qwen2.5-Coder-1.5B-Instruct
.dgx-station/bin/dgx-assist recipe show --recipe-id RECIPE_ID
.dgx-station/bin/dgx-assist recipe preflight --resolution-id RESOLUTION_ID
.dgx-station/bin/dgx-assist recipe run --resolution-id RESOLUTION_ID --dry-run
```

Copy the labeled recipe and resolution IDs from one command to the next; in
agent use the skill does this for you. Without `--dry-run` or `--yes`, an
interactive terminal shows the action preview and asks for confirmation. A
non-interactive caller repeats the command with `--yes` only after showing that
preview and obtaining approval. Add `--allow-download` only after every
required model and image download is disclosed and approved. If you set a
non-local `--bind-host` on `recipe resolve`, the matching `recipe run` also
requires an explicit `--allow-external-bind`.

The bundled v1 catalog contains one published Software 1.0 smoke recipe —
`Qwen/Qwen2.5-Coder-1.5B-Instruct` on vLLM — bound to its exact model revision,
immutable NGC image digest, backend version, release profile, and checksummed
qualification evidence. This is a functional smoke claim, not a performance
benchmark.

These candidates are bundled but deliberately non-runnable, and will refuse to
resolve:

- `nvidia/nemotron-3.5-nano` on vLLM
- `qwen/qwen3.6-27b` on vLLM
- `nvidia/nemotron-3-super-120b-a12b` on vLLM
- `qwen/qwen3.6-27b` on SGLang

Verify and manage a running service, then stop it when finished:

```bash
.dgx-station/bin/dgx-assist recipe status
.dgx-station/bin/dgx-assist recipe stop --service-id SERVICE_ID
```

`recipe stop` revalidates ownership labels and sends SIGTERM only. It never
force-kills, and it only ever stops resources recorded as owned by
`dgx-assist`.

## Step 10. Plan a MIG layout

Inspection is available on recognized Software 1.0 and Software 2.0 profiles.
Planning and apply additionally require `mig_mutation=true` and your approval.

```bash
.dgx-station/bin/dgx-assist mig inspect
.dgx-station/bin/dgx-assist mig profiles
.dgx-station/bin/dgx-assist mig plan --layout "DRIVER_PROFILE_IDS_OR_NAMES"
.dgx-station/bin/dgx-assist mig apply --plan-id PLAN_ID --dry-run
```

Expected output should show driver-discovered profiles, the disruption and
restoration information for the plan, and — under `--dry-run` — no change to
the GPUs. `dgx-assist` never stops active GPU clients to apply a layout.

## Step 11. Run diagnostics

Diagnosis is read-only. Each allowlisted fix is separately previewed,
confirmed, verified, and recorded.

```bash
.dgx-station/bin/dgx-assist diagnose run
.dgx-station/bin/dgx-assist diagnose bundle --report-id REPORT_ID
.dgx-station/bin/dgx-assist diagnose fix --report-id REPORT_ID --finding FINDING_ID --dry-run
```

Expected output should show findings correlated to pinned playbook content, and
a redacted support bundle path. Bundles and receipts persist redacted argv and
credential variable names only, never secret values.

## Step 12. Cleanup

Remove the skills, the CLI, and the managed context block from a project.

> [!WARNING]
> This deletes every file recorded in `.dgx-station/install-manifest.json` and
> removes the delimited NVIDIA block from your context file. Managed files you
> modified are preserved, and unrelated context is left untouched.

```bash
assets/install.sh uninstall --target /path/to/project --dry-run
assets/install.sh uninstall --target /path/to/project
```

To remove a user-scope CLI and the local caches as well:

```bash
rm -f ~/.local/bin/dgx-assist
rm -rf ~/.cache/dgx-assist ~/.local/state/dgx-assist
```

## Step 13. Next steps

Keep an installation current, or migrate one made by an older release:

```bash
assets/install.sh update --target /path/to/project --dry-run
assets/install.sh update --target /path/to/project
assets/install.sh migrate --target /path/to/project --dry-run
assets/install.sh migrate --target /path/to/project
```

Migration removes a released legacy skill only when its exact artifact hash is
known; modified legacy skills remain with a warning. Update and uninstall
operate only on manifest-owned files and delimited context blocks.

1. **Automate with the JSON envelope.** Every command accepts `--json`
   anywhere in its arguments and returns `schema_version`, `command`, `ok`,
   `data`, `warnings`, and `provenance`. Errors use the same envelope with an
   `error` object. Never parse the human display in automation.

   ```bash
   .dgx-station/bin/dgx-assist system inspect --json |
     jq '.data.compatibility | {profile_id, support_level, capabilities}'
   ```

2. **Branch on stable exit classes.** Inspect both the process exit code and
   the JSON `error.code`.

   | Exit code | Meaning |
   |---:|---|
   | `0` | Success |
   | `2` | Invalid command, configuration, or input |
   | `3` | Unsupported platform or operation |
   | `4` | No eligible exact recipe |
   | `5` | Safety, approval, staleness, conflict, or policy block |
   | `6` | Authorized action or internal operation failed |

3. **Relocate configuration and state.** Unless XDG environment variables
   override them, configuration lives at `~/.config/dgx-assist/config.json`,
   caches at `~/.cache/dgx-assist/`, and resolutions, diagnostics, service
   ownership, receipts, and MIG plans at `~/.local/state/dgx-assist/`.

4. **Point at a live content endpoint.** The public package uses its bundled
   snapshot and is entirely offline by default. An internal pilot can supply
   endpoint configuration in the XDG config file or via `--config PATH`:

   ```json
   {
     "catalog": {
       "manifest_url": "https://approved.example/catalog/manifest.json",
       "allowed_hosts": ["approved.example"]
     },
     "playbook": {
       "manifest_url": "https://approved.example/playbook/manifest.json",
       "allowed_hosts": ["approved.example"]
     }
   }
   ```

   Refresh accepts HTTPS from explicit hosts, validates schema, key ID, Ed25519
   signature through OpenSSL, digest, generation time, and expiry, then
   atomically activates the artifact. A failed refresh retains the
   last-known-good content. Use `--offline` to suppress the 24-hour bounded
   refresh attempt, or `catalog refresh --dry-run` and
   `playbook refresh --dry-run` to inspect the configured host and cache impact
   without network access.

5. **Tune inference with sourced guidance.** Name the exact model and describe
   the workload's ISL, generated-output distribution, target concurrency,
   latency and throughput goals, and repeated-prefix rate. The inference skill
   explains NGC versus upstream containers, GPU-memory headroom, weight and KV
   offload, HBM placement, KV-cache sizing, prefix caching, and chunked
   prefill — without turning those recommendations into launch flags. Changed
   parameters stay non-executable until an exact recipe is physically
   validated.

A successful end state is an agent in your project that inspects the real
Station before advising, cites pinned NVIDIA guidance with provenance, and
stops for your approval before every mutation.

## Troubleshooting

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `system inspect` reports restricted or unknown actions | Untrusted or unknown release identity, missing GB300 `10.3` compute capability, or an observed power-budget violation | Run `.dgx-station/bin/dgx-assist system inspect` and read the compatibility profile, capabilities, and restrictions. With `--json`, inspect `data.compatibility` and `data.rejection_reasons`. Do not bypass qualification or rely on product-name strings alone — a different build is not automatically compatible |
| Software 1.0 profile blocks MIG mutation and platform fixes | Expected behavior: the Software 1.0 profile is capability-scoped to guidance, diagnostics, read-only MIG inspection, and explicitly validated recipes | No action. Qualified recipe execution still works; MIG mutation and platform fixes require a profile that enables them |
| No recipe resolves for your model | The bundled v1 catalog has one runnable Software 1.0 smoke model, `Qwen/Qwen2.5-Coder-1.5B-Instruct`; larger candidates are deliberately non-runnable | Run `.dgx-station/bin/dgx-assist recipe models` and `catalog status` to list exact IDs. Other stable reasons: experimental lifecycle, missing LaunchSpec, mutable model or image references, stale recipe hash, expired evidence, wrong release or hardware, ambiguity, or a hard preflight conflict. No error path substitutes a different model |
| `playbook search` returns nothing, with `retrieval_trace.abstained=true` | Query terms did not overlap the pinned content | Narrow the query using Station-specific terms. Do not invent a command to fill the abstention |
| `playbook status` reports a missing or corrupt index | Damaged bundled search index | Run `.dgx-station/bin/dgx-assist diagnose run`, then `diagnose fix --report-id REPORT --finding content.playbook.index --dry-run`. Approve the fix before repeating it with `--yes` |
| Mixed-coherency GPU ordering looks wrong in a container | Containers do not automatically inherit host device exposure or ordinals from `/etc/mixed-coherency-gpu-select/env` | Inspect `compatibility.capabilities` first and report observed NVML addressing modes and UUIDs. Check the ordering service only when `mixed_coherency_service` is true. Pass explicit UUIDs; never assume `nvidia-smi` index `0` or `1` is the GB300 CUDA ordinal. See the [NVIDIA mixed-coherency guide](https://docs.nvidia.com/dgx/dgx-station-development-guide/coherency.html) |
| Older guidance says one CUDA context cannot use both GPUs | Superseded bring-up passage that predates the current Development Guide | Do not follow it on the qualified R610+ Software 2.0 profile. The signed retrieval snapshot excludes that passage and records the current Development Guide as the superseding source. Software 2.0 can access ATS and HMM devices; do not import that behavior into Software 1.0 |
| A power check fails or `vsloshd` is missing | `dynamic_power_sloshing` is not enabled on this profile, or a real budget violation was observed | When the capability is true, report the `vsloshd` service and mode. On Software 1.0 its absence is not a fault. Report observed caps and violations; never attempt an ad hoc power-cap fix. See the [NVIDIA power-sloshing guide](https://docs.nvidia.com/dgx/dgx-station-development-guide/dynamic-power-sloshing.html) |
| vLLM tuning advice seems too generic | The request lacked an exact model and workload shape | Name the model and give ISL, generated-output distribution, target concurrency, TTFT/inter-token-latency/throughput goals, and repeated-prefix rate, then run `playbook search "gpu_memory_utilization KV cache preemption"` or `playbook search "prefix caching chunked prefill concurrency"`. Do not maximize `gpu_memory_utilization` blindly, treat CPU memory as HBM-equivalent, promise concurrency from ISL alone, or claim a prefix-cache speedup for decode-heavy workloads |
| Preflight reports a port or GPU conflict | Another listener or GPU client owns the resource | Resolve the conflict yourself — `dgx-assist` never stops, kills, or takes over an unknown resource. Create a fresh resolution afterwards if any input changed |
| `recipe stop` leaves the service running | The graceful SIGTERM timeout expired; `dgx-assist` does not implicitly send SIGKILL | The receipt reports a degraded stop and leaves the resource for explicit operator review. Stop it manually after confirming what it is |
| A MIG plan is rejected as stale | Clients, mode, instances, installed profiles, release identity, or the driver changed after planning | Run `mig inspect` and `mig profiles`, then create a new plan. Do not replay or edit the old commands |
| Skills are not discovered by the agent | Incomplete skill folder, wrong directory, or a harness without native skill support | Verify the native path contains the full folder and the exact uppercase filename — `.claude/skills/dgx-station/SKILL.md`, `.agents/skills/…`, `.gemini/skills/…`, or `.cursor/skills/…` — then restart the agent in that project. Upgrade a harness that lacks native skill support rather than transforming the bundle |
| `install.sh status` reports `MODIFIED` files | You edited a managed file after installation | Expected and safe: the installer preserves a modified managed file during update and uninstall. Review it manually, or delete it and re-run `install` to restore the shipped version |
| Install fails with `unmanaged skill collision` or `unmanaged CLI collision` | A same-name file exists that the installer does not own | The installer never overwrites unmanaged files. Move or delete the existing file, then re-run `install` |
| Install fails partway with an OS error | A write failed mid-installation | The installer records a recovery manifest covering the files that landed. Re-run `install` to finish, or `uninstall` to remove them |
| `install-cli --scope user` refuses to write | `~/.local/bin/dgx-assist` already exists or is a symlink | Remove or rename the existing destination, then re-run. The installer refuses to replace a symlink destination or an existing user CLI |
| Context file changes look unexpected | Only the delimited NVIDIA block is managed | Unrelated content is never touched, and the original file is backed up under `.dgx-station/backups/` before the first managed edit. Restore from that backup if needed |
