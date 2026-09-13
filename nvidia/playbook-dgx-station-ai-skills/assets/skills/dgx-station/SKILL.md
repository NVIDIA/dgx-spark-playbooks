---
name: dgx-station
description: Inspect and guide NVIDIA DGX Station GB300 development using the local dgx-assist CLI and pinned NVIDIA playbooks. Use for general Station platform questions, Software 1.0 or 2.0 compatibility, GB300 or RTX GPU selection, UUID ordering, mixed ATS/HMM coherency, CDMM, general containers, CDI, CUDA visibility, or vsloshd power-sloshing behavior. Do not use for vLLM or SGLang container selection or tuning, serving a named model, changing MIG, or troubleshooting a reported failure when the dedicated Station skill applies.
---

# DGX Station

Ground every Station-specific answer in current host evidence and a retrieved NVIDIA passage.

## Workflow

1. Run `scripts/dgx-assist system inspect --json`.
2. Read [references/software-compatibility.md](references/software-compatibility.md). Use `compatibility.profile_id`, `support_level`, and `capabilities`; do not branch on a version prefix.
3. Form a narrow playbook query from the user's question and run `scripts/dgx-assist playbook search "<query>" --json`.
4. Apply the source-precedence rules in [references/sources.md](references/sources.md). Use only passages applicable to the detected profile. Cite their URL, heading, line span, source commit, and source-file digest.
5. On the Software 1.0 capability-scoped profile, provide the supported inspection, diagnostics, compatibility guidance, and exact recipes qualified for that profile. Do not treat an absent Software 2.0-only service as a fault.
6. Before any platform action, require its named capability to be `true`. If it is false, explain the profile restriction separately from relevant read-only guidance.
7. If `version_specific_guidance` is false or retrieval abstains, say no applicable version-specific passage was found and avoid inventing a platform command.
8. Answer from the combined profile and applicable passage. Label unknown evidence as unknown.

## Safety requirements

- Before any mutating action, display the exact command to be run and obtain explicit user approval immediately before executing it. Never batch approvals or carry one forward to a later action.
- Never assume an `nvidia-smi` index is a CUDA ordinal.
- Use GPU UUIDs for every proposed launch. When multiple GPUs are deliberately visible, place the GB300 UUID first.
- Do not apply either legacy or Software 2.0 mixed-device behavior without checking the detected profile and observed ATS/HMM state.
- Do not tell the user to install or operate Fabric Manager as a normal Station requirement.
- When `mixed_coherency_service` is true, inspect `mixed-coherency-gpu-select.service`, its generated environment, and container exposure separately.
- When `dynamic_power_sloshing` is true, inspect `vsloshd`. Always inspect observed caps and never propose an ad hoc power-cap change.
- Never install or change the driver, kernel, OS, firmware, or packages.
- Never display credential values.

Read [references/platform.md](references/platform.md) when interpreting coherency, container ordering, power, memory placement, or qualification evidence. Read [references/software-compatibility.md](references/software-compatibility.md) before applying version-specific guidance. Read [references/sources.md](references/sources.md) when guidance may come from the Development Guide or bring-up guide. Read [references/cli.md](references/cli.md) for command and JSON details.
