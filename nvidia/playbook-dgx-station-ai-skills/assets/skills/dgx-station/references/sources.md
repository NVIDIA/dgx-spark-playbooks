# Guidance sources and precedence

The bundled retrieval snapshot uses these exact source revisions:

| Role | Source | Revision |
|---|---|---|
| ISV development and Software 2.0 behavior | DGX Station Development Guide | `76a1f6adf1a740699c2efff201377947d90f7fd8` |
| Deployment, firmware verification, and bring-up troubleshooting | DGX Station GB300 Bring-Up Guide | `2f2d22b2fee4b6a2964045a97b786b86b366b65b` |
| vLLM engine semantics matching NVIDIA vLLM 26.06 | upstream vLLM | `0decac0d96c42b49572498019f0a0e3600f50398` (`v0.22.1`) |
| NVIDIA vLLM container contents and compatibility | NVIDIA vLLM 26.06 release notes | updated 2026-07-09 |

`playbook search` and `playbook show` return the repository, revision,
source-file SHA-256, heading, and line span for every record.

## Precedence

1. Use the pinned DGX Station Development Guide for Software 2.0 runtime
   behavior, coherency, GPU selection, containers, memory placement, and
   power sloshing.
   Apply its CDMM, ordering-service, and `vsloshd` passages only when the
   detected compatibility capability is true.
2. Use the pinned bring-up guide for physical deployment, BMC and firmware
   verification, driver bring-up, hardware power-brake evidence, and support
   collection where it does not conflict with the Development Guide.
3. Use NVIDIA vLLM release notes and the framework support matrix for the
   NVIDIA container. Use upstream vLLM documentation for engine semantics.
4. Guidance never becomes executable merely because it is authoritative.
   Only a current, digest-bound, physically validated recipe can supply
   launch arguments.

## Known conflicts and exclusions

The bring-up repository's `rtx-considerations.md` predates the current R610+
Software 2.0 mixed-coherency behavior. Do not retrieve or repeat its claim
that one CUDA context cannot use the RTX and GB300 together. Do not copy its
GPU-index launch examples. The current Development Guide permits mixed access
with supported CUDA and driver releases and warns that NVML indices need not
match CUDA ordinals. On Software 1.0, do not replace that excluded claim with
Software 2.0 behavior; report observed addressing and abstain where the
profile-specific source is insufficient.

The bring-up guide also contains credential literals in examples. Never copy
them. Ask the user to supply protected BMC credentials through an appropriate
credential mechanism, and never persist their values.

The bring-up hardware overview is not a recipe-fit qualification source.
Recipe resolution uses the production GB300 limit and current free-memory
evidence defined by the qualified Station profile.
