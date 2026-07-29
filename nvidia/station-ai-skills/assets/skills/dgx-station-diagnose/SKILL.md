---
name: dgx-station-diagnose
description: Run and interpret the complete read-only dgx-assist diagnostic suite for NVIDIA DGX Station GB300, correlate findings with pinned NVIDIA playbooks, export a redacted support bundle, and apply one separately approved allowlisted fix. Use when the user reports a Station, CUDA, GPU health, coherency, vsloshd, Docker, CDI, MIG, cache, port, or owned inference-service failure.
---

# DGX Station diagnostics

Diagnose first. Do not mutate as part of diagnosis.

## Workflow

1. Run `scripts/dgx-assist diagnose run --json`.
2. Report the detected compatibility profile, then findings in severity order with their stable IDs and evidence. Preserve `unknown` states. On Software 1.0, do not reinterpret intentionally skipped Software 2.0 service checks as faults.
3. Search the pinned playbooks for each high or critical finding; cite the relevant URL, heading, lines, and commit.
4. If no passage overlaps, say so and avoid inventing a platform fix.
5. Offer `diagnose bundle --report-id "<id>"` when escalation is appropriate.
6. Offer at most one automatic fix at a time, and only when `fix_id` is present.
7. Preview with `diagnose fix --report-id "<id>" --finding "<id>" --dry-run`.
8. Explain exact actions, impact, privilege, and reboot state. Obtain explicit approval.
9. Repeat with `--yes` only after approval and report the action receipt.

## Safety requirements

- Keep `diagnose run` read-only.
- Never install packages, rewrite Docker configuration, change power caps or driver parameters, kill workloads, or modify MIG through a diagnostic fix.
- Never stop a service without current `dgx-assist` ownership evidence.
- Never use Fabric Manager as a routine Station check or remediation.
- Re-run diagnostics when finding evidence is stale.
- Never reveal secrets or unredacted home paths in a bundle.
- Do not execute an unregistered remediation.
- Treat `--yes` only as approval already obtained.

Read [references/findings.md](references/findings.md) before proposing a fix or support bundle. Read [references/bringup.md](references/bringup.md) when the problem concerns physical deployment, BMC or firmware verification, driver bring-up, power braking, or support escalation.
