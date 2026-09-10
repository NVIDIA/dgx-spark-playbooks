---
name: dgx-station-mig
description: Inspect NVIDIA DGX Station GB300 MIG state and installed-driver profiles, create a digest-bound layout plan, disclose disruption and restoration, and apply an approved still-valid plan. Use when the user asks to enable, disable, partition, reconfigure, inspect, or troubleshoot MIG instances or needs MIG UUIDs. Never assume static profile IDs or terminate GPU clients.
---

# DGX Station MIG

Treat inspection and planning as read-only. Treat every apply as disruptive.

## Workflow

1. Run `scripts/dgx-assist system inspect --json`.
2. Search the playbooks for the user's MIG concern and cite relevant results.
3. Read `compatibility.capabilities.mig_inspection`. If it is false, stop and explain that the detected release has no MIG inspection profile.
4. Run `mig inspect` and `mig profiles`. Use only profiles reported by the installed driver. This read-only step is available on the recognized Software 1.0 and Software 2.0 profiles.
5. Before planning, require `compatibility.capabilities.mig_mutation` to be `true`. If it is false, report the observed state but do not suggest a layout or work around the restriction.
6. Require the desired exact layout. Do not choose a layout from model names or nominal memory sums.
7. Run `mig plan --layout "<layout>"`.
8. Present current and proposed state, exact commands, active clients, privilege, workload impact, reset/reboot possibility, and restoration commands.
9. If any client is active, stop. Tell the user dgx-assist will not terminate it.
10. Obtain explicit confirmation immediately before mutation.
11. Run `mig apply --plan-id "<id>" --yes`.
12. Report every verified MIG UUID. On partial failure, stop and present only the recorded restoration plan.

## Safety requirements

- Never use a hardcoded GPU index, profile table, or assumed placement.
- Never destroy instances, reset a GPU, reboot, or change MIG mode outside an approved plan.
- Never kill a GPU process or take over a service.
- Invalidate the plan when clients, mode, profiles, release, driver, or instances change.
- Do not improvise further mutations after partial failure.
- Do not treat Fabric Manager as a normal Station prerequisite.
- Treat `--yes` only as approval already obtained.

Read [references/workflow.md](references/workflow.md) before planning or applying a layout.
