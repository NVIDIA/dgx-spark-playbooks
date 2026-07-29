# Software compatibility profiles

Use the trusted release marker and the `compatibility` object returned by
`system inspect`. Never infer a profile from a marketing name or version prefix
alone.

## Supported profiles

| Profile | Exact release identity | Support level |
|---|---|---|
| Software 1.0 | `7.4.1` or `7.4.1-GB300ws`; build `2026-02-20-05-22-42`; `NVIDIA DGX GB300WS` | Version-aware guidance, inspection, diagnostics, read-only MIG inspection, and exact recipes validated for this profile |
| Software 2.0 | `7.5.0`; build `2026-06-16-11-48-10`; `NVIDIA DGX GB300WS` | Qualified workflows, subject to every live safety and content check |

An exact identity mismatch has support level `unknown`. On an unknown profile,
use read-only host evidence and general guidance only. Do not provide a
version-specific command.

## Capability differences

| Capability | Software 1.0 | Software 2.0 |
|---|---:|---:|
| Host inspection and read-only diagnostics | Yes | Yes |
| Version-specific guidance | Yes | Yes |
| Read-only MIG inspection and installed-driver profile listing | Yes | Yes |
| Software 2.0 mixed-coherency ordering service checks | No | Yes |
| CDMM-specific checks and guidance | No | Yes |
| `vsloshd` service checks | No | Yes |
| Recipe execution | Yes, only when current trusted validation explicitly names Software 1.0 | Yes, only when current trusted validation explicitly names Software 2.0 |
| MIG or other platform mutation | No | Yes, after preview and approval |
| Packaged platform-service fixes | No | Yes, after preview and approval |

Software 1.0 still reports observed GPU addressing modes, UUIDs, driver state,
containers, CDI state, power caps, clients, and MIG state. Do not report an
absent Software 2.0-only service as a Software 1.0 fault. Continue to report
hardware, driver, container, storage, active-client, and combined-power-budget
problems that are directly observed. Its recipe capability does not enable MIG
mutation, platform mutation, or packaged service fixes.

The current Development Guide describes Software 2.0 and R610+ behavior. On
Software 1.0, use its general hardware and operational guidance only when the
passage is not tied to CDMM, the Software 2.0 ordering service, or `vsloshd`.
If applicability is uncertain, say so and abstain from a command.
