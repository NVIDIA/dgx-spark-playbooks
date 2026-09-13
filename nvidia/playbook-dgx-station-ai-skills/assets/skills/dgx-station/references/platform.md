# Station platform evidence

Use this reference only after reading `system inspect` evidence.

## Action-qualified profiles

Station v1 recognizes these exact action-qualified release identities:

- Software 1.0: `DGX_SWBUILD_VERSION=7.4.1` or `7.4.1-GB300ws`,
  `DGX_SWBUILD_DATE=2026-02-20-05-22-42`
- Software 2.0: `DGX_SWBUILD_VERSION=7.5.0`,
  `DGX_SWBUILD_DATE=2026-06-16-11-48-10`
- `DGX_PRETTY_NAME=NVIDIA DGX GB300WS`
- a GB300 reporting compute capability `10.3`

The release marker must be a root-owned regular file and must not be a symlink
or group/world writable. Product and DMI names are supporting evidence only.
Actions are capability-scoped: Software 1.0 enables only recipes explicitly
validated for its profile and keeps MIG/platform mutation disabled. See
[software-compatibility.md](software-compatibility.md).

## Mixed coherency

Software 2.0 with R610+ uses CDMM on the GB300 by default and supports access
to ATS and HMM devices. Do not preserve the legacy blanket prohibition on
mixed-device CUDA access.

These CDMM and packaged ordering-service expectations do not apply to the
Software 1.0 profile. On that profile, report observed UUIDs and
addressing modes without importing a Software 2.0 command.

The boot service writes host ordering to
`/etc/mixed-coherency-gpu-select/env`. Its values affect login environments,
not automatic container device exposure. Container runtimes re-enumerate
their exposed subset. Use physical GPU UUIDs; put the GB300 first only when a
multi-GPU recipe intentionally wants it as `cuda:0`.

Primary source:
https://docs.nvidia.com/dgx/dgx-station-development-guide/coherency.html

## Power

`vsloshd` owns a combined 1600 W budget for the GB110 module and optional RTX
card. Dynamic mode reallocates that budget. Inspect service, mode, and observed
caps. Do not suggest manual caps.

The `vsloshd` service check is a Software 2.0 capability. Its absence is not a
Software 1.0 fault; continue to report directly observed caps and budget
violations.

Primary source:
https://docs.nvidia.com/dgx/dgx-station-development-guide/dynamic-power-sloshing.html

## Memory placement and optimization

Coherence expands the addressable working set; it does not make CPU memory
equivalent to GB300 HBM. GPU access to CPU memory is not cached. Keep
performance-critical allocations in HBM where capacity permits, and use
system memory or UVM deliberately for ease of programming or capacity
overflow. Profile the actual workload before making a performance claim.

Primary sources:

- https://docs.nvidia.com/dgx/dgx-station-development-guide/coherency.html#uvm-pitfalls
- https://docs.nvidia.com/dgx/dgx-station-development-guide/optimization.html#taking-advantage-of-uma
