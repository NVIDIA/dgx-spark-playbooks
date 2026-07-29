# Bring-up evidence

The diagnostic workflow uses the NVIDIA DGX Station GB300 Bring-Up Guide at
commit `2f2d22b2fee4b6a2964045a97b786b86b366b65b` for deployment, firmware
inventory, driver bring-up, hardware power-brake evidence, and escalation.
Search the bundled playbook for the exact symptom before advising.

The Development Guide at
`76a1f6adf1a740699c2efff201377947d90f7fd8` takes precedence for Software 2.0
mixed coherency, CUDA/NVML ordering, container exposure, CDMM, memory
placement, and power sloshing.

## Read-only triage

- Verify release-marker integrity and the detected compatibility profile first.
- For a missing GPU, capture driver and kernel/XID evidence. Do not reinstall
  the driver from this skill.
- For unexpectedly low performance, inspect thermal and hardware power-brake
  evidence. Inspect `vsloshd` only when the detected profile enables that
  capability.
- Treat the BMC as an initial-setup and recovery interface, not normal service
  management.
- FirmwareInventory may require BMC HTTPS and protected credentials. Never
  place credential values in commands, transcripts, reports, or bundles.
- Support artifacts are exported only when the user requests a bundle. Never
  upload automatically.

The current automatic diagnostic suite does not authenticate to the BMC or
install or update firmware. If firmware inventory is required, cite the
pinned bring-up passage, describe the protected read-only collection
separately, and state that it is outside automatic `diagnose run`.

Known-stale RTX context and GPU-index examples in the bring-up repository are
excluded from retrieval. Do not use them during diagnosis.
