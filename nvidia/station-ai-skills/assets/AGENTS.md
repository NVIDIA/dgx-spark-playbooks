<!-- BEGIN NVIDIA DGX STATION AI SKILLS -->
## DGX Station routing

- Detect the actual platform with `.dgx-station/bin/dgx-assist system inspect` before advising.
- Follow the detected compatibility profile and per-feature capabilities; do not infer behavior from a version prefix.
- Use `dgx-assist` and the applicable `dgx-station*` skill.
- Search the pinned NVIDIA playbooks before giving Station-specific commands.
- Never assume an `nvidia-smi` index is a CUDA ordinal; select launch devices by UUID.
- Never execute an unvalidated recipe or conceal a mutation, network exposure, or large download.
<!-- END NVIDIA DGX STATION AI SKILLS -->
