# Diagnostic findings and fixes

`diagnose run` records stable finding IDs for release integrity and
compatibility, driver/GPU health, applicable addressing/coherency/CDMM and
power checks, Docker/Toolkit/CDI, MIG, cache, processes, ports, content
freshness, and owned service health. Software 1.0 intentionally skips
Software 2.0-only service and CDMM checks.

Only findings with `fix_id` are executable. The v1 registry is limited to:

- rebuilding the local playbook index from verified content;
- gracefully stopping a failed service whose ownership labels are rechecked;
- restarting the packaged mixed-coherency ordering service on the exact
  qualified release;
- invoking the packaged CDI refresh service on the exact qualified release.

Preview one fix:

```text
scripts/dgx-assist diagnose fix --report-id REPORT --finding FINDING --dry-run --json
```

After explicit approval:

```text
scripts/dgx-assist diagnose fix --report-id REPORT --finding FINDING --yes --json
```

Export for support:

```text
scripts/dgx-assist diagnose bundle --report-id REPORT --json
```

The bundle is explicit local export. There is no telemetry or automatic upload.
