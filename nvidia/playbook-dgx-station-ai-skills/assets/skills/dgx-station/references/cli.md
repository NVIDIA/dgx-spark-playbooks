# General CLI reference

Use the local launcher:

```text
scripts/dgx-assist system inspect --json
scripts/dgx-assist playbook status --json
scripts/dgx-assist playbook search "mixed coherency containers" --json
scripts/dgx-assist playbook show RESULT_ID --json
```

Every JSON response has schema version, command, success state, data, warnings,
and provenance. Preserve explicit rejection reasons and unknown evidence.

`system inspect` returns `compatibility.profile_id`, `software_release`,
`support_level`, a boolean `capabilities` mapping, and short version-specific
`guidance`. Treat capability values as authoritative workflow gates.
`qualified` remains the narrower signal that all action-qualification checks
passed.

`playbook search` returns at most five passages and no more than two per file.
An empty result with `retrieval_trace.abstained=true` is a required abstention,
not a reason to fall back to remembered Station commands.
