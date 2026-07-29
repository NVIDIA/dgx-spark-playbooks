# Inference command workflow

Begin with this read-only sequence:

```text
scripts/dgx-assist recipe models --json
scripts/dgx-assist system inspect --json
scripts/dgx-assist playbook search "MODEL BACKEND inference" --json
```

Continue only when `compatibility.capabilities.recipe_execution` is true:

```text
scripts/dgx-assist recipe resolve --model MODEL --json
scripts/dgx-assist recipe show --recipe-id RECIPE_ID --json
scripts/dgx-assist recipe preflight --resolution-id RESOLUTION_ID --json
scripts/dgx-assist recipe run --resolution-id RESOLUTION_ID --dry-run --json
```

Only after the user approves the dry-run:

```text
scripts/dgx-assist recipe run --resolution-id RESOLUTION_ID --yes --json
```

Add `--allow-download` only after separately disclosing and approving every
required model or image download. Add `--allow-external-bind` only after the
user requested a non-local `--bind-host` and approved the disclosed exposure.

For lifecycle:

```text
scripts/dgx-assist recipe status --json
scripts/dgx-assist recipe stop --service-id SERVICE_ID --dry-run --json
scripts/dgx-assist recipe stop --service-id SERVICE_ID --yes --json
```

Resolution records expire after one hour and become invalid whenever their
system, catalog, recipe, validation, port, device, cache, or network inputs
change.
