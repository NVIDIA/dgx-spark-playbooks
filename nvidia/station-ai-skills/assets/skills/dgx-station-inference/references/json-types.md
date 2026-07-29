# Inference JSON interpretation

## Resolution

Verify the canonical model equals the user's requested model. Report recipe
and launch IDs, system/catalog/recipe/validation digests, GPU UUIDs, cache,
network policy, rejected same-model candidates, and expiry.

## Preflight

Treat any failed `severity=error` check as a hard block. Disclose warning
checks, required confirmations, download and storage estimates, port and GPU
conflicts, and readiness.

## Receipt

Report action, actor mode, timestamps, input digests, redacted argv,
environment variable names, owned resource IDs, result, model identity, and
smoke verification. A receipt never authorizes a different future action.

