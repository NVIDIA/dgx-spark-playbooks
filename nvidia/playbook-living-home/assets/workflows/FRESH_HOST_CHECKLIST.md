# Fresh ARM64 Spark validation

No second Spark is available yet. Check boxes are intentionally unchecked; this
is the acceptance checklist for the later clean-machine test, not a test receipt.

## Local source and packaging checks

- [ ] Run packaging, workflow, collector and installer tests against temporary
  synthetic files and fake transports; record exact outputs.
- [ ] Assemble reviewed source/dependency components from exact file lists.
- [ ] Confirm no `.env`, account/device IDs, OAuth artifacts, `.storage`, sessions,
  cookies, DBs, recordings, property records or model-plan ledgers enter the ZIP.
- [ ] Verify each packaged source hash and artifact license; detect reparse points
  and unsafe/colliding Windows archive names.
- [ ] Complete runtime readiness gates before producing a release manifest that
  claims to install a complete stack. A development bundle is not that release.
- [ ] Verify model sizes/hashes and declared archive size/unpacked byte totals.
- [ ] Run the bootstrapper against a local manifest before choosing a hosting URL.

## Hardware and dependencies

- [ ] Identify Windows ARM64, available NVIDIA GPU/driver/CUDA compatibility,
  RAM/VRAM, disk space and permissions on the actual target Spark.
- [ ] Validate signed/pinned ARM64 Python, Node, llama.cpp and any needed container
  runtime independently; do not copy this machine's private compatibility tree.
- [ ] Download model artifacts from verified pinned sources, verify hashes, and
  load the local model using measured settings for that hardware.
- [ ] Confirm interrupted download recovery, insufficient disk handling and a
  corrupted artifact refusal without promoting a partial installation.

## Own accounts, records and device scope

- [ ] Decide existing HA or new local HA. For new HA, complete owner account and
  hardware integrations. For existing HA, verify origin/token against that server.
- [ ] Initialize a new household profile without Director Reset or sample seeds.
- [ ] Keep private credentials in the current user's protected data directory.
- [ ] Verify the property store is empty and has a unique own household identity.
- [ ] Select exact own HA entities; verify remote/local coverage and unavailable
  or missing entities remain explicit unknowns.
- [ ] Wire the portable collector to this selected entity scope and own `.env`.
- [ ] If Google is requested, authorize the own account and configure its exact
  label/folder. Verify an empty account remains empty, with no sample bootstrap.
- [ ] Configure native local OpenClaw first and verify a local chat/tool trace.
  Connect Discord separately only if the user requests it.

## Intent automation

- [ ] Ask for one small intended automation using actual observed capabilities.
- [ ] Verify the model reads capabilities and existing loaded HA rules before
  authoring a plan. Confirm no demo-only labels, entities or preset actions.
- [ ] Review entity IDs, values, triggers, timezone, run/profile and plan hash.
- [ ] Apply only the reviewed authorized plan and verify native HA loaded/enabled
  rule state; validate last-triggered and action readback separately.
- [ ] For a remote HA instance, prove the native rule was installed there rather
  than merely written to a local Spark YAML file.
- [ ] Restore the original device state after any authorized physical test.
- [ ] Verify stale drafts/revisions are rejected and uncertain writes are resolved
  through status without replaying physical effects.

## Status report schedule

- [ ] Preview `configure_home_report.py` with the own timezone/profile/Gateway.
- [ ] Apply once and verify a disabled, exact daily job with delivery mode none.
- [ ] Apply again; prove no duplicate and no unnecessary mutation.
- [ ] Manually run the observed job, inspect collector evidence, local model
  report write and matching `report_read` receipt.
- [ ] Verify latest-health retrieval reads stored prose without collecting again.
- [ ] Enable explicitly, verify next run timestamp in the selected timezone, and
  inspect the first scheduled run and report.
- [ ] Confirm no Discord/Google/email receipt exists in local-only mode.
- [ ] Change schedule/timezone; verify the same job becomes disabled for testing.
- [ ] Validate optional delivery only with the user's configured own destination.

## Own maintenance incident

- [ ] Create/import an own incident from actual supplied asset/recording evidence.
- [ ] Preserve source time, missing evidence and immutable evidence revision.
- [ ] Verify analyze/report/draft/read operations use its observed ID and revision.
- [ ] Confirm local drafts remain unsent and public sample scenarios are absent.

## Remaining release work

Implementation work that can proceed locally: clean backend/run-store wiring,
sample-free capability and report routes, portable health plugin, native OpenClaw
model/profile config, own Google OAuth/scope, own incident ingestion, native HA
automation installation, and a reviewed ARM64 dependency inventory.

Later environment checks: actual fresh-Spark GPU/model execution, first own-account
OAuth, own HA physical actuation, scheduled run receipts, and a chosen hosting
origin. Hosting is deliberately deferred until the local complete build passes.
