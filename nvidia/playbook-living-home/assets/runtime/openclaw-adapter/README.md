# Portable OpenClaw adapter

This plugin connects OpenClaw to Living Home for selected-device plans, local records, reports and device-status collection. It registers `living_home_fast` and `living_home_property`. Google publication and email sending are not supported.

Start the Living Home workspace before using the plugin. Its API normally uses `http://127.0.0.1:18881`. Use the workspace's household directory and its `api-token.txt`; the Home Assistant token is a different credential. The plugin and OpenClaw model configuration must be completed separately before using scheduled reports.

Build with Node 24.16.0 or newer; no npm dependencies are needed:

```sh
node scripts/build.mjs
node --test test/*.test.mjs
```

The generated entry is `dist/index.js`. OpenClaw supplies its SDK at plugin load time. The manifest ID is `living-home`; the package name is `@living-home/openclaw-adapter`.

Trusted plugin configuration requires `baseUrl`, `pythonExecutable`, `collectorPath`, `dataDirectory`, and `apiTokenFile`. All four filesystem paths must be absolute, with existing local files and data directory. `collectorPath` must name the installed `.py` collector. `baseUrl` must be an HTTP(S) origin on `127.0.0.1` or `[::1]`, with the selected port; it cannot include URL credentials, paths, queries or fragments. `propertyBasePath` optionally changes the property API prefix and defaults to `/property`. Credentials are never tool parameters.

The backend token is read from `apiTokenFile` for each API call. It must be 32–8192 characters without whitespace and is sent as a bearer token. Transport and HTTP errors are generic, without response bodies, headers, token values or executable paths. Responses are redacted for this token. Redirects are refused and responses are limited to 8 MiB. Writes are never automatically retried. A host tool call ID produces an idempotency key, and changed arguments for a reused ID are rejected. The in-memory ledger stops accepting new writes at 1000 IDs instead of evicting safety bindings; backend receipts provide persistence across plugin restarts.

`living_home_property` with `operation: "health_snapshot"` invokes Python directly, without a shell, using:

```text
<pythonExecutable> <collectorPath> --env-file <dataDirectory>/.env --config <dataDirectory>/health-config.json --output-dir <dataDirectory>/health/snapshots
```

The child process runs in `dataDirectory`, hides its window on Windows, limits its output, and honors cancellation and a 35-second timeout. Inherited `HA_URL` and `HA_TOKEN` are removed so the designated household `.env` is authoritative. Tool arguments cannot override the executable, collector, credentials, output path or backend origin. The collector returns selected-entity coverage and never classifies entity totals as physical-device counts.

Backend route contract: GET `/device-plans/capabilities`; POST `/device-plans/preview`, `/device-plans/apply`, `/device-plans/execute`; GET `/device-plans/status?draft_id=...`. Plans retain the exact authored effects and the observed `expected_run_id`, with `device_profile: "physical"`. Preview proposes effects. Apply uses the exact returned `draft_id` and 64-character `plan_hash`; apply and immediate execute require `authorization: {confirmed: true, user_request: "<actual user instruction>"}`. The backend must enforce capabilities, run guards, plan hashes and authorization. Future rules require preview/apply. Missing or stale evidence is not rewritten or bypassed.

Property routes under `propertyBasePath`: GET `/status`; GET `/search?q=...&limit=...`; GET `/asset/{id}`, `/incident/{id}`, `/maintenance/{id|current}` (or `/brief`), `/reports/{id|latest}`; POST `/report`, `/repair-draft`. Report reads accept bounded metadata filters and body paging. Reports have `title`, `body`, and optional `report_type` of `general`, `health`, or `maintenance`. Maintenance reports require exact `incident_id` and `maintenance_revision`; repair drafts require exact `asset_id`, `incident_id`, `subject`, and `body`. Optional draft recipients are stored only as authored text. `publish_to_drive: true` or `save_to_gmail: true` rejects before any request. Saved prose must preserve the reported source coverage and uncertainties.

Tests use injected process runners, injected fetchers, and temporary files; they never call the homeowner's running backend, Home Assistant, models, or OpenClaw service. The SDK smoke script takes a separately supplied `OPENCLAW_PACKAGE_ROOT`, imports only its SDK, and checks plugin registration without executing tools or starting services. Local fake-server execution of the Python collector is tested separately in `runtime/health/tests`.
