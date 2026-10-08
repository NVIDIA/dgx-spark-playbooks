# Portable Home Assistant evidence

`collect_home_status.py` uses Python 3.11+ and its standard library. It makes authenticated GET requests to Home Assistant's `/api/config` and `/api/states`. No local Home Assistant files or entity/device registries are required. Its API calls follow the [Home Assistant REST API](https://developers.home-assistant.io/docs/api/rest/).

Copy `health-config.example.json` to the installation root as `health-config.json`. Replace the empty `entity_ids` array with exact entity IDs from your own Home Assistant. Select battery sensors explicitly if you want their battery reports included. There are no sample-home entity defaults; the empty configuration reports `status: "unknown"` and incomplete selection coverage. Selection accepts up to 500 unique exact IDs, without wildcard or domain expansion.

Set `HA_URL` and `HA_TOKEN` in the install root `.env`, or set those two environment variables. Environment variables take precedence. The script reads the file literally and never sources or executes it. The URL must be an HTTP(S) base URL, including any reverse proxy prefix, without credentials, query parameters, or fragments. Redirects are refused so an Authorization header is never forwarded to another URL. The collector never prints the URL or token and redacts any token echoed in retained API fields.

Run explicitly, substituting your installation and Python paths:

```powershell
& '<PythonPath>' '<InstallRoot>\runtime\health\collect_home_status.py' --env-file '<InstallRoot>\.env' --config '<InstallRoot>\health-config.json' --output-dir '<InstallRoot>\health\snapshots'
```

On Linux, the equivalent is:

```sh
python3 '<InstallRoot>/runtime/health/collect_home_status.py' --env-file '<InstallRoot>/.env' --config '<InstallRoot>/health-config.json' --output-dir '<InstallRoot>/health/snapshots'
```

An explicit invocation prints one JSON snapshot and saves a UTC timestamped JSON evidence file, `latest.json`, and `latest-summary.txt`. Each file is replaced atomically; the checked timestamp identifies its evidence run. On Linux newly created files use owner-only permissions and the snapshot directory uses mode 0700. Retained entity names/state information can still be personal household data. The invocation returns 0 after a saved valid snapshot, including `attention` or `unknown`; 2 means configuration could not be used, and 3 means evidence files could not be saved. Library use via `collect(...)` returns fresh JSON-compatible data and performs no file writes.

Each selected entity has a friendly `name`, reported `state`, `availability`, `availability_reason`, reported change/update timestamps when present, and `physical_operation: "not_tested"`. A missing selected ID appears as unknown with `missing_entity`. When state collection fails, all selected IDs appear unknown with `states_not_collected`; they are not claimed missing. States such as off, idle, closed, and locked count as available. A fresh snapshot does not establish the freshness of device telemetry.

Battery warnings use selected entities whose reported `device_class` is `battery`: numeric percentage reports at or below `battery_warning_threshold`, or a binary battery sensor reporting `on`. Non-percentage values, invalid/out-of-range percentages, and unknown/unavailable battery reports are not converted to a battery percentage.

The snapshot timezone uses `timezone` in the health configuration when supplied, then Home Assistant's configured `time_zone`, then UTC. If an IANA timezone is invalid or unavailable on the Python host, the snapshot uses UTC and records the limitation. Linux Python commonly uses system timezone data; a Windows Python host may require an installed `tzdata` package for non-UTC IANA zones. UTC remains functional without that package.

`check_integrations: true` opts into one additional authenticated read of `/api/config/config_entries/entry`. This optional endpoint is not part of the documented general REST contract and can be unavailable or forbidden on a given Home Assistant version/account. A 404/405/501 marks `coverage.integrations: "unsupported"`, keeps failing-integration count null, and does not fabricate healthy integration status. Other collection failures are visible in `check_errors`. Integration errors report only domain and state, not credentials or configuration payloads. This endpoint is off by default.

The shape retains the existing local report keys `checked_at`, `timezone`, `status`, `home_assistant`, `summary`, `inventory`, `unavailable_controls`, `battery_attention`, `integration_issues`, `integration_evidence`, `check_errors`, and `limitations`. Additional `coverage` and `unavailable_entities` fields make partial coverage explicit. `summary.tracked_control_entities` covers only selected control-domain entities, and `source_counts` uses `home_assistant_entity`; it never infers a physical, staged, phone, or simulated device classification. Report rendering must not treat this count as physical-device inventory.

Tests use only a temporary local fake HTTP server:

```powershell
& '<PythonPath>' -m unittest discover -s '<InstallRoot>\runtime\health\tests' -v
```
