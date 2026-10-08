# Clean release builder

## Runnable local-workspace preview

`build_preview.py` builds a separate, deliberately bounded Windows ARM64 app.
It includes the guided browser workspace, household API, health collector,
optional plugin, launcher, native CPython 3.14.8 and tzdata 2026.5. It requires
the exact pinned upstream ZIP/wheel and preserves their license files. It copies
an explicit reviewed source list and rejects credential-shaped content and
linked source paths. No household data, Home Assistant, model weights or OpenClaw
runtime is included. Its receipt explicitly sets `fullStackRelease=false`.

Build the launcher with `runtime/ui/build.ps1`, then run:

```powershell
python packaging/build_preview.py --output <new-output-directory> --python-archive <verified-python-3.14.8-embed-arm64.zip> --tzdata-wheel <verified-tzdata-2026.5-py2.py3-none-any.whl>
```

Sources and publisher links are in `THIRD-PARTY-NOTICES.md`. The resulting ZIP
contains `LivingHome-Workspace/START-HERE.md` and `LivingHome.exe`; distribute the
complete archive, not the launcher alone. The exact output inventory and archive
SHA256 are emitted alongside it. Hosting remains unset.

## Full Spark release

`build_release.py` uses only exact reviewed files; it does not recursively copy
an installation or take snapshots of live state. Each file needs an exact source
path, destination, SHA256, classification and `reviewed=true`. Private state paths,
credential-shaped content, symbolic links/junctions, unsafe Windows paths and
case-insensitive collisions are rejected.

`release-source.inventory.json` records unresolved runtime gates and deliberately
refuses a full release. Closing a gate requires its implementation and verification;
changing a boolean alone is not evidence. Do not use the original private demo
builder: it copies tokens, sessions, property records and account configuration.

After the clean runtime and its approved inventory are ready:

```powershell
& 'C:\Path\To\Own\python.exe' 'C:\LivingHome-installer\packaging\build_release.py' --inventory 'C:\Path\To\Approved\inventory.json' --source-root 'C:\Path\To\Clean\Source' --output 'C:\Path\To\New\Release'
```

The output directory must be new. The builder stages and verifies exact bytes,
creates `living-home-arm64.zip`, and emits the installer's schemaVersion-1
`release.json` contract: releaseVersion, architecture arm64,
payload URL/size/SHA256/unpackedBytes/entryPoint, models path/URL/size/SHA256 and
minimumFreeBytes. A local build uses a relative ZIP filename; no public origin or
upload is invented. Future `--payload-url` accepts an exact HTTPS URL without
performing an upload. Model downloads require real pinned HTTPS URLs.

The unpacked-byte total is the sum of ZIP file-entry lengths. The entrypoint is a
packaged `.ps1`, `.exe`, `.cmd` or `.bat` called from the install root with no extra
arguments; it must derive its paths from its own location.

```powershell
& 'C:\Path\To\Own\python.exe' -m unittest discover -s 'C:\LivingHome-installer\packaging' -p 'test_*.py' -v
& 'C:\Path\To\Own\python.exe' -m unittest discover -s 'C:\LivingHome-installer\workflows' -p 'test_*.py' -v
```

These tests use synthetic local fixtures and fake cron responses. They do not
start/restart services, change live devices, run existing schedules or publish.

`build_development.py` separately assembles exact new-project source paths from
`development-source.files.json`. It checks source hashes/exclusions and emits
`development-source.zip` plus `development-bundle.json` with `runnableStack=false`.
This component bundle includes the clean backend, portable collector, new plugin
and setup tools. It has no full-runtime `release.json` and cannot masquerade as a
complete installer payload. Optional future files are recorded as missing.

```powershell
& 'C:\Path\To\Own\python.exe' 'C:\LivingHome-installer\packaging\build_development.py' --source-root 'C:\LivingHome-installer' --approved-list 'C:\LivingHome-installer\packaging\development-source.files.json' --output 'C:\Path\To\New\DevelopmentBundle'
```

`dependencies.candidate.json` records four pre-existing pinned model downloads
(94,586,588,224 bytes total), two pinned official llama/CUDA ARM64 archives and
eight explicitly named native files rehashed locally. Node is ARM64; the existing
Python binaries are **x64**. These observations do not constitute a complete
native ARM64 dependency inventory or a fresh-Spark compatibility test. Model and
ZIP pins were transcribed from existing manifests; large weights were not rehashed
during this work. No credential-bearing runtime directories were copied.
