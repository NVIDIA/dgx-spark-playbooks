# Living Home Windows bootstrapper

`dist/LivingHomeSetup.exe` is a compiled, single-file .NET Framework WinForms bootstrapper with a GUI and command-line readiness/install modes. It requires a publisher-supplied release manifest. No public hosting, runtime payload, real model download, or real installation is included in these tests.

The executable targets managed AnyCPU and detects the native machine through `IsWow64Process2`, including on Windows ARM64 under emulation. A release installation requires Windows 11 ARM64, .NET Framework 4.8.1, a detected NVIDIA GPU, enough Windows commit, and enough disk space. This executable does not install .NET, NVIDIA drivers, CUDA, Docker, Home Assistant, or the release's Python/OpenClaw dependencies. The trusted release setup entry point handles its declared runtime prerequisites. Fresh-machine completion still requires testing on the second Spark.

## Build and verify

Run from the separate project root in normal Windows PowerShell 5.1:

```powershell
& .\installer\build.ps1 -Tests
```

The build uses the installed `C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe` (or its Framework fallback). It downloads no compiler or package. Sources are `InstallerCore.cs`, `SetupApp.cs`, and `tests/InstallerTests.cs`. Output is `dist/LivingHomeSetup.exe` and its SHA256 sidecar. `tests/InstallerTests.exe` is test tooling and is not part of the shipped bootstrapper.

Latest recorded command: `& C:\LivingHome-installer\installer\build.ps1 -Tests`. Result: **13 test groups passed, 0 failed**. Machine-check evidence is in `tests/check-only-host.json`; test evidence is in `tests/test-results.json`. Test assets were generated under a unique `%TEMP%\LivingHomeSetupTests-*` directory and removed. No release setup entry point or runtime service was launched.

The tests validate manifest pins/HTTPS/schema/path requirements; valid ZIP extraction; traversal, Windows path aliases, ADS and reserved device names; ZIP symlinks and directory/file conflicts; actual Windows junction/reparse escape guards; existing installation and file preservation; wrong sizes and hashes; corrupt-cache repair; cancellation and retained partial data; an actual interrupted HTTP body followed by a validated Range resume; servers ignoring Range; invalid Content-Range and Content-Length; and the production executable's check-only, missing-pin, wrong-pin, and HTTP-source gates. Localhost HTTP is enabled only in the separately compiled test binary through `INSTALLER_TESTS`; the production executable rejects it.

## GUI

Run `dist/LivingHomeSetup.exe` without arguments. Select an explicit JSON manifest file or enter its HTTPS URL, select a new installation directory under an existing local parent, and optionally supply an independently obtained manifest SHA256 pin. Check readiness, review the displayed release and publisher, then check the trust box to enable installation. Changes to release or destination clear the check. The installer rereads the manifest and refuses a changed release before installation.

The GUI reports progress, supports cancellation while downloading/extracting, and can save a JSON report. Cancellation retains verified and partial downloads for the next attempt. Once trusted release setup starts, cancellation is disabled to avoid killing setup midway. Release output is saved to `.livinghome-setup.log` in the new install directory. The release must not print credentials or secrets to its setup output.

`tests/setup-preview.png` is an image of the actual form, rendered without displaying a window, checks, downloads, or service actions. To regenerate it:

```powershell
$preview = Start-Process -FilePath .\dist\LivingHomeSetup.exe -ArgumentList '--render-preview', 'C:\LivingHome-installer\installer\tests\setup-preview.png' -WindowStyle Hidden -Wait -PassThru
$preview.ExitCode
```

## Command-line use

These example paths represent a separately approved release, not files bundled here. `--check-only` reads the manifest and checks the device, storage and destination. It creates neither an installation directory nor an asset cache and launches no release code. A remote manifest is the only possible network fetch in this mode.

```powershell
$check = Start-Process -FilePath .\dist\LivingHomeSetup.exe -ArgumentList '--check-only --manifest "C:\Releases\release.json" --install-dir "C:\LivingHome-New" --report "C:\Releases\readiness.json"' -WindowStyle Hidden -Wait -PassThru
$check.ExitCode
```

For unattended bootstrap installation, replace the hash below with the release publisher's independently obtained manifest SHA256. The hash is over the exact manifest file bytes. Do not obtain a trust pin only by hashing an otherwise untrusted manifest.

```powershell
$install = Start-Process -FilePath .\dist\LivingHomeSetup.exe -ArgumentList '--unattended --manifest "C:\Releases\release.json" --manifest-sha256 PUBLISHER_SUPPLIED_64_HEX_PIN --install-dir "C:\LivingHome-New" --report "C:\Releases\installation.json"' -WindowStyle Hidden -Wait -PassThru
$install.ExitCode
```

Exit codes: `0` succeeded; `2` invalid input/download/setup error; `3` readiness failed; `4` cancelled. Use `--help` for syntax. GUI publisher confirmation and the unattended SHA256 pin protect the decision to execute remote code; manifest payload hashes alone establish integrity, not publisher identity. HTTPS URLs and every redirect are checked, using normal Windows certificate validation. Credentials in URLs and remote file shares are rejected. A local manifest can reference local files or HTTPS assets; a remote manifest can reference only HTTPS assets.

The release setup contract is noninteractive and log-producing. A `.ps1` entry point runs with normal Windows PowerShell 5.1, `-NoProfile -NonInteractive -ExecutionPolicy Bypass -File`; `.exe`, `.cmd`, and `.bat` are also supported. No arguments are supplied, stdin is closed, and working directory is the final install root. Derive the installation root from the script location. GUI application entry points can still open their own UI; the bootstrapper's unattended mode cannot suppress a release's own application dialogs. It does not elevate automatically.

## Manifest contract

The packaging builder owns creation of the real manifest. Required fields:

| Field | Meaning |
|---|---|
| `schemaVersion` | Integer `1` |
| `releaseVersion` | Nonempty publisher release identifier |
| `architecture` | `arm64` |
| `payload.url` | HTTPS URL, or local file reference when the manifest is local |
| `payload.size` | Exact compressed ZIP byte length, positive integer |
| `payload.sha256` | SHA256 of ZIP bytes, exactly 64 hexadecimal characters |
| `payload.unpackedBytes` | Exact sum of all ZIP file-entry lengths, positive integer |
| `payload.entryPoint` | Windows-safe relative `.ps1`, `.exe`, `.cmd`, or `.bat` path within the ZIP |
| `models` | Array, possibly empty, of `{path,url,size,sha256}` objects; each size is exact and positive |
| `minimumFreeBytes` | Optional minimum free target-drive bytes; defaults to zero |

Model `path` is relative to the final installation root. Rooted paths, `.`/`..`, blank components, ADS colons, control/invalid filename characters, trailing dots/spaces, Windows device names, ZIP symlinks and reparse metadata, duplicate case-insensitive paths, file/directory collisions, reserved installer filenames, and overly long paths are refused. Windows batch entry point paths cannot contain `%` or `!`. The archive must include the declared entry point. ZIP preflight validates every entry before extraction.

## Installation and safety behavior

The destination must not exist, even if empty. A new staging directory is created beside it; the ZIP is extracted using create-new file semantics, models are copied and rehashed, a receipt is written, and a final directory move commits the install. The move fails if the final destination appeared meanwhile. Existing targets are never removed. `C:\LivingHome` and its descendants are explicitly reserved for the current workspace; an existing `.livinghome-install.json` ancestor also prevents nested installs. Drive letters and cache/install roots otherwise follow the selected machine and destination; no old `D:` root is assumed.

Reparse points are rejected along manifest, local asset, cache, staging and destination paths. Partial downloads use validated HTTP byte ranges; a server ignoring Range causes a safe restart. Exact byte count and SHA256 are required before a file becomes verified. Downloads are stored under `%LOCALAPPDATA%\LivingHome\Installer\Cache\<manifest-hash>`, keyed by asset digest. The cache is retained intentionally for retries; it is not cleared automatically.

Readiness reserves target space for uncompressed payload + all models + 4 GiB, with `minimumFreeBytes` as a lower bound. Cache space reserves compressed payload + all models + 1 GiB; needs are added on the same drive. Available commit must exceed the largest declared model + 8 GiB. These are conservative capacity checks, not a CUDA/model benchmark. Windows-visible physical memory may be lower than nominal shared RAM due to GPU reservation; setup does not change it.

Before commit, failed extraction is cleaned up only if the staging ownership marker matches and the complete tree contains no reparse points. Unsafe/unowned staging is preserved with an actionable error. After commit, setup failures preserve the install, log and receipt for diagnosis. Receipts contain release/hash/status metadata, not credentials. Setup does not change GPU carveout, Defender settings, PAIR/device pairing, or existing services.

## Remaining release and fresh-machine validation

- Assemble a reviewed, neutral runtime payload with ARM64-compatible Python/CUDA/model dependencies and exact verified inventory. Current packaging blockers must be resolved before publishing a real release.
- Test the real manifest and models in a new directory on the second Spark; validate exact range behavior from the chosen HTTPS host, interruptions, free-space/commit gates, release setup exit handling, logs, reboot and first run.
- Confirm driver/model inference compatibility, service registration, ports, and private household setup on the second machine. The local synthetic test does not prove these runtime functions.
- Choose hosting, establish an independent trusted manifest-pin distribution channel, and sign the executable before broader distribution. The current locally reviewable build is unsigned.
