# Validation results

## Windows workspace UX, October 5, 2026

The packaged Windows ARM64 workspace passed **11/11 real-browser integration checks** against the isolated Home Assistant 2026.9.4 lab. The package includes CPython 3.14.8 ARM64 (publisher SHA256 verified; Python Software Foundation executable signature valid) and tzdata 2026.5. Headless Edge exercised the actual packaged application, not a static mockup.

The checks covered first run, invalid-token recovery without creating state, explicit selection, preview with no device effect, approved lamp control, a saved native automation's actual state trigger, low-battery report creation/readback, truthful schedule preparation, 390-pixel layout, restart persistence, and browser JavaScript errors. The first run found a report-contract bug; the corrected package passed. Screenshots and results are in `evidence/ux-browser-v2/`.

The combined source check command passed 60 Python tests, 15 adapter tests and 13 bootstrapper test groups. The Git-visible content audit found no credential-pattern or private-path matches after a clearly synthetic test value was constructed at runtime. This focused scan is not a guarantee against every possible secret. CI is prepared but has not run on the internal NVIDIA service.

The Windows launcher was compiled; the browser integration launches the bundled Python entry point with an isolated data directory. Clean-Windows installation, model-generated intent, OpenClaw cron execution, Google OAuth and physical-device effects are still untested. The workspace archive is not a full Spark installer.

## Backend integration, October 4, 2026

**18 of 18 virtual-device integration checks passed. Windows installer execution in a clean Windows guest remains pending.** The lab uses its own generated credentials and fresh household data. It does not use the reference installation's configuration, devices, services, or accounts.

## What ran

An isolated ARM64 WSL2 distribution named `LivingHome-Validation` runs Alpine 3.24.2, Python 3.14.8, Node 24.18.1, and Podman 5.8.8. Podman runs real Home Assistant 2026.9.4 with software-defined lamp, plug, motion, and battery entities. The application sources were copied from this project using an explicit inventory; the Windows executable did not install them.

The harness calls the actual portable plugin tools, authenticated household API, and real Home Assistant REST, configuration, and service endpoints. It passed:

- Unauthenticated API refusal, selected-entity scope, no side effects during preview, and confirmation before applying a plan.
- Lamp control with HA readback; a saved and loaded native automation; a real virtual-plug state transition that triggered that automation and turned on the virtual lamp.
- Low-battery detection, treating an off device as available, and preserving unavailable/missing status instead of reusing stale battery values.
- Local report creation/readback, reads that do not refresh evidence, and durable idempotency with changed-payload refusal.
- Empty household records, import of a newly supplied maintenance source and incident, a local unsent repair draft, and refusal of external report publication.

The test harness authored the plan and report. It did **not** test an AI intent conversation, OpenClaw cron execution, Google OAuth, Windows installation, GPU inference, or physical devices. These results validate the backend integration beneath those user workflows, not their complete end-to-end experience.

Evidence:

- `evidence/virtual-home-results.json`: all 18 checks, timestamps, automation trigger evidence, and fixture cleanup.
- `evidence/guest-provisioning.json`: source file hashes and source-deployment method.
- `evidence/ha-provisioning.json`: fresh lab provisioning receipt.
- `evidence/host-before.json` and `evidence/host-after.json`: host isolation check.
- `evidence/hyperv-enable-result.json`: approved Hyper-V installation, `RestartNeeded: true`, no automatic reboot.

Evidence, downloads, virtual disks, credentials, and generated guest kits are excluded from Git and are not release inputs. Credentials are under `private/` with a restricted Windows ACL; guest household state is under `/opt/livinghome-validation/household`. Do not attach either to a public issue or distribute them.

## Lab state and reproduction

The test cleanup disabled its saved automation, turned the virtual lamp and plug off, and restored the battery sensor's availability and original health scope. The lab is then stopped to free host resources; its virtual disk and evidence are retained. Only `LivingHome-Validation` is terminated; Docker Desktop and the reference household are left untouched.

The image used was `ghcr.io/home-assistant/home-assistant:2026.9.4`, manifest digest `sha256:35e6df56a9ce632c9b15df869ac73a17af6cdd2cfb99830527ffac9cc5218ba2`. The Alpine rootfs was downloaded from the official Alpine mirror and verified against its release index: SHA256 `9bf70a7f18ea44094cbb5f70c58f9af129c8214745743db0e68e5502cc2ce773`.

`Start-VirtualHome.sh`, `provision_virtual_home.py`, and `provision_guest_runtime.py` are one-time provisioning scripts; they deliberately refuse to overwrite an existing lab. `test_virtual_home.mjs` expects a newly initialized household and is not an idempotent replay of the entire suite. To repeat from a clean baseline, create a separate fresh lab or deliberately reset only this lab after preserving evidence. Do not run household-reset scripts against `C:\LivingHome`.

## Pending Windows guest

Hyper-V was enabled after explicit owner approval. Windows reported that a reboot is required; the reboot remains deferred as requested. No Windows guest has been created or booted.

The official [Windows 11 ARM64 ISO download](https://www.microsoft.com/en-us/software-download/windows11arm64) page was reachable, but its download-link API rejected the automated request. No ISO was downloaded. Use the normal Microsoft browser download and its displayed language-specific SHA256. The recorded English-US checksum on October 4 was `DBF606CEB4CE1390F9F21AD1D96331848BC6B701CBAB0012785881A1A1C0EC63`; reconfirm it if the offered build changes.

Once the owner has rebooted and supplied the ISO:

1. Run `New-WindowsLabVM.ps1 -IsoPath <official-iso> -IsoSha256 <publisher-sha256>` to review the plan. With `-Apply` from an administrator PowerShell, it creates only `LivingHome-Win11-InstallerLab`: Windows ARM64, generation 2, 8 GiB RAM, 4 CPUs, 96 GiB dynamic disk, Secure Boot, TPM, and the internal NAT Default Switch. It does not start the VM or assign a GPU. The script is syntax-checked; guest creation is not yet tested.
2. Boot it deliberately, install Windows, and create a clean checkpoint. Do not sign in with the reference household's Google, Discord, or Home Assistant accounts. Use a guest-only local test account where Windows permits, or an authorized test account.
3. Run `Build-WindowsGuestKit.ps1` on the host if the installer changes. Transfer its exact generated ZIP into the guest through an intentional file-copy mechanism, verify its host-reported SHA256, and extract it to a local writable directory. Do not expose the host drive or credentials to the guest.
4. Run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run-WindowsGuestChecks.ps1` from the extracted kit. It refuses a non-Hyper-V host, verifies its binaries, runs the 13 bootstrapper test groups and production check-only gates, and saves JSON and log evidence. It launches no release setup entry point. The test kit has only been assembled and syntax-checked; it has not yet run in a Windows guest.
5. Inspect every readiness result. The expected NVIDIA-GPU refusal is evidence that the production requirement remains enforced, not evidence of a successful full installation. Additional .NET, architecture, capacity, or destination failures must be recorded independently.
6. Once a full application release exists, test its supported installation path, interrupted downloads, first run, restart, fresh identity, user-selected HA connection/devices, and absence of reference data. A regular Hyper-V guest without a supported GPU cannot validate the full Spark model runtime. Do not weaken production requirements to label a VM test successful.
7. On a supported Spark, complete the real local conversation, intent-to-automation review/approval and actual trigger, disabled cron test and saved report readback, timezone-aware schedule enablement, GPU inference, and physical-device checks from `workflows/FRESH_HOST_CHECKLIST.md`.

The installer is still an unsigned bootstrapper requiring a complete release manifest and payload. The first-run GUI, native OpenClaw configuration/startup, own-account Google OAuth, and full dependency assembly remain unfinished. A Windows VM will not resolve these packaging gaps by itself.
