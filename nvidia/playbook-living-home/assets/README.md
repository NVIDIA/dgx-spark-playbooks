# Living Home

Connect your Home Assistant devices, review changes before applying them, and keep local device-status reports.

## Run from this playbook checkout

From this `assets` directory on Windows, use Python 3.12 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install tzdata
.\.venv\Scripts\python.exe .\runtime\ui\app.py
```

The app opens an authenticated browser window. Keep the PowerShell process running and use **Quit** to stop the app. Its household directory is the same as the packaged workspace described below. The parent playbook's **Instructions** tab walks through connection, device selection, automation verification, and reports.

## Before you start with a packaged workspace

- A Windows ARM64 computer and the complete `LivingHome-Workspace-Windows-ARM64.zip` package.
- A running Home Assistant instance with your devices already connected.
- A Home Assistant long-lived access token. Saving automations also requires permission to manage Home Assistant automations.

Living Home includes its Python runtime. Install Home Assistant separately. A local model is optional for requests in ordinary language; scheduled reports additionally require a configured OpenClaw installation.

## Get started

1. Extract the complete ZIP into a folder and open **LivingHome.exe**.
2. Enter your Home Assistant address and access token.
3. Select your lights, switches and sensors, then confirm the time zone.
4. Choose **Open my home**.

The executable is currently unsigned. Keep it with the extracted runtime folder.

[Setup and troubleshooting](runtime/ui/START-HERE.md)

## Use your home

| Task | Where to start |
| --- | --- |
| Turn a device on or off | **Devices** → choose an action → **Approve and apply** |
| Create an automation | **Automations** → build a rule or describe your intent → review and approve |
| Check device availability and batteries | **Reports** → **Check devices and save report** |
| Read or download a saved report | **Reports** → choose a report |
| Prepare a daily report | **Reports** → **Prepare schedule**, then complete the OpenClaw setup below |

[Automation and reporting playbook](workflows/WORKFLOWS.md)

## Your data and settings

Your selected devices, credentials and reports stay in `%LOCALAPPDATA%\LivingHome\Household`. Updating the application folder preserves them. Keep this private data folder out of shared archives and source repositories.

Choose **Quit** to stop Living Home. Closing the browser tab leaves it running. Automations saved in Home Assistant continue running until you disable them there.

Device selection is currently set during first run. Changing it afterward requires manual configuration; there is no add/remove-device screen yet. Preparing a report schedule saves a configuration file; it does not start recurring reports.

[Security and privacy](SECURITY.md) · [MIT license](LICENSE) · [Third-party notices](THIRD-PARTY-NOTICES.md)
