# Connect Your Home with Living Home

> Reviewed device actions, automations, and local reports with Home Assistant

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Optional: Generate a plan from your intent](#optional-generate-a-plan-from-your-intent)
- [Scheduled reports](#scheduled-reports)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Living Home connects a local browser workspace to your existing Home Assistant. Select the lights, switches, and sensors it can use, review each proposed change, and save device-status reports on your computer.

The workspace has four views: **Devices**, **Automations**, **Reports**, and **Settings**. A local model can turn an ordinary-language request into an automation for review. OpenClaw can separately run recurring report jobs after its model and Living Home plugin are configured.

> [!NOTE]
> This playbook covers the Living Home workspace. Home Assistant, a local model server, and OpenClaw are separate installations; the relevant prerequisites are listed below.

## What you'll accomplish

- Connect your own Home Assistant and explicitly select the devices to include.
- Review and apply a lamp action, then create and check a device-state automation.
- Save, reopen, and download a local device-status report.
- Optionally connect a local model or prepare recurring reports with OpenClaw.

## What to know before starting

**Required:**

- Basic use of Windows PowerShell and a web browser.
- Access to your own Home Assistant account and familiarity with its device names.
- Ability to check a real device after approving an action.

**Optional:**

- Experience running a local model server for natural-language automation requests.
- Experience configuring OpenClaw plugins and scheduled jobs for recurring reports.

## Supported hardware platforms

| Hardware platform | OS | Memory | Multi-node capable hardware |
| :---- | :---- | :---- | :---- |
| **RTX Spark** | Windows ARM64 | Depends on the optional local model; a workspace-only minimum has not been established | — |

## Prerequisites

**Hardware requirements**

- The hardware platform listed above, with network access to Home Assistant.
- A light, or a lamp on a smart plug, already connected to Home Assistant. Keep it in view for the first test.
- A second selected switch or sensor if you want to test a device-state trigger.
- Storage for the source, Python environment, and local reports. Optional model storage and memory depend on the model you choose.

**Software requirements**

- An existing Home Assistant installation with its account and device setup complete. If needed, follow the [Home Assistant installation guide](https://www.home-assistant.io/installation/) and return once you can control the lamp there.
- A Home Assistant long-lived access token for your account. Saving automations also requires permission to manage automations.
- Git and Python 3.12 or newer for the source launch in the **Instructions** tab. Windows Python also needs the `tzdata` package, installed during setup.
- A current browser and unused local ports **18880** (workspace) and **18881** (household API).
- For optional natural-language requests: a separately running local HTTP model server with `/v1/models` and `/v1/chat/completions` endpoints. Enter its numeric loopback address in Settings, such as `http://127.0.0.1:8000`.
- For optional recurring reports: a configured OpenClaw Gateway and model, plus the Living Home plugin. See the **Scheduled reports** tab.

## Ancillary files

The [source assets](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-living-home/assets/) include the workspace, household API, OpenClaw plugin, report helper, tests, and packaging tools. The source launch does not require a prebuilt installer.

If your deployment provides the complete `LivingHome-Workspace-Windows-ARM64.zip`, you can extract it and open **LivingHome.exe** instead. That package includes Python. Keep the executable with its runtime folder; the launcher is currently unsigned. No hosted package download is configured in this playbook.

## Time & risk

- **Estimated time:** 30 MIN for the workspace walkthrough with Home Assistant and devices already configured. This is a planning estimate; model setup, downloads, and OpenClaw onboarding take additional time.
- **Device changes:** Only approve actions for devices you selected and can check. Saved Home Assistant automations continue running after Living Home quits.
- **Local data:** Credentials, selected devices, and saved reports are stored in `%LOCALAPPDATA%\LivingHome\Household`. Keep this directory and downloaded household configuration files private.
- **Background operation:** Closing the browser tab leaves the workspace running. Use **Quit** to stop it. Recurring reports require the workspace, Home Assistant, model server, and OpenClaw Gateway to remain running.

## Instructions

## Step 1. Check your Home Assistant device

In Home Assistant, open **Settings → Devices & services**. Add your light or smart plug using its [device integration guide](https://www.home-assistant.io/getting-started/integration/) if needed, then return here once the lamp works.

Turn the lamp on and off from Home Assistant and check that the physical lamp responds. Note its name and starting state. For the automation step, also identify a switch or sensor whose state you can change, such as a desk smart plug or motion sensor.

## Step 2. Launch the workspace

Open PowerShell on your hardware platform and verify the source-launch prerequisites:

```powershell
git --version
python --version
```

Confirm that Python reports **3.12 or newer**. If either command is missing, install it before continuing.

Clone the playbooks and create a Python environment for Living Home:

```powershell
git clone https://github.com/NVIDIA/dgx-spark-playbooks
Set-Location dgx-spark-playbooks/nvidia/playbook-living-home/assets
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install tzdata
.\.venv\Scripts\python.exe -c "from zoneinfo import ZoneInfo; print(ZoneInfo('America/Los_Angeles'))"
.\.venv\Scripts\python.exe .\runtime\ui\app.py
```

The timezone check should print `America/Los_Angeles`. The app then opens **Connect your home** in your default browser. Keep its PowerShell window open while using the workspace. The browser opens with a private session link; use that window instead of manually typing the address.

If you already have the complete `LivingHome-Workspace-Windows-ARM64.zip`, extract it into a new folder and open **LivingHome.exe** instead of running the source commands. Python is included in that package. The current launcher is unsigned; keep it beside the extracted runtime folder.

## Step 3. Connect and select your devices

1. Enter your **Home Assistant address**.
2. In your Home Assistant profile, open **Security → Long-lived access tokens**, create a token named Living Home, and paste it into **Access token**. See the [Home Assistant authentication documentation](https://developers.home-assistant.io/docs/auth_api/#long-lived-access-token) for token details.
3. Choose **Connect to Home Assistant**.
4. Search and select your light, the switch or sensor used for the test trigger, and any reporting sensors. Include battery sensors if you want battery observations in reports.
5. Confirm your time zone and choose **Open my home**.

Nothing is selected automatically. If you select a light group, include all its members, including nested members. Device selection is currently fixed at first run; changing it afterward requires manual household configuration.

On **Devices**, compare the lamp name and reported state with Home Assistant. Do not continue with an unavailable or unexpected device.

## Step 4. Review and apply a lamp action

1. On the lamp card, choose **Turn on** or **Turn off**.
2. Review the device and requested action. Choose **Cancel** if either is wrong.
3. Choose **Approve and apply**.
4. Check the physical lamp and the state shown in Home Assistant. Choose **Refresh status** in Living Home if needed.

The result should match the action you approved. If the response is unclear, inspect the device state before repeating the action.

## Step 5. Create and verify an automation

Use the rule builder first; it works without a model.

1. Open **Automations** and name the test rule, for example `Desk lamp test`.
2. Under **When this, do that**, choose your trigger device and the state to watch for, such as a desk plug becoming `on`.
3. Choose your lamp as the target and **Turn on** as the action.
4. Choose **Review rule**, check the trigger and action, and choose **Approve and apply**.
5. In Home Assistant, open **Settings → Automations & scenes**, find the saved rule by name, and check that it is enabled.
6. Cause the actual trigger, such as changing the desk plug from off to on. Watch the lamp, choose **Check trigger status** in Living Home, and inspect the Home Assistant trace.

A saved rule does not establish that its trigger fired. Home Assistant's **Run actions** command skips the trigger; check the actual state transition as well.

### Optional: Generate a plan from your intent

First start a local model server using the [Running Models playbook](https://build.nvidia.com/playbooks/running-llamacpp). Return with its HTTP API listening on this computer and a model available.

1. Open **Settings**, enter the model server origin, for example `http://127.0.0.1:8000`, and choose **Find models**. Use the server's actual port; omit `/v1` from this field.
2. Select the model and choose **Use this model**.
3. In **Automations**, describe your trigger and outcome using the selected device names:

   > When my desk smart plug turns on, turn on my desk lamp. Show me the trigger and action so I can review them.

4. Choose **Create a plan to review**. Check the exact devices, trigger, and action before approving. Cancel and revise any proposal that does not match your intent.
5. Verify the saved automation and its actual trigger as above.

Living Home connects to an existing model server. It does not download or start one. The current connector accepts a numeric loopback HTTP origin without credentials, such as `127.0.0.1` or `[::1]`.

## Step 6. Save and read a device-status report

1. Open **Reports** and choose **Check devices and save report**.
2. Read the collection time, selected-device coverage, availability, and battery observations.
3. Compare several entries with Home Assistant. An off lamp can still be available; an unavailable or missing reading needs investigation.
4. Reopen the saved report, then choose **Download report** if you want a copy.

The report stays on this computer. Its timestamp shows when Home Assistant was checked; reopening a report does not collect fresh observations. A status report alone cannot prove that a physical appliance is working correctly.

## Step 7. Quit and reopen

Choose **Quit** to stop Living Home. Closing a browser tab alone leaves the app running.

For the source launch, run the same `.\.venv\Scripts\python.exe .\runtime\ui\app.py` command from the `assets` directory to reopen it after quitting. For a packaged workspace, open **LivingHome.exe** again. Confirm that your selected devices and saved report are still present.

Your household is saved separately in `%LOCALAPPDATA%\LivingHome\Household`. Updating the application folder preserves it.

## Cleanup

Disable or delete `Desk lamp test` in **Home Assistant → Settings → Automations & scenes** and return the lamp and trigger device to their starting states. Quitting Living Home does not disable Home Assistant automations.

If you enabled a report schedule, disable that job using the **Scheduled reports** tab before stopping the services it needs. Keep your household directory if you intend to resume later; removing the source folder does not remove credentials or reports from it. Revoke the Living Home token in Home Assistant when you permanently stop using the integration.

## Optional — next steps / advanced usage

- Use the **Scheduled reports** tab to configure a daily device-status report with an existing OpenClaw installation.
- Add other Home Assistant integrations, then review how their entities should be included in your household configuration. The preview has no guided add/remove-device screen yet.
- Explore the [source and development checks](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-living-home/assets/CONTRIBUTING.md) before changing the workspace or building a package.

Google records, guided maintenance records, and a one-time natural-language lamp schedule remain future walkthroughs. They are not required to complete this workspace preview.

## Scheduled reports

## Before you start

> [!NOTE]
> The helper targets the OpenClaw 2026.9.4 CLI contract recorded in its source. Check your installed version against the [OpenClaw cron reference](https://docs.openclaw.ai/cli/cron) before continuing.

Scheduled reports require a running OpenClaw Gateway, a configured model, and the [Living Home plugin](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-living-home/assets/runtime/openclaw-adapter/README.md) connected to this household. Complete those prerequisites before enabling a schedule. This setup uses PowerShell; the workspace prepares the schedule file.

Use the [Getting Started with Agents playbook](https://build.nvidia.com/playbooks/start-with-agent) for OpenClaw setup and return with a working model and Gateway. This helper uses the native Windows `node.exe` and `openclaw.mjs`; a separate WSL installation needs its own path and connectivity setup.

Follow the [OpenClaw plugin configuration guide](https://docs.openclaw.ai/tools/plugin) to load the local `runtime/openclaw-adapter` directory in the household's OpenClaw profile, enable plugin ID `living-home`, and allow its tools. Configure the following values for that plugin using absolute paths:

| Setting | Source launch | Packaged workspace |
| --- | --- | --- |
| `baseUrl` | `http://127.0.0.1:18881` | `http://127.0.0.1:18881` |
| `pythonExecutable` | `<assets>\.venv\Scripts\python.exe` | `<app>\runtime\python\python.exe` |
| `collectorPath` | `<assets>\runtime\health\collect_home_status.py` | `<app>\runtime\health\collect_home_status.py` |
| `dataDirectory` | Your expanded `%LOCALAPPDATA%\LivingHome\Household` path | Your expanded `%LOCALAPPDATA%\LivingHome\Household` path |
| `apiTokenFile` | `api-token.txt` inside that household directory | `api-token.txt` inside that household directory |

Replace `<assets>` or `<app>` with the real absolute directory. Expand `%LOCALAPPDATA%` to its actual path in the configuration; these are descriptions, not literal configuration values. Use the API port you selected if you changed it. The backend token file differs from the Home Assistant access token.

Restart the configured Gateway to load the plugin. Before creating a schedule, confirm that the agent can call `living_home_property` to collect selected-device status and save and read back a local health report. Keep the report's collection time and coverage visible. Resolve any plugin or model error first.

Keep Living Home, Home Assistant, the model server and the OpenClaw Gateway running at the scheduled time. The current workspace does not install a Windows service or start automatically when you sign in.

## Step 1. Prepare the schedule

1. In **Reports**, choose the daily time and time zone.
2. Enter your OpenClaw Gateway port.
3. Choose **Prepare schedule**, then **Save schedule configuration**.
4. Keep the downloaded `home-report.json` in a private folder.

The saved file contains your household's OpenClaw profile, Gateway address and schedule. Preparing it does not create or enable a job.

## Step 2. Review and create the job

Open a new PowerShell window. Set these paths to your source `assets` directory, downloaded configuration, and existing native Windows OpenClaw installation:

```powershell
$app = 'C:\Path\To\playbook-living-home\assets'
$reportConfig = 'C:\Path\To\home-report.json'
$node = 'C:\Path\To\node.exe'
$openclaw = 'C:\Path\To\openclaw.mjs'
$python = Join-Path $app '.venv\Scripts\python.exe'

$reportSettings = Get-Content -LiteralPath $reportConfig -Raw | ConvertFrom-Json
$profile = $reportSettings.openclawProfile
$gateway = $reportSettings.gatewayUrl

& $python "$app\workflows\configure_home_report.py" --config $reportConfig
```

For a packaged workspace, set `$app` to its extracted directory and `$python` to `Join-Path $app 'runtime\python\python.exe'` instead.

Check the time zone, schedule and OpenClaw profile in the preview. Confirm that this profile and Gateway belong to the household you connected. Reports are local by default.

Create the job:

```powershell
& $python "$app\workflows\configure_home_report.py" --config $reportConfig --node $node --openclaw $openclaw --apply
```

A new job is created disabled. Changing an existing job disables it for review; applying an unchanged configuration preserves its current enabled state. Setup does not run the job.

## Step 3. Run it once and read the report

Copy the `jobId` returned by setup:

```powershell
$jobId = '<jobId returned by setup>'
& $node $openclaw --profile $profile cron show $jobId --json --url $gateway
& $node $openclaw --profile $profile cron run $jobId --wait --url $gateway
& $node $openclaw --profile $profile cron runs $jobId --limit 5 --json --url $gateway
```

In Living Home, choose **Refresh status**, open **Reports**, and read the new report. Check its collection time and device coverage. Resolve a failed run or missing report before enabling the schedule.

The job collects device status and asks the model to write and save a local report. It does not control devices or send email.

## Step 4. Enable recurring reports

```powershell
& $node $openclaw --profile $profile cron enable $jobId --json --url $gateway
& $node $openclaw --profile $profile cron show $jobId --json --url $gateway
```

Check the next run time. After that time, confirm a new report appears.

## Cleanup and schedule changes

To pause:

```powershell
& $node $openclaw --profile $profile cron disable $jobId --json --url $gateway
```

To change the time or time zone, prepare a new configuration and repeat setup for the same household. Run the changed job once, inspect its report, and enable it again.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Home Assistant connection fails | The address is unreachable, or the token is invalid | Open the address in your browser, confirm Home Assistant is running, and create a new long-lived access token if needed. Retry on the connection screen. |
| No devices appear | Home Assistant has no matching devices connected | Add and test the devices in Home Assistant first, then reconnect Living Home. |
| A device is unavailable | Home Assistant cannot currently read its state | Check the integration and device connection in Home Assistant, then choose **Refresh status**. |
| A group cannot be controlled | One or more nested members were not selected | Review the group's membership and permitted selection. All members must be included; the preview requires manual configuration to change the selection after setup. |
| A rule saves but does not run | The actual trigger has not occurred, the rule is disabled, or its condition does not match | Check the saved automation in Home Assistant, cause the expected state transition, choose **Check trigger status**, and inspect the trace. **Run actions** alone does not test the trigger. |
| A device action has no clear result | The response was lost or the device state has not refreshed | Inspect the device and Home Assistant state before retrying the action. |
| **Find models** fails | The model server is stopped or the address is unsupported | Start the server and enter its numeric loopback HTTP origin with its actual port, for example `http://127.0.0.1:8000`. Omit `/v1`; the workspace adds API paths. |
| Natural-language requests are unavailable | No local model has been selected | Connect one in **Settings**, or use the rule builder and direct report checks without a model. |
| `ZoneInfoNotFoundError` appears | Windows Python has no IANA timezone data | From the source `assets` directory, run `.\.venv\Scripts\python.exe -m pip install tzdata`, then restart the app. |
| Opening the workspace address manually gives an authorization error | The browser does not have the private session link | Use the browser window opened by the launcher. For the source launch, quit the existing instance and start it again to open a new authenticated window. |
| A daily report does not run | Only the configuration was prepared, a dependency is stopped, or the job is disabled | Complete the OpenClaw plugin/model setup, run the job once, inspect its saved report, and then enable it. Keep the workspace and required services running. |
| OpenClaw cannot use Living Home | The API port, plugin paths, household directory, or backend token file is wrong | Check the plugin configuration against the running workspace. Use the household's `api-token.txt`, which is different from the Home Assistant token. |
| A saved household will not open | Its configuration or local files are incomplete | Preserve `%LOCALAPPDATA%\LivingHome\Household` and have your deployment administrator inspect it. Reinstalling the application does not reset household data. |
| The launcher is missing after copying files | Only part of the package was copied, or you have the source checkout | Extract the complete workspace ZIP if available, or use the source launch in **Instructions**. The source checkout does not contain compiled executables. |

## If the default ports are busy

Choose two different unused ports. From the source `assets` directory, run:

```powershell
.\.venv\Scripts\python.exe .\runtime\ui\app.py --port 18890 --api-port 18891
```

For a packaged workspace, use `.\runtime\python\python.exe` instead of `.\.venv\Scripts\python.exe`. Update the OpenClaw plugin's `baseUrl` to the same household API port if you use scheduled reports.

## If OpenClaw commands differ

The report helper targets the OpenClaw **2026.9.4** CLI contract recorded in the supplied source. Check your installed version and its `cron --help` against the [OpenClaw cron reference](https://docs.openclaw.ai/cli/cron). Resolve unsupported flags before applying a schedule. Do not enable a job until a manual run has saved a report you can read back.
