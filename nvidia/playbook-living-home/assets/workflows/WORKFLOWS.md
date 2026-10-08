# Living Home playbook

Start with [Set up Living Home](../runtime/ui/START-HERE.md). The steps below use the devices you selected during setup.

## Control a device

1. Open **Devices** and choose **Turn on** or **Turn off**.
2. Review the device and requested action.
3. Choose **Approve and apply**.
4. Check the device, then use **Refresh status** if needed.

## Create an automation

### Describe your intent

This option requires an OpenAI-compatible model server already running on the same computer.

1. Open **Settings** and enter the local model server address.
2. Choose **Find models**, select a model, and choose **Use this model**.
3. Open **Automations** and describe the trigger and outcome using your device names. For example: “When the hallway motion sensor turns on, turn on the desk lamp.”
4. Choose **Create a plan to review**.
5. Check the proposed devices, trigger and action. If they do not match your intent, cancel and revise your request.
6. Choose **Approve and apply**.
7. Trigger the condition, choose **Check trigger status**, and check the device.

Living Home proposes changes for your approval. It does not download or start the model server.

### Build a simple rule

The rule builder works without a model.

1. Open **Automations** and give the rule a name.
2. Select the triggering device and the state to watch for, such as `on`.
3. Select the device to change and choose **Turn on** or **Turn off**.
4. Choose **Review rule**, check the proposal, and choose **Approve and apply**.
5. Trigger the condition and choose **Check trigger status**.

Saved automations run in Home Assistant, including after you quit Living Home. To pause or remove one, disable or delete it in Home Assistant.

## Check device status and save a report

1. Open **Reports**.
2. Choose **Check devices and save report**.
3. Read the saved report. It includes availability and battery observations for your selected devices.
4. Choose a previous report to reopen it, or choose **Download report** to save a copy.

Reports stay on this computer. Their timestamps show when Home Assistant was checked. An unavailable reading needs investigation; it does not by itself establish a hardware fault.

## Set up a daily status report

Scheduled reports require a running OpenClaw Gateway, a configured model, and the [Living Home plugin](../runtime/openclaw-adapter/README.md) connected to this household. Complete those prerequisites before enabling a schedule. This setup uses PowerShell; the workspace prepares the schedule file.

Keep Living Home, Home Assistant, the model server and the OpenClaw Gateway running at the scheduled time. The current workspace does not install a Windows service or start automatically when you sign in.

### 1. Prepare the schedule

1. In **Reports**, choose the daily time and time zone.
2. Enter your OpenClaw Gateway port.
3. Choose **Prepare schedule**, then **Save schedule configuration**.
4. Keep the downloaded `home-report.json` in a private folder.

The saved file contains your household's OpenClaw profile, Gateway address and schedule. Preparing it does not create or enable a job.

### 2. Review and create the job

Open PowerShell. Set these paths to your extracted app, downloaded configuration and existing OpenClaw installation:

```powershell
$app = 'C:\Path\To\LivingHome-Workspace'
$reportConfig = 'C:\Path\To\home-report.json'
$node = 'C:\Path\To\node.exe'
$openclaw = 'C:\Path\To\openclaw.mjs'
$python = Join-Path $app 'runtime\python\python.exe'

$reportSettings = Get-Content -LiteralPath $reportConfig -Raw | ConvertFrom-Json
$profile = $reportSettings.openclawProfile
$gateway = $reportSettings.gatewayUrl

& $python "$app\workflows\configure_home_report.py" --config $reportConfig
```

Check the time zone, schedule and OpenClaw profile in the preview. Confirm that this profile and Gateway belong to the household you connected. Reports are local by default.

Create the job:

```powershell
& $python "$app\workflows\configure_home_report.py" --config $reportConfig --node $node --openclaw $openclaw --apply
```

A new job is created disabled. Changing an existing job disables it for review; applying an unchanged configuration preserves its current enabled state. Setup does not run the job.

### 3. Run it once and read the report

Copy the `jobId` returned by setup:

```powershell
$jobId = '<jobId returned by setup>'
& $node $openclaw --profile $profile cron show $jobId --json --url $gateway
& $node $openclaw --profile $profile cron run $jobId --wait --url $gateway
& $node $openclaw --profile $profile cron runs $jobId --limit 5 --json --url $gateway
```

In Living Home, choose **Refresh status**, open **Reports**, and read the new report. Check its collection time and device coverage. Resolve a failed run or missing report before enabling the schedule.

The job collects device status and asks the model to write and save a local report. It does not control devices or send email.

### 4. Enable recurring reports

```powershell
& $node $openclaw --profile $profile cron enable $jobId --json --url $gateway
& $node $openclaw --profile $profile cron show $jobId --json --url $gateway
```

Check the next run time. After that time, confirm a new report appears.

### Pause or change the schedule

To pause:

```powershell
& $node $openclaw --profile $profile cron disable $jobId --json --url $gateway
```

To change the time or time zone, prepare a new configuration and repeat setup for the same household. Run the changed job once, inspect its report, and enable it again.
