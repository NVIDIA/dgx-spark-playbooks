# Set up Living Home

You need a Windows ARM64 computer, the complete Living Home workspace ZIP, and an existing Home Assistant with your devices connected. Python is included. A model and OpenClaw are separate, optional installations.

## Connect your home

1. Extract the ZIP into a new folder and open **LivingHome.exe**. Keep the executable beside its runtime folder. The executable is currently unsigned.
2. Enter your **Home Assistant address**.
3. In Home Assistant, open your profile → **Security** → **Long-lived access tokens**. Create a token named Living Home and paste it into **Access token**.
4. Choose **Connect to Home Assistant**.
5. Search the device list and select the devices you want Living Home to use. Include battery sensors if you want battery status in reports.
6. Confirm the time zone and choose **Open my home**.

Nothing is selected automatically. Choose your devices carefully: changing this selection later requires manual configuration.

## Try one device

On **Devices**, choose **Turn on** or **Turn off**. Review the device and action, then choose **Approve and apply**. Check the device to confirm the result.

For automations, reports and daily checks, follow the [playbook](../../workflows/WORKFLOWS.md).

## Reopen and quit

Open **LivingHome.exe** to return to your workspace. Choose **Quit** to stop it. Closing the browser tab alone leaves it running.

Your credentials, selected devices and reports are saved in `%LOCALAPPDATA%\LivingHome\Household`, separately from the application. Replacing the application folder preserves this data. Keep backups private.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| Cannot connect to Home Assistant | Confirm its address, check that it is running, and create a new token if yours is invalid. |
| No devices appear | Add your devices to Home Assistant first, then reconnect. |
| A device is unavailable | Check its connection in Home Assistant, then choose **Refresh status**. |
| A group cannot be controlled | Select all of its members during setup, including members of nested groups. |
| A rule saves but does not run | Trigger its condition, choose **Check trigger status**, and check that the rule is enabled in Home Assistant. |
| Requests in your own words are unavailable | Connect an existing local model in **Settings**, or use the rule builder. |
| A daily report does not run | Complete the OpenClaw setup, run the job once, then enable it. **Prepare schedule** alone does not enable recurring reports. |
| An action gives no clear result | Refresh the device state and check automation status before repeating the action. |
| The saved household will not open | Preserve the household folder and contact your deployment administrator. Reinstalling the app does not reset it. |

Home Assistant permissions must allow access to the selected devices and, for saved rules, automation configuration.

### If the default ports are busy

Living Home uses ports 18880 and 18881 on this computer. Close the other application if appropriate, or open PowerShell in the extracted Living Home folder and choose different unused ports:

```powershell
.\runtime\python\python.exe .\runtime\ui\app.py --port 18890 --api-port 18891
```

If you use OpenClaw, update its Living Home plugin connection to the same API port.
