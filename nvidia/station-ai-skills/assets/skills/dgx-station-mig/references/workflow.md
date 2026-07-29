# MIG command workflow

```text
scripts/dgx-assist mig inspect --json
scripts/dgx-assist mig profiles --json
scripts/dgx-assist playbook search "GB300 MIG requested concern" --json
scripts/dgx-assist mig plan --layout "DRIVER_PROFILE_IDS_OR_NAMES" --json
scripts/dgx-assist mig apply --plan-id PLAN_ID --dry-run --json
```

Explain that nominal profile memory does not prove a combination has legal
placement. The driver profile and placement evidence is authoritative.

Before approval, report:

- selected GB300 UUID and current mode/instances;
- exact requested driver profile IDs and names;
- active clients;
- every proposed command;
- privilege, disruption, and reset/reboot possibility;
- restoration commands.

After approval:

```text
scripts/dgx-assist mig apply --plan-id PLAN_ID --yes --json
```

If the plan is stale, create a new plan. If apply partially fails, perform no
additional mutation and present the receipt's restoration commands.

