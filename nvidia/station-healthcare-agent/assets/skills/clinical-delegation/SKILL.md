---
name: clinical-delegation
description: How to delegate clinical tasks to specialist agents. Always use sub-agent runtime with explicit agentId — never ACP. Never call FHIR via web_fetch.
metadata:
  openclaw:
    requires:
      bins: ["python3"]
---

# Clinical delegation (sessions_spawn)

When you delegate work to specialist agents, you **must** use the **sub-agent** runtime, not ACP.

## Do not call FHIR yourself

**Do not use `web_fetch` or any HTTP tool to call FHIR APIs from the coordinator.** The coordinator does not have the correct FHIR base URL in context when it invents URLs. Specialist agents (patient-data, labs-vitals, medications, analyst) have the configured FHIR endpoint and write Python that runs on the server.

- For "find patient", "first patient", "list conditions", "demographics" → **sessions_spawn** with **agentId: "patient-data"**
- For labs, vitals, BP, observations → **agentId: "labs-vitals"**
- For medications → **agentId: "medications"**
- For analysis, care gaps, charts → **agentId: "analyst"**

The only FHIR server in use is **https://r4.smarthealthit.org**. Never use placeholder domains like fhir.example.com.

## Rule

Use `sessions_spawn` with:

- **agentId** — set to the specialist agent id: `patient-data`, `labs-vitals`, `medications`, `analyst`, or `molecular`
- **task** — clear description of the task
- **Do not set** `runtime: "acp"`. Omit `runtime` or use `runtime: "subagent"` so the request goes to the correct clinical agent.

## Correct examples

```json
{ "agentId": "patient-data", "task": "Find the first patient and return demographics and active conditions.", "mode": "run" }
```

```json
{ "agentId": "labs-vitals", "task": "Get latest HbA1c and eGFR for patient abc123", "mode": "run" }
```

```json
{ "agentId": "analyst", "task": "Run care gap analysis for diabetic patients with A1c > 9%", "mode": "run" }
```

## Wrong

- Using **web_fetch** to call FHIR (e.g. https://fhir.example.com/Patient) — coordinator must delegate instead; fhir.example.com is not a real server and will fail.
- `"runtime": "acp"` without agentId — causes "ACP target agent is not configured"
- Omitting agentId when delegating — the system will not know which specialist to run

## Specialist agents

| agentId        | Use for |
|----------------|---------|
| patient-data   | Find patients, get demographics, list conditions |
| labs-vitals    | Labs, vitals, BP, observations |
| medications    | Active medications, drug classes |
| analyst       | Python analysis, care gaps, charts |
| molecular     | Drug molecular structure, OpenFold3 NIM 3D visualization |
