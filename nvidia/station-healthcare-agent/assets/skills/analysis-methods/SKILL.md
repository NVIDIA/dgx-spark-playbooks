---
name: analysis-methods
description: Teaches the analyst agent how to write correct, robust Python analysis code for FHIR clinical data using pandas, matplotlib, and scipy.
metadata:
  openclaw:
    requires:
      bins: ["python3", "pip"]
---

# Analysis Code Guidelines

## FHIR Helpers Library

**Always import the helpers library at the top of every analysis script:**

```python
import sys
sys.path.insert(0, '/sandbox/clinical-intelligence/skills/analysis-methods/scripts')
from fhir_helpers import *
```

### Available functions

| Function | Use for | HTTP calls |
|----------|---------|------------|
| `get_patients_with_condition(snomed_code)` | Find patients with a condition → list of IDs | 1-2 |
| `get_latest_labs_batch(loinc_code, patient_ids)` | Labs for a cohort → dict: pid → (value, unit, date) | 1-2 |
| `get_all_medications_batch(patient_ids)` | Meds for a cohort → dict: pid → [med names] | 1-2 |
| `build_cohort_df(patient_ids, loinc, lab_name, drug_check_fn)` | Full DataFrame with labs + meds | 2-3 |
| `get_latest_lab(patient_id, loinc_code)` | Lab for ONE patient → (value, unit, date) | 1 |
| `get_medications(patient_id)` | Meds for ONE patient → [names] | 1 |
| `get_latest_bp(patient_id)` | BP for ONE patient → (sys, dia, date) | 1-2 |
| `check_drug_class(med_list, drug_names)` | Check if any med matches drug list → bool | 0 |
| `fhir_get(path, params)` | Raw FHIR GET → parsed JSON | 1 |
| `get_all_pages(path, params)` | Paginated FHIR GET → all entries | 1+ |
| `save_chart_to_canvas(fig, filename)` | Save matplotlib figure to canvas directory | 0 |

### Performance rules

- **Cohort queries (2+ patients):** Use `get_latest_labs_batch()` and `get_all_medications_batch()`. These make 1-2 HTTP calls total regardless of patient count.
- **Single patient:** Use `get_latest_lab()`, `get_medications()`, `get_latest_bp()`.
- **NEVER** loop over patients calling `get_latest_lab()` per patient. Each HTTP call through the sandbox proxy adds 1-3s. For 48 patients = 48 calls = 2+ minutes. The batch function does it in one call.

## Execution Rules

- Run scripts with `python` (NOT `python3`)
- Write a SINGLE Python script for the entire task
- Write the script to `/tmp/<name>.py`, then execute it
- All HTTP inside the sandbox must use `subprocess.run(["curl", ...])` — the `requests` library does NOT work

## Mandatory Workflow

```
STEP 1 - WRITE SCRIPT (import fhir_helpers, write analysis)
STEP 2 - VALIDATE: python /sandbox/clinical-intelligence/scripts/validate_and_run.py --validate-only /tmp/<name>.py
STEP 3 - EXECUTE: python /tmp/<name>.py
STEP 4 - INTERPRET: explain results using clinical-knowledge skill
```

## Code Structure

1. Imports (always start with fhir_helpers import)
2. Data collection (use batch functions)
3. DataFrame construction
4. Analysis (filters, aggregations)
5. Visualization -- use `save_chart_to_canvas(fig, filename)` (NOT plt.savefig)
6. Summary (print findings)
7. Disclaimer

## Care Gap Analysis Pattern

```python
# Example: diabetes care gap
patients = get_patients_with_condition("44054006")  # SNOMED for diabetes
df = build_cohort_df(patients, "4548-4", "HbA1c",
                     lambda meds: check_drug_class(meds, ["metformin", "insulin", "glipizide"]))

gap = df[(df['HbA1c'] > 9) & (~df['on_target_med'])]
denom = len(df[df['HbA1c'].notna()])
pct = f"{len(gap)/denom*100:.1f}%" if denom > 0 else "N/A (no HbA1c data)"
print(f"Care gap: {len(gap)}/{denom} ({pct})")
```

## Visualization

Always use dark theme. Use `save_chart_to_canvas()` instead of `plt.savefig()` directly.

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')

# Histogram with NVIDIA green
ax.hist(values, bins=15, color='#76B900', edgecolor='#1a1a1a', alpha=0.85)
ax.axvline(x=threshold, color='#ff4444', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
ax.set_title("Title", fontsize=14, fontweight='bold', color='white')
ax.legend()
ax.grid(axis='y', alpha=0.2, color='#444444')
ax.text(0.98, 0.95, f"N = {len(values)}", transform=ax.transAxes, fontsize=11, color='#888888', ha='right', va='top')

# MANDATORY: use save_chart_to_canvas (NOT plt.savefig)
save_chart_to_canvas(fig, "chart.png")
plt.close()
```

## Guardrails

- Never compute statistics on fewer than 5 data points
- Always report sample size: "45.0% (27 out of 60)"
- Flag data quality issues if >30% missing
- Do not fabricate data — report what exists, flag what's missing
- All charts must include N annotation

## Output Format

End every script with:

```python
print(f"\nDisclaimer: This analysis is for research and operational purposes.")
print("Clinical decisions should be made by qualified clinicians.")
```
