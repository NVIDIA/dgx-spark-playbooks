---
name: cohort-compare
description: Analyze a cohort of patients from FHIR endpoints to find care gaps and patterns. Use when asked to compare patients, find quality gaps, or analyze a population.
metadata:
  openclaw:
    requires:
      bins: ["python3"]
---

Analyze a patient cohort: $ARGUMENTS

Use your fhir-basics skill to query FHIR endpoints. Use your clinical-knowledge skill to identify care gaps and apply correct thresholds. Use your analysis-methods skill to write correct Python analysis code.

## Execution Rules

- Do NOT explore the workspace or list files. Begin the analysis immediately.
- Write ONE Python script that does everything: FHIR queries, analysis, chart, and summary.
- Write the script correctly the first time. Do NOT write a draft and then edit it.
- Run it with `python` (NOT `python3`).
- All HTTP calls must use `subprocess.run(["curl", "-sf", "--max-time", "30", url], capture_output=True, text=True)` -- the `requests` library does NOT work through the sandbox proxy. See the fhir-basics skill for the `fhir_get` helper pattern.
- Save the script to `/tmp/<name>.py`, run it once, interpret the output.

## Steps

1. **Identify the cohort** -- Query `GET /Condition?code={snomed_code}&_count=200` and follow pagination links to get all matching Condition resources. Extract unique patient IDs from `entry[].resource.subject.reference`. Report the cohort size.

2. **Pull clinical data in BATCHED queries (do NOT loop per-patient)**:
   - **Lab values**: Use `get_latest_labs_batch(loinc_code, patient_ids)` to fetch ALL observations for the LOINC code in one call and filter client-side. This queries `GET /Observation?code={loinc_code}&_count=500&_sort=-date` without a patient filter, then builds a dict keyed by patient ID. Handle both `valueQuantity` (numeric) and `valueString` (text) formats. For blood pressure, query the BP panel code `85354-9` in batch and parse components.
   - **Medications**: Use `get_all_medications_batch(patient_ids)` to fetch `GET /MedicationRequest?status=active&_count=500` in one call, then filter to cohort patients client-side.
   - **NEVER write a `for pid in patient_ids:` loop that makes FHIR HTTP calls inside the loop.** The sandbox proxy adds 1-3s latency per call. With 24 patients x 4 LOINC codes = 96 calls = 5+ minutes. Batching brings this to 4-6 total calls = 30 seconds.

3. **Build a pandas DataFrame** with one row per patient:
   - `patient_id` (string)
   - `{lab_name}` (float or None)
   - `lab_date` (string)
   - `on_target_med` (boolean -- True if the patient is on the specified medication class)
   - `medications` (comma-separated string of all active med names)
   - `med_count` (int)

4. **Data quality check**:
   - Report how many patients have the lab recorded vs. missing
   - If > 30% missing, flag as a data quality issue but continue analysis
   - If < 5 patients have data, warn that the sample is too small for meaningful statistics

5. **Identify care gap patients**: Apply the threshold and medication check:
   - Gap = lab value exceeds threshold AND patient is NOT on the specified medication class
   - Report: total with condition, total with lab data, total above threshold, total in care gap
   - Compute gap rate as percentage of patients with lab data (not total cohort)

6. **Generate visualization**:
   - Histogram of the lab value distribution with threshold line
   - Use NVIDIA dark theme: `plt.style.use('dark_background')`, primary color `#76B900`, background `#1a1a1a`
   - Annotate with sample size (N = ...) and gap count
   - Save as PNG with `dpi=150`

7. **Write a plain-English summary** including:
   - Cohort size and data completeness
   - Distribution statistics (mean, median, range)
   - Gap patient count and percentage with absolute numbers
   - Comparison to relevant CMS quality measure if applicable
   - Any notable patterns (e.g., "All gap patients had no medications recorded at all")

8. **Disclaimer**: "This analysis is for research and operational purposes. Clinical decisions should be made by qualified clinicians."

## Example: Diabetes Gap Analysis (CMS122)

Condition: Type 2 Diabetes (SNOMED 44054006)
Lab: HbA1c (LOINC 4548-4)
Threshold: > 9.0%
Gap medication: insulin or GLP-1 agonist
Quality measure: CMS122v12 (poor glycemic control)

```python
INSULIN_AND_GLP1 = ["insulin", "liraglutide", "semaglutide", "dulaglutide",
                     "exenatide", "tirzepatide", "victoza", "ozempic",
                     "trulicity", "byetta", "mounjaro", "rybelsus"]

def is_on_insulin_or_glp1(med_list):
    med_lower = [m.lower() for m in med_list]
    return any(drug in med_text for drug in INSULIN_AND_GLP1 for med_text in med_lower)
```

## Example: Hypertension Gap Analysis (CMS165)

Condition: Essential Hypertension (SNOMED 38341003)
Lab: Systolic BP (LOINC 8480-6) -- **use component Observation pattern**
Threshold: >= 140 mmHg
Gap medication: any antihypertensive
Quality measure: CMS165v12 (controlling high blood pressure)

Note: Use `get_latest_bp()` from the analysis-methods skill to handle both BP panel (85354-9) and standalone systolic Observations.

```python
ANTIHYPERTENSIVES = ["lisinopril", "enalapril", "ramipril", "benazepril",
    "losartan", "valsartan", "irbesartan", "olmesartan", "telmisartan",
    "amlodipine", "nifedipine", "diltiazem",
    "metoprolol", "atenolol", "carvedilol", "bisoprolol",
    "hydrochlorothiazide", "hctz", "chlorthalidone",
    "furosemide", "spironolactone"]

def is_on_antihypertensive(med_list):
    med_lower = [m.lower() for m in med_list]
    return any(drug in med_text for drug in ANTIHYPERTENSIVES for med_text in med_lower)
```
