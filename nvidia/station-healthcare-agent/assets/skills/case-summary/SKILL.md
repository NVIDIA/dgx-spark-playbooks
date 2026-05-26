---
name: case-summary
description: Prepare a complete clinical case summary for a patient from FHIR endpoints. Use when asked to summarize a patient, compile a case, or prepare for tumor board.
metadata:
  openclaw:
    requires:
      bins: ["python3"]
---

Prepare a complete case summary for $ARGUMENTS

Use your fhir-basics skill to query the FHIR endpoints. Use your clinical-knowledge skill to flag abnormal values and identify relevant clinical context.

## Steps

1. **Find the patient** -- Query `GET /Patient?name={name}&_count=5` or `GET /Patient?_count=1` for "first patient". Extract `id`, full name, birthDate, gender.

2. **Get active conditions** -- Query `GET /Condition?patient={id}&clinical-status=active`. Extract each condition's display name, SNOMED/ICD code, onset date, and verification status. Also query for resolved conditions and list them separately (they provide clinical history context).

3. **Get recent labs** -- Query `GET /Observation?patient={id}&category=laboratory&_sort=-date&_count=50`. "Recent" means the most recent value for each distinct LOINC code within the past 12 months. For each lab, report: name, value, unit, date, and whether it's normal/abnormal per the clinical-knowledge skill. If the Observation includes a `referenceRange`, use that for flagging.

4. **Get recent vitals** -- Query `GET /Observation?patient={id}&category=vital-signs&_sort=-date&_count=20`. For blood pressure, handle the component Observation format (LOINC 85354-9 panel with systolic/diastolic in `component[]`). Report the most recent BP, heart rate, BMI, temperature.

5. **Get current medications** -- Query `GET /MedicationRequest?patient={id}&status=active`. For each medication, report: drug name, dosage text, and drug class (per clinical-knowledge skill). Organize by drug class when possible.

6. **Get recent encounters** (optional but adds context) -- Query `GET /Encounter?patient={id}&_sort=-date&_count=5`. Report type, date, and reason if available. This shows how recently the patient was seen.

7. **Compile the case summary** with these sections:
   - **Demographics**: Name, age (calculated from birthDate), gender, address if available
   - **Active Conditions**: With codes and onset dates
   - **Resolved Conditions** (if any): Brief list for clinical history
   - **Recent Labs**: Grouped by category (metabolic, lipids, renal, hematology), abnormal values highlighted
   - **Recent Vitals**: Most recent BP, HR, BMI
   - **Current Medications**: With drug class annotations
   - **Clinical Flags**: Any abnormal values, potential care gaps (e.g., diabetic patient without A1c in 12 months, hypertensive with uncontrolled BP), or comorbidity patterns worth noting

8. **Disclaimer**: "This summary is auto-generated from FHIR data for research and operational use. Verify all information before clinical decision-making."

## Output Format

```
============================================================
PATIENT CASE SUMMARY
============================================================
Generated: {date/time}
Source: {FHIR endpoint URL}

DEMOGRAPHICS
  Name:     {full name}
  Age:      {age} years (DOB: {birthDate})
  Gender:   {gender}

ACTIVE CONDITIONS ({count})
  1. {display} (SNOMED {code}) -- onset {date}
  2. ...

RESOLVED CONDITIONS ({count})
  1. {display} -- resolved

RECENT LABS (most recent per test, past 12 months)
  Metabolic:
    HbA1c:          {value}% ({date})     ⚠ ABOVE TARGET (>7.0%)
    Fasting Glucose: {value} mg/dL ({date}) -- Normal
  Renal:
    Creatinine:     {value} mg/dL ({date}) -- Normal
    eGFR:           {value} mL/min ({date}) -- Normal
  Lipids:
    LDL:            {value} mg/dL ({date})  ⚠ ELEVATED (>100)
    ...

RECENT VITALS
  Blood Pressure: {systolic}/{diastolic} mmHg ({date})
  Heart Rate:     {value} bpm ({date})

CURRENT MEDICATIONS ({count})
  Diabetes:
    - Metformin 1000mg twice daily
  Antihypertensive:
    - Lisinopril 20mg daily
  Statin:
    - Atorvastatin 40mg daily

CLINICAL FLAGS
  ⚠ HbA1c 9.2% indicates poor glycemic control (CMS122 gap)
  ⚠ LDL 165 mg/dL above target for diabetic patient
  ✓ On statin therapy (appropriate for diabetes + elevated LDL)
  ✓ On ACE inhibitor (appropriate for diabetes + hypertension)

============================================================
Disclaimer: This summary is auto-generated from FHIR data
for research and operational use. Verify all information
before clinical decision-making.
============================================================
```
