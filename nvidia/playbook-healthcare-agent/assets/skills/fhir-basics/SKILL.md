---
name: fhir-basics
description: Teaches agents how FHIR R4 APIs work, what resources are available, how to query them with search parameters, and how to correctly parse all response formats including component Observations.
metadata:
  openclaw:
    requires:
      bins: ["python3"]
---

# FHIR Data Retrieval

**Important**: In this sandbox, run Python scripts with `python` (not `python3`). Use `subprocess.run(["curl", "-sf", url], capture_output=True, text=True)` for all FHIR HTTP calls — the `requests` library does NOT work through the sandbox proxy. Always parse the output with `json.loads()`.

FHIR (Fast Healthcare Interoperability Resources) R4 is the standard API format mandated by the 21st Century Cures Act for US healthcare interoperability. ~70% of US hospitals expose FHIR R4 endpoints (ONC 2024). All queries use REST GET requests returning JSON Bundles.

## Default FHIR Endpoint

Unless the user specifies a different FHIR server, always use: `https://r4.smarthealthit.org`

This is the SMART on FHIR public test server with synthetic (Synthea) patient data. No authentication required.

**Query format for this server**: Use bare codes without system URIs. Example: `code=44054006` NOT `code=http://snomed.info/sct|44054006`. The test server does not support system-qualified code searches and will return empty results.

## Authentication

- **Public test servers** (e.g., `https://r4.smarthealthit.org`): No authentication required. Synthetic data, real FHIR format.
- **Production hospital endpoints**: Use SMART on FHIR OAuth2 flows. Requires a `client_id`, redirect URI, and scope negotiation. Access tokens are passed as `Authorization: Bearer {token}` headers.

## Resource Endpoints and Search Parameters

### Patient

```
GET /Patient                              -- all patients (paginated)
GET /Patient?name=Smith                   -- search by family or given name
GET /Patient?name=John&name=Smith         -- search by given AND family name
GET /Patient?birthdate=1970-01-01         -- exact birthdate
GET /Patient?birthdate=ge1960-01-01&birthdate=le1970-12-31  -- date range
GET /Patient?gender=male                  -- filter by gender
GET /Patient/{id}                         -- get a specific patient by ID
GET /Patient?_count=50                    -- control page size
```

### Condition (Diagnoses)

```
GET /Condition?patient={id}                           -- all conditions for a patient
GET /Condition?patient={id}&clinical-status=active     -- only active conditions
GET /Condition?code={snomed_code}                      -- all patients with a condition (cohort query)
GET /Condition?code=44054006&_count=200                -- paginated cohort (bare code for test server)
```

For the default test server (`r4.smarthealthit.org`), always use bare codes (e.g., `code=44054006`). Production servers may require the full system URI (`code=http://snomed.info/sct|44054006`).

### Observation (Labs and Vitals)

```
GET /Observation?patient={id}                            -- all observations
GET /Observation?patient={id}&code={loinc}               -- specific lab by LOINC
GET /Observation?patient={id}&code={loinc}&_sort=-date&_count=1  -- most recent only
GET /Observation?patient={id}&category=vital-signs       -- vitals only
GET /Observation?patient={id}&category=laboratory        -- labs only
GET /Observation?patient={id}&date=ge2023-01-01          -- after a date
```

### MedicationRequest (Prescriptions)

```
GET /MedicationRequest?patient={id}                  -- all prescriptions
GET /MedicationRequest?patient={id}&status=active    -- current prescriptions only
GET /MedicationRequest?patient={id}&_count=100       -- increase page size
```

### Encounter (Visits)

```
GET /Encounter?patient={id}                          -- all encounters
GET /Encounter?patient={id}&_sort=-date&_count=5     -- 5 most recent visits
GET /Encounter?patient={id}&type=office              -- office visits only
```

### DiagnosticReport (Lab Reports, Imaging)

```
GET /DiagnosticReport?patient={id}                   -- all reports
GET /DiagnosticReport?patient={id}&category=LAB      -- lab reports
GET /DiagnosticReport?patient={id}&category=imaging  -- imaging reports
```

## Key LOINC Codes

| LOINC | Lab/Vital | Notes |
|-------|-----------|-------|
| 4548-4 | Hemoglobin A1c (HbA1c) | Primary diabetes monitoring |
| 2345-7 | Glucose | Fasting or random |
| 2160-0 | Creatinine | Kidney function |
| 33914-3 | eGFR (CKD-EPI) | Kidney function staging |
| 2093-3 | Total Cholesterol | Lipid panel |
| 2571-8 | Triglycerides | Lipid panel |
| 2085-9 | HDL Cholesterol | Lipid panel |
| 18262-6 | LDL Cholesterol | Lipid panel |
| 85354-9 | Blood Pressure panel | Component observation (see below) |
| 8480-6 | Systolic Blood Pressure | Component of BP panel, or standalone |
| 8462-4 | Diastolic Blood Pressure | Component of BP panel, or standalone |
| 42637-9 | BNP (B-type Natriuretic Peptide) | Heart failure marker |
| 33762-6 | NT-proBNP | Heart failure marker (alternative to BNP) |
| 6690-2 | WBC Count | Infection/inflammation |
| 718-7 | Hemoglobin | Anemia screening |
| 2823-3 | Potassium | Electrolyte; critical for ACEi/ARB/MRA monitoring |
| 2951-2 | Sodium | Electrolyte |
| 1742-6 | ALT | Liver function |
| 14959-1 | Microalbumin/Creatinine Ratio (urine) | Diabetic nephropathy screening |

## Parsing FHIR JSON Responses

### Bundle Structure

Every search returns a Bundle:
```json
{
  "resourceType": "Bundle",
  "type": "searchset",
  "total": 42,
  "entry": [ { "resource": { ... } }, ... ],
  "link": [
    { "relation": "self", "url": "..." },
    { "relation": "next", "url": "..." }
  ]
}
```

Always check `bundle.get('entry', [])` before iterating -- an empty result returns a Bundle with no `entry` key.

### Patient Resource

```python
patient = entry['resource']
patient_id = patient['id']
given = patient['name'][0].get('given', [''])[0]
family = patient['name'][0].get('family', '')
full_name = f"{given} {family}"
birth_date = patient.get('birthDate', 'Unknown')
gender = patient.get('gender', 'Unknown')

# Address (optional)
address = patient.get('address', [{}])[0]
city = address.get('city', '')
state = address.get('state', '')
```

### Condition Resource

```python
condition = entry['resource']

# Code -- check ALL coding entries, not just [0]
codings = condition.get('code', {}).get('coding', [])
for coding in codings:
    system = coding.get('system', '')
    code = coding.get('code', '')
    display = coding.get('display', '')
    if 'snomed' in system:
        snomed_code = code
    elif 'icd' in system.lower():
        icd_code = code

# Clinical status
status_codings = condition.get('clinicalStatus', {}).get('coding', [])
clinical_status = status_codings[0]['code'] if status_codings else 'unknown'
# Values: "active", "recurrence", "relapse", "inactive", "remission", "resolved"

# Onset
onset = condition.get('onsetDateTime', condition.get('onsetPeriod', {}).get('start', 'Unknown'))

# Verification status (confirmed, unconfirmed, provisional, differential, refuted)
verification = condition.get('verificationStatus', {}).get('coding', [{}])[0].get('code', 'unknown')
```

### Observation Resource -- Simple (single value)

Most labs return a single value in `valueQuantity`:
```python
obs = entry['resource']
lab_name = obs['code']['coding'][0]['display']
loinc_code = obs['code']['coding'][0]['code']
date = obs.get('effectiveDateTime', 'Unknown')

# Value -- multiple possible formats
if 'valueQuantity' in obs:
    value = obs['valueQuantity']['value']
    unit = obs['valueQuantity'].get('unit', '')
elif 'valueString' in obs:
    value = obs['valueString']    # qualitative result like "negative"
    unit = ''
elif 'valueCodeableConcept' in obs:
    value = obs['valueCodeableConcept'].get('text', 'See coding')
    unit = ''
else:
    value = None  # check component (see below)

# Reference range (from the lab, more accurate than general tables)
ref_ranges = obs.get('referenceRange', [])
if ref_ranges:
    low = ref_ranges[0].get('low', {}).get('value')
    high = ref_ranges[0].get('high', {}).get('value')
```

### Observation Resource -- Component (Blood Pressure)

Blood pressure in FHIR is typically a **component Observation** with LOINC `85354-9` (BP panel). Systolic and diastolic are nested inside `component[]`, NOT in `valueQuantity`:

```python
obs = entry['resource']
panel_code = obs['code']['coding'][0]['code']

if panel_code == '85354-9' or 'component' in obs:
    systolic = None
    diastolic = None
    for comp in obs.get('component', []):
        comp_code = comp['code']['coding'][0]['code']
        if comp_code == '8480-6':  # systolic
            systolic = comp['valueQuantity']['value']
        elif comp_code == '8462-4':  # diastolic
            diastolic = comp['valueQuantity']['value']
```

**Critical**: When querying for systolic BP (LOINC 8480-6), some FHIR servers return the panel Observation (85354-9) where systolic is inside `component`. Others return a standalone Observation with `valueQuantity`. Your code must handle both:

```python
def get_bp(patient_id, base_url):
    """Get most recent blood pressure, handling both panel and standalone formats."""
    # Try panel first
    r = fhir_get("Observation",
                     params={"patient": patient_id, "code": "85354-9",
                             "_sort": "-date", "_count": "1"}).json()
    if r.get('entry'):
        obs = r['entry'][0]['resource']
        systolic = diastolic = None
        for comp in obs.get('component', []):
            c = comp['code']['coding'][0]['code']
            if c == '8480-6':
                systolic = comp['valueQuantity']['value']
            elif c == '8462-4':
                diastolic = comp['valueQuantity']['value']
        if systolic is not None:
            return systolic, diastolic, obs.get('effectiveDateTime', 'Unknown')

    # Fallback: standalone systolic
    r = fhir_get("Observation",
                     params={"patient": patient_id, "code": "8480-6",
                             "_sort": "-date", "_count": "1"}).json()
    if r.get('entry'):
        obs = r['entry'][0]['resource']
        systolic = obs.get('valueQuantity', {}).get('value')
        return systolic, None, obs.get('effectiveDateTime', 'Unknown')

    return None, None, None
```

### MedicationRequest Resource

```python
med = entry['resource']

# Medication name -- check multiple locations
med_name = (
    med.get('medicationCodeableConcept', {}).get('text')
    or med.get('medicationCodeableConcept', {}).get('coding', [{}])[0].get('display')
    or 'Unknown medication'
)

# Status
status = med.get('status', 'unknown')  # active, on-hold, cancelled, completed, stopped

# Dosage
dosage_instructions = med.get('dosageInstruction', [{}])
dosage_text = dosage_instructions[0].get('text', 'No dosage recorded') if dosage_instructions else 'No dosage recorded'

# Authored date
authored = med.get('authoredOn', 'Unknown')
```

## Pagination

FHIR responses default to 20 results per page (server-dependent). Always handle pagination for cohort queries:

```python
def get_all_pages(url, params=None):
    """Fetch all pages of a FHIR Bundle search."""
    all_entries = []
    if params:
        r = fhir_get(url, params)
    else:
        r = fhir_get(url)
    all_entries.extend(r.get('entry', []))

    # Follow 'next' links
    while True:
        next_url = None
        for link in r.get('link', []):
            if link.get('relation') == 'next':
                next_url = link['url']
                break
        if not next_url:
            break
        r = fhir_get(next_url)
        all_entries.extend(r.get('entry', []))

    return all_entries
```

For large cohorts, set `_count=200` to reduce the number of pages. The SMART test server caps at ~1000 results regardless.

## Batched and Multi-Patient Queries (CRITICAL for Performance)

**Never loop over patients making individual FHIR calls.** Each HTTP call through the sandbox proxy adds 1-3 seconds of latency. For 24 patients x 4 LOINC codes, that is 96 sequential calls = 5+ minutes. Instead, fetch all observations for a LOINC code in one request and filter client-side in Python.

### Pattern 1: Fetch all Observations for a LOINC code (preferred)

Query Observation by code alone (no patient filter) to get results for ALL patients in one call:

```
GET /Observation?code={loinc}&_count=500&_sort=-date
```

Then filter in Python by patient reference:

```python
def get_all_obs_for_code(loinc_code, count=500):
    """Fetch ALL observations for a LOINC code across all patients in one call."""
    entries = get_all_pages("Observation", {"code": loinc_code, "_count": str(count), "_sort": "-date"})
    # Build dict: patient_id -> list of observations (already sorted newest first)
    by_patient = {}
    for e in entries:
        obs = e['resource']
        ref = obs.get('subject', {}).get('reference', '')
        pid = ref.split('/')[-1] if '/' in ref else ref
        if pid not in by_patient:
            by_patient[pid] = obs  # keep only the most recent per patient
    return by_patient
```

### Pattern 2: Multi-patient query parameter

Some FHIR servers accept comma-separated patient references:

```
GET /Observation?patient=Patient/X,Patient/Y,Patient/Z&code={loinc}&_count=500
```

The SMART test server supports this. Use it when you have a specific list of patient IDs and want to avoid fetching observations for patients not in your cohort.

### Pattern 3: FHIR Batch Bundle

For heterogeneous queries (different resource types per patient), POST a Bundle of type `batch` to the server root:

```python
def fhir_batch(requests_list):
    """Execute multiple FHIR queries in a single HTTP call using a batch Bundle.
    requests_list: list of {"method": "GET", "url": "Observation?patient=X&code=Y"} dicts.
    """
    bundle = {
        "resourceType": "Bundle",
        "type": "batch",
        "entry": [{"request": req} for req in requests_list]
    }
    import subprocess, json
    r = subprocess.run(
        ["curl", "-sf", "--max-time", "60",
         "-X", "POST", "-H", "Content-Type: application/fhir+json",
         "-d", json.dumps(bundle), f"{BASE_URL}"],
        capture_output=True, text=True, timeout=65
    )
    if r.returncode != 0 or not r.stdout.strip():
        return []
    result = json.loads(r.stdout)
    return result.get('entry', [])
```

### When to use which pattern

| Scenario | Pattern | Why |
|----------|---------|-----|
| Labs for a cohort (same LOINC, many patients) | Pattern 1: code-only query | One call gets everything; filter in Python |
| Labs for a specific patient list | Pattern 2: comma-separated patients | Scoped to your cohort; one call |
| Mixed data (labs + meds + conditions per patient) | Pattern 3: batch Bundle | Multiple queries, single HTTP round-trip |
| Single patient lookup | Individual GET | Fine for 1-2 patients |

## Bulk FHIR (Production-Scale)

For production population health workflows (not used in this demo, but important context): FHIR Bulk Data Access (SMART/HL7) allows exporting entire patient populations as NDJSON files via an async `$export` operation. This is how real quality measure engines work at scale -- they don't query patient-by-patient. The demo uses individual queries for clarity and because the public test server doesn't support Bulk FHIR.

## Error Handling

- Always check `response.status_code` before parsing JSON
- Check if `entry` exists in the response before iterating: `bundle.get('entry', [])`
- Some fields may be missing -- use `.get()` with sensible defaults
- Never fabricate data. If a field is absent, report "Not recorded"
- Handle `OperationOutcome` responses (FHIR error format):
  ```python
  if response.json().get('resourceType') == 'OperationOutcome':
      issues = response.json().get('issue', [])
      error_msg = issues[0].get('diagnostics', 'Unknown FHIR error') if issues else 'Unknown error'
  ```
- Rate limiting: The public SMART test server has no rate limits, but production endpoints may. Add a small delay (0.1-0.5s) between calls in tight loops
- Timeout: Use `--max-time 30` with curl; FHIR servers can be slow under load
