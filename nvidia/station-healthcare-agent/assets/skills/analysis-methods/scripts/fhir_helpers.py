"""FHIR query helpers for Clinical Intelligence.

Import this in analysis scripts:
    import sys
    sys.path.insert(0, '/sandbox/clinical-intelligence/skills/analysis-methods/scripts')
    from fhir_helpers import *

All HTTP calls use subprocess/curl (requests library does NOT work in sandbox).
"""

import subprocess
import json
import os
import pandas as pd
from urllib.parse import urlencode

BASE_URL = "https://r4.smarthealthit.org"
CANVAS_DIR = os.path.expanduser("~/.openclaw/canvas")
CANVAS_URL_BASE = "http://localhost:18789/__openclaw__/canvas"


def save_chart_to_canvas(fig, filename, dpi=150, facecolor='#1a1a1a'):
    """Save a matplotlib figure to the canvas directory and print the access URL."""
    os.makedirs(CANVAS_DIR, exist_ok=True)
    path = os.path.join(CANVAS_DIR, filename)
    fig.savefig(path, dpi=dpi, facecolor=facecolor, bbox_inches='tight')
    print(f"\nCanvas URL: {CANVAS_URL_BASE}/{filename}")
    return path


# ── Core FHIR functions ────────────────────────────────────────

def fhir_get(path, params=None):
    """GET a FHIR endpoint via curl. Returns parsed JSON."""
    url = f"{BASE_URL}/{path}" if not path.startswith("http") else path
    if params:
        url = f"{url}?{urlencode(params)}"
    r = subprocess.run(["curl", "-sf", "--max-time", "30", url],
                       capture_output=True, text=True, timeout=35)
    if r.returncode != 0 or not r.stdout.strip():
        import sys
        print(f"WARNING: curl failed for {url} (exit code {r.returncode})", file=sys.stderr)
        if r.stderr:
            print(f"  stderr: {r.stderr[:200]}", file=sys.stderr)
        return {"entry": []}
    return json.loads(r.stdout)


def get_all_pages(path, params=None):
    """Fetch all pages of a FHIR Bundle."""
    all_entries = []
    data = fhir_get(path, params)
    all_entries.extend(data.get('entry', []))
    while True:
        next_url = None
        for link in data.get('link', []):
            if link.get('relation') == 'next':
                next_url = link['url']
                break
        if not next_url:
            break
        data = fhir_get(next_url)
        all_entries.extend(data.get('entry', []))
    return all_entries


# ── Patient cohort ─────────────────────────────────────────────

def get_patients_with_condition(snomed_code):
    """Get unique patient IDs for a SNOMED condition code.
    Uses bare codes (no system URI) for the SMART test server."""
    entries = get_all_pages("Condition", {"code": snomed_code, "_count": "200"})
    patient_ids = list(set(
        e['resource']['subject']['reference'].split('/')[-1]
        for e in entries
        if 'subject' in e.get('resource', {})
    ))
    print(f"Found {len(patient_ids)} patients with condition {snomed_code}")
    return patient_ids


# ── Lab retrieval (BATCHED) ────────────────────────────────────

def get_latest_labs_batch(loinc_code, patient_ids=None):
    """Fetch the most recent observation for a LOINC code for ALL patients in one call.
    Returns dict: patient_id -> (value, unit, date).

    This makes 1-2 FHIR calls regardless of patient count.
    Do NOT use get_latest_lab() in a loop — use this instead."""
    entries = get_all_pages("Observation", {
        "code": loinc_code, "_count": "500", "_sort": "-date"
    })
    result = {}
    pid_set = set(patient_ids) if patient_ids else None
    for e in entries:
        obs = e['resource']
        ref = obs.get('subject', {}).get('reference', '')
        pid = ref.split('/')[-1] if '/' in ref else ref
        if pid in result:
            continue  # already have most recent (sorted by -date)
        if pid_set is not None and pid not in pid_set:
            continue
        date = obs.get('effectiveDateTime', 'Unknown')
        if 'valueQuantity' in obs:
            result[pid] = (obs['valueQuantity']['value'],
                           obs['valueQuantity'].get('unit', ''), date)
        elif 'valueString' in obs:
            result[pid] = (obs['valueString'], '', date)
        else:
            result[pid] = (None, None, date)
    return result


def get_latest_lab(patient_id, loinc_code):
    """Get the most recent lab for ONE patient. Returns (value, unit, date).
    Only use for single-patient lookups. For cohorts, use get_latest_labs_batch()."""
    r = fhir_get("Observation", {"patient": patient_id, "code": loinc_code,
                                  "_sort": "-date", "_count": "1"})
    if not r.get('entry'):
        return None, None, None
    obs = r['entry'][0]['resource']
    date = obs.get('effectiveDateTime', 'Unknown')
    if 'valueQuantity' in obs:
        return obs['valueQuantity']['value'], obs['valueQuantity'].get('unit', ''), date
    elif 'valueString' in obs:
        return obs['valueString'], '', date
    return None, None, None


# ── Blood pressure (component observation) ─────────────────────

def get_latest_bp(patient_id):
    """Get most recent BP. Handles both panel (85354-9) and standalone formats.
    Returns (systolic, diastolic, date)."""
    r = fhir_get("Observation", {"patient": patient_id, "code": "85354-9",
                                  "_sort": "-date", "_count": "1"})
    if r.get('entry'):
        obs = r['entry'][0]['resource']
        systolic = diastolic = None
        for comp in obs.get('component', []):
            code = comp.get('code', {}).get('coding', [{}])[0].get('code', '')
            if code == '8480-6':
                systolic = comp.get('valueQuantity', {}).get('value')
            elif code == '8462-4':
                diastolic = comp.get('valueQuantity', {}).get('value')
        if systolic is not None:
            return systolic, diastolic, obs.get('effectiveDateTime', 'Unknown')

    r = fhir_get("Observation", {"patient": patient_id, "code": "8480-6",
                                  "_sort": "-date", "_count": "1"})
    if r.get('entry'):
        obs = r['entry'][0]['resource']
        return obs.get('valueQuantity', {}).get('value'), None, obs.get('effectiveDateTime', 'Unknown')

    return None, None, None


def get_latest_bp_batch(patient_ids):
    """Fetch most recent BP for all patients in a cohort.
    Returns dict: patient_id -> (systolic, diastolic, date).

    Makes 1-2 FHIR calls regardless of patient count.
    Do NOT use get_latest_bp() in a loop — use this instead."""
    pid_set = set(patient_ids)
    result = {}

    # First try BP panel (85354-9)
    entries = get_all_pages("Observation", {"code": "85354-9", "_count": "500", "_sort": "-date"})
    for e in entries:
        obs = e['resource']
        ref = obs.get('subject', {}).get('reference', '')
        pid = ref.split('/')[-1] if '/' in ref else ref
        if pid in result or pid not in pid_set:
            continue
        systolic = diastolic = None
        for comp in obs.get('component', []):
            code = comp.get('code', {}).get('coding', [{}])[0].get('code', '')
            if code == '8480-6':
                systolic = comp.get('valueQuantity', {}).get('value')
            elif code == '8462-4':
                diastolic = comp.get('valueQuantity', {}).get('value')
        if systolic is not None:
            result[pid] = (systolic, diastolic, obs.get('effectiveDateTime', 'Unknown'))

    # Fallback: standalone systolic (8480-6) for patients without a panel
    missing = pid_set - set(result.keys())
    if missing:
        entries = get_all_pages("Observation", {"code": "8480-6", "_count": "500", "_sort": "-date"})
        for e in entries:
            obs = e['resource']
            ref = obs.get('subject', {}).get('reference', '')
            pid = ref.split('/')[-1] if '/' in ref else ref
            if pid in result or pid not in missing:
                continue
            val = obs.get('valueQuantity', {}).get('value')
            if val is not None:
                result[pid] = (val, None, obs.get('effectiveDateTime', 'Unknown'))

    return result


# ── Medications (BATCHED) ──────────────────────────────────────

def get_all_medications_batch(patient_ids):
    """Fetch active medications for all patients in a cohort.
    Returns dict: patient_id -> list of medication name strings.

    Makes 1-2 FHIR calls regardless of patient count."""
    entries = get_all_pages("MedicationRequest", {"status": "active", "_count": "500"})
    result = {}
    pid_set = set(patient_ids)
    for e in entries:
        med = e['resource']
        ref = med.get('subject', {}).get('reference', '')
        pid = ref.split('/')[-1] if '/' in ref else ref
        if pid not in pid_set:
            continue
        name = (
            med.get('medicationCodeableConcept', {}).get('text')
            or med.get('medicationCodeableConcept', {}).get('coding', [{}])[0].get('display')
            or 'Unknown'
        )
        result.setdefault(pid, []).append(name)
    return result


def get_medications(patient_id):
    """Get active medications for ONE patient. Returns list of names.
    Only use for single-patient lookups. For cohorts, use get_all_medications_batch()."""
    r = fhir_get("MedicationRequest", {"patient": patient_id, "status": "active",
                                        "_count": "100"})
    meds = []
    for e in r.get('entry', []):
        med = e['resource']
        name = (
            med.get('medicationCodeableConcept', {}).get('text')
            or med.get('medicationCodeableConcept', {}).get('coding', [{}])[0].get('display')
            or 'Unknown'
        )
        meds.append(name)
    return meds


def check_drug_class(med_list, drug_names):
    """Check if any medication in med_list matches any drug in drug_names (case-insensitive)."""
    med_lower = [m.lower() for m in med_list]
    return any(drug in med_text for drug in drug_names for med_text in med_lower)


# ── Cohort DataFrame builder ──────────────────────────────────

def build_cohort_df(patient_ids, loinc_code, lab_name, drug_check_fn=None):
    """Build a DataFrame with lab values and medications for a patient cohort.
    Uses batched FHIR queries — only 2-3 HTTP calls total.

    Args:
        patient_ids: list of patient ID strings
        loinc_code: LOINC code for the lab (e.g. '4548-4' for HbA1c)
        lab_name: column name for the lab value (e.g. 'HbA1c')
        drug_check_fn: optional function(med_list) -> bool for medication check
    """
    print(f"Fetching {lab_name} for {len(patient_ids)} patients (batched)...")
    labs = get_latest_labs_batch(loinc_code, patient_ids)
    print(f"Fetching medications (batched)...")
    meds_map = get_all_medications_batch(patient_ids)

    rows = []
    for pid in patient_ids:
        lab_val, unit, date = labs.get(pid, (None, None, None))
        meds = meds_map.get(pid, [])
        row = {
            "patient_id": pid,
            lab_name: lab_val,
            "lab_date": date,
            "medications": ", ".join(meds) if meds else "None",
            "med_count": len(meds)
        }
        if drug_check_fn is not None:
            row["on_target_med"] = drug_check_fn(meds)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Built DataFrame: {len(df)} patients, {df[lab_name].notna().sum()} with {lab_name}")
    return df
