"""
Quick test script to verify the FHIR test server is reachable
and has usable patient data for the demo.

Run: python scripts/test-fhir.py
"""

import json
import sys

import requests

BASE = "https://r4.smarthealthit.org"


def test_endpoint(path, label):
    url = f"{BASE}{path}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        count = len(data.get("entry", []))
        print(f"  {label}: {count} results")
        return data
    except Exception as e:
        print(f"  {label}: FAILED -- {e}")
        return None


def main():
    print(f"Testing FHIR server at {BASE}\n")

    patients = test_endpoint("/Patient?_count=10", "Patients")
    if not patients or not patients.get("entry"):
        print("\nNo patients found. Server may be down.")
        sys.exit(1)

    pid = patients["entry"][0]["resource"]["id"]
    name = patients["entry"][0]["resource"].get("name", [{}])[0]
    display = f"{name.get('given', ['?'])[0]} {name.get('family', '?')}"
    print(f"\nUsing test patient: {display} (ID: {pid})\n")

    test_endpoint(f"/Condition?patient={pid}&_count=50", "Conditions")
    test_endpoint(f"/Observation?patient={pid}&_count=50", "Observations (labs/vitals)")
    test_endpoint(f"/MedicationRequest?patient={pid}&_count=50", "Medications")
    test_endpoint(f"/Encounter?patient={pid}&_count=50", "Encounters")

    print(f"\nGood patient for demo: {display} (ID: {pid})")
    print("Use this patient ID in your demo commands.\n")

    conditions = requests.get(f"{BASE}/Condition?patient={pid}&_count=50", timeout=10).json()
    if conditions.get("entry"):
        print("Active conditions:")
        for e in conditions["entry"]:
            code = e["resource"]["code"]["coding"][0]
            status = (
                e["resource"]
                .get("clinicalStatus", {})
                .get("coding", [{}])[0]
                .get("code", "?")
            )
            print(f"  [{status}] {code.get('display', '?')} ({code.get('code', '?')})")


if __name__ == "__main__":
    main()
