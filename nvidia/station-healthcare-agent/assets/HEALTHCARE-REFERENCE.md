# Healthcare Reference Supplement

Clinical code tables, full CMS quality measure specifications, drug classification libraries, FHIR resource reference, and glossary.

---

## SNOMED CT Condition Codes

| Code | Condition | ICD-10 | US Prevalence | CMS Measure |
|------|-----------|--------|---------------|-------------|
| 44054006 | Type 2 Diabetes Mellitus | E11.x | ~37M adults | CMS122, CMS134 |
| 46635009 | Type 1 Diabetes Mellitus | E10.x | ~1.6M adults | CMS122, CMS134 |
| 38341003 | Essential Hypertension | I10 | ~116M adults | CMS165 |
| 84114007 | Heart Failure | I50.x | ~6.7M adults | CMS135 |
| 40055000 | Chronic Kidney Disease | N18.x | ~37M adults | CMS134 |
| 53741008 | Coronary Artery Disease | I25.x | ~20M adults | — |
| 13645005 | COPD | J44.x | ~16M adults | — |
| 195967001 | Asthma | J45.x | ~25M adults | — |
| 49436004 | Atrial Fibrillation | I48.x | ~6M adults | — |
| 431856006 | CKD Stage 2 | N18.2 | — | — |
| 55822004 | Hyperlipidemia | E78.5 | — | — |

> **Note:** Use bare codes when querying the test server. Do not use system-qualified URIs (e.g., use `code=44054006`, not `code=http://snomed.info/sct|44054006`).

## LOINC Laboratory and Vitals Codes

| Code | Display | Category | Unit | Clinical Context |
|------|---------|----------|------|-----------------|
| 4548-4 | Hemoglobin A1c | Metabolic | % | Diabetes monitoring (CMS122) |
| 2345-7 | Glucose | Metabolic | mg/dL | Diabetes screening |
| 2160-0 | Creatinine | Renal | mg/dL | Kidney function |
| 33914-3 | eGFR (CKD-EPI) | Renal | mL/min/1.73m² | CKD staging |
| 85354-9 | Blood Pressure panel | Vital Signs | — | Component observation (CMS165) |
| 8480-6 | Systolic Blood Pressure | Vital Signs | mmHg | Component of 85354-9, or standalone |
| 8462-4 | Diastolic Blood Pressure | Vital Signs | mmHg | Component of 85354-9, or standalone |
| 2093-3 | Total Cholesterol | Lipids | mg/dL | Cardiovascular risk |
| 2571-8 | Triglycerides | Lipids | mg/dL | Cardiovascular risk |
| 2085-9 | HDL Cholesterol | Lipids | mg/dL | Cardiovascular risk |
| 18262-6 | LDL Cholesterol | Lipids | mg/dL | Statin therapy target |
| 42637-9 | BNP | Cardiac | pg/mL | Heart failure marker |
| 33762-6 | NT-proBNP | Cardiac | pg/mL | Heart failure marker |
| 2823-3 | Potassium | Electrolyte | mEq/L | ACEi/ARB/MRA safety |
| 2951-2 | Sodium | Electrolyte | mEq/L | HF fluid status |
| 718-7 | Hemoglobin | Hematology | g/dL | Anemia (CKD) |
| 1742-6 | ALT | Hepatic | U/L | Statin monitoring |
| 14959-1 | Urine Albumin/Creatinine Ratio | Renal | mg/g | Nephropathy screening (CMS134) |

## FHIR Coding Systems

| System URI | Code System | Example |
|-----------|-------------|---------|
| `http://snomed.info/sct` | SNOMED CT | 44054006 (Type 2 Diabetes) |
| `http://hl7.org/fhir/sid/icd-10-cm` | ICD-10-CM | E11.9 (T2DM, unspecified) |
| `http://loinc.org` | LOINC | 4548-4 (HbA1c) |
| `http://www.nlm.nih.gov/research/umls/rxnorm` | RxNorm | Medication identifiers |
| `http://hl7.org/fhir/sid/ndc` | NDC | National Drug Codes |

---

## CMS Quality Measure Specifications

### CMS122v12 — Simplified vs. Full eCQM

| Component | This System (Simplified) | Full eCQM Specification |
|-----------|------------------------|------------------------|
| **Denominator** | Patients 18–75 with diabetes (SNOMED 44054006 or 46635009) | + at least 2 outpatient encounters during measurement year; specific encounter value sets (CPT 99201-99215, HCPCS G0438-G0439) |
| **Numerator** | Most recent HbA1c > 9.0% OR no HbA1c recorded | Same threshold, but HbA1c must be during measurement period; specific LOINC value set (4548-4, 4549-2, 17856-6) |
| **Exclusions** | Hospice, palliative care, advanced illness + frailty | + specific value sets for hospice (SNOMED 385763009+), palliative care (ICD-10 Z51.5), advanced illness (200+ ICD-10 codes), frailty indicators, dementia medications (specific RxNorm codes) |
| **Measurement period** | Any recent HbA1c | January 1 – December 31 of the performance year |

### CMS165v12 — Simplified vs. Full eCQM

| Component | This System (Simplified) | Full eCQM Specification |
|-----------|------------------------|------------------------|
| **Denominator** | Patients 18–85 with hypertension (SNOMED 38341003) | + diagnosed before or during first 6 months of measurement period; at least 2 outpatient encounters |
| **Numerator** | Most recent BP < 140/90 | BP measured during an outpatient encounter within measurement period |
| **Exclusions** | Hospice, ESRD, pregnancy | + kidney transplant, palliative care, advanced illness + frailty |

### CMS135v12 — Simplified vs. Full eCQM

| Component | This System (Simplified) | Full eCQM Specification |
|-----------|------------------------|------------------------|
| **Denominator** | Patients 18+ with HF (SNOMED 84114007) + LVEF < 40% | + specific HF diagnosis value set (ICD-10 I50.1, I50.20-I50.23, I50.40-I50.43, I50.9) |
| **Numerator** | Prescribed ACEi, ARB, or ARNI | + specific RxNorm medication value sets; at least 1 prescription during measurement period |
| **LVEF source** | Not queryable via standard FHIR Observation | Typically in DiagnosticReport or CarePlan |

---

## Drug Classification Tables

### Diabetes Medications

| Class | Generic Names | Brand Names | Matching Strings |
|-------|-------------|-------------|-----------------|
| Biguanide | metformin | Glucophage, Fortamet, Riomet | `metformin` |
| Sulfonylureas | glipizide, glyburide, glimepiride | Glucotrol, DiaBeta, Amaryl | `glipizide`, `glyburide`, `glimepiride` |
| Insulin | lispro, glargine, aspart, detemir, degludec, NPH | Humalog, Lantus, Novolog, Levemir, Tresiba | `insulin` |
| GLP-1 RA | liraglutide, semaglutide, dulaglutide, exenatide, tirzepatide | Victoza, Ozempic, Wegovy, Rybelsus, Trulicity, Byetta, Mounjaro | `liraglutide`, `semaglutide`, `dulaglutide`, `exenatide`, `tirzepatide`, `victoza`, `ozempic`, `trulicity`, `byetta`, `mounjaro`, `rybelsus` |
| SGLT2i | empagliflozin, dapagliflozin, canagliflozin, ertugliflozin | Jardiance, Farxiga, Invokana, Steglatro | `empagliflozin`, `dapagliflozin`, `canagliflozin`, `ertugliflozin`, `jardiance`, `farxiga`, `invokana`, `steglatro` |
| DPP-4i | sitagliptin, saxagliptin, linagliptin, alogliptin | Januvia, Onglyza, Tradjenta, Nesina | `sitagliptin`, `saxagliptin`, `linagliptin`, `alogliptin` |

### Antihypertensives

| Class | Generic Names | Matching Strings |
|-------|-------------|-----------------|
| ACE Inhibitors | lisinopril, enalapril, ramipril, benazepril, fosinopril, quinapril | `lisinopril`, `enalapril`, `ramipril`, `benazepril`, `fosinopril`, `quinapril` |
| ARBs | losartan, valsartan, irbesartan, olmesartan, telmisartan, candesartan, azilsartan | `losartan`, `valsartan`, `irbesartan`, `olmesartan`, `telmisartan`, `candesartan`, `azilsartan` |
| ARNIs | sacubitril/valsartan | `entresto`, `sacubitril` |
| CCBs | amlodipine, nifedipine, diltiazem, verapamil | `amlodipine`, `nifedipine`, `diltiazem`, `verapamil` |
| Beta-Blockers | metoprolol, atenolol, carvedilol, bisoprolol, propranolol, nebivolol | `metoprolol`, `atenolol`, `carvedilol`, `bisoprolol`, `propranolol`, `nebivolol` |
| Thiazides | HCTZ, chlorthalidone, indapamide | `hydrochlorothiazide`, `hctz`, `chlorthalidone`, `indapamide` |
| Loop Diuretics | furosemide, bumetanide, torsemide | `furosemide`, `bumetanide`, `torsemide` |
| Aldosterone Antagonists | spironolactone, eplerenone | `spironolactone`, `eplerenone` |

### Heart Failure GDMT (Four Pillars)

| Pillar | Drug Class | Specific Agents | Evidence |
|--------|-----------|----------------|----------|
| 1 | ACEi/ARB/ARNI | Sacubitril/valsartan preferred; lisinopril, enalapril, losartan, valsartan alternatives | ACC/AHA 2022 Class I |
| 2 | Beta-Blocker | Carvedilol, metoprolol succinate, bisoprolol ONLY | ACC/AHA 2022 Class I |
| 3 | Aldosterone Antagonist | Spironolactone, eplerenone (if eGFR > 30, K+ < 5.0) | ACC/AHA 2022 Class I |
| 4 | SGLT2 Inhibitor | Dapagliflozin, empagliflozin (regardless of diabetes) | ACC/AHA 2022 Class I |

### Statins

| Intensity | Drugs |
|-----------|-------|
| High | atorvastatin 40-80mg, rosuvastatin 20-40mg |
| Moderate | atorvastatin 10-20mg, rosuvastatin 5-10mg, simvastatin 20-40mg, pravastatin 40-80mg |
| Low | simvastatin 10mg, pravastatin 10-20mg, lovastatin 20mg |

---

## FHIR Resource Quick Reference

### Endpoints and Search Parameters

| Resource | Endpoint | Common Search Parameters | Key JSON Paths |
|----------|----------|------------------------|----------------|
| Patient | `GET /Patient` | `name`, `birthdate`, `gender`, `_id`, `_count` | `.name[0].given[0]`, `.name[0].family`, `.birthDate`, `.gender` |
| Condition | `GET /Condition` | `patient`, `code` (SNOMED), `clinical-status`, `_count` | `.code.coding[].code`, `.clinicalStatus.coding[0].code`, `.onsetDateTime` |
| Observation | `GET /Observation` | `patient`, `code` (LOINC), `category`, `_sort`, `_count`, `date` | `.valueQuantity.value`, `.effectiveDateTime`, `.component[]` |
| MedicationRequest | `GET /MedicationRequest` | `patient`, `status`, `_count` | `.medicationCodeableConcept.text`, `.status` |
| Encounter | `GET /Encounter` | `patient`, `_sort`, `_count`, `type` | `.type[0].text`, `.period.start`, `.class.code` |

### Bundle Navigation

```python
import subprocess, json
r = subprocess.run(["curl", "-sf", "--max-time", "30", url],
                   capture_output=True, text=True, timeout=35)
bundle = json.loads(r.stdout) if r.returncode == 0 else {"entry": []}
entries = bundle.get('entry', [])
total = bundle.get('total', len(entries))

for link in bundle.get('link', []):
    if link.get('relation') == 'next':
        next_url = link['url']
```

---

## Glossary

| Term | Definition |
|------|-----------|
| **FHIR** | Fast Healthcare Interoperability Resources. HL7 standard for exchanging healthcare data via REST APIs. R4 is the current normative version. |
| **SNOMED CT** | Systematized Nomenclature of Medicine. Standardized clinical terminology (~350K concepts). Used for diagnosis coding. |
| **LOINC** | Logical Observation Identifiers Names and Codes. Standard for identifying lab tests and vital signs. |
| **ICD-10-CM** | International Classification of Diseases, 10th Revision. Used for billing and reimbursement. |
| **RxNorm** | Standardized nomenclature for clinical drugs. |
| **eCQM** | Electronic Clinical Quality Measure. Machine-readable quality measure specs published by CMS. |
| **CMS** | Centers for Medicare & Medicaid Services. Administers Medicare, Medicaid, and quality programs. |
| **MIPS** | Merit-based Incentive Payment System. Adjusts Medicare payments based on quality measure performance (up to ±9%). |
| **PHI** | Protected Health Information. Any individually identifiable health information covered by HIPAA. |
| **HIPAA** | Health Insurance Portability and Accountability Act (1996). Governs privacy/security of health information. |
| **BAA** | Business Associate Agreement. HIPAA-required contract when sharing PHI with a third party. |
| **HCC** | Hierarchical Condition Category. Risk adjustment model for Medicare Advantage payments. |
| **GDMT** | Guideline-Directed Medical Therapy. Evidence-based treatment per clinical practice guidelines. |
| **SaMD** | Software as a Medical Device. FDA classification for software meeting medical device definition. |
| **Bulk FHIR** | FHIR Bulk Data Access spec. Async API (`$export`) for large-volume patient data extraction as NDJSON. |
| **Synthea** | Open-source synthetic patient data generator. Creates realistic fictional records in FHIR format. |
