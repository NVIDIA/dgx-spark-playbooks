---
name: clinical-knowledge
description: Teaches agents clinical reference ranges, condition codes, quality measure definitions, drug classifications, and regulatory context so they can flag abnormal values and identify care gaps.
metadata:
  openclaw:
    requires:
      bins: ["python3"]
---

# Clinical Reference Knowledge

## Regulatory Context

The **21st Century Cures Act** (2016) and ONC's Interoperability Final Rule (2020) require US healthcare organizations to expose patient data through standardized FHIR APIs. As of 2024, ~70% of US hospitals support FHIR R4 (source: ONC). CMS ties quality measure reporting to reimbursement through programs like MIPS (Merit-based Incentive Payment System) and the Hospital Value-Based Purchasing Program. Failure to report -- or poor performance -- results in payment adjustments of up to 9%.

**HIPAA** (Health Insurance Portability and Accountability Act) prohibits transmitting Protected Health Information (PHI) to external services without a BAA (Business Associate Agreement). This is the primary reason clinical AI must run locally: cloud LLM APIs are not BAA-covered by default, and even those that offer BAAs (e.g., Azure OpenAI) face institutional resistance from hospital compliance teams.

## Common Lab Reference Ranges

These are general adult reference ranges. Values vary by lab, assay, and patient characteristics. Always defer to the performing laboratory's reference range when available via FHIR `referenceRange`.

| Lab | Normal Range | Concerning | Unit | Notes |
|-----|-------------|------------|------|-------|
| HbA1c | < 5.7% (non-diabetic), < 7.0% (diabetic target) | > 9.0% = poor control | % | ADA 2024 guidelines; target may be relaxed to < 8.0% for elderly/frail |
| Fasting Glucose | 70-100 | 100-125 = prediabetes, >= 126 = diabetic | mg/dL | Must be fasting; random glucose >= 200 also diagnostic |
| Creatinine | 0.7-1.3 (male), 0.6-1.1 (female) | > 1.5 = impaired renal | mg/dL | Affected by muscle mass; less reliable in elderly |
| eGFR | > 90 (normal), 60-89 (mild decrease) | 30-59 = moderate CKD, 15-29 = severe, < 15 = kidney failure | mL/min/1.73m2 | CKD-EPI 2021 equation (race-neutral) |
| BUN | 7-20 | > 20 with rising creatinine = renal concern | mg/dL | Elevated by dehydration, high protein diet |
| Total Cholesterol | < 200 | 200-239 = borderline, >= 240 = high | mg/dL | |
| LDL | < 100 (general), < 70 (high-risk ASCVD) | > 160 = high | mg/dL | ACC/AHA 2018 |
| HDL | > 40 (male), > 50 (female) | < 40 = cardiovascular risk factor | mg/dL | |
| Triglycerides | < 150 | 150-499 = elevated, >= 500 = severe (pancreatitis risk) | mg/dL | |
| Systolic BP | < 120 (normal), 120-129 (elevated) | 130-139 = Stage 1 HTN, >= 140 = Stage 2 HTN | mmHg | ACC/AHA 2017; CMS165 uses 140/90 threshold |
| Diastolic BP | < 80 | 80-89 = Stage 1 HTN, >= 90 = Stage 2 HTN | mmHg | |
| BNP | < 100 | 100-400 = possible HF, > 400 = likely HF | pg/mL | Age-adjusted: higher cutoffs in elderly; obesity lowers BNP |
| NT-proBNP | < 300 (rule-out) | Age-stratified: >450 (<50y), >900 (50-75y), >1800 (>75y) | pg/mL | More stable than BNP; renal clearance affects levels |
| Potassium | 3.5-5.0 | < 3.5 = hypokalemia, > 5.5 = hyperkalemia (cardiac risk) | mEq/L | Critical for patients on ACEi/ARB/spironolactone |
| Sodium | 136-145 | < 130 = moderate hyponatremia | mEq/L | Common in HF patients |
| ALT | 7-56 | > 3x ULN = significant hepatotoxicity | U/L | Monitor with statin therapy |
| Hemoglobin | 13.5-17.5 (male), 12.0-16.0 (female) | < 12 (male) or < 11 (female) = anemia | g/dL | Common in CKD (erythropoietin deficiency) |

When reporting lab values:
- Always flag values outside the normal range with the severity (mild / moderate / severe)
- Note the date of the observation -- a result from 2 years ago has different clinical significance than one from yesterday
- If the FHIR Observation includes a `referenceRange`, use that instead of the table above

## Condition Codes

### SNOMED CT Codes (Primary)

| Code | Condition | ICD-10 Equivalent | Prevalence (US adults) |
|------|-----------|-------------------|----------------------|
| 44054006 | Type 2 Diabetes Mellitus | E11.x | ~11% (37M) |
| 46635009 | Type 1 Diabetes Mellitus | E10.x | ~0.5% (1.6M) |
| 38341003 | Essential Hypertension | I10 | ~47% (116M) |
| 84114007 | Heart Failure | I50.x | ~2.4% (6.7M) |
| 40055000 | Chronic Kidney Disease | N18.x | ~15% (37M) |
| 53741008 | Coronary Artery Disease | I25.x | ~7% (20M) |
| 13645005 | COPD | J44.x | ~6% (16M) |
| 195967001 | Asthma | J45.x | ~8% (25M) |
| 49436004 | Atrial Fibrillation | I48.x | ~2% (6M) |
| 73211009 | Diabetes (unspecified) | E11.9 | Used in older records |

### ICD-10-CM to SNOMED Crosswalk

When FHIR Condition resources use ICD-10 coding (system `http://hl7.org/fhir/sid/icd-10-cm`), map as follows:
- E11.* → Type 2 Diabetes (SNOMED 44054006)
- E10.* → Type 1 Diabetes (SNOMED 46635009)
- I10 → Essential Hypertension (SNOMED 38341003)
- I50.* → Heart Failure (SNOMED 84114007)
- N18.* → Chronic Kidney Disease (SNOMED 40055000)

Note: A Condition resource may have both SNOMED and ICD-10 codes in the `coding` array. Always check all entries, not just `coding[0]`.

## CMS Quality Measures

### CMS122v12 -- Diabetes: Hemoglobin A1c (HbA1c) Poor Control (> 9%)

| Component | Definition |
|-----------|-----------|
| **Denominator** | Patients 18-75 with diabetes (Type 1 or Type 2) and at least 2 encounters during the measurement period |
| **Numerator** | Patients with most recent HbA1c > 9.0%, OR no HbA1c recorded during the measurement period |
| **Exclusions** | Hospice care, palliative care, advanced illness with frailty (2+ encounters for advanced illness AND frailty diagnosis), dementia medications (donepezil, rivastigmine, memantine, galantamine) |
| **Performance rate** | Lower is better (inverse measure) |
| **Payment impact** | Part of MIPS quality reporting; affects Medicare reimbursement |

### CMS165v12 -- Controlling High Blood Pressure

| Component | Definition |
|-----------|-----------|
| **Denominator** | Patients 18-85 with essential hypertension diagnosed before or during the measurement period |
| **Numerator** | Patients with most recent BP < 140/90 mmHg |
| **Exclusions** | Hospice, palliative care, ESRD, kidney transplant, advanced illness with frailty, pregnancy |
| **Performance rate** | Higher is better |
| **Note** | BP must be measured during an outpatient encounter; home BP readings are not counted in the standard measure |

### CMS135v12 -- Heart Failure: ACEi/ARB/ARNI Therapy for LVEF < 40%

| Component | Definition |
|-----------|-----------|
| **Denominator** | Patients 18+ with heart failure AND documented LVEF < 40% (HFrEF) |
| **Numerator** | Patients prescribed ACE inhibitor, ARB, or ARNI (sacubitril/valsartan) |
| **Exclusions** | Hospice, allergy/intolerance to all three classes, bilateral renal artery stenosis, pregnancy, hyperkalemia > 5.5 |
| **Note** | LVEF data often in DiagnosticReport or CarePlan, not always queryable via Condition alone |

### CMS134v12 -- Diabetes: Medical Attention for Nephropathy

| Component | Definition |
|-----------|-----------|
| **Denominator** | Patients 18-75 with diabetes |
| **Numerator** | Patients with nephropathy screening (urine albumin test) OR evidence of nephropathy treatment (ACEi/ARB) OR nephropathy diagnosis |
| **Exclusions** | Hospice, palliative care, advanced illness with frailty |

## Drug Classifications

When checking medication coverage, recognize these drug class groupings. Matching should be **case-insensitive partial string matching** on the medication name from FHIR `medicationCodeableConcept.text` or `.coding[].display`.

### Diabetes Medications

| Class | Drugs | Notes |
|-------|-------|-------|
| Biguanide | metformin | First-line therapy |
| Sulfonylureas | glipizide, glyburide, glimepiride | Hypoglycemia risk |
| Insulin | insulin lispro, insulin glargine, insulin aspart, insulin detemir, insulin degludec, NPH insulin | Match any string containing "insulin" |
| GLP-1 Receptor Agonists | liraglutide (Victoza), semaglutide (Ozempic/Wegovy/Rybelsus), dulaglutide (Trulicity), exenatide (Byetta/Bydureon), tirzepatide (Mounjaro) | Weight loss benefit; cardiovascular benefit |
| SGLT2 Inhibitors | empagliflozin (Jardiance), dapagliflozin (Farxiga), canagliflozin (Invokana), ertugliflozin (Steglatro) | Cardiovascular + renal benefit; monitor for DKA |
| DPP-4 Inhibitors | sitagliptin (Januvia), saxagliptin, linagliptin, alogliptin | Weight-neutral |
| Thiazolidinediones | pioglitazone, rosiglitazone | HF risk; edema |

### Antihypertensives

| Class | Drugs | Notes |
|-------|-------|-------|
| ACE Inhibitors | lisinopril, enalapril, ramipril, benazepril, fosinopril, quinapril | Cough side effect; monitor K+ and creatinine |
| ARBs | losartan, valsartan, irbesartan, olmesartan, telmisartan, candesartan, azilsartan | Alternative if ACEi cough |
| ARNIs | sacubitril/valsartan (Entresto) | HFrEF guideline-directed; do NOT combine with ACEi |
| CCBs | amlodipine, nifedipine, diltiazem, verapamil | Diltiazem/verapamil contraindicated in HFrEF |
| Beta-Blockers | metoprolol (tartrate or succinate), atenolol, carvedilol, bisoprolol, propranolol, nebivolol | Only carvedilol, metoprolol succinate, bisoprolol for HF |
| Thiazide Diuretics | hydrochlorothiazide (HCTZ), chlorthalidone, indapamide | First-line for HTN |
| Loop Diuretics | furosemide, bumetanide, torsemide | Volume management in HF, not primary HTN therapy |
| Aldosterone Antagonists | spironolactone, eplerenone | HF benefit; monitor K+ |

### Statins (HMG-CoA Reductase Inhibitors)

| Intensity | Drugs |
|-----------|-------|
| High | atorvastatin 40-80mg, rosuvastatin 20-40mg |
| Moderate | atorvastatin 10-20mg, rosuvastatin 5-10mg, simvastatin 20-40mg, pravastatin 40-80mg |
| Low | simvastatin 10mg, pravastatin 10-20mg, lovastatin 20mg |

### Heart Failure Medications (Guideline-Directed Medical Therapy)

The four pillars of HFrEF therapy (ACC/AHA 2022):
1. **ACEi/ARB/ARNI** -- sacubitril/valsartan preferred over ACEi/ARB
2. **Beta-Blocker** -- carvedilol, metoprolol succinate, or bisoprolol only
3. **Aldosterone Antagonist** -- spironolactone or eplerenone (if eGFR > 30, K+ < 5.0)
4. **SGLT2 Inhibitor** -- dapagliflozin or empagliflozin (regardless of diabetes status)

### Matching Strategy

```
medication_name = fhir_med_text.lower()

is_on_insulin = "insulin" in medication_name
is_on_glp1 = any(drug in medication_name for drug in
    ["liraglutide", "semaglutide", "dulaglutide", "exenatide", "tirzepatide",
     "victoza", "ozempic", "trulicity", "byetta", "mounjaro", "rybelsus"])
is_on_sglt2 = any(drug in medication_name for drug in
    ["empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin",
     "jardiance", "farxiga", "invokana", "steglatro"])
is_on_acei = any(drug in medication_name for drug in
    ["lisinopril", "enalapril", "ramipril", "benazepril", "fosinopril", "quinapril"])
is_on_arb = any(drug in medication_name for drug in
    ["losartan", "valsartan", "irbesartan", "olmesartan", "telmisartan",
     "candesartan", "azilsartan"])
is_on_betablocker = any(drug in medication_name for drug in
    ["metoprolol", "atenolol", "carvedilol", "bisoprolol", "propranolol", "nebivolol"])
is_on_statin = any(drug in medication_name for drug in
    ["atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "lovastatin"])
```

## Clinical Comorbidity Patterns

Common co-occurring conditions to watch for during analysis:
- **Cardiorenal-metabolic overlap**: Diabetes + Hypertension + CKD occur together in ~30% of diabetic patients
- **Heart failure + CKD**: eGFR < 30 limits medication options (spironolactone, SGLT2i dose adjustment)
- **Diabetes + CAD**: Statin therapy should be high-intensity; GLP-1/SGLT2i have cardiovascular benefit
- **Atrial fibrillation + Heart failure**: Common pairing; rate control with beta-blocker preferred

## Guardrails

- These reference ranges are general guidelines based on published clinical guidelines (ADA, ACC/AHA, KDIGO), not diagnostic criteria
- Never state that a patient "has" a condition based on a lab value alone
- Always include the disclaimer: "This is for informational and research purposes, not clinical decision-making"
- When flagging care gaps, use language like "may warrant review" not "requires treatment"
- Acknowledge that CMS measure logic here is simplified -- production implementations use the full eCQM (electronic Clinical Quality Measure) specifications from CMS
- Lab reference ranges should defer to the performing laboratory's range when available
