---
name: molecular-viz
description: Visualize drug-protein complexes using build_viewer.py, PubChem, and OpenFold3 NIM. Use when asked to show a molecular structure, drug target, or protein visualization.
metadata:
  openclaw:
    requires:
      bins: ["python3", "curl"]
---

# Molecular Visualization

Generate 3D protein-ligand visualizations using the `build_viewer.py` script. The script handles the full pipeline:

1. **Drug SMILES** -- looked up automatically from PubChem
2. **Protein target** -- resolved from a built-in drug-target table (or pass `--sequence` manually)
3. **Structure prediction** -- protein + drug sent to OpenFold3 NIM for co-structure prediction
4. **3D viewer** -- self-contained HTML with jQuery + 3Dmol.js inlined, saved to canvas

## Usage

Simplest form (target auto-resolved):
```
python /sandbox/clinical-intelligence/scripts/build_viewer.py --drug metformin
```

With explicit sequence (for drugs not in the built-in table):
```
python /sandbox/clinical-intelligence/scripts/build_viewer.py --drug drugname --sequence AMINOACIDSEQ --title "Custom Title"
```

### Options

| Flag | Required | Description |
|------|----------|-------------|
| `--drug` | Yes | Drug name for PubChem SMILES lookup (e.g. `metformin`) |
| `--sequence` | No | Amino acid sequence of protein target. Auto-resolved if omitted. |
| `--title` | No | Custom viewer title |
| `--output` | No | Custom output path (defaults to `~/.openclaw/canvas/{drug}_complex.html`) |
| `--openfold-host` | No | Override OpenFold3 host IP (defaults to `172.17.0.1`) |

### Built-in drug targets

The script knows these drugs and auto-resolves their protein targets:

| Drug | Target protein |
|------|---------------|
| metformin | Insulin B-chain |
| atorvastatin | HMG-CoA reductase |
| rosuvastatin | HMG-CoA reductase |
| lisinopril | ACE |
| enalapril | ACE |
| losartan | Angiotensin II receptor type 1 |
| amlodipine | L-type calcium channel Cav1.2 |
| empagliflozin | SGLT2 |
| semaglutide | GLP-1 receptor |

For any drug in this table, just pass `--drug` and the script does the rest.

### Drugs NOT in the table

If the drug is not listed, the script exits with an error and prints the list of known drugs. In that case, you need to provide `--sequence` explicitly. Tell the user the drug is not in the built-in table and that you need a protein target sequence to proceed.

### Drugs that cannot be visualized

Biologics, enzyme mixtures, or complex formulations that PubChem cannot resolve to a single SMILES (e.g. pancrelipase, insulin glargine) will still get protein-only structure prediction -- the script handles this gracefully by predicting without a ligand.

## Output

The script saves an HTML viewer to canvas. Link it in your response as a markdown hyperlink:
```
[View 3D structure](http://localhost:18789/__openclaw__/canvas/metformin_complex.html)
```

## Confidence Scores

The viewer header displays OpenFold3 scores:
- **Confidence** -- overall prediction confidence (higher = better)
- **pLDDT** -- per-residue local confidence (0-100, >70 is good)
- **pTM** -- predicted template modeling score (0-1)
- **ipTM** -- interface predicted TM-score (complexes only, measures protein-ligand interface quality)
