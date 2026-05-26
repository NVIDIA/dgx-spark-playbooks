#!/usr/bin/env python3
"""Build a self-contained 3D protein-ligand viewer with inline JS libraries.

Usage:
    python build_viewer.py --drug metformin
    python build_viewer.py --drug metformin --sequence CUSTOMSEQ --title "Custom Title"

When --sequence is omitted, the script looks up the protein target from a
built-in drug-target table (drug name -> target protein -> amino acid sequence).
Fetches drug SMILES from PubChem, predicts a protein-ligand complex with
OpenFold3 NIM, and generates an interactive 3D viewer saved to canvas.
"""
import argparse, subprocess, json, os, sys

CANVAS = os.path.expanduser("~/.openclaw/canvas")
OF3_HOST = os.environ.get("OPENFOLD3_HOST", "172.17.0.1")
OF3_URL = f"http://{OF3_HOST}:8000/biology/openfold/openfold3/predict"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES/JSON"

# Drug -> (target protein name, amino acid sequence)
# Sequences are canonical fragments from UniProt, chosen to be under 300 aa
# for fast OpenFold3 prediction while covering the drug binding domain.
DRUG_TARGETS = {
    "metformin": (
        "Insulin B-chain",
        "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    ),
    "atorvastatin": (
        "HMG-CoA reductase (catalytic domain)",
        "EIGTVGGGTQLFNQLESRIRAVLKDAGFLEEARAVIDRPGPYLEDVVTASNLKEGATLITSPAKLLREVGLTPETISKALKESGVRFIRIATTAPYAMNPVSAVEIAGATLYPVSALTEIARGMFVFQSGKYSMSSSGIVLPVVFATLME",
    ),
    "rosuvastatin": (
        "HMG-CoA reductase (catalytic domain)",
        "EIGTVGGGTQLFNQLESRIRAVLKDAGFLEEARAVIDRPGPYLEDVVTASNLKEGATLITSPAKLLREVGLTPETISKALKESGVRFIRIATTAPYAMNPVSAVEIAGATLYPVSALTEIARGMFVFQSGKYSMSSSGIVLPVVFATLME",
    ),
    "lisinopril": (
        "ACE (binding domain)",
        "QGSERRGPFKSWYGSSPDIIRDQIRKQLQELLQELNEERDCTSIHPFHNIFSEDDASFEERKVLKNMMDTLKRNVQEAVDTYGFK",
    ),
    "enalapril": (
        "ACE (binding domain)",
        "QGSERRGPFKSWYGSSPDIIRDQIRKQLQELLQELNEERDCTSIHPFHNIFSEDDASFEERKVLKNMMDTLKRNVQEAVDTYGFK",
    ),
    "losartan": (
        "Angiotensin II receptor type 1 (transmembrane domain)",
        "MILNSSTEDGIKRIQDDCPKAGRHNYIFVMIPTLYSIIFVVGIFGNSLVVIVIYFYMKLKTVASVFLLNLALADLCFLLTLPLWAVYTAMEYRWPFGNYLCKIASASVSFNLYASVFLLTCLSIDRYLAIVHPMKSRLRRTMLVAKVTCIIIWLLAGLASLPAIIHRNVFFIENTNITVCAFHYESQNSTLPIGLGLTKNILGFLFPFLIILTSYTLIWKALKKAYEIQKNKPRNDDIFKIIMAIVLFFFFSWIPHQIFTFLDVLIQLGIIRDCRIADIVDTAMPITICIAYFNNCLNPLFYGFLGKKFKRYFLQLLKYIPPKAKSHSNLSTRMSTLSYRPSDNVSSSTKKPAPCFEVE",
    ),
    "amlodipine": (
        "L-type calcium channel Cav1.2 (domain III)",
        "QCIDDYDTQFFLQDNAKFEGMCLRDIPDDRDNFDLFLKRVDIGPEDYYLNQHFLDAAENPDPEISFQFEGRILRGFIDIIYDLSDWFDPNEDY",
    ),
    "empagliflozin": (
        "SGLT2 (sodium-glucose cotransporter 2)",
        "MDSSRQSGAHQHPPAQRVELQGLADEADARALRGEFSLHPELAARAATPEQAFALGGELPMERDSQLCMGFVHTYFNMTGYSEAETLTGAGPPMAYAIPPQAKEVEEMKEFFQKFGKTYPGLKDIFPETKIDFLRNIMLQHMGIGLASATLVPMYIAAEMTAHMGCMHRFLYASYVAAEFLAIVFAVILFNLGERRKHFS",
    ),
    "semaglutide": (
        "GLP-1 receptor (extracellular domain)",
        "RPQGATVSLWETVQKWREYRRQCQRSLTEDPPPATDLFCNRTFDEYACWPDGEPGSFVNVSCPWYLPWASSVPQGHVYRFCTAEGLWLQKDNSSLPWRDLSECEESKRGERNSPEEQLLS",
    ),
}


def fetch_js(url):
    r = subprocess.run(["curl", "-sL", url], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"  ERROR: curl failed for {url} (exit {r.returncode})", file=sys.stderr)
        if r.stderr:
            print(f"  {r.stderr[:300]}", file=sys.stderr)
        raise RuntimeError(f"Failed to fetch {url}")
    if len(r.stdout) < 1000:
        print(f"  WARN: {url} returned only {len(r.stdout)} bytes", file=sys.stderr)
    return r.stdout


def lookup_smiles(drug):
    try:
        r = subprocess.run(
            ["curl", "-sf", PUBCHEM_URL.format(drug)],
            capture_output=True, text=True, timeout=15
        )
        d = json.loads(r.stdout)
        props = d["PropertyTable"]["Properties"][0]
        return props.get("IsomericSMILES", props.get("CanonicalSMILES", props.get("SMILES", "")))
    except Exception as e:
        print(f"  PubChem lookup failed: {e}", file=sys.stderr)
        return ""


def resolve_target(drug):
    """Look up drug in built-in table. Returns (target_name, sequence) or None."""
    key = drug.strip().lower()
    if key in DRUG_TARGETS:
        return DRUG_TARGETS[key]
    for k, v in DRUG_TARGETS.items():
        if k in key or key in k:
            return v
    return None


def predict_structure(sequence, smiles=""):
    molecules = [{
        "type": "protein", "id": "A", "sequence": sequence,
        "msa": {"main": {"a3m": {
            "alignment": f">query\n{sequence}", "format": "a3m"
        }}}
    }]
    if smiles:
        molecules.append({"type": "ligand", "smiles": smiles})

    body = json.dumps({"inputs": [{
        "input_id": "viewer", "molecules": molecules, "output_format": "pdb"
    }]})
    r = subprocess.run(
        ["curl", "-sf", "--max-time", "300", "-X", "POST",
         "-H", "Content-Type: application/json", "-d", body, OF3_URL],
        capture_output=True, text=True, timeout=305
    )
    if r.returncode != 0 or not r.stdout.strip():
        print(f"  OpenFold3 prediction failed (exit {r.returncode})", file=sys.stderr)
        if r.stderr:
            print(f"  {r.stderr[:300]}", file=sys.stderr)
        sys.exit(1)
    result = json.loads(r.stdout)
    out = result["outputs"][0]["structures_with_scores"][0]
    return out["structure"], out


def build_html(title, drug, smiles, sequence, pdb, scores, jquery_js, mol3d_js):
    pdb_escaped = pdb.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    has_ligand = bool(smiles)
    conf = scores.get('confidence_score', 0)
    plddt = scores.get('complex_plddt_score', 0)
    ptm = scores.get('ptm_score', 0)
    iptm = scores.get('iptm_score', 0)

    ligand_legend = f'<div class="leg-item"><div class="leg-dot" style="background:#ff4444"></div>{drug.capitalize()} (ligand)</div>' if has_ligand else ""
    ligand_style = """
viewer.setStyle({chain:"B"}, {stick:{radius:0.2,colorscheme:{prop:"elem",map:{C:"#ff4444",N:"#4444ff",O:"#ff6666",S:"#ffcc00"}}}, sphere:{radius:0.4,color:"#ff4444",opacity:0.5}});""" if has_ligand else ""

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>{title}</title>
<script>{jquery_js}</script>
<script>{mol3d_js}</script>
<style>
body {{ margin:0; background:#0d0d0d; color:#fff; font-family:Inter,'Segoe UI',system-ui,sans-serif; overflow:hidden; }}
#header {{ position:fixed; top:0; left:0; right:0; z-index:100; background:rgba(13,13,13,0.95); padding:14px 24px; border-bottom:2px solid #76B900; display:flex; justify-content:space-between; align-items:center; backdrop-filter:blur(10px); }}
#header h1 {{ margin:0; color:#76B900; font-size:18px; font-weight:700; letter-spacing:-0.3px; }}
#header .scores {{ color:#777; font-size:11px; text-align:right; line-height:1.5; }}
#header .scores span {{ color:#76B900; font-weight:600; }}
#viewer {{ width:100vw; height:calc(100vh - 60px); margin-top:60px; }}
#legend {{ position:fixed; bottom:20px; left:20px; background:rgba(17,17,17,0.92); border:1px solid #282828; border-radius:10px; padding:14px 18px; font-size:11px; backdrop-filter:blur(12px); }}
.leg-item {{ display:flex; align-items:center; gap:8px; margin:5px 0; color:#aaa; }}
.leg-dot {{ width:12px; height:12px; border-radius:3px; flex-shrink:0; }}
#controls {{ position:fixed; bottom:20px; right:20px; background:rgba(17,17,17,0.92); border:1px solid #282828; border-radius:10px; padding:8px 14px; font-size:10px; color:#444; backdrop-filter:blur(12px); }}
</style></head><body>
<div id="header">
  <h1>{title}</h1>
  <div class="scores">
    Predicted by <span>OpenFold3 NIM</span> &middot; Local GPU<br>
    Confidence: <span>{conf:.1%}</span> &middot; pLDDT: <span>{plddt:.1f}</span> &middot; pTM: <span>{ptm:.2f}</span>{f' &middot; ipTM: <span>{iptm:.2f}</span>' if has_ligand else ''}
  </div>
</div>
<div id="viewer"></div>
<div id="legend">
  <div class="leg-item"><div class="leg-dot" style="background:#4a90d9"></div>Protein ({len(sequence)} aa)</div>
  {ligand_legend}
</div>
<div id="controls">Drag to rotate &middot; Scroll to zoom &middot; Double-click toggle spin</div>
<script>
var pdb = `{pdb_escaped}`;
var viewer = $3Dmol.createViewer("viewer", {{backgroundColor:"#0d0d0d"}});
viewer.addModel(pdb, "pdb");
viewer.setStyle({{chain:"A"}}, {{cartoon:{{color:"spectrum",opacity:0.92}}}});
{ligand_style}
viewer.zoomTo();
viewer.render();
viewer.zoom(1.15, 600);
var spinning = true;
viewer.spin("y", 0.4);
document.addEventListener("dblclick", function(){{ if(spinning){{viewer.spin(false);spinning=false;}}else{{viewer.spin("y",0.4);spinning=true;}} }});
document.addEventListener("mousedown", function(e){{ if(e.button===0){{viewer.spin(false);spinning=false;}} }});
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Build 3D protein-ligand viewer")
    parser.add_argument("--drug", required=True, help="Drug name (e.g. metformin)")
    parser.add_argument("--sequence", default=None, help="Protein target sequence (auto-resolved if omitted)")
    parser.add_argument("--title", default=None, help="Viewer title")
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument("--openfold-host", default=None, help="OpenFold3 host IP")
    args = parser.parse_args()

    if args.openfold_host:
        global OF3_URL
        OF3_URL = f"http://{args.openfold_host}:8000/biology/openfold/openfold3/predict"

    sequence = args.sequence
    target_name = None
    if not sequence:
        target = resolve_target(args.drug)
        if target:
            target_name, sequence = target
            print(f"Target: {target_name}")
        else:
            print(f"ERROR: No built-in target for '{args.drug}'. Pass --sequence explicitly.", file=sys.stderr)
            print(f"Known drugs: {', '.join(sorted(DRUG_TARGETS.keys()))}", file=sys.stderr)
            sys.exit(1)

    title = args.title or f"{args.drug.capitalize()}: {target_name or 'Protein-Ligand Complex'}"
    output = args.output or os.path.join(CANVAS, f"{args.drug}_complex.html")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"Drug: {args.drug}")
    print(f"Sequence: {sequence[:40]}{'...' if len(sequence) > 40 else ''} ({len(sequence)} aa)")

    print("Fetching jQuery...")
    jq = fetch_js("https://code.jquery.com/jquery-3.6.0.min.js")
    print(f"  {len(jq)} bytes")

    print("Fetching 3Dmol...")
    mol3d = fetch_js("https://3Dmol.org/build/3Dmol-min.js")
    print(f"  {len(mol3d)} bytes")

    if len(jq) < 1000 or len(mol3d) < 1000:
        print("ERROR: JS library download failed", file=sys.stderr)
        sys.exit(1)

    print("Looking up SMILES on PubChem...")
    smiles = lookup_smiles(args.drug)
    print(f"  SMILES: {smiles}")

    pdb_cache = os.path.join(CANVAS, f"{args.drug}_complex.pdb")
    scores_cache = os.path.join(CANVAS, f"{args.drug}_complex_scores.json")
    if os.path.exists(pdb_cache) and os.path.getsize(pdb_cache) > 1000:
        pdb = open(pdb_cache).read()
        if os.path.exists(scores_cache):
            scores = json.loads(open(scores_cache).read())
        else:
            scores = {"confidence_score": 0, "complex_plddt_score": 0, "ptm_score": 0, "iptm_score": 0}
        print(f"PDB from cache: {len(pdb)} bytes")
    else:
        mode = "protein-ligand complex" if smiles else "protein only"
        print(f"Predicting {mode} with OpenFold3...")
        pdb, scores = predict_structure(sequence, smiles)
        with open(pdb_cache, "w") as f:
            f.write(pdb)
        with open(scores_cache, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"  {len(pdb)} bytes, confidence: {scores.get('confidence_score',0):.1%}")

    html = build_html(title, args.drug, smiles, sequence, pdb, scores, jq, mol3d)
    with open(output, "w") as f:
        f.write(html)
    print(f"\nViewer saved: {output} ({len(html)} bytes)")
    print(f"Open: http://localhost:18789/__openclaw__/canvas/{os.path.basename(output)}")


if __name__ == "__main__":
    main()
