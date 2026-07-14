#!/usr/bin/env python3
"""
Test script for the molecular visualization agent flow:
  PubChem (drug -> SMILES) + OpenFold3 NIM local (protein structure) -> Dash + py3Dmol viewer.

Usage:
  # OpenFold3 on localhost:8000 (default):
  python scripts/test_molecular_viz_agent.py

  # Custom drug / port / OpenFold3 URL:
  python scripts/test_molecular_viz_agent.py --drug "lisinopril" --port 8051 --openfold-url http://localhost:8000

Requires: requests, dash, py3Dmol.
OpenFold3 NIM must be running locally (host port defaults to 8000; set
OPENFOLD_PORT to avoid a clash with NemoClaw's nemoclaw-vllm on 8000):
  docker run --rm -p "${OPENFOLD_PORT:-8000}":8000 --gpus all --shm-size=16g \\
    -e NGC_API_KEY=$NGC_API_KEY -e NIM_OPTIMIZED_BACKEND=torch_baseline \\
    nvcr.io/nim/openfold/openfold3:latest
"""

import argparse
import os
import sys
import time
import urllib.parse

import requests

try:
    import dash
    from dash import html
    import py3Dmol
except ImportError as e:
    print("Missing dependency for viewer:", e, file=sys.stderr)
    print("Install: pip install dash py3Dmol", file=sys.stderr)
    py3Dmol = None
    dash = None


DEMO_PROTEIN_SEQUENCE = "MKTVRQERLKSIVR"


# ---------------------------------------------------------------------------
# PubChem
# ---------------------------------------------------------------------------
def drug_name_to_smiles(drug_name: str, timeout: int = 30) -> dict:
    name_enc = urllib.parse.quote(drug_name.strip())
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name_enc}"
        "/property/IsomericSMILES,MolecularFormula,MolecularWeight/JSON"
    )
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    props = r.json()["PropertyTable"]["Properties"][0]
    smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES") or ""
    return {
        "smiles": smiles,
        "formula": props.get("MolecularFormula", ""),
        "weight": props.get("MolecularWeight"),
    }


def pubchem_get_sdf(drug_name: str, timeout: int = 30) -> str | None:
    name_enc = urllib.parse.quote(drug_name.strip())
    cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name_enc}/cids/JSON"
    r = requests.get(cid_url, timeout=timeout)
    if not r.ok:
        return None
    cids = r.json().get("IdentifierList", {}).get("CID", [])
    if not cids:
        return None
    cid = cids[0]
    for path in [f"compound/cid/{cid}/record/3D/SDF", f"compound/cid/{cid}/record/SDF"]:
        sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/{path}?response_type=display"
        r = requests.get(sdf_url, timeout=timeout)
        if r.ok and len(r.content) > 100:
            return r.text
    return None


# ---------------------------------------------------------------------------
# OpenFold3 NIM (local)
# ---------------------------------------------------------------------------
def openfold3_predict(
    sequence: str,
    base_url: str = "http://localhost:8000",
    timeout: int = 300,
) -> str | None:
    """Call local OpenFold3 NIM. Returns PDB string or None."""
    url = f"{base_url.rstrip('/')}/biology/openfold/openfold3/predict"
    msa_content = f">query\n{sequence}"
    data = {
        "inputs": [
            {
                "input_id": "molecular_viz_test",
                "molecules": [
                    {
                        "type": "protein",
                        "sequence": sequence,
                        "msa": {
                            "main": {
                                "a3m": {
                                    "alignment": msa_content,
                                    "format": "a3m",
                                }
                            }
                        },
                    }
                ],
                "output_format": "pdb",
            }
        ]
    }
    try:
        r = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if not r.ok:
            print(f"  OpenFold3 error: {r.status_code} {r.text[:500]}", file=sys.stderr)
            return None
        out = r.json()
        structure = (
            out.get("outputs", [{}])[0]
            .get("structures_with_scores", [{}])[0]
            .get("structure")
        )
        return structure
    except Exception as e:
        print(f"  OpenFold3 request failed: {e}", file=sys.stderr)
        return None


def openfold3_health(base_url: str = "http://localhost:8000", timeout: int = 10) -> bool:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/v1/health/ready", timeout=timeout)
        return r.ok
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Dash + py3Dmol viewer
# ---------------------------------------------------------------------------
def run_viewer(
    drug_name: str,
    smiles: str,
    ligand_sdf: str | None,
    protein_pdb: str | None,
    port: int = 8051,
    host: str = "0.0.0.0",
) -> None:
    if not py3Dmol or not dash:
        print("Skipping viewer (dash/py3Dmol not available).")
        return

    def _wrap(frag: str) -> str:
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
            "<body style='margin:0;background:#1a1a1a'>"
            + frag + "</body></html>"
        )

    app = dash.Dash(__name__)
    panels = []

    # Panel 1: Protein structure from OpenFold3
    if protein_pdb:
        v1 = py3Dmol.view(width=620, height=480)
        v1.addModel(protein_pdb, "pdb")
        v1.setStyle({"cartoon": {"color": "spectrum", "opacity": 0.9}})
        v1.zoomTo()
        panels.append(html.Div([
            html.H3("Protein target (OpenFold3 NIM — local GPU)", style={"color": "#76B900"}),
            html.Iframe(srcDoc=_wrap(v1._make_html()), style={
                "width": "620px", "height": "480px", "border": "none",
                "borderRadius": "6px", "background": "#1a1a1a",
            }),
        ], style={"marginBottom": "20px"}))
    else:
        panels.append(html.Div([
            html.H3("Protein target", style={"color": "#76B900"}),
            html.P("OpenFold3 NIM not available or prediction failed.", style={"color": "#f88"}),
        ], style={"marginBottom": "20px"}))

    # Panel 2: Drug molecule
    if ligand_sdf:
        v2 = py3Dmol.view(width=400, height=350)
        v2.addModel(ligand_sdf, "sdf")
        v2.setStyle({"stick": {"colorscheme": "greenCarbon"}})
        v2.zoomTo()
        panels.append(html.Div([
            html.H3(f"Drug: {drug_name}", style={"color": "#76B900"}),
            html.P(f"SMILES: {smiles}", style={"color": "#aaa", "fontSize": "11px", "wordBreak": "break-all"}),
            html.Iframe(srcDoc=_wrap(v2._make_html()), style={
                "width": "400px", "height": "350px", "border": "none",
                "borderRadius": "6px", "background": "#1a1a1a",
            }),
        ], style={"marginBottom": "20px"}))
    else:
        panels.append(html.Div([
            html.H3(f"Drug: {drug_name}", style={"color": "#76B900"}),
            html.P(f"SMILES: {smiles}", style={"color": "#e0e0e0"}),
            html.P("No 3D SDF from PubChem.", style={"color": "#888"}),
        ], style={"marginBottom": "20px"}))

    app.layout = html.Div([
        html.H2("Molecular Visualization — OpenFold3 NIM (local)", style={"color": "#76B900", "marginBottom": "4px"}),
        html.P(
            "Drug from PubChem  |  Protein structure from OpenFold3 NIM running on local GB300 GPU",
            style={"color": "#aaa", "fontSize": "12px", "marginBottom": "16px"},
        ),
        html.Div(panels, style={"display": "flex", "flexWrap": "wrap", "gap": "24px", "alignItems": "flex-start"}),
        html.P(
            "Molecular visualization is for education and orientation. Not a substitute for clinical or chemical validation.",
            style={"color": "#666", "fontSize": "11px", "marginTop": "24px"},
        ),
    ], style={"background": "#1a1a1a", "color": "#e0e0e0", "padding": "24px", "fontFamily": "sans-serif", "minHeight": "100vh"})

    print(f"\nViewer: http://{host}:{port}/")
    app.run(host=host, port=port, debug=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Molecular viz: PubChem + local OpenFold3 NIM")
    p.add_argument("--drug", default="atorvastatin", help="Drug name (default: atorvastatin)")
    p.add_argument("--openfold-url", default=os.environ.get("OPENFOLD_NIM_URL", "http://localhost:8000"),
                   help="OpenFold3 NIM base URL (default: http://localhost:8000)")
    p.add_argument("--port", type=int, default=8051, help="Viewer port (default: 8051)")
    p.add_argument("--host", default="0.0.0.0", help="Viewer host (default: 0.0.0.0)")
    p.add_argument("--no-viewer", action="store_true", help="Only run API steps, skip Dash viewer")
    args = p.parse_args()

    drug_name = args.drug

    # Step 1: PubChem
    print(f"Drug: {drug_name}")
    print("Step 1: PubChem (SMILES + SDF)...")
    try:
        smiles_result = drug_name_to_smiles(drug_name)
        smiles = smiles_result["smiles"]
        print(f"  SMILES: {smiles}")
        print(f"  Formula: {smiles_result.get('formula')}  MW: {smiles_result.get('weight')}")
    except Exception as e:
        print(f"  PubChem failed: {e}")
        sys.exit(1)

    ligand_sdf = pubchem_get_sdf(drug_name)
    print(f"  SDF: {'yes (' + str(len(ligand_sdf)) + ' bytes)' if ligand_sdf else 'no'}")

    # Step 2: OpenFold3 local NIM
    print(f"Step 2: OpenFold3 NIM ({args.openfold_url})...")
    if not openfold3_health(args.openfold_url):
        print("  OpenFold3 NIM not ready (health check failed). Is the container running?")
        protein_pdb = None
    else:
        print("  Health: OK")
        t0 = time.time()
        protein_pdb = openfold3_predict(
            DEMO_PROTEIN_SEQUENCE,
            base_url=args.openfold_url,
            timeout=300,
        )
        dt = time.time() - t0
        if protein_pdb:
            print(f"  PDB: yes ({len(protein_pdb)} bytes) in {dt:.1f}s")
        else:
            print(f"  PDB: no (prediction failed, {dt:.1f}s)")

    # Step 3: Viewer
    if not args.no_viewer:
        print("Step 3: Dash + py3Dmol viewer...")
        run_viewer(drug_name, smiles, ligand_sdf, protein_pdb, port=args.port, host=args.host)
    else:
        print("Skipping viewer (--no-viewer).")


if __name__ == "__main__":
    main()
