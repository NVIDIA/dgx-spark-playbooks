#!/usr/bin/env python3
"""
OpenFold3 integration test suite.
Verifies every layer: infrastructure, network, agent, end-to-end.

Usage (on DGX host):
    python3 scripts/test_openfold3_integration.py

Usage (inside sandbox, for T5-T7 only):
    python scripts/test_openfold3_integration.py --sandbox
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.parse


def _docker_bridge_ip():
    """Auto-detect the docker0 bridge IP (override with DOCKER_BRIDGE_IP).

    The bridge IP varies by host; a hard-coded value breaks the
    sandbox->OpenFold3 reachability tests on machines whose docker0 differs.
    """
    override = os.environ.get("DOCKER_BRIDGE_IP")
    if override:
        return override
    try:
        out = subprocess.run(
            ["ip", "-4", "addr", "show", "docker0"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for token in out.split():
            if token.startswith("inet"):
                continue
            if token.count(".") == 3:
                return token.split("/")[0]
    except Exception:
        pass
    return "172.17.0.1"


OPENFOLD_PORT = os.environ.get("OPENFOLD_PORT", "8000")
OPENFOLD_URL = f"http://localhost:{OPENFOLD_PORT}"
OPENFOLD_BRIDGE_URL = f"http://{_docker_bridge_ip()}:{OPENFOLD_PORT}"
OLLAMA_URL = "http://localhost:11434"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov"
TEST_SEQUENCE = "MKTVRQERLKSIVR"
TEST_DRUG = "metformin"
METFORMIN_SMILES = "CN(C)C(=N)NC(=N)N"

results = []


def report(test_id, name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((test_id, name, status, detail))
    print(f"  [{status}] {test_id}: {name}" + (f" -- {detail}" if detail else ""))


def http_get(url, timeout=30):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode()


def http_post_json(url, data, timeout=120):
    payload = json.dumps(data).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode()


# ── Layer 1: Infrastructure (run on host) ──

def t1_openfold_health():
    try:
        status, _ = http_get(f"{OPENFOLD_URL}/v1/health/ready", timeout=10)
        report("T1", "OpenFold3 NIM health", status == 200, f"HTTP {status}")
    except Exception as e:
        report("T1", "OpenFold3 NIM health", False, str(e))


def t2_openfold_prediction():
    try:
        data = {
            "request_id": "test",
            "inputs": [{
                "input_id": "test",
                "molecules": [{
                    "type": "protein",
                    "id": "A",
                    "sequence": TEST_SEQUENCE,
                    "msa": {
                        "main_db": {
                            "csv": {
                                "alignment": f"key,sequence\n-1,{TEST_SEQUENCE}",
                                "format": "csv"
                            }
                        }
                    }
                }],
                "output_format": "pdb"
            }]
        }
        status, body = http_post_json(
            f"{OPENFOLD_URL}/biology/openfold/openfold3/predict", data, timeout=180
        )
        has_atom = "ATOM" in body or "atom" in body.lower()
        report("T2", "OpenFold3 prediction", status == 200 and has_atom,
               f"HTTP {status}, PDB={'yes' if has_atom else 'no'}, {len(body)} bytes")
    except Exception as e:
        report("T2", "OpenFold3 prediction", False, str(e))


def t3_gpu_memory():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        total_used = 0
        total_mem = 0
        for line in r.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2:
                total_used += int(parts[0])
                total_mem += int(parts[1])
        ok = total_used < 250000
        report("T3", "GPU memory coexistence", ok, f"{total_used} MiB used / {total_mem} MiB total (across {len(r.stdout.strip().splitlines())} GPU(s))")
    except Exception as e:
        report("T3", "GPU memory coexistence", False, str(e))


def t4_ollama_coexistence():
    try:
        data = {"model": "nemotron-3-super", "messages": [{"role": "user", "content": "ping"}], "stream": False}
        status, body = http_post_json(f"{OLLAMA_URL}/api/chat", data, timeout=120)
        has_response = "message" in body
        report("T4", "Ollama still works", status == 200 and has_response, f"HTTP {status}")
    except Exception as e:
        report("T4", "Ollama still works", False, str(e))


# ── Layer 2: Network/Sandbox ──

def t5_pubchem_reachable():
    try:
        url = f"{PUBCHEM_URL}/rest/pug/compound/name/{TEST_DRUG}/property/IsomericSMILES/JSON"
        status, body = http_get(url, timeout=15)
        has_smiles = "IsomericSMILES" in body
        report("T5", "PubChem reachable", status == 200 and has_smiles, f"HTTP {status}")
    except Exception as e:
        report("T5", "PubChem reachable", False, str(e))


def t6_openfold_reachable_from_sandbox():
    try:
        status, _ = http_get(f"{OPENFOLD_BRIDGE_URL}/v1/health/ready", timeout=10)
        report("T6", "OpenFold3 reachable (bridge IP)", status == 200, f"HTTP {status}")
    except Exception as e:
        report("T6", "OpenFold3 reachable (bridge IP)", False, str(e))


def t7_security_enforced():
    try:
        http_get("https://google.com", timeout=5)
        report("T7", "Security (google blocked)", False, "Google was reachable -- sandbox is leaking")
    except Exception:
        report("T7", "Security (google blocked)", True, "Blocked as expected")


# ── Layer 3: Agent ──

def t8_drug_resolution():
    try:
        url = f"{PUBCHEM_URL}/rest/pug/compound/name/{TEST_DRUG}/property/IsomericSMILES/JSON"
        status, body = http_get(url, timeout=15)
        data = json.loads(body)
        smiles = data["PropertyTable"]["Properties"][0].get("IsomericSMILES", "")
        ok = METFORMIN_SMILES in smiles or len(smiles) > 5
        report("T8", "Drug SMILES resolution", ok, f"SMILES={smiles[:50]}")
    except Exception as e:
        report("T8", "Drug SMILES resolution", False, str(e))


def t9_openfold_from_sandbox():
    try:
        data = {
            "request_id": "sandbox-test",
            "inputs": [{
                "input_id": "sandbox-test",
                "molecules": [{
                    "type": "protein",
                    "id": "A",
                    "sequence": TEST_SEQUENCE,
                    "msa": {
                        "main_db": {
                            "csv": {
                                "alignment": f"key,sequence\n-1,{TEST_SEQUENCE}",
                                "format": "csv"
                            }
                        }
                    }
                }],
                "output_format": "pdb"
            }]
        }
        status, body = http_post_json(
            f"{OPENFOLD_BRIDGE_URL}/biology/openfold/openfold3/predict", data, timeout=180
        )
        has_atom = "ATOM" in body
        report("T9", "OpenFold3 from sandbox", status == 200 and has_atom,
               f"HTTP {status}, PDB={'yes' if has_atom else 'no'}")
    except Exception as e:
        report("T9", "OpenFold3 from sandbox", False, str(e))


# ── Summary ──

def print_summary():
    print("\n" + "=" * 60)
    print("OPENFOLD3 INTEGRATION TEST RESULTS")
    print("=" * 60)
    passed = sum(1 for _, _, s, _ in results if s == "PASS")
    failed = sum(1 for _, _, s, _ in results if s == "FAIL")
    for tid, name, status, detail in results:
        print(f"  {status:4s}  {tid:4s}  {name}" + (f"  ({detail})" if detail else ""))
    print(f"\n{passed} passed, {failed} failed out of {len(results)} tests")
    print("=" * 60)
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="OpenFold3 integration tests")
    parser.add_argument("--sandbox", action="store_true", help="Run sandbox-only tests (T5-T9)")
    parser.add_argument("--infra", action="store_true", help="Run infrastructure tests only (T1-T4)")
    args = parser.parse_args()

    if args.sandbox:
        print("Running sandbox tests (T5-T9)...")
        t5_pubchem_reachable()
        t6_openfold_reachable_from_sandbox()
        t7_security_enforced()
        t8_drug_resolution()
        t9_openfold_from_sandbox()
    elif args.infra:
        print("Running infrastructure tests (T1-T4)...")
        t1_openfold_health()
        if not any(s == "FAIL" for _, _, s, _ in results):
            t2_openfold_prediction()
        else:
            print("  [SKIP] T2: OpenFold3 not healthy, skipping prediction test")
        t3_gpu_memory()
        t4_ollama_coexistence()
    else:
        print("Running all tests (T1-T9)...")
        t1_openfold_health()
        if any(s == "FAIL" for _, _, s, _ in results):
            print("  [SKIP] T2-T4: OpenFold3 not healthy, skipping remaining infra tests")
        else:
            t2_openfold_prediction()
            t3_gpu_memory()
            t4_ollama_coexistence()
            t5_pubchem_reachable()
            t6_openfold_reachable_from_sandbox()
            t7_security_enforced()
            t8_drug_resolution()
            t9_openfold_from_sandbox()

    all_passed = print_summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
