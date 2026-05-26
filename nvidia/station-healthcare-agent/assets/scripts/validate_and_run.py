#!/usr/bin/env python3
"""
Code validation and execution for clinical intelligence workflows.
Used by OpenClaw agents to validate LLM-generated Python before running it.

Usage:
    python scripts/validate_and_run.py <script_path>
    python scripts/validate_and_run.py --validate-only <script_path>
    echo "print('hello')" | python scripts/validate_and_run.py --stdin
"""

import argparse
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED_IMPORTS = {
    "subprocess", "pandas", "pd", "matplotlib", "matplotlib.pyplot", "plt",
    "json", "time", "pathlib", "Path", "os", "sys", "datetime", "math",
    "collections", "re", "scipy", "scipy.stats", "numpy", "np",
}

BLOCKED_MODULES = {
    "shutil", "socket", "http.server", "ftplib", "smtplib",
    "ctypes", "multiprocessing", "threading", "signal", "pickle", "shelve",
    "importlib", "code",
}


def validate_code(code: str) -> tuple[bool, list[str], list[str]]:
    """Validate generated Python before execution.
    Returns (passed, warnings, errors)."""
    warnings = []
    errors = []

    # AST-based import detection
    imports_found = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_root = alias.name.split(".")[0]
                    imports_found.add(module_root)
                    if module_root in BLOCKED_MODULES:
                        errors.append(f"Line {node.lineno}: blocked import '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_root = node.module.split(".")[0]
                    imports_found.add(module_root)
                    if module_root in BLOCKED_MODULES:
                        errors.append(f"Line {node.lineno}: blocked import '{node.module}'")
    except SyntaxError:
        # Fall back to line-based scanning if code has syntax errors
        for i, line in enumerate(code.split("\n"), 1):
            stripped = line.strip()
            for blocked in BLOCKED_MODULES:
                if f"import {blocked}" in stripped or f"from {blocked}" in stripped:
                    errors.append(f"Line {i}: blocked import '{blocked}'")

    # AST-based checks for dangerous builtins
    DANGEROUS_BUILTINS = {"__import__", "eval", "exec", "compile"}
    try:
        tree_for_builtins = ast.parse(code)
        for node in ast.walk(tree_for_builtins):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name in DANGEROUS_BUILTINS:
                    errors.append(f"Line {node.lineno}: dangerous builtin '{name}()'")
                # Reject subprocess calls with shell=True
                if isinstance(func, ast.Attribute) and func.attr == "run":
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            errors.append(f"Line {node.lineno}: subprocess with shell=True is not allowed")
                elif isinstance(func, ast.Name) and func.id == "run":
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            errors.append(f"Line {node.lineno}: subprocess with shell=True is not allowed")
    except SyntaxError:
        # Fall back to line-based scanning if code has syntax errors
        for i, line in enumerate(code.split("\n"), 1):
            stripped = line.strip()
            for dangerous in ("__import__", "compile(", "eval(", "exec("):
                if dangerous in stripped and not stripped.startswith("#"):
                    errors.append(f"Line {i}: dangerous builtin '{dangerous}'")

    if "requests" in imports_found:
        warnings.append("The requests library does not work in the sandbox. Use subprocess.run(['curl', ...]) + json.loads() instead (see fhir_helpers.py).")

    if "85354-9" not in code and "8480-6" in code:
        warnings.append("Queries systolic BP (8480-6) without BP panel (85354-9) check")

    if "_count=" not in code and "_count =" not in code:
        if "Condition?code=" in code or "Condition?" in code:
            warnings.append("Cohort query without _count parameter -- defaults to 20 results")

    if "get('entry'" not in code and '.get("entry"' not in code:
        if "['entry']" in code:
            warnings.append("Uses ['entry'] instead of .get('entry', []) -- crashes on empty bundles")

    return len(errors) == 0, warnings, errors


def run_code(code: str, work_dir: str | None = None) -> tuple[int, str, str, list[str]]:
    """Execute validated Python code. Returns (exit_code, stdout, stderr, chart_paths)."""
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="clinical_")

    script_path = os.path.join(work_dir, "_analysis.py")
    with open(script_path, "w") as f:
        f.write(code)

    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True, timeout=120,
            cwd=work_dir,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired:
        return (1, "", "ERROR: Script execution timed out after 120 seconds", [])

    pngs = sorted(Path(work_dir).glob("*.png"))
    return result.returncode, result.stdout, result.stderr, [str(p) for p in pngs]


def main():
    parser = argparse.ArgumentParser(description="Validate and run clinical analysis code")
    parser.add_argument("script", nargs="?", help="Path to Python script to validate/run")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't execute")
    parser.add_argument("--stdin", action="store_true", help="Read code from stdin")
    parser.add_argument("--work-dir", help="Working directory for execution")
    args = parser.parse_args()

    if args.stdin:
        code = sys.stdin.read()
    elif args.script:
        code = Path(args.script).read_text()
    else:
        parser.print_help()
        sys.exit(1)

    passed, warnings, errors = validate_code(code)

    if warnings:
        for w in warnings:
            print(f"⚠ WARNING: {w}", file=sys.stderr)

    if errors:
        for e in errors:
            print(f"✗ BLOCKED: {e}", file=sys.stderr)
        print(f"\nValidation failed: {len(errors)} error(s). Code was NOT executed.", file=sys.stderr)
        sys.exit(1)

    if args.validate_only:
        print(f"✓ Validation passed ({len(warnings)} warning(s))")
        sys.exit(0)

    exit_code, stdout, stderr, charts = run_code(code, args.work_dir)
    print(stdout)
    if stderr:
        print(f"\nSTDERR:\n{stderr}", file=sys.stderr)
    if charts:
        print(f"\nCharts saved: {', '.join(charts)}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
