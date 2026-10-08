"""Check the Git-visible source inventory without echoing credential values."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packaging'))
from build_release import SECRET_PATTERNS

PRIVATE_DIRS = {'private', 'evidence', 'downloads', 'machines', 'linux', 'review', 'output',
                'node_modules', '__pycache__', '.storage', '.ssh', 'sessions', 'household'}
PRIVATE_NAMES = {'api-token.txt', 'connection.json', 'ha-credentials.json', 'home-settings.json',
                 'health-config.json', 'household.json', 'openclaw.json'}
PRIVATE_SUFFIXES = {'.exe', '.dll', '.pyd', '.zip', '.gguf', '.vhdx', '.iso', '.sqlite', '.sqlite3', '.db', '.pem', '.pfx', '.log'}


def inspect_bytes(name, content):
    findings = []
    for i, pattern in enumerate(SECRET_PATTERNS):
        for match in pattern.finditer(content):
            findings.append({'file': name, 'line': content[:match.start()].count(b'\n') + 1, 'rule': 'credential-pattern-' + str(i + 1)})
    # An actual local user's home path is not a portable public/source default.
    for match in re.finditer(rb'(?i)[A-Z]:[\\/]+Users[\\/]+(?!Public\b|Default\b|<|\$|\{)[A-Za-z0-9_.-]+', content):
        findings.append({'file': name, 'line': content[:match.start()].count(b'\n') + 1, 'rule': 'personal-home-path'})
    return findings


def audit(root=ROOT):
    listing = subprocess.run(['git', '-C', str(root), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                             check=True, capture_output=True)
    names = sorted(set(n.decode('utf-8') for n in listing.stdout.split(b'\0') if n))
    findings, inventory = [], []
    for name in names:
        path = root / name
        parts = {p.lower() for p in Path(name).parts}
        if parts & PRIVATE_DIRS or path.name.lower() in PRIVATE_NAMES or path.name.startswith('.env') or path.suffix.lower() in PRIVATE_SUFFIXES:
            findings.append({'file': name, 'rule': 'private-or-generated-path'})
            continue
        linked = any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)() for p in (path, *path.parents) if p != root.parent)
        if linked or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            findings.append({'file': name, 'rule': 'nonregular-source-file'})
            continue
        data = path.read_bytes()
        findings.extend(inspect_bytes(name, data))
        inventory.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    return {'ok': not findings, 'filesChecked': len(inventory), 'findings': findings, 'inventory': inventory,
            'scope': 'Current Git-visible source; ignored private files are not read. This is a focused scan, not proof that every possible secret is absent.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = audit()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'inventory'}, indent=2))
    raise SystemExit(0 if result['ok'] else 1)
