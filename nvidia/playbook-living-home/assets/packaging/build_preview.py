"""Assemble the bounded Windows ARM64 workspace preview; never a full-stack release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from build_release import _checked_file, SECRET_PATTERNS

PYTHON_SHA = '155be84ccb57c6331cf0e39001c78a1dfac3be62f403f3ff5f2e29b80dda7ebe'
TZDATA_SHA = 'b683bd1b6659ddcd810ff02ad09ba821d4bf1065072805063eb35c49617905ac'
SOURCE_FILES = [
    'LICENSE', 'README.md', 'SECURITY.md', 'THIRD-PARTY-NOTICES.md',
    'runtime/ui/START-HERE.md',
    'runtime/ui/app.py', 'runtime/ui/web/index.html', 'runtime/ui/web/app.js', 'runtime/ui/web/style.css',
    'runtime/backend/server.py', 'runtime/backend/household_plans.py', 'runtime/backend/plan_journal.py',
    'runtime/backend/plan_evidence.py', 'runtime/backend/records_store.py', 'runtime/backend/saved_reports.py',
    'runtime/backend/PROVENANCE.json', 'runtime/health/collect_home_status.py',
    'workflows/initialize_household.py', 'workflows/household_state.py', 'workflows/configure_home_report.py',
    'workflows/Setup-HomeReport.ps1', 'workflows/WORKFLOWS.md', 'workflows/home-report.example.json',
    'runtime/openclaw-adapter/package.json', 'runtime/openclaw-adapter/openclaw.plugin.json',
    'runtime/openclaw-adapter/README.md', 'runtime/openclaw-adapter/PROVENANCE.md',
    *['runtime/openclaw-adapter/dist/' + name + '.js' for name in ('index', 'client', 'tools', 'schemas', 'validation')],
]


def extract_pinned(path, digest, root):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Dependency checksum mismatch: ' + path.name)
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            target = root / item.filename
            if '\\' in item.filename or not target.resolve().is_relative_to(root.resolve()) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe dependency ZIP entry')
        archive.extractall(root)


def build(source, output, python, tzdata):
    source = source.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    app = output / 'LivingHome-Workspace'
    app.mkdir()
    for name in SOURCE_FILES:
        original = _checked_file(source, name)
        if any(pattern.search(original.read_bytes()) for pattern in SECRET_PATTERNS):
            raise ValueError('Credential-shaped source content rejected: ' + name)
        target = app / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    shutil.copyfile(_checked_file(source, 'dist/LivingHome.exe'), app / 'LivingHome.exe')
    guide = _checked_file(source, 'runtime/ui/START-HERE.md').read_text(encoding='utf-8')
    (app / 'START-HERE.md').write_text(
        guide.replace('../../workflows/WORKFLOWS.md', 'workflows/WORKFLOWS.md'),
        encoding='utf-8',
    )
    runtime = app / 'runtime/python'
    extract_pinned(python, PYTHON_SHA, runtime)
    extract_pinned(tzdata, TZDATA_SHA, runtime)
    inventory = [{'path': p.relative_to(app).as_posix(), 'bytes': p.stat().st_size,
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(app.rglob('*')) if p.is_file()]
    meta = {'kind': 'living-home-local-workspace-preview', 'architecture': 'arm64',
            'fullStackRelease': False, 'includesModel': False, 'includesHomeAssistant': False,
            'includesOpenClaw': False, 'signedLauncher': False, 'files': inventory,
            'dependencies': {'python': {'version': '3.14.8', 'sha256': PYTHON_SHA}, 'tzdata': {'version': '2026.5', 'sha256': TZDATA_SHA}}}
    (app / 'package-inventory.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    zip_path = output / 'LivingHome-Workspace-Windows-ARM64.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(app.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    result = {'archive': str(zip_path), 'bytes': zip_path.stat().st_size,
              'sha256': hashlib.sha256(zip_path.read_bytes()).hexdigest(), 'fullStackRelease': False}
    (output / 'preview-receipt.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python-archive', type=Path, required=True)
    parser.add_argument('--tzdata-wheel', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_root, args.output, args.python_archive, args.tzdata_wheel), indent=2))
