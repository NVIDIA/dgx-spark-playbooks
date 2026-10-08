"""Assemble an explicit clean component source bundle, never a full release.

The approved list contains exact own-project source paths; no recursive copy or
external/private runtime discovery occurs. Missing declared future components
are recorded. No release.json or installer-entrypoint contract is emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from build_release import ReleaseError, SECRET_PATTERNS, _checked_file, clean_path, digest


def build_development(root: Path, approved: dict, output: Path) -> dict:
    if not isinstance(approved, dict) or approved.get("schemaVersion") != 1 or approved.get("kind") != "development-source":
        raise ReleaseError("An explicit development-source list is required")
    if not isinstance(approved.get("files"), list) or not approved["files"]:
        raise ReleaseError("Development source list is empty")
    exclusions = approved.get("excluded", [])
    if not isinstance(exclusions, list):
        raise ReleaseError("Excluded components must be an explicit list")
    for row in exclusions:
        if not isinstance(row, dict) or set(row) != {"path", "reason"} or not isinstance(row["reason"], str):
            raise ReleaseError("Excluded component requires path and explanation")
        clean_path(row["path"])
    root = root.resolve(strict=True)
    files, missing, names = [], [], set()
    for row in approved["files"]:
        if not isinstance(row, dict) or set(row) != {"path", "classification", "optional"} or type(row["optional"]) is not bool:
            raise ReleaseError("Source list entries require path/classification/optional")
        if row["classification"] not in {"code", "clean-template", "license", "public-pin-metadata"}:
            raise ReleaseError("Development files must be explicitly approved source/template/license/pin metadata")
        name = clean_path(row["path"])
        if name.casefold() in names:
            raise ReleaseError("Duplicate source destination")
        names.add(name.casefold())
        try:
            source = _checked_file(root, name)
        except FileNotFoundError:
            if not row["optional"]:
                raise
            missing.append(name)
            continue
        files.append((name, source, digest(source), row["classification"]))
    for name in names:
        if any("/".join(name.split("/")[:i]) in names for i in range(1, len(name.split("/")))):
            raise ReleaseError("File/directory source collision")
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    try:
        snapshot = []
        with tempfile.TemporaryDirectory(prefix="component-stage-", dir=output) as stage_dir:
            stage = Path(stage_dir)
            for name, source, expected, classification in files:
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                h, carry, count = hashlib.sha256(), b"", 0
                with source.open("rb") as reader, target.open("xb") as writer:
                    for block in iter(lambda: reader.read(1024 * 1024), b""):
                        sample = carry + block
                        if any(p.search(sample) for p in SECRET_PATTERNS):
                            raise ReleaseError("Credential-shaped content rejected: " + name)
                        carry = sample[-4096:]
                        writer.write(block)
                        h.update(block)
                        count += len(block)
                if h.hexdigest() != expected:
                    raise ReleaseError("Source changed while assembling development bundle: " + name)
                snapshot.append({"path": name, "sha256": expected, "size": count, "classification": classification})
            archive = output / "development-source.zip"
            with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
                for row in snapshot:
                    z.write(stage / row["path"], row["path"])
        metadata = {"schemaVersion": 1, "kind": "development-source", "runnableStack": False,
                    "archive": {"path": archive.name, "size": archive.stat().st_size, "sha256": digest(archive)},
                    "files": snapshot, "missingDeclaredComponents": missing, "excludedDeclaredComponents": exclusions,
                    "limits": ["Component source snapshot; not a complete runtime payload or release manifest",
                               "No services started and no fresh-host integration verification claimed",
                               "Dependencies/model runtime installation and a tested first-run entrypoint remain required"]}
        (output / "development-bundle.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return metadata
    except Exception:
        (output / "BUILD_FAILED.txt").write_text("Incomplete development bundle. Do not distribute.\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--approved-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build_development(args.source_root, json.loads(args.approved_list.read_text("utf-8-sig")), args.output)
        print(f"Assembled {len(result['files'])} exact source files; {len(result['missingDeclaredComponents'])} future components absent.")
        print("Development source bundle only; runnableStack=false; no release.json emitted.")
        return 0
    except (ValueError, OSError, TypeError) as exc:
        print("Development assembly refused: " + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
