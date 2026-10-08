"""Extract only public artifact pins and explicitly named clean native-file metadata.

Never copies a runtime tree, reads OAuth/config state, or hashes model weights.
Manifest model pins are transcribed; they are not a new download verification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

from build_release import ReleaseError, SHA256, digest, relative_path
from urllib.parse import urlsplit


def pe_machine(path: Path) -> str:
    with path.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ReleaseError("Native executable lacks a PE header")
        stream.seek(0x3C)
        offset = struct.unpack("<I", stream.read(4))[0]
        stream.seek(offset)
        if stream.read(4) != b"PE\0\0":
            raise ReleaseError("Invalid PE signature")
        machine = struct.unpack("<H", stream.read(2))[0]
    return {0xAA64: "arm64", 0x8664: "x64", 0xA641: "arm64ec", 0x14C: "x86"}.get(machine, hex(machine))


def manifest_rows(path: Path, key: str, prefix="") -> list[dict]:
    value = json.loads(path.read_text("utf-8-sig"))
    rows = value[key]
    if not isinstance(rows, list):
        raise ReleaseError("Artifact manifest list missing")
    output = []
    for row in rows:
        relative = relative_path(prefix + row["path"].replace("\\", "/"))
        url = urlsplit(row["url"])
        if url.scheme != "https" or not url.netloc or url.username or url.password:
            raise ReleaseError("Artifact URL is not a clean public HTTPS origin")
        if type(row["size"]) is not int or row["size"] <= 0 or not SHA256.fullmatch(row["sha256"]):
            raise ReleaseError("Artifact pin missing")
        output.append({"path": relative, "url": row["url"], "size": row["size"], "sha256": row["sha256"].lower(),
                       "verification": "transcribed_from_existing_manifest_not_rehashed_this_run"})
    return output


def native_rows(directory: Path, names: list[str], prefix: str) -> list[dict]:
    output = []
    for name in names:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ReleaseError("Named native component missing or symlinked")
        machine = pe_machine(path) if path.suffix.lower() in {".exe", ".dll"} else None
        output.append({"path": prefix + "/" + name, "size": path.stat().st_size, "sha256": digest(path),
                       "architecture": machine, "url": None, "verification": "local_exact_file_rehashed_and_PE_checked"})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--llama-manifest", type=Path, required=True)
    parser.add_argument("--node-directory", type=Path, required=True)
    parser.add_argument("--python-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        model_rows = manifest_rows(args.model_manifest, "downloadArtifacts")
        llama_rows = [r for r in manifest_rows(args.llama_manifest, "artifacts", "llamacpp/flash-next/") if r["path"].endswith(".zip")]
        native = native_rows(args.node_directory, ["node.exe", "LICENSE"], "runtime/node")
        native += native_rows(args.python_directory,
            ["python.exe", "python3.dll", "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt"], "runtime/python")
        output = {"schemaVersion": 1, "completeRuntimeInventory": False,
                  "modelBytes": sum(r["size"] for r in model_rows), "modelArtifacts": model_rows,
                  "llamaArchiveArtifacts": llama_rows, "localNativeComponents": native,
                  "remaining": ["Python standard library and reviewed dependencies",
                    "OpenClaw npm/plugin build dependency inventory and licenses", "Container/HA runtime and own HA provisioning",
                    "Model runtime choice and measured target-Spark settings", "A complete clean runtime entrypoint and release validation"],
                  "sourceManifestSha256": {"models": digest(args.model_manifest), "llama": digest(args.llama_manifest)}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(output, stream, indent=2)
            stream.write("\n")
        print(f"Exported {len(model_rows)} model pins, {len(llama_rows)} native archive pins, {len(native)} rehashed native files.")
        print("Architecture observations: " + ", ".join(sorted({r["architecture"] for r in native if r["architecture"]})))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("Metadata export refused: " + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
