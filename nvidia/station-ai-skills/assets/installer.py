#!/usr/bin/env python3
"""Safe project installer for DGX Station AI Skills."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
BUNDLED_CLI = "dgx-assist.pyz"
SKILLS = (
    "dgx-station",
    "dgx-station-inference",
    "dgx-station-mig",
    "dgx-station-diagnose",
)
HARNESS_PATHS = {
    "claude": Path(".claude/skills"),
    "codex": Path(".agents/skills"),
    "gemini": Path(".gemini/skills"),
    "cursor": Path(".cursor/skills"),
    "lah": Path(".lah/skills"),
}
HARNESS_CONTEXTS = {
    "claude": "CLAUDE.md",
    "codex": "AGENTS.md",
    "gemini": "GEMINI.md",
    "cursor": "AGENTS.md",
    "lah": "AGENTS.md",
}
BEGIN = "<!-- BEGIN NVIDIA DGX STATION AI SKILLS -->"
END = "<!-- END NVIDIA DGX STATION AI SKILLS -->"
BLOCK_BODY = """\
<!-- BEGIN NVIDIA DGX STATION AI SKILLS -->
## DGX Station routing

- Detect the actual platform with `.dgx-station/bin/dgx-assist system inspect` before advising.
- Follow the detected compatibility profile and per-feature capabilities; do not infer behavior from a version prefix.
- Use `dgx-assist` and the applicable `dgx-station*` skill.
- Search the pinned NVIDIA playbooks before giving Station-specific commands.
- Never assume an `nvidia-smi` index is a CUDA ordinal; select launch devices by UUID.
- Never execute an unvalidated recipe or conceal a mutation, network exposure, or large download.
<!-- END NVIDIA DGX STATION AI SKILLS -->
"""
LEGACY_HASHES = {
    "d00e4f5f375f63a61e90f4757c6170967812eef03c4041123e20ad445f721ffd",
    "6e81f15965bbacba4ed300aeba17315ebd2212fc458e7666817e36843059e7c0",
    "0981a3ead0260d0acb9f41c4a5bfe2b7600ed823b8c11b0642d0a60d99339804",
    "6f5cf2697cefca1ae72de3847b95d04168572548722e2b1c9ae525e16a317a45",
    "07bb59985183d79d418762a93c322c4bceacd949d3a1c8bf70e8c2eb4d2be5eb",
    "273df07d27bfc208b897f524a46515e4ce09a697b0ca9c0b1ff84a929ce6037d",
    "675bc02c33804ab74280fb9478f78eda257be2844301bfcb296fc93f4f895a03",
    "797fa36005e7d0d1c57b7868dc767aab755d0351be7119ebcec235b83797d876",
    "802923fffabd32aee7c93f88f9a577ee925aedd9edcba225ee2e2d68e0af349b",
    "9b58df3ad7ab7056877bc916bcde7376e20e1e09c39a07e227a4c22d3a625509",
    "f362be13c14d5551b2273ee9c23d0330f565e2e4e1eab1409f9663c5db4649ca",
    "ffabe0f22b96dae4acd1f121e364c2e7e647ef40b75c1351e8ce03e97b7ea316",
}
LEGACY_NAMES = ("vllm-setup", "sglang-setup", "mig-configure", "dgx-diagnose")


class InstallError(RuntimeError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assets_dir() -> Path:
    return Path(__file__).resolve().parent


def manifest_path(target: Path) -> Path:
    return target / ".dgx-station" / "install-manifest.json"


def read_manifest(target: Path) -> dict[str, Any] | None:
    path = manifest_path(target)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError(f"cannot read installation manifest: {exc}") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or not isinstance(value.get("files"), dict)
        or not isinstance(value.get("contexts"), dict)
        or not isinstance(value.get("harnesses"), list)
        or any(
            not isinstance(harness, str) or harness not in HARNESS_PATHS
            for harness in value["harnesses"]
        )
        or any(
            not isinstance(name, str)
            or not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
            for name, checksum in value["files"].items()
        )
        or any(
            not isinstance(name, str)
            or not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
            for name, checksum in value["contexts"].items()
        )
    ):
        raise InstallError("installation manifest schema is unsupported")
    return value


def write_manifest(target: Path, value: dict[str, Any]) -> None:
    path = manifest_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".manifest.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def normalize_target(value: str, *, create: bool, allow_missing: bool = False) -> Path:
    candidate = Path(value).expanduser().absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise InstallError(
                f"target path may not traverse a symbolic link: {component}"
            )
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    elif not candidate.is_dir() and not allow_missing:
        raise InstallError("target directory does not exist")
    return candidate.resolve()


def safe_path(target: Path, relative: str | Path) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise InstallError(f"unsafe relative path: {relative}")
    candidate = target / relative_path
    current = target
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise InstallError(f"refusing symbolic-link destination: {current}")
    resolved_parent = candidate.parent.resolve(strict=False)
    if target != resolved_parent and target not in resolved_parent.parents:
        raise InstallError(f"destination escapes target: {candidate}")
    return candidate


def replace_block(original: str) -> str:
    if original.count(BEGIN) != original.count(END):
        raise InstallError("managed context markers are unbalanced")
    if original.count(BEGIN) > 1:
        raise InstallError("managed context block appears more than once")
    if BEGIN in original:
        pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
        return pattern.sub(BLOCK_BODY.strip(), original).rstrip() + "\n"
    separator = (
        ""
        if not original or original.endswith("\n\n")
        else ("\n" if original.endswith("\n") else "\n\n")
    )
    return original + separator + BLOCK_BODY


def remove_block(original: str) -> str:
    if BEGIN not in original:
        return original
    if original.count(BEGIN) != 1 or original.count(END) != 1:
        raise InstallError("managed context markers are ambiguous")
    pattern = re.compile(
        r"\n?" + re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", re.DOTALL
    )
    value = pattern.sub("\n", original)
    return value.lstrip("\n") if value.strip() else ""


def block_hash(value: str) -> str | None:
    match = re.search(re.escape(BEGIN) + r".*?" + re.escape(END), value, re.DOTALL)
    return (
        hashlib.sha256(match.group(0).strip().encode()).hexdigest() if match else None
    )


def show_diff(path: Path, old: str, new: str) -> None:
    relative = str(path)
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"{relative} (current)",
        tofile=f"{relative} (planned)",
        n=0,
    )
    rendered = "".join(diff)
    if rendered:
        print(rendered, end="")


def _source_files(source: Path) -> list[Path]:
    values: list[Path] = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise InstallError(f"skill source contains a symbolic link: {path}")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            values.append(path)
    return values


def _copy_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return digest(destination)


def _remove_empty_parents(path: Path, target: Path) -> None:
    current = path
    while current != target and target in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _harnesses(value: str) -> list[str]:
    return list(HARNESS_PATHS) if value == "all" else [value]


def _check_native_support(harnesses: list[str]) -> None:
    if "gemini" in harnesses and shutil.which("gemini"):
        result = subprocess.run(
            ["gemini", "skills", "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if result.returncode != 0:
            raise InstallError(
                "installed Gemini CLI does not expose native Agent Skills; upgrade Gemini CLI "
                "to a release with `gemini skills` support"
            )


def _manifest(
    harnesses: list[str], files: dict[str, str], contexts: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "installed_at": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_bundle": "station-ai-skills-v1",
        "harnesses": sorted(harnesses),
        "files": files,
        "contexts": contexts,
    }


def bundled_cli() -> Path:
    path = assets_dir() / BUNDLED_CLI
    if not path.is_file():
        raise InstallError(f"missing bundled CLI: {path}")
    return path


def install(
    target: Path, harness_value: str, *, dry_run: bool, update: bool = False
) -> None:
    harnesses = _harnesses(harness_value)
    _check_native_support(harnesses)
    old_manifest = read_manifest(target)
    if update and old_manifest is None:
        raise InstallError("no managed installation exists; run install first")
    if update:
        assert old_manifest is not None
        harnesses = list(old_manifest.get("harnesses", []))
    elif old_manifest is not None:
        harnesses = sorted(set(harnesses) | set(old_manifest.get("harnesses", [])))
    managed_files = dict(old_manifest.get("files", {})) if old_manifest else {}
    contexts = dict(old_manifest.get("contexts", {})) if old_manifest else {}
    planned_files: dict[str, str] = {}
    changed_contexts: list[tuple[Path, str, str]] = []
    copy_plan: list[tuple[Path, Path]] = []

    source_skills = assets_dir() / "skills"
    for harness in harnesses:
        base = HARNESS_PATHS[harness]
        for skill in SKILLS:
            source = source_skills / skill
            if not source.is_dir():
                raise InstallError(f"missing bundled skill: {skill}")
            for source_file in _source_files(source):
                suffix = source_file.relative_to(source)
                relative = base / skill / suffix
                destination = safe_path(target, relative)
                relative_string = relative.as_posix()
                source_hash = digest(source_file)
                if destination.exists() and relative_string not in managed_files:
                    raise InstallError(f"unmanaged skill collision: {destination}")
                if (
                    destination.exists()
                    and relative_string in managed_files
                    and digest(destination) != managed_files[relative_string]
                ):
                    print(
                        f"warning: preserving modified managed file {destination}",
                        file=sys.stderr,
                    )
                    planned_files[relative_string] = managed_files[relative_string]
                    continue
                planned_files[relative_string] = source_hash
                if not destination.exists() or digest(destination) != source_hash:
                    copy_plan.append((source_file, destination))

    context_names = sorted({HARNESS_CONTEXTS[harness] for harness in harnesses})
    for name in context_names:
        path = safe_path(target, name)
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        expected_block = contexts.get(name)
        actual_block = block_hash(old)
        if actual_block is not None and expected_block is None:
            raise InstallError(f"unmanaged context block collision: {path}")
        if expected_block is not None and actual_block != expected_block:
            print(
                f"warning: preserving modified managed context block {path}",
                file=sys.stderr,
            )
            continue
        new = replace_block(old)
        show_diff(path, old, new)
        if old != new:
            changed_contexts.append((path, old, new))
        contexts[name] = hashlib.sha256(BLOCK_BODY.strip().encode()).hexdigest()

    cli_relative = Path(".dgx-station/bin/dgx-assist")
    cli_destination = safe_path(target, cli_relative)
    cli_relative_string = cli_relative.as_posix()
    if cli_destination.exists() and cli_relative_string not in managed_files:
        raise InstallError(f"unmanaged CLI collision: {cli_destination}")
    preserve_cli = bool(
        cli_destination.exists()
        and cli_relative_string in managed_files
        and digest(cli_destination) != managed_files[cli_relative_string]
    )
    cli_source = bundled_cli()
    if preserve_cli:
        print(
            f"warning: preserving modified managed CLI {cli_destination}",
            file=sys.stderr,
        )
        planned_files[cli_relative_string] = managed_files[cli_relative_string]
    elif dry_run:
        planned_files[cli_relative_string] = digest(cli_source)

    if dry_run:
        for _source_file, destination in copy_plan:
            print(f"WOULD WRITE {destination}")
        if not preserve_cli:
            print(f"WOULD WRITE {cli_destination}")
        print("DRY-RUN: no files changed")
        return

    pending = {
        destination.relative_to(target).as_posix() for _source, destination in copy_plan
    }
    if not preserve_cli:
        pending.add(cli_relative_string)
    # An old managed file stays on disk holding its old content until its
    # overwrite actually lands, so it has to stay in the recovery manifest.
    # Dropping it would make a partially failed update look like an unmanaged
    # collision and block every later install, update, and uninstall.
    written_files = {
        name: checksum
        for name, checksum in managed_files.items()
        if safe_path(target, name).exists()
    }
    written_files.update(
        {
            name: checksum
            for name, checksum in planned_files.items()
            if name not in pending
        }
    )
    written_contexts = dict(old_manifest.get("contexts", {})) if old_manifest else {}
    try:
        for source_file, destination in copy_plan:
            relative_written = destination.relative_to(target).as_posix()
            written_files[relative_written] = _copy_file(source_file, destination)
            print(f"WROTE {destination}")
        if not preserve_cli:
            written_files[cli_relative_string] = _copy_file(cli_source, cli_destination)
            os.chmod(cli_destination, 0o755)
            print(f"WROTE {cli_destination}")
        if changed_contexts:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            backup_root = target / ".dgx-station" / "backups" / stamp
            for path, old, new in changed_contexts:
                if path.exists():
                    backup = backup_root / path.name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_text(old, encoding="utf-8")
                    os.chmod(backup, 0o600)
                    print(f"BACKUP {backup}")
                path.write_text(new, encoding="utf-8")
                written_contexts[path.name] = contexts[path.name]
                print(f"WROTE {path}")
    except OSError:
        # Claim ownership of whatever landed so the target stays recoverable:
        # without a manifest those files read as unmanaged collisions and block
        # every later install, update, and uninstall.
        write_manifest(
            target, _manifest(harnesses, written_files, written_contexts)
        )
        print(
            f"error: installation failed partway; recorded {manifest_path(target)} "
            "for completed files. Re-run install to finish, or uninstall to "
            "remove them.",
            file=sys.stderr,
        )
        raise
    write_manifest(target, _manifest(harnesses, written_files, contexts))
    print(f"WROTE {manifest_path(target)}")


def status(target: Path) -> int:
    manifest = read_manifest(target)
    if manifest is None:
        print("not installed")
        return 1
    unhealthy = False
    for relative, expected in sorted(manifest.get("files", {}).items()):
        path = safe_path(target, relative)
        if not path.exists():
            print(f"MISSING {path}")
            unhealthy = True
        elif digest(path) != expected:
            print(f"MODIFIED {path}")
            unhealthy = True
        else:
            print(f"OK {path}")
    for name in sorted(manifest.get("contexts", {})):
        path = safe_path(target, name)
        if not path.exists():
            print(f"MISSING-BLOCK {path}")
            unhealthy = True
            continue
        text = path.read_text(encoding="utf-8")
        actual = block_hash(text)
        if actual != manifest["contexts"][name]:
            print(f"MODIFIED-BLOCK {path}")
            unhealthy = True
        else:
            print(f"OK-BLOCK {path}")
    return 1 if unhealthy else 0


def migrate(target: Path, *, dry_run: bool) -> None:
    removed = 0
    preserved = 0
    candidates: list[Path] = []
    for base in HARNESS_PATHS.values():
        for name in LEGACY_NAMES:
            candidates.append(safe_path(target, base / name / "SKILL.md"))
    for name in LEGACY_NAMES:
        candidates.extend(
            [
                safe_path(target, Path(".gemini/commands") / f"{name}.md"),
                safe_path(target, Path(".cursor/rules") / f"{name}.mdc"),
            ]
        )
    for path in candidates:
        if not path.exists():
            continue
        if digest(path) not in LEGACY_HASHES:
            print(f"warning: preserving modified legacy skill {path}", file=sys.stderr)
            preserved += 1
            continue
        print(f"{'WOULD REMOVE' if dry_run else 'REMOVED'} {path}")
        if not dry_run:
            path.unlink()
            _remove_empty_parents(path.parent, target)
        removed += 1
    print(f"legacy migration: matched={removed} modified_preserved={preserved}")


def uninstall(target: Path, *, dry_run: bool) -> None:
    manifest = read_manifest(target)
    if manifest is None:
        raise InstallError("no managed installation exists")
    for relative, expected in sorted(manifest.get("files", {}).items(), reverse=True):
        path = safe_path(target, relative)
        if not path.exists():
            continue
        if digest(path) != expected:
            print(f"warning: preserving modified managed file {path}", file=sys.stderr)
            continue
        print(f"{'WOULD REMOVE' if dry_run else 'REMOVED'} {path}")
        if not dry_run:
            path.unlink()
            _remove_empty_parents(path.parent, target)
    for name in sorted(manifest.get("contexts", {})):
        path = safe_path(target, name)
        if not path.exists():
            continue
        old = path.read_text(encoding="utf-8")
        if block_hash(old) != manifest["contexts"][name]:
            print(
                f"warning: preserving modified managed context block {path}",
                file=sys.stderr,
            )
            continue
        new = remove_block(old)
        show_diff(path, old, new)
        if not dry_run and old != new:
            if new:
                path.write_text(new, encoding="utf-8")
            else:
                path.unlink()
    if dry_run:
        print("DRY-RUN: no files changed")
        return
    manifest_path(target).unlink()
    print(f"REMOVED {manifest_path(target)}")


def install_cli_user() -> None:
    destination = (
        Path(os.environ.get("XDG_BIN_HOME", Path.home() / ".local/bin")) / "dgx-assist"
    )
    if destination.is_symlink():
        raise InstallError("refusing to replace a symbolic-link CLI destination")
    if destination.exists():
        raise InstallError(
            f"refusing to overwrite existing user CLI destination: {destination}"
        )
    source = bundled_cli()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage inside the destination directory and rename, so a failed copy or
    # chmod cannot leave a partial file that the existence check above would
    # then refuse to overwrite on the next run.
    staged = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, staged)
        os.chmod(staged, 0o755)
        staged.replace(destination)
    except OSError:
        staged.unlink(missing_ok=True)
        raise
    print(f"WROTE {destination}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="install.sh",
        description=(
            "Safely install DGX Station skills and dgx-assist into a project. "
            "The target is the project that should receive the skills."
        ),
        epilog=(
            "Safe first step: install.sh install --harness codex "
            "--target /path/to/project --dry-run"
        ),
    )
    commands = result.add_subparsers(dest="command", required=True, title="commands")
    install_parser = commands.add_parser(
        "install",
        help="Install skills and a project-local CLI.",
        description="Install skills and a project-local CLI into the target project.",
    )
    install_parser.add_argument(
        "--harness",
        required=True,
        choices=(*HARNESS_PATHS, "all"),
        help="Agent harness to configure, or `all` for every supported harness.",
    )
    install_parser.add_argument(
        "--target",
        required=True,
        help="Project directory that should receive the skills.",
    )
    install_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview every planned write without changing the target.",
    )
    descriptions = {
        "update": "Update unmodified files in an existing managed installation.",
        "status": "Check every managed file and context block.",
        "migrate": "Remove only checksum-matched legacy artifacts.",
        "uninstall": "Remove only unmodified installer-owned content.",
    }
    for name, description in descriptions.items():
        child = commands.add_parser(name, help=description, description=description)
        child.add_argument(
            "--target", required=True, help="Project directory to inspect or change."
        )
        if name in {"update", "migrate", "uninstall"}:
            child.add_argument(
                "--dry-run",
                action="store_true",
                help="Preview planned changes without changing the target.",
            )
    cli = commands.add_parser(
        "install-cli",
        help="Install only dgx-assist for the current user.",
        description="Install only dgx-assist for direct terminal use.",
    )
    cli.add_argument(
        "--scope",
        required=True,
        choices=("user",),
        help="Install into XDG_BIN_HOME or ~/.local/bin for the current user.",
    )
    return result


def translate_legacy(argv: list[str]) -> list[str]:
    if not argv or argv[0] not in {*HARNESS_PATHS, "all"}:
        return argv
    harness = argv[0]
    target = "."
    remainder: list[str] = []
    for value in argv[1:]:
        if value == "--force":
            print(
                "warning: --force is ignored by the safe v1 installer", file=sys.stderr
            )
        elif value.startswith("-"):
            remainder.append(value)
        else:
            target = value
    print("warning: legacy positional installer syntax is deprecated", file=sys.stderr)
    return ["install", "--harness", harness, "--target", target, *remainder]


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = parser().parse_args(
            translate_legacy(list(sys.argv[1:] if argv is None else argv))
        )
        if arguments.command == "install":
            install(
                normalize_target(
                    arguments.target,
                    create=not arguments.dry_run,
                    allow_missing=arguments.dry_run,
                ),
                arguments.harness,
                dry_run=arguments.dry_run,
            )
        elif arguments.command == "update":
            target = normalize_target(arguments.target, create=False)
            manifest = read_manifest(target)
            if manifest is None:
                raise InstallError("no managed installation exists")
            install(target, "all", dry_run=arguments.dry_run, update=True)
        elif arguments.command == "status":
            return status(normalize_target(arguments.target, create=False))
        elif arguments.command == "migrate":
            migrate(
                normalize_target(arguments.target, create=False),
                dry_run=arguments.dry_run,
            )
        elif arguments.command == "uninstall":
            uninstall(
                normalize_target(arguments.target, create=False),
                dry_run=arguments.dry_run,
            )
        elif arguments.command == "install-cli":
            install_cli_user()
        return 0
    except (InstallError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
