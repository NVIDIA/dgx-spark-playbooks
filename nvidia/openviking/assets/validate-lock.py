#!/usr/bin/env python3
"""Verify that a venv contains exactly the locked runtime dependency closure."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
from pathlib import Path

from packaging.utils import canonicalize_name


LOCKED_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ;\\]+)")
BOOTSTRAP_PACKAGES: set[str] = set()


def locked_versions(lock_paths: list[Path]) -> dict[str, str]:
    locked: dict[str, str] = {}
    for lock_path in lock_paths:
        for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
            match = LOCKED_REQUIREMENT.match(raw_line)
            if match:
                name = canonicalize_name(match.group(1))
                version = match.group(2)
                if name in locked and locked[name] != version:
                    raise SystemExit(
                        f"Conflicting locks for {name}: {locked[name]} and {version}"
                    )
                locked[name] = version
    if not locked:
        raise SystemExit(f"No pinned requirements found in {lock_paths}")
    return locked


def installed_versions() -> dict[str, str]:
    return {
        canonicalize_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("locks", nargs="+", type=Path)
    args = parser.parse_args()

    locked = locked_versions(args.locks)
    installed = installed_versions()
    missing = sorted(set(locked) - set(installed))
    mismatched = sorted(
        f"{name}: locked={locked[name]}, installed={installed[name]}"
        for name in set(locked) & set(installed)
        if locked[name] != installed[name]
    )
    unexpected = sorted(set(installed) - set(locked) - BOOTSTRAP_PACKAGES)
    if missing or mismatched or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if mismatched:
            details.append(f"mismatched={mismatched}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise SystemExit(
            "Installed environment does not match lock: " + "; ".join(details)
        )

    print(
        f"LOCK_CLOSURE_OK: {len(locked)} locked packages; "
        f"bootstrap={sorted(set(installed) & BOOTSTRAP_PACKAGES)}"
    )


if __name__ == "__main__":
    main()
