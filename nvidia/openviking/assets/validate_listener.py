#!/usr/bin/env python3
"""Validate loopback TCP listeners and, optionally, their owning process."""

from __future__ import annotations

import argparse
import ipaddress
import re
import sys


PID_PATTERN = re.compile(r"\bpid=(\d+)\b")


def _split_endpoint(endpoint: str) -> tuple[str, int]:
    if endpoint.startswith("["):
        closing = endpoint.rfind("]:")
        if closing < 0:
            raise ValueError(f"could not parse listener endpoint {endpoint!r}")
        host = endpoint[1:closing]
        port_text = endpoint[closing + 2 :]
    else:
        try:
            host, port_text = endpoint.rsplit(":", 1)
        except ValueError as error:
            raise ValueError(
                f"could not parse listener endpoint {endpoint!r}"
            ) from error
    try:
        port = int(port_text)
    except ValueError as error:
        raise ValueError(f"listener port is not numeric: {endpoint!r}") from error
    return host, port


def validate_ss_listeners(
    output: str,
    *,
    expected_port: int,
    label: str,
    expected_pid: int | None = None,
) -> list[str]:
    """Validate every listener in numeric ``ss -H -ltn[p]`` output."""
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"no {label} TCP listener found on port {expected_port}")

    endpoints: list[str] = []
    for line in lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"could not parse {label} listener: {line!r}")
        endpoint = fields[3]
        host, port = _split_endpoint(endpoint)
        if port != expected_port:
            raise ValueError(
                f"unexpected {label} listener port: {endpoint!r}; expected {expected_port}"
            )
        try:
            address = ipaddress.ip_address(host)
        except ValueError as error:
            raise ValueError(f"non-loopback {label} listener: {endpoint}") from error
        if not address.is_loopback:
            raise ValueError(f"non-loopback {label} listener: {endpoint}")

        if expected_pid is not None:
            owner_pids = {int(value) for value in PID_PATTERN.findall(line)}
            if owner_pids != {expected_pid}:
                raise ValueError(
                    f"{label} listener {endpoint} owners {sorted(owner_pids)} do not match "
                    f"service MainPID {expected_pid}"
                )
        endpoints.append(endpoint)
    return endpoints


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-pid", type=int)
    args = parser.parse_args()
    try:
        endpoints = validate_ss_listeners(
            sys.stdin.read(),
            expected_port=args.port,
            label=args.label,
            expected_pid=args.expected_pid,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"LISTENER_OK: {args.label} {','.join(endpoints)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
