#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-platform control-host wrapper for DGX Station CX8 setup.

The DUT-side helpers remain Bash because the DUT OS is Linux. This wrapper lets
Windows, macOS, and Linux control hosts run the same steps through Python +
OpenSSH without requiring Bash locally.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent


def timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        try:
            parsed = shlex.split(val, posix=True)
            values[key] = parsed[0] if parsed else ""
        except ValueError:
            values[key] = val.strip("'\"")
    return values


@dataclass
class Config:
    pair_name: str
    os_user: str
    a_name: str
    a_host: str
    a_role: str
    b_name: str
    b_host: str
    b_role: str
    remote_base: str
    asset_src: Path
    log_root: Path
    rail0_dev: str
    rail1_dev: str
    mtu: str
    roce_tos: str
    gpu_bdf: str
    try_nvidia_peermem: str
    perftest_repo: str
    perftest_ref: str
    perftest_prefix: str
    perftest_server_ready_timeout: str
    a_rail0_cidr: str
    a_rail1_cidr: str
    b_rail0_cidr: str
    b_rail1_cidr: str
    ssh_strict_host_key_checking: str

    @property
    def remote_a(self) -> str:
        return f"{self.os_user}@{self.a_host}"

    @property
    def remote_b(self) -> str:
        return f"{self.os_user}@{self.b_host}"


def load_config() -> Config:
    env = dict(os.environ)
    env.update(parse_env_file(ROOT / "00_env.local"))
    def get(key: str, default: str) -> str:
        return env.get(key, default)

    os_user = get("CX8_OS_USER", "nvidia")
    cfg = Config(
        pair_name=get("CX8_PAIR_NAME", "dgx-station-cx8-setup"),
        os_user=os_user,
        a_name=get("CX8_A_NAME", "station-a"),
        a_host=get("CX8_A_HOST", ""),
        a_role=get("CX8_A_ROLE", "station-a"),
        b_name=get("CX8_B_NAME", "station-b"),
        b_host=get("CX8_B_HOST", ""),
        b_role=get("CX8_B_ROLE", "station-b"),
        remote_base=get("CX8_REMOTE_BASE", f"/home/{os_user}/cx8-two-station"),
        asset_src=Path(get("CX8_ASSET_SRC", str(ROOT / "dut-assets"))),
        log_root=Path(get("CX8_LOCAL_LOG_ROOT", str(ROOT / "logs"))),
        rail0_dev=get("RAIL0_DEV", "mlx5_0"),
        rail1_dev=get("RAIL1_DEV", "mlx5_1"),
        mtu=get("MTU", "9000"),
        roce_tos=get("ROCE_TOS", "106"),
        gpu_bdf=get("CX8_GPU_BDF", ""),
        try_nvidia_peermem=get("CX8_TRY_NVIDIA_PEERMEM", "0"),
        perftest_repo=get("CX8_PERFTEST_REPO", "https://github.com/linux-rdma/perftest.git"),
        perftest_ref=get("CX8_PERFTEST_REF", "26.04.17"),
        perftest_prefix=get("CX8_PERFTEST_PREFIX", "/usr/local"),
        perftest_server_ready_timeout=get("CX8_PERFTEST_SERVER_READY_TIMEOUT", "30"),
        a_rail0_cidr=get("STATION_A_RAIL0_CIDR", "192.168.100.1/24"),
        a_rail1_cidr=get("STATION_A_RAIL1_CIDR", "192.168.101.1/24"),
        b_rail0_cidr=get("STATION_B_RAIL0_CIDR", "192.168.100.2/24"),
        b_rail1_cidr=get("STATION_B_RAIL1_CIDR", "192.168.101.2/24"),
        ssh_strict_host_key_checking=get("CX8_SSH_STRICT_HOST_KEY_CHECKING", "ask"),
    )
    if cfg.ssh_strict_host_key_checking not in {"yes", "ask", "accept-new"}:
        raise SystemExit(
            "ERROR: CX8_SSH_STRICT_HOST_KEY_CHECKING must be one of: yes, ask, accept-new"
        )
    if not cfg.perftest_server_ready_timeout.isdigit():
        raise SystemExit("ERROR: CX8_PERFTEST_SERVER_READY_TIMEOUT must be seconds")
    if not cfg.a_host or not cfg.b_host:
        raise SystemExit(
            "ERROR: set CX8_A_HOST and CX8_B_HOST in 00_env.local before running this wrapper"
        )
    return cfg


def ssh_base_args(cfg: Config, tty: bool = False) -> list[str]:
    ssh = shutil.which("ssh")
    if not ssh:
        raise SystemExit("ERROR: OpenSSH client 'ssh' was not found in PATH")
    args = [
        ssh,
        "-o",
        f"StrictHostKeyChecking={cfg.ssh_strict_host_key_checking}",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=4",
    ]
    if tty:
        args.insert(1, "-tt")
    return args


def q(value: str) -> str:
    return shlex.quote(value)


def remote_env(cfg: Config, role: str | None = None) -> str:
    items = {
        "RAIL0_DEV": cfg.rail0_dev,
        "RAIL1_DEV": cfg.rail1_dev,
        "MTU": cfg.mtu,
        "ROCE_TOS": cfg.roce_tos,
        "GPU_BDF": cfg.gpu_bdf,
        "TRY_NVIDIA_PEERMEM": cfg.try_nvidia_peermem,
        "PERFTEST_REPO": cfg.perftest_repo,
        "PERFTEST_REF": cfg.perftest_ref,
        "PERFTEST_PREFIX": cfg.perftest_prefix,
        "STATION_A_RAIL0_CIDR": cfg.a_rail0_cidr,
        "STATION_A_RAIL1_CIDR": cfg.a_rail1_cidr,
        "STATION_B_RAIL0_CIDR": cfg.b_rail0_cidr,
        "STATION_B_RAIL1_CIDR": cfg.b_rail1_cidr,
    }
    if role is not None:
        items["ROLE"] = role
    return " ".join(f"{k}={q(v)}" for k, v in items.items())


def log_dir(cfg: Config, name: str) -> Path:
    path = cfg.log_root / f"{timestamp()}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_logged(
    cfg: Config,
    host: str,
    command: str,
    log: Path,
    *,
    stdin: bytes | None = None,
    tty: bool = False,
) -> int:
    print(f"\n### {host}\n")
    args = ssh_base_args(cfg, tty=tty) + [host, command]
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE if stdin is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    with log.open("wb") as f:
        if stdin is not None:
            assert proc.stdin is not None
            proc.stdin.write(stdin)
            proc.stdin.close()
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            f.write(chunk)
    return proc.wait()


def make_assets_tar(cfg: Config) -> bytes:
    assets = sorted(cfg.asset_src.glob("*.sh"))
    if not assets:
        raise SystemExit(f"ERROR: no .sh assets found under {cfg.asset_src}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for path in assets:
            data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            info = tarfile.TarInfo(name=f"./{path.name}")
            info.size = len(data)
            info.mode = 0o755
            info.mtime = int(time.time())
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def for_each_station(cfg: Config) -> Iterable[tuple[str, str, str]]:
    yield cfg.a_name, cfg.remote_a, cfg.a_role
    yield cfg.b_name, cfg.remote_b, cfg.b_role


def cmd_probe_access(cfg: Config, _args: argparse.Namespace) -> int:
    d = log_dir(cfg, "probe_access")
    remote = r'''
set -u
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname -f 2>/dev/null || hostname)"
echo "user=$(id -un)"
echo "kernel=$(uname -r)"
test -r /etc/os-release && . /etc/os-release && echo "os=${PRETTY_NAME}"
echo "[required commands]"
for cmd in bash sudo tar lspci nvidia-smi ibdev2netdev ibv_devinfo ip ethtool show_gids rdma_topo mlnx_qos cma_roce_tos ib_write_bw; do
  if command -v "${cmd}" >/dev/null 2>&1; then echo "ok ${cmd}=$(command -v "${cmd}")"; else echo "missing ${cmd}"; fi
done
echo "[optional MFT commands]"
for cmd in flint mlxconfig; do
  if command -v "${cmd}" >/dev/null 2>&1; then echo "optional_ok ${cmd}=$(command -v "${cmd}")"; else echo "optional_missing ${cmd}"; fi
done
if sudo -n true >/dev/null 2>&1; then echo "sudo_status=passwordless-or-cached"; else echo "sudo_status=will-prompt-or-require-password"; fi
'''
    rc = 0
    for label, host, _role in for_each_station(cfg):
        rc |= run_logged(cfg, host, remote, d / f"{label}_probe_access.log")
    print(f"\nLogs: {d}")
    return rc


def cmd_push_assets(cfg: Config, _args: argparse.Namespace) -> int:
    d = log_dir(cfg, "push_assets")
    tar_bytes = make_assets_tar(cfg)
    remote = (
        f"set -euo pipefail; mkdir -p {q(cfg.remote_base + '/assets')} {q(cfg.remote_base + '/logs')}; "
        f"tar -C {q(cfg.remote_base + '/assets')} -xf -; "
        f"chmod +x {q(cfg.remote_base + '/assets')}/*.sh; "
        f"ls -l {q(cfg.remote_base + '/assets')}"
    )
    rc = 0
    for label, host, _role in for_each_station(cfg):
        rc |= run_logged(cfg, host, f"bash -lc {q(remote)}", d / f"{label}_push_assets.log", stdin=tar_bytes)
    print(f"\nLogs: {d}")
    return rc


def remote_asset(cfg: Config, script: str, role: str | None = None, sudo: bool = False, extra: str = "") -> str:
    env = remote_env(cfg, role)
    base = f"cd {q(cfg.remote_base)} && env {env} ./assets/{script} {extra}".strip()
    if sudo:
        return f"cd {q(cfg.remote_base)} && sudo env {env} ./assets/{script} {extra}".strip()
    return base


def cmd_prereq_check(cfg: Config, _args: argparse.Namespace) -> int:
    d = log_dir(cfg, "prereq_check")
    rc = 0
    for label, host, _role in for_each_station(cfg):
        command = f"cd {q(cfg.remote_base)} && CX8_PRIVILEGED=0 ./assets/check_prereqs.sh 2>&1"
        rc |= run_logged(cfg, host, command, d / f"{label}_prereq_check.log")
    print(f"\nLogs: {d}")
    return rc


def cmd_cable_check(cfg: Config, _args: argparse.Namespace) -> int:
    d = log_dir(cfg, "cable_presence")
    rc = 0
    for label, host, _role in for_each_station(cfg):
        rc |= run_logged(cfg, host, remote_asset(cfg, "check_cable_presence_on_dut.sh"), d / f"{label}_cable_presence.log")
    print(f"\nLogs: {d}")
    return rc


def cmd_configure_rails(cfg: Config, args: argparse.Namespace) -> int:
    d = log_dir(cfg, "configure_rails")
    rc = 0
    extra = "--persist" if args.persist else ""
    for label, host, role in for_each_station(cfg):
        rc |= run_logged(
            cfg,
            host,
            remote_asset(cfg, "configure_rails.sh", role=role, sudo=True, extra=extra),
            d / f"{label}_configure_rails.log",
            tty=True,
        )
    print(f"\nLogs: {d}")
    return rc


def cmd_configure_roce_gdr(cfg: Config, _args: argparse.Namespace) -> int:
    d = log_dir(cfg, "configure_roce_gdr")
    rc = 0
    for label, host, _role in for_each_station(cfg):
        rc |= run_logged(
            cfg,
            host,
            remote_asset(cfg, "configure_roce_gdr.sh", sudo=True),
            d / f"{label}_configure_roce_gdr.log",
            tty=True,
        )
    print(f"\nLogs: {d}")
    return rc


def cmd_validate(cfg: Config, _args: argparse.Namespace) -> int:
    d = log_dir(cfg, "validate_setup")
    rc = 0
    for label, host, role in for_each_station(cfg):
        rc |= run_logged(cfg, host, remote_asset(cfg, "validate_setup.sh", role=role), d / f"{label}_validate_setup.log", tty=True)
    print(f"\nLogs: {d}")
    return rc


def cmd_perftest(cfg: Config, args: argparse.Namespace) -> int:
    if args.gpu_bdf:
        cfg.gpu_bdf = args.gpu_bdf
    d = log_dir(cfg, f"perftest_rail{args.rail}_{'gdr' if args.gdr else 'hostmem'}")
    run_id = f"{timestamp()}_rail{args.rail}_{'gdr' if args.gdr else 'hostmem'}"
    gdr = "--gdr" if args.gdr else ""
    remote_server_log = f"{cfg.remote_base}/logs/{run_id}_server.log"
    remote_server_pid = f"{cfg.remote_base}/logs/{run_id}_server.pid"
    server_hca = cfg.rail0_dev if args.rail == "0" else cfg.rail1_dev
    server_cmd = (
        f"mkdir -p {q(cfg.remote_base + '/logs')}; "
        f"pkill -f {q('ib_write_bw.*-d ' + server_hca)} >/dev/null 2>&1 || true; "
        f"cd {q(cfg.remote_base)}; "
        f"nohup env {remote_env(cfg, cfg.a_role)} ./assets/run_perftest.sh --server --rail {q(args.rail)} {gdr} "
        f"--duration {q(str(args.duration))} --size {q(str(args.size))} >{q(remote_server_log)} 2>&1 & "
        f"server_pid=$!; echo $server_pid >{q(remote_server_pid)}; "
        f"start=$(date +%s); timeout={q(cfg.perftest_server_ready_timeout)}; "
        f"while :; do "
        f"if grep -Eq 'Waiting for client to connect|local address:' {q(remote_server_log)} 2>/dev/null; then exit 0; fi; "
        f"if ! kill -0 $server_pid >/dev/null 2>&1; then echo 'ERROR: perftest server exited before accepting a client'; cat {q(remote_server_log)} 2>/dev/null || true; exit 1; fi; "
        f"now=$(date +%s); if [ $((now - start)) -ge $timeout ]; then echo \"ERROR: perftest server was not ready within ${{timeout}}s\"; cat {q(remote_server_log)} 2>/dev/null || true; exit 1; fi; "
        f"sleep 1; "
        f"done"
    )
    rc = run_logged(cfg, cfg.remote_a, server_cmd, d / f"{cfg.a_name}_server_start.log")
    if rc != 0:
        print(
            f"ERROR: perftest server was not ready on {cfg.a_name}; "
            "not starting the client. Inspect the server-start log first.",
            file=sys.stderr,
        )
        run_logged(
            cfg,
            cfg.remote_a,
            f"cat {q(remote_server_log)} 2>/dev/null || true",
            d / f"{cfg.a_name}_server_rail{args.rail}.log",
        )
        run_logged(
            cfg,
            cfg.remote_a,
            f"if test -s {q(remote_server_pid)}; then pid=$(cat {q(remote_server_pid)}); "
            'kill "$pid" >/dev/null 2>&1 || true; fi',
            d / f"{cfg.a_name}_server_cleanup.log",
        )
        print(f"\nLogs: {d}")
        return rc
    print(f"Server is ready on {cfg.a_name}; starting client")
    client_cmd = (
        f"cd {q(cfg.remote_base)} && env {remote_env(cfg, cfg.b_role)} ./assets/run_perftest.sh "
        f"--client --rail {q(args.rail)} {gdr} --duration {q(str(args.duration))} --size {q(str(args.size))}"
    )
    rc |= run_logged(cfg, cfg.remote_b, client_cmd, d / f"{cfg.b_name}_client_rail{args.rail}.log", tty=True)
    run_logged(cfg, cfg.remote_a, f"cat {q(remote_server_log)} 2>/dev/null || true", d / f"{cfg.a_name}_server_rail{args.rail}.log")
    run_logged(cfg, cfg.remote_a, f"if test -s {q(remote_server_pid)}; then pid=$(cat {q(remote_server_pid)}); kill \"$pid\" >/dev/null 2>&1 || true; fi", d / f"{cfg.a_name}_server_cleanup.log")
    print(f"\nLogs: {d}")
    return rc


def cmd_install_perftest_gdr(cfg: Config, args: argparse.Namespace) -> int:
    if args.ref:
        cfg.perftest_ref = args.ref
    d = log_dir(cfg, "install_perftest_gdr")
    extra = []
    if args.check_only:
        extra.append("--check-only")
    if args.source_only:
        extra.append("--source-only")
    if args.ref:
        extra.extend(["--ref", args.ref])
    rc = 0
    for label, host, _role in for_each_station(cfg):
        rc |= run_logged(
            cfg,
            host,
            remote_asset(cfg, "install_perftest_gdr.sh", sudo=True, extra=" ".join(q(x) for x in extra)),
            d / f"{label}_install_perftest_gdr.log",
            tty=True,
        )
    print(f"\nLogs: {d}")
    return rc


def cmd_configure_acs_grub(cfg: Config, args: argparse.Namespace) -> int:
    d = log_dir(cfg, "configure_acs_grub")
    extra = "--apply" if args.apply else ""
    rc = 0
    for label, host, _role in for_each_station(cfg):
        rc |= run_logged(
            cfg,
            host,
            remote_asset(cfg, "configure_acs.sh", sudo=True, extra=extra),
            d / f"{label}_configure_acs_grub.log",
            tty=True,
        )
    if args.apply and rc == 0:
        print(
            "\nNext after both DUTs reboot:\n"
            "  Confirm both DUTs are idle and downtime/workload-owner approval is in place.\n"
            "  python3 ./cx8_setup.py 05 --persist\n"
            "  python3 ./cx8_setup.py 06\n"
            "  python3 ./cx8_setup.py 07\n"
            "  python3 ./cx8_setup.py 08 --rail 0 --gdr\n"
            "  python3 ./cx8_setup.py 08 --rail 1 --gdr\n\n"
            "Step 3 and Step 4 are optional after reboot unless OS/packages changed "
            "or cable state is uncertain.\n"
            "Step 5 --persist is recommended for this post-reboot GDR path so rail IP/MTU "
            "settings remain stable during longer Data Direct validation. It is not required "
            "for the normal temporary bring-up flow."
        )
    elif args.apply:
        print(
            "\nACTION: ACS/Data Direct GRUB apply did not complete on both DUTs. "
            "Do not reboot or continue the pair until the failed DUT is fixed and "
            "Step 11 --apply passes on both systems.",
            file=sys.stderr,
        )
    print(f"\nLogs: {d}")
    return rc


def cmd_cleanup(cfg: Config, args: argparse.Namespace) -> int:
    d = log_dir(cfg, "cleanup_runtime")
    extra = []
    if args.remove_persist:
        extra.append("--remove-persist")
    if args.down:
        extra.append("--down")
    rc = 0
    for label, host, role in for_each_station(cfg):
        rc |= run_logged(
            cfg,
            host,
            remote_asset(cfg, "cleanup_runtime_on_dut.sh", role=role, sudo=True, extra=" ".join(extra)),
            d / f"{label}_cleanup_runtime.log",
            tty=True,
        )
    print(f"\nLogs: {d}")
    return rc


def cmd_collect_logs(cfg: Config, _args: argparse.Namespace) -> int:
    cfg.log_root.mkdir(parents=True, exist_ok=True)
    tarball = cfg.log_root.parent / f"{cfg.log_root.name}_manual_logs_{timestamp()}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(cfg.log_root, arcname=cfg.log_root.name)
    print(f"Tarball: {tarball}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross-platform DGX Station CX8 setup wrapper")
    sp = p.add_subparsers(dest="command", required=True)
    sp.add_parser("probe-access", aliases=["01"]).set_defaults(func=cmd_probe_access)
    sp.add_parser("push-assets", aliases=["02"]).set_defaults(func=cmd_push_assets)
    sp.add_parser("prereq-check", aliases=["03"]).set_defaults(func=cmd_prereq_check)
    sp.add_parser("check-cables", aliases=["04"]).set_defaults(func=cmd_cable_check)
    p5 = sp.add_parser("configure-rails", aliases=["05"])
    p5.add_argument("--persist", action="store_true")
    p5.set_defaults(func=cmd_configure_rails)
    sp.add_parser("configure-roce-gdr", aliases=["06"]).set_defaults(func=cmd_configure_roce_gdr)
    sp.add_parser("validate", aliases=["07"]).set_defaults(func=cmd_validate)
    p8 = sp.add_parser("perftest", aliases=["08"])
    p8.add_argument("--rail", choices=["0", "1"], required=True)
    p8.add_argument("--gdr", action="store_true")
    p8.add_argument("--duration", default=20, type=int)
    p8.add_argument("--size", default=1048576, type=int)
    p8.add_argument("--gpu-bdf", default="")
    p8.set_defaults(func=cmd_perftest)
    sp.add_parser("collect-logs", aliases=["09"]).set_defaults(func=cmd_collect_logs)
    p10 = sp.add_parser("install-perftest-gdr", aliases=["10"])
    p10.add_argument("--check-only", action="store_true")
    p10.add_argument("--source-only", action="store_true")
    p10.add_argument("--ref", default="")
    p10.set_defaults(func=cmd_install_perftest_gdr)
    p11 = sp.add_parser("configure-acs-grub", aliases=["11"])
    p11.add_argument("--apply", action="store_true")
    p11.set_defaults(func=cmd_configure_acs_grub)
    p99 = sp.add_parser("cleanup", aliases=["99"])
    p99.add_argument("--remove-persist", action="store_true")
    p99.add_argument("--down", action="store_true")
    p99.set_defaults(func=cmd_cleanup)
    return p


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config()
    return int(args.func(cfg, args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
