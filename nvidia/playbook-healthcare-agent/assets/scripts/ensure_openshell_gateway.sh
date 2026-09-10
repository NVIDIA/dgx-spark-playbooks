#!/usr/bin/env bash
# Configure and verify the package-managed OpenShell gateway.
#
# This intentionally does not start openshell-gateway directly. The packaged
# systemd user unit owns process lifecycle and certificate generation.
set -euo pipefail
umask 077

export PATH="$HOME/.local/bin:$PATH"

CHECK_ONLY=false
case "${1:-}" in
    "") ;;
    --check-only) CHECK_ONLY=true ;;
    *) echo "Usage: $0 [--check-only]" >&2; exit 64 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_SOURCE="${OPENSHELL_GATEWAY_CONFIG_SOURCE:-$SCRIPT_DIR/../openshell-gateway.toml}"
OPENSHELL_CONFIG_DIR="${OPENSHELL_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/openshell}"
CONFIG_TARGET="${OPENSHELL_GATEWAY_CONFIG_TARGET:-$OPENSHELL_CONFIG_DIR/gateway.toml}"
GATEWAY_NAME="${OPENSHELL_GATEWAY_NAME:-openshell}"
GATEWAY_ENDPOINT="${OPENSHELL_GATEWAY_ENDPOINT:-https://127.0.0.1:17670}"
DOCKER_SOCKET="${OPENSHELL_DOCKER_SOCKET:-/var/run/docker.sock}"
PROC_ROOT="${OPENSHELL_PROC_ROOT:-/proc}"
ACTIVE_GATEWAY_FILE="$OPENSHELL_CONFIG_DIR/active_gateway"
REGISTRATION_DIR="$OPENSHELL_CONFIG_DIR/gateways/$GATEWAY_NAME"
CMD_TIMEOUT="${OPENSHELL_GATEWAY_CMD_TIMEOUT:-20s}"
READY_TIMEOUT_SECONDS="${OPENSHELL_GATEWAY_READY_TIMEOUT_SECONDS:-30}"
READY_PROBE_TIMEOUT_SECONDS="${OPENSHELL_GATEWAY_READY_PROBE_TIMEOUT_SECONDS:-5}"
LOCK_FILE="${OPENSHELL_GATEWAY_LOCK_FILE:-$OPENSHELL_CONFIG_DIR/.ensure_openshell_gateway.lock}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

run_timed_for() {
    local duration="$1"
    shift
    timeout --foreground --kill-after=5s "$duration" "$@"
}

run_timed() {
    run_timed_for "$CMD_TIMEOUT" "$@"
}

check_systemd_docker_access() {
    local docker_bin refresh_guidance linger_state
    docker_bin="$(command -v docker)" || die "Docker CLI is required."
    command -v systemd-run >/dev/null 2>&1 || die "systemd-run is required."
    command -v timeout >/dev/null 2>&1 || die "timeout is required."

    # Run the probe as a transient user service. It therefore has the same
    # inherited groups as openshell-gateway.service, not merely this shell.
    if ! run_timed systemd-run --user --quiet --wait --pipe --collect \
        --property=Type=exec \
        "$docker_bin" --host "unix://$DOCKER_SOCKET" info \
        >/dev/null 2>&1; then
        refresh_guidance="Fully end every login session and log back in, or reboot, after Docker-group membership is assigned."
        if command -v loginctl >/dev/null 2>&1; then
            linger_state="$(run_timed loginctl show-user "$USER" -p Linger --value 2>/dev/null || true)"
            if [[ "$linger_state" == "yes" ]]; then
                refresh_guidance="Linger is enabled, so logout/login will not recreate this manager. Reboot during a maintenance window after Docker-group membership is assigned."
            fi
        fi
        cat >&2 <<EOF
ERROR: the systemd user manager cannot access Docker at $DOCKER_SOCKET.
Your current shell may have the docker group while the older user manager does not.
$refresh_guidance
Then rerun this command. Do not use newgrp, chmod the socket, sudo the gateway,
or launch openshell-gateway manually.
EOF
        exit 1
    fi
}

validate_gateway() {
    local deadline="${1:-}"
    local status_json info_json probe_timeout remaining

    probe_timeout="$CMD_TIMEOUT"
    if [[ -n "$deadline" ]]; then
        remaining=$((deadline - SECONDS))
        (( remaining > 0 )) || return 1
        if (( remaining < READY_PROBE_TIMEOUT_SECONDS )); then
            probe_timeout="${remaining}s"
        else
            probe_timeout="${READY_PROBE_TIMEOUT_SECONDS}s"
        fi
    fi

    # `openshell status` exits zero even when it prints Authentication: Failed,
    # so its structured authentication field must be checked explicitly.
    status_json="$(run_timed_for "$probe_timeout" openshell -g "$GATEWAY_NAME" status --output json 2>&1)" || {
        printf '%s\n' "$status_json" >&2
        return 1
    }
    if ! OPENSHELL_EXPECTED_ENDPOINT="$GATEWAY_ENDPOINT" \
         OPENSHELL_EXPECTED_NAME="$GATEWAY_NAME" \
         python3 -c '
import json, os, sys
d = json.load(sys.stdin)
ok = (
    d.get("gateway") == os.environ["OPENSHELL_EXPECTED_NAME"]
    and d.get("server") == os.environ["OPENSHELL_EXPECTED_ENDPOINT"]
    and d.get("status") == "connected"
    and d.get("authentication", {}).get("status") == "authenticated"
)
raise SystemExit(0 if ok else 1)
' <<<"$status_json"; then
        printf '%s\n' "$status_json" >&2
        return 1
    fi

    # This is a protected RPC. Success proves the credentials work; checking
    # the payload also proves the Docker compute driver initialized.
    probe_timeout="$CMD_TIMEOUT"
    if [[ -n "$deadline" ]]; then
        remaining=$((deadline - SECONDS))
        (( remaining > 0 )) || return 1
        if (( remaining < READY_PROBE_TIMEOUT_SECONDS )); then
            probe_timeout="${remaining}s"
        else
            probe_timeout="${READY_PROBE_TIMEOUT_SECONDS}s"
        fi
    fi
    info_json="$(run_timed_for "$probe_timeout" openshell -g "$GATEWAY_NAME" gateway info --output json 2>&1)" || {
        printf '%s\n' "$info_json" >&2
        return 1
    }
    if ! OPENSHELL_EXPECTED_ENDPOINT="$GATEWAY_ENDPOINT" python3 -c '
import json, os, sys
d = json.load(sys.stdin)
ok = (
    d.get("server") == os.environ["OPENSHELL_EXPECTED_ENDPOINT"]
    and d.get("status") == "healthy"
    and any(x.get("name") == "docker" for x in d.get("compute_drivers", []))
)
raise SystemExit(0 if ok else 1)
' <<<"$info_json"; then
        printf '%s\n' "$info_json" >&2
        return 1
    fi
}

REGISTRATION_STATE=""
REGISTRATION_ACTIVE="false"

inspect_local_registration() {
    local registrations_json

    registrations_json="$(run_timed openshell gateway list --output json 2>&1)" || {
        printf '%s\n' "$registrations_json" >&2
        die "Could not inspect existing OpenShell gateway registrations."
    }

    # Gateway names are immutable in OpenShell 0.0.97/0.0.98: `gateway add`
    # fails when the name already exists. Replace only the expected local
    # registration and refuse to clobber an operator-owned endpoint.
    local classification
    classification="$(
        OPENSHELL_EXPECTED_NAME="$GATEWAY_NAME" \
        OPENSHELL_EXPECTED_ENDPOINT="$GATEWAY_ENDPOINT" \
        python3 -c '
import json, os, sys
items = json.load(sys.stdin)
name = os.environ["OPENSHELL_EXPECTED_NAME"]
expected = os.environ["OPENSHELL_EXPECTED_ENDPOINT"]
legacy = expected.replace("https://", "http://", 1)
matches = [x for x in items if x.get("name") == name]
if not matches:
    print("absent:false")
elif len(matches) == 1:
    item = matches[0]
    operator_owned = (
        item.get("is_remote", False)
        or item.get("source") != "user"
        or item.get("type") != "local"
    )
    is_active = str(bool(item.get("active", False))).lower()
    if not operator_owned and item.get("endpoint") == legacy and item.get("auth") == "plaintext":
        print("legacy:" + is_active)
    elif not operator_owned and item.get("endpoint") == expected and item.get("auth") == "mtls":
        print("desired:" + is_active)
    else:
        raise SystemExit(1)
else:
    raise SystemExit(1)
' <<<"$registrations_json"
    )" || {
        printf '%s\n' "$registrations_json" >&2
        die "Gateway name '$GATEWAY_NAME' belongs to an unexpected endpoint; refusing to replace it."
    }
    REGISTRATION_STATE="${classification%%:*}"
    REGISTRATION_ACTIVE="${classification#*:}"
}

legacy_plaintext_gateway_running() {
    local gateway_exe proc_path process_exe arg
    gateway_exe="$(readlink -f "$(command -v openshell-gateway)")" || \
        die "openshell-gateway is required."

    # Inspect only processes whose executable is the real gateway binary.
    # A broad `pgrep -f` can match an SSH wrapper or test command containing
    # the words "openshell-gateway --disable-tls" and produce a false blocker.
    for proc_path in "$PROC_ROOT"/[0-9]*; do
        [[ -r "$proc_path/cmdline" ]] || continue
        process_exe="$(readlink -f "$proc_path/exe" 2>/dev/null)" || continue
        [[ "$process_exe" == "$gateway_exe" ]] || continue
        while IFS= read -r arg; do
            [[ "$arg" == "--disable-tls" ]] && return 0
        done < <(tr '\0' '\n' <"$proc_path/cmdline")
    done
    return 1
}

check_systemd_docker_access

[[ "$READY_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
    die "OPENSHELL_GATEWAY_READY_TIMEOUT_SECONDS must be a positive integer."
[[ "$READY_PROBE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
    die "OPENSHELL_GATEWAY_READY_PROBE_TIMEOUT_SECONDS must be a positive integer."
command -v flock >/dev/null 2>&1 || die "flock is required."
command -v install >/dev/null 2>&1 || die "install is required."
if [[ ! -d "$OPENSHELL_CONFIG_DIR" ]]; then
    install -d -m 0700 "$OPENSHELL_CONFIG_DIR"
fi
[[ ! -L "$LOCK_FILE" ]] || die "Refusing symlink lock file: $LOCK_FILE"
exec 9>"$LOCK_FILE"
flock -n 9 || die "Another ensure_openshell_gateway.sh invocation is already running."

if $CHECK_ONLY; then
    validate_gateway || die "Gateway validation failed."
    echo "Gateway: authenticated, healthy, Docker driver ready"
    exit 0
fi

# Complete every fail-fast, non-mutating check before the first write.
[[ -f "$CONFIG_SOURCE" ]] || die "Missing OpenShell config: $CONFIG_SOURCE"
[[ "$GATEWAY_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
    die "Invalid gateway name: $GATEWAY_NAME"
for required in openshell openshell-gateway python3 install cmp systemctl readlink tr mktemp cp mv rm timeout flock; do
    command -v "$required" >/dev/null 2>&1 || die "$required is required."
done

# Never overwrite an unrelated operator-owned OpenShell configuration.
if [[ -e "$CONFIG_TARGET" ]] && \
   ! grep -Eq '^# Managed by (station-healthcare-agent|playbook-healthcare-agent)\.$' "$CONFIG_TARGET"; then
    die "Existing $CONFIG_TARGET is not managed by this playbook; merge $CONFIG_SOURCE manually."
fi

legacy_env="$OPENSHELL_CONFIG_DIR/gateway.env"
if [[ -f "$legacy_env" ]] && \
   grep -Eqi '^[[:space:]]*OPENSHELL_DISABLE_TLS[[:space:]]*=[[:space:]]*(1|true|yes|on)' "$legacy_env"; then
    die "$legacy_env still forces plaintext. Remove that legacy override before continuing."
fi

if legacy_plaintext_gateway_running; then
    die "A legacy manually launched plaintext gateway is running. Stop it once, then rerun."
fi

run_timed systemctl --user cat openshell-gateway.service >/dev/null 2>&1 || \
    die "The package-managed openshell-gateway.service is not installed."
SERVICE_UNIT_STATE="$(run_timed systemctl --user is-enabled openshell-gateway.service 2>/dev/null || true)"
SERVICE_ACTIVE_STATE="$(run_timed systemctl --user show openshell-gateway.service -p ActiveState --value 2>/dev/null || true)"
inspect_local_registration

case "$SERVICE_UNIT_STATE" in
    enabled|enabled-runtime|disabled) ;;
    *) die "Unsupported openshell-gateway.service enablement state: $SERVICE_UNIT_STATE" ;;
esac
case "$SERVICE_ACTIVE_STATE" in
    active|reloading|inactive|failed|activating|deactivating|maintenance|refreshing) ;;
    *) die "Could not determine openshell-gateway.service active state." ;;
esac

# A healthy repeat run performs no service or registration mutation.
if [[ "$REGISTRATION_STATE" == "desired" ]] && \
   [[ "$REGISTRATION_ACTIVE" == "true" ]] && \
   [[ "$SERVICE_UNIT_STATE" == "enabled" ]] && \
   [[ -f "$CONFIG_TARGET" ]] && cmp -s "$CONFIG_SOURCE" "$CONFIG_TARGET" && \
   run_timed systemctl --user is-active --quiet openshell-gateway.service && \
   validate_gateway >/dev/null 2>&1; then
    echo "Gateway: authenticated, healthy, Docker driver ready (no changes)"
    exit 0
fi

# Transaction state used to restore the preimage if any later step fails.
TXN_DIR="$(mktemp -d)"
chmod 0700 "$TXN_DIR"
CONFIG_WAS_PRESENT=false
if [[ -e "$CONFIG_TARGET" ]]; then
    CONFIG_WAS_PRESENT=true
    cp -p "$CONFIG_TARGET" "$TXN_DIR/gateway.toml"
fi
ACTIVE_GATEWAY_WAS_PRESENT=false
if [[ -e "$ACTIVE_GATEWAY_FILE" ]]; then
    ACTIVE_GATEWAY_WAS_PRESENT=true
    cp -p "$ACTIVE_GATEWAY_FILE" "$TXN_DIR/active_gateway"
fi
REGISTRATION_WAS_PRESENT=false
if [[ -d "$REGISTRATION_DIR" ]]; then
    REGISTRATION_WAS_PRESENT=true
    cp -a "$REGISTRATION_DIR" "$TXN_DIR/registration"
fi
REGISTRATION_MUTATED=false
TRANSACTION_COMMITTED=false

# shellcheck disable=SC2329  # Invoked indirectly by the EXIT trap below.
rollback_partial_migration() {
    local rc=$?
    if ! $TRANSACTION_COMMITTED; then
        set +e
        run_timed systemctl --user stop openshell-gateway.service >/dev/null 2>&1

        if $REGISTRATION_MUTATED; then
            run_timed openshell gateway remove "$GATEWAY_NAME" >/dev/null 2>&1
            if [[ -e "$REGISTRATION_DIR" ]]; then
                mv "$REGISTRATION_DIR" "$TXN_DIR/registration-candidate" >/dev/null 2>&1
            fi
            if $REGISTRATION_WAS_PRESENT; then
                install -d -m 0700 "$(dirname "$REGISTRATION_DIR")"
                cp -a "$TXN_DIR/registration" "$REGISTRATION_DIR"
            fi
            if $ACTIVE_GATEWAY_WAS_PRESENT; then
                install -d -m 0700 "$(dirname "$ACTIVE_GATEWAY_FILE")"
                cp -p "$TXN_DIR/active_gateway" "$ACTIVE_GATEWAY_FILE"
            else
                rm -f -- "$ACTIVE_GATEWAY_FILE"
            fi
        fi

        if $CONFIG_WAS_PRESENT; then
            install -d -m 0700 "$(dirname "$CONFIG_TARGET")"
            cp -p "$TXN_DIR/gateway.toml" "$CONFIG_TARGET"
        else
            rm -f -- "$CONFIG_TARGET"
        fi

        case "$SERVICE_UNIT_STATE" in
            enabled)
                run_timed systemctl --user enable openshell-gateway.service >/dev/null 2>&1
                ;;
            enabled-runtime)
                run_timed systemctl --user disable openshell-gateway.service >/dev/null 2>&1
                run_timed systemctl --user enable --runtime openshell-gateway.service >/dev/null 2>&1
                ;;
            disabled)
                run_timed systemctl --user disable openshell-gateway.service >/dev/null 2>&1
                ;;
        esac
        case "$SERVICE_ACTIVE_STATE" in
            active|activating|reloading)
                run_timed systemctl --user start openshell-gateway.service >/dev/null 2>&1
                ;;
        esac
        rm -rf -- "$TXN_DIR"
        printf 'ERROR: OpenShell gateway migration failed; rollback was attempted. Verify the prior config, registration, active gateway, and unit state.\n' >&2
    fi
    exit "$rc"
}
trap rollback_partial_migration EXIT

CONFIG_CHANGED=false
if [[ ! -f "$CONFIG_TARGET" ]] || ! cmp -s "$CONFIG_SOURCE" "$CONFIG_TARGET"; then
    install -d -m 0700 "$(dirname "$CONFIG_TARGET")"
    install -m 0600 "$CONFIG_SOURCE" "$CONFIG_TARGET"
    CONFIG_CHANGED=true
fi

run_timed systemctl --user daemon-reload
case "$SERVICE_UNIT_STATE" in
    enabled) ;;
    *) run_timed systemctl --user enable openshell-gateway.service ;;
esac

if $CONFIG_CHANGED; then
    run_timed systemctl --user restart openshell-gateway.service
elif ! run_timed systemctl --user is-active --quiet openshell-gateway.service; then
    run_timed systemctl --user start openshell-gateway.service
fi

# HTTPS + --local imports the client bundle generated by ExecStartPre and
# records auth_mode=mtls. OpenShell requires an existing legacy name to be
# removed explicitly; unrelated endpoints were rejected by preflight.
case "$REGISTRATION_STATE" in
    legacy)
        REGISTRATION_MUTATED=true
        run_timed openshell gateway remove "$GATEWAY_NAME"
        run_timed openshell gateway add "$GATEWAY_ENDPOINT" --local --name "$GATEWAY_NAME"
        ;;
    absent)
        REGISTRATION_MUTATED=true
        run_timed openshell gateway add "$GATEWAY_ENDPOINT" --local --name "$GATEWAY_NAME"
        ;;
    desired)
        if [[ "$REGISTRATION_ACTIVE" != "true" ]]; then
            REGISTRATION_MUTATED=true
            run_timed openshell gateway select "$GATEWAY_NAME"
        fi
        ;;
    *) die "Unexpected registration state: $REGISTRATION_STATE" ;;
esac

# Certificate generation and driver initialization can take a moment. Use a
# single overall deadline; each probe receives only the remaining time.
READY_DEADLINE=$((SECONDS + READY_TIMEOUT_SECONDS))
while (( SECONDS < READY_DEADLINE )); do
    if validate_gateway "$READY_DEADLINE" >/dev/null 2>&1; then
        TRANSACTION_COMMITTED=true
        trap - EXIT
        rm -rf -- "$TXN_DIR"
        echo "Gateway: authenticated, healthy, Docker driver ready"
        exit 0
    fi
    (( SECONDS < READY_DEADLINE )) && sleep 1
done

run_timed_for "${READY_PROBE_TIMEOUT_SECONDS}s" \
    journalctl --user -u openshell-gateway.service --no-pager -n 30 >&2 || true
FINAL_DIAGNOSTIC_DEADLINE=$((SECONDS + READY_PROBE_TIMEOUT_SECONDS))
validate_gateway "$FINAL_DIAGNOSTIC_DEADLINE" || true
die "Gateway did not become ready within ${READY_TIMEOUT_SECONDS}s."
