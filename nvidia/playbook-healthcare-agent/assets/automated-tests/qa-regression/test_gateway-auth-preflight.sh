#!/usr/bin/env bash
# Regression test for the OpenShell authenticated readiness gate.
set -euo pipefail

PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"
CHECK="$PB_DIR/assets/scripts/ensure_openshell_gateway.sh"
[[ -x "$CHECK" || -f "$CHECK" ]] || { echo "FAIL: missing $CHECK"; exit 2; }

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
mkdir -p "$TMP_DIR/bin" "$TMP_DIR/home" "$TMP_DIR/proc-self/456"

cat >"$TMP_DIR/bin/docker" <<'MOCK'
#!/usr/bin/env bash
exit 0
MOCK

cat >"$TMP_DIR/bin/openshell-gateway" <<'MOCK'
#!/usr/bin/env bash
exit 0
MOCK

cat >"$TMP_DIR/bin/systemd-run" <<'MOCK'
#!/usr/bin/env bash
exit "${MOCK_SYSTEMD_RUN_RC:-0}"
MOCK

cat >"$TMP_DIR/bin/timeout" <<'MOCK'
#!/usr/bin/env bash
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --foreground|--kill-after=*) shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done
[[ "$#" -ge 2 ]] || exit 64
duration="$1"
shift
if [[ -n "${MOCK_TIMEOUT_LOG:-}" ]]; then
    printf '%s %s\n' "$duration" "$*" >>"$MOCK_TIMEOUT_LOG"
fi
if [[ -n "${MOCK_TIMEOUT_FAIL_MATCH:-}" && "$*" == *"$MOCK_TIMEOUT_FAIL_MATCH"* ]]; then
    exit 124
fi
exec "$@"
MOCK

cat >"$TMP_DIR/bin/flock" <<'MOCK'
#!/usr/bin/env bash
if [[ -n "${MOCK_FLOCK_LOG:-}" ]]; then
    printf '%s\n' "$*" >>"$MOCK_FLOCK_LOG"
fi
exit "${MOCK_FLOCK_RC:-0}"
MOCK

cat >"$TMP_DIR/bin/openshell" <<'MOCK'
#!/usr/bin/env bash
case "$*" in
    "gateway list --output json")
        printf '%s\n' "${MOCK_GATEWAY_LIST_JSON:-[]}"
        ;;
    "gateway remove openshell")
        printf '%s\n' "$*" >>"$MOCK_CALL_LOG"
        if [[ -n "${MOCK_REGISTRATION_DIR:-}" ]]; then
            rm -rf -- "$MOCK_REGISTRATION_DIR"
        fi
        ;;
    "gateway add https://127.0.0.1:17670 --local --name openshell")
        printf '%s\n' "$*" >>"$MOCK_CALL_LOG"
        if [[ -n "${MOCK_REGISTRATION_DIR:-}" ]]; then
            mkdir -p "$MOCK_REGISTRATION_DIR"
            printf 'candidate\n' >"$MOCK_REGISTRATION_DIR/metadata.json"
        fi
        if [[ -n "${MOCK_ACTIVE_GATEWAY_FILE:-}" ]]; then
            mkdir -p "$(dirname "$MOCK_ACTIVE_GATEWAY_FILE")"
            printf 'openshell\n' >"$MOCK_ACTIVE_GATEWAY_FILE"
        fi
        exit "${MOCK_ADD_RC:-0}"
        ;;
    "gateway add http://127.0.0.1:17670 --name openshell")
        printf '%s\n' "$*" >>"$MOCK_CALL_LOG"
        ;;
    "gateway select openshell")
        printf '%s\n' "$*" >>"$MOCK_CALL_LOG"
        if [[ -n "${MOCK_ACTIVE_GATEWAY_FILE:-}" ]]; then
            mkdir -p "$(dirname "$MOCK_ACTIVE_GATEWAY_FILE")"
            printf 'openshell\n' >"$MOCK_ACTIVE_GATEWAY_FILE"
        fi
        ;;
    "-g openshell status --output json")
        if [[ -n "${MOCK_STATUS_COUNTER_FILE:-}" ]]; then
            status_count=0
            [[ ! -f "$MOCK_STATUS_COUNTER_FILE" ]] || \
                read -r status_count <"$MOCK_STATUS_COUNTER_FILE"
            status_count=$((status_count + 1))
            printf '%s\n' "$status_count" >"$MOCK_STATUS_COUNTER_FILE"
            if (( status_count <= ${MOCK_STATUS_FAILURE_COUNT:-0} )); then
                exit 1
            fi
        fi
        printf '%s\n' "$MOCK_STATUS_JSON"
        ;;
    "-g openshell gateway info --output json")
        printf '%s\n' "$MOCK_INFO_OUTPUT"
        exit "${MOCK_INFO_RC:-0}"
        ;;
    *) echo "unexpected openshell invocation: $*" >&2; exit 64 ;;
esac
MOCK
chmod +x "$TMP_DIR/bin/docker" "$TMP_DIR/bin/systemd-run" \
    "$TMP_DIR/bin/timeout" "$TMP_DIR/bin/flock" \
    "$TMP_DIR/bin/openshell" "$TMP_DIR/bin/openshell-gateway"

cat >"$TMP_DIR/bin/systemctl" <<'MOCK'
#!/usr/bin/env bash
case "$*" in
    "--user cat openshell-gateway.service") exit 0 ;;
    "--user is-active --quiet openshell-gateway.service")
        exit "${MOCK_SERVICE_ACTIVE_RC:-1}"
        ;;
    "--user is-enabled openshell-gateway.service")
        printf '%s\n' "${MOCK_SERVICE_UNIT_STATE:-enabled}"
        ;;
    "--user show openshell-gateway.service -p ActiveState --value")
        printf '%s\n' "${MOCK_SERVICE_ACTIVE_STATE:-inactive}"
        ;;
    *) printf '%s\n' "$*" >>"$MOCK_CALL_LOG" ;;
esac
MOCK

chmod +x "$TMP_DIR/bin/systemctl"

# A non-gateway wrapper containing the legacy words must never self-match.
ln -s /usr/bin/bash "$TMP_DIR/proc-self/456/exe"
printf 'bash\0-c\0openshell-gateway --disable-tls\0' >"$TMP_DIR/proc-self/456/cmdline"
export OPENSHELL_PROC_ROOT="$TMP_DIR/proc-self"

GOOD_STATUS='{"gateway":"openshell","server":"https://127.0.0.1:17670","status":"connected","authentication":{"status":"authenticated","provider":"mTLS transport"}}'
BAD_AUTH_STATUS='{"gateway":"openshell","server":"http://127.0.0.1:17670","status":"connected","authentication":{"status":"failed","error":"missing authorization header"}}'
GOOD_INFO='{"server":"https://127.0.0.1:17670","status":"healthy","compute_drivers":[{"name":"docker"}]}'
NO_DOCKER_INFO='{"server":"https://127.0.0.1:17670","status":"healthy","compute_drivers":[]}'

run_check() {
    env \
        HOME="$TMP_DIR/home" \
        PATH="$TMP_DIR/bin:/usr/bin:/bin" \
        MOCK_SYSTEMD_RUN_RC="${MOCK_SYSTEMD_RUN_RC:-0}" \
        MOCK_STATUS_JSON="${MOCK_STATUS_JSON:-$GOOD_STATUS}" \
        MOCK_INFO_OUTPUT="${MOCK_INFO_OUTPUT:-$GOOD_INFO}" \
        MOCK_INFO_RC="${MOCK_INFO_RC:-0}" \
        MOCK_GATEWAY_LIST_JSON="${MOCK_GATEWAY_LIST_JSON:-[]}" \
        MOCK_TIMEOUT_FAIL_MATCH="${MOCK_TIMEOUT_FAIL_MATCH:-}" \
        MOCK_TIMEOUT_LOG="$TMP_DIR/timeout.log" \
        MOCK_FLOCK_RC="${MOCK_FLOCK_RC:-0}" \
        MOCK_FLOCK_LOG="$TMP_DIR/flock.log" \
        MOCK_CALL_LOG="$TMP_DIR/calls.log" \
        OPENSHELL_GATEWAY=some-unrelated-active-gateway \
        bash "$CHECK" --check-only >/dev/null 2>&1
}

expect_failure() {
    local name="$1"
    if run_check; then
        echo "FAIL: $name was accepted"
        exit 1
    fi
    echo "PASS: $name was rejected"
}

MOCK_SYSTEMD_RUN_RC=1 expect_failure "systemd manager without Docker access"

MOCK_TIMEOUT_FAIL_MATCH="systemd-run" \
expect_failure "timed-out systemd Docker probe"

MOCK_TIMEOUT_FAIL_MATCH="-g openshell status" \
expect_failure "timed-out OpenShell status probe"

MOCK_TIMEOUT_FAIL_MATCH="gateway info --output json" \
expect_failure "timed-out protected gateway-info probe"

: >"$TMP_DIR/calls.log"
MOCK_FLOCK_RC=1 expect_failure "concurrent gateway helper invocation"
if [[ -s "$TMP_DIR/calls.log" ]]; then
    echo "FAIL: lock contention allowed a gateway or service mutation"
    exit 1
fi
echo "PASS: lock contention failed before gateway and service mutations"

MOCK_SYSTEMD_RUN_RC=0 \
MOCK_STATUS_JSON="$BAD_AUTH_STATUS" \
expect_failure "Connected plus Authentication: Failed"

MOCK_STATUS_JSON="$GOOD_STATUS" \
MOCK_INFO_OUTPUT="$NO_DOCKER_INFO" \
expect_failure "authenticated gateway without Docker driver"

MOCK_INFO_OUTPUT="$GOOD_INFO"
if ! run_check; then
    echo "FAIL: authenticated healthy Docker gateway was rejected"
    exit 1
fi
echo "PASS: authenticated healthy Docker gateway was accepted"
for timed_probe in \
    'systemd-run --user --quiet --wait --pipe --collect' \
    'openshell -g openshell status --output json' \
    'openshell -g openshell gateway info --output json'; do
    if ! grep -Fq "$timed_probe" "$TMP_DIR/timeout.log"; then
        echo "FAIL: probe did not run through timeout: $timed_probe"
        exit 1
    fi
done
echo "PASS: Docker, status, and protected info probes are timeout-bounded"

# Registration inspection is also a fail-fast probe and must abort before any
# gateway config, service, or registration mutation.
LIST_TIMEOUT_HOME="$TMP_DIR/home-list-timeout"
mkdir -p "$LIST_TIMEOUT_HOME"
: >"$TMP_DIR/calls.log"
if env \
    HOME="$LIST_TIMEOUT_HOME" \
    PATH="$TMP_DIR/bin:/usr/bin:/bin" \
    MOCK_SYSTEMD_RUN_RC=0 \
    MOCK_TIMEOUT_FAIL_MATCH="gateway list --output json" \
    MOCK_GATEWAY_LIST_JSON='[]' \
    MOCK_CALL_LOG="$TMP_DIR/calls.log" \
    OPENSHELL_GATEWAY_CONFIG_SOURCE="$PB_DIR/assets/openshell-gateway.toml" \
    OPENSHELL_GATEWAY_CONFIG_TARGET="$LIST_TIMEOUT_HOME/.config/openshell/gateway.toml" \
    bash "$CHECK" >/dev/null 2>&1; then
    echo "FAIL: timed-out gateway-list probe was accepted"
    exit 1
fi
if [[ -e "$LIST_TIMEOUT_HOME/.config/openshell/gateway.toml" ]] || \
   [[ -s "$TMP_DIR/calls.log" ]]; then
    echo "FAIL: timed-out gateway-list probe caused a mutation"
    exit 1
fi
echo "PASS: timed-out gateway-list probe failed before all gateway mutations"

# Full setup must explicitly replace the expected legacy local registration;
# OpenShell does not overwrite an existing gateway name.
: >"$TMP_DIR/calls.log"
MOCK_GATEWAY_LIST_JSON='[{"name":"openshell","endpoint":"http://127.0.0.1:17670","auth":"plaintext","source":"user","type":"local","is_remote":false,"active":true}]'
env \
    HOME="$TMP_DIR/home" \
    PATH="$TMP_DIR/bin:/usr/bin:/bin" \
    MOCK_SYSTEMD_RUN_RC=0 \
    MOCK_STATUS_JSON="$GOOD_STATUS" \
    MOCK_INFO_OUTPUT="$GOOD_INFO" \
    MOCK_GATEWAY_LIST_JSON="$MOCK_GATEWAY_LIST_JSON" \
    MOCK_TIMEOUT_LOG="$TMP_DIR/timeout.log" \
    MOCK_CALL_LOG="$TMP_DIR/calls.log" \
    MOCK_SERVICE_ACTIVE_RC=1 \
    MOCK_SERVICE_ACTIVE_STATE=inactive \
    MOCK_SERVICE_UNIT_STATE=enabled \
    OPENSHELL_GATEWAY_CONFIG_SOURCE="$PB_DIR/assets/openshell-gateway.toml" \
    OPENSHELL_GATEWAY_CONFIG_TARGET="$TMP_DIR/home/.config/openshell/gateway.toml" \
    bash "$CHECK" >/dev/null

grep -qx 'gateway remove openshell' "$TMP_DIR/calls.log" || {
    echo "FAIL: legacy registration was not removed explicitly"
    exit 1
}
grep -qx 'gateway add https://127.0.0.1:17670 --local --name openshell' "$TMP_DIR/calls.log" || {
    echo "FAIL: mTLS registration was not added"
    exit 1
}
echo "PASS: expected legacy local registration was replaced with HTTPS/mTLS"
grep -Fq 'openshell gateway list --output json' "$TMP_DIR/timeout.log" || {
    echo "FAIL: gateway registration inspection did not run through timeout"
    exit 1
}
echo "PASS: gateway registration inspection is timeout-bounded"

# Configurations written from the former repository remain playbook-owned and
# are upgraded in place to the consolidated repository marker.
{
    printf '# Managed by station-healthcare-agent.\n'
    tail -n +2 "$PB_DIR/assets/openshell-gateway.toml"
} >"$TMP_DIR/home/.config/openshell/gateway.toml"
: >"$TMP_DIR/calls.log"
MOCK_GATEWAY_LIST_JSON='[{"name":"openshell","endpoint":"https://127.0.0.1:17670","auth":"mtls","source":"user","type":"local","is_remote":false,"active":true}]'
env \
    HOME="$TMP_DIR/home" \
    PATH="$TMP_DIR/bin:/usr/bin:/bin" \
    MOCK_SYSTEMD_RUN_RC=0 \
    MOCK_STATUS_JSON="$GOOD_STATUS" \
    MOCK_INFO_OUTPUT="$GOOD_INFO" \
    MOCK_GATEWAY_LIST_JSON="$MOCK_GATEWAY_LIST_JSON" \
    MOCK_CALL_LOG="$TMP_DIR/calls.log" \
    MOCK_SERVICE_ACTIVE_RC=0 \
    MOCK_SERVICE_ACTIVE_STATE=active \
    MOCK_SERVICE_UNIT_STATE=enabled \
    OPENSHELL_GATEWAY_CONFIG_SOURCE="$PB_DIR/assets/openshell-gateway.toml" \
    OPENSHELL_GATEWAY_CONFIG_TARGET="$TMP_DIR/home/.config/openshell/gateway.toml" \
    bash "$CHECK" >/dev/null
cmp -s "$PB_DIR/assets/openshell-gateway.toml" \
    "$TMP_DIR/home/.config/openshell/gateway.toml" || {
    echo "FAIL: legacy playbook marker was not upgraded"
    exit 1
}
echo "PASS: legacy playbook marker was accepted and upgraded"

# A healthy repeat run must not restart the shared gateway or rewrite its
# registration. This protects managed sandbox workloads from needless stops.
: >"$TMP_DIR/calls.log"
MOCK_GATEWAY_LIST_JSON='[{"name":"openshell","endpoint":"https://127.0.0.1:17670","auth":"mtls","source":"user","type":"local","is_remote":false,"active":true}]'
for _ in 1 2; do
    env \
        HOME="$TMP_DIR/home" \
        PATH="$TMP_DIR/bin:/usr/bin:/bin" \
        MOCK_SYSTEMD_RUN_RC=0 \
        MOCK_STATUS_JSON="$GOOD_STATUS" \
        MOCK_INFO_OUTPUT="$GOOD_INFO" \
        MOCK_GATEWAY_LIST_JSON="$MOCK_GATEWAY_LIST_JSON" \
        MOCK_CALL_LOG="$TMP_DIR/calls.log" \
        MOCK_SERVICE_ACTIVE_RC=0 \
        MOCK_SERVICE_ACTIVE_STATE=active \
        MOCK_SERVICE_UNIT_STATE=enabled \
        OPENSHELL_GATEWAY=some-unrelated-active-gateway \
        OPENSHELL_GATEWAY_CONFIG_SOURCE="$PB_DIR/assets/openshell-gateway.toml" \
        OPENSHELL_GATEWAY_CONFIG_TARGET="$TMP_DIR/home/.config/openshell/gateway.toml" \
        bash "$CHECK" >/dev/null
done
if grep -Eq '(^| )(restart|start|enable|remove|add)( |$)' "$TMP_DIR/calls.log"; then
    echo "FAIL: healthy repeat run mutated the service or registration"
    exit 1
fi
echo "PASS: healthy repeat runs made no service or registration changes"

# Runtime-only enablement is not durable across reboot. Promote it to a
# persistent enablement without restarting an already healthy gateway.
: >"$TMP_DIR/calls.log"
MOCK_GATEWAY_LIST_JSON='[{"name":"openshell","endpoint":"https://127.0.0.1:17670","auth":"mtls","source":"user","type":"local","is_remote":false,"active":true}]'
env \
    HOME="$TMP_DIR/home" \
    PATH="$TMP_DIR/bin:/usr/bin:/bin" \
    MOCK_SYSTEMD_RUN_RC=0 \
    MOCK_STATUS_JSON="$GOOD_STATUS" \
    MOCK_INFO_OUTPUT="$GOOD_INFO" \
    MOCK_GATEWAY_LIST_JSON="$MOCK_GATEWAY_LIST_JSON" \
    MOCK_CALL_LOG="$TMP_DIR/calls.log" \
    MOCK_SERVICE_ACTIVE_RC=0 \
    MOCK_SERVICE_ACTIVE_STATE=active \
    MOCK_SERVICE_UNIT_STATE=enabled-runtime \
    OPENSHELL_GATEWAY_CONFIG_SOURCE="$PB_DIR/assets/openshell-gateway.toml" \
    OPENSHELL_GATEWAY_CONFIG_TARGET="$TMP_DIR/home/.config/openshell/gateway.toml" \
    bash "$CHECK" >/dev/null
grep -qx -- '--user enable openshell-gateway.service' "$TMP_DIR/calls.log" || {
    echo "FAIL: runtime-only unit enablement was not made persistent"
    exit 1
}
if grep -Eq '(^| )(restart|start)( |$)' "$TMP_DIR/calls.log"; then
    echo "FAIL: persistent enablement restarted the healthy service"
    exit 1
fi
echo "PASS: runtime-only unit enablement was made persistent without restart"

# A healthy desired registration that is not active must be selected without
# restarting the already healthy shared gateway.
: >"$TMP_DIR/calls.log"
MOCK_GATEWAY_LIST_JSON='[{"name":"openshell","endpoint":"https://127.0.0.1:17670","auth":"mtls","source":"user","type":"local","is_remote":false,"active":false}]'
env \
    HOME="$TMP_DIR/home" \
    PATH="$TMP_DIR/bin:/usr/bin:/bin" \
    MOCK_SYSTEMD_RUN_RC=0 \
    MOCK_STATUS_JSON="$GOOD_STATUS" \
    MOCK_INFO_OUTPUT="$GOOD_INFO" \
    MOCK_GATEWAY_LIST_JSON="$MOCK_GATEWAY_LIST_JSON" \
    MOCK_CALL_LOG="$TMP_DIR/calls.log" \
    MOCK_SERVICE_ACTIVE_RC=0 \
    MOCK_SERVICE_ACTIVE_STATE=active \
    MOCK_SERVICE_UNIT_STATE=enabled \
    OPENSHELL_GATEWAY_CONFIG_SOURCE="$PB_DIR/assets/openshell-gateway.toml" \
    OPENSHELL_GATEWAY_CONFIG_TARGET="$TMP_DIR/home/.config/openshell/gateway.toml" \
    bash "$CHECK" >/dev/null
grep -qx 'gateway select openshell' "$TMP_DIR/calls.log" || {
    echo "FAIL: desired inactive registration was not selected"
    exit 1
}
if grep -Eq '(^| )(restart|start)( |$)' "$TMP_DIR/calls.log"; then
    echo "FAIL: selecting the desired registration restarted the service"
    exit 1
fi
echo "PASS: desired registration was selected without a service restart"

# An unexpected endpoint using the same name is operator-owned and must not be
# removed or overwritten.
: >"$TMP_DIR/calls.log"
MOCK_GATEWAY_LIST_JSON='[{"name":"openshell","endpoint":"https://gateway.example.test:17670","is_remote":true}]'
if env \
    HOME="$TMP_DIR/home" \
    PATH="$TMP_DIR/bin:/usr/bin:/bin" \
    MOCK_SYSTEMD_RUN_RC=0 \
    MOCK_STATUS_JSON="$GOOD_STATUS" \
    MOCK_INFO_OUTPUT="$GOOD_INFO" \
    MOCK_GATEWAY_LIST_JSON="$MOCK_GATEWAY_LIST_JSON" \
    MOCK_CALL_LOG="$TMP_DIR/calls.log" \
    OPENSHELL_GATEWAY_CONFIG_SOURCE="$PB_DIR/assets/openshell-gateway.toml" \
    OPENSHELL_GATEWAY_CONFIG_TARGET="$TMP_DIR/home/.config/openshell/gateway.toml" \
    bash "$CHECK" >/dev/null 2>&1; then
    echo "FAIL: unexpected operator-owned registration was replaced"
    exit 1
fi
if grep -q '^gateway \(remove\|add\)' "$TMP_DIR/calls.log"; then
    echo "FAIL: unexpected registration caused a gateway mutation"
    exit 1
fi
echo "PASS: unexpected operator-owned registration was preserved"

# Matching endpoints are still operator-owned when auth or source differs.
for unexpected in \
    '[{"name":"openshell","endpoint":"http://127.0.0.1:17670","auth":"oidc","source":"user","type":"local","is_remote":false,"active":true}]' \
    '[{"name":"openshell","endpoint":"https://127.0.0.1:17670","auth":"mtls","source":"system","type":"local","is_remote":false,"active":true}]'; do
    : >"$TMP_DIR/calls.log"
    if env \
        HOME="$TMP_DIR/home-unexpected" \
        PATH="$TMP_DIR/bin:/usr/bin:/bin" \
        MOCK_SYSTEMD_RUN_RC=0 \
        MOCK_STATUS_JSON="$GOOD_STATUS" \
        MOCK_INFO_OUTPUT="$GOOD_INFO" \
        MOCK_GATEWAY_LIST_JSON="$unexpected" \
        MOCK_CALL_LOG="$TMP_DIR/calls.log" \
        OPENSHELL_GATEWAY_CONFIG_SOURCE="$PB_DIR/assets/openshell-gateway.toml" \
        OPENSHELL_GATEWAY_CONFIG_TARGET="$TMP_DIR/home-unexpected/.config/openshell/gateway.toml" \
        bash "$CHECK" >/dev/null 2>&1; then
        echo "FAIL: unexpected auth/source registration was replaced"
        exit 1
    fi
    if grep -q '^gateway \(remove\|add\|select\)' "$TMP_DIR/calls.log"; then
        echo "FAIL: unexpected auth/source caused a gateway mutation"
        exit 1
    fi
done
echo "PASS: unexpected auth and system-owned registrations were preserved"

# A known legacy daemon blocker must be detected before gateway.toml or the
# package service is touched.
PREFLIGHT_HOME="$TMP_DIR/home-preflight"
mkdir -p "$PREFLIGHT_HOME" "$TMP_DIR/proc-legacy/123"
ln -s "$TMP_DIR/bin/openshell-gateway" "$TMP_DIR/proc-legacy/123/exe"
printf 'openshell-gateway\0--disable-tls\0--drivers\0docker\0' \
    >"$TMP_DIR/proc-legacy/123/cmdline"
: >"$TMP_DIR/calls.log"
if env \
    HOME="$PREFLIGHT_HOME" \
    PATH="$TMP_DIR/bin:/usr/bin:/bin" \
    MOCK_SYSTEMD_RUN_RC=0 \
    MOCK_STATUS_JSON="$GOOD_STATUS" \
    MOCK_INFO_OUTPUT="$GOOD_INFO" \
    MOCK_GATEWAY_LIST_JSON='[]' \
    MOCK_CALL_LOG="$TMP_DIR/calls.log" \
    OPENSHELL_PROC_ROOT="$TMP_DIR/proc-legacy" \
    OPENSHELL_GATEWAY_CONFIG_SOURCE="$PB_DIR/assets/openshell-gateway.toml" \
    OPENSHELL_GATEWAY_CONFIG_TARGET="$PREFLIGHT_HOME/.config/openshell/gateway.toml" \
    bash "$CHECK" >/dev/null 2>&1; then
    echo "FAIL: legacy daemon preflight was accepted"
    exit 1
fi
if [[ -e "$PREFLIGHT_HOME/.config/openshell/gateway.toml" ]] || \
   [[ -s "$TMP_DIR/calls.log" ]]; then
    echo "FAIL: legacy daemon preflight caused a mutation"
    exit 1
fi
echo "PASS: legacy daemon preflight failed before all mutations"

# If mTLS registration fails after the legacy entry is removed, the helper
# must restore the absent config preimage and the legacy registration.
ROLLBACK_HOME="$TMP_DIR/home-rollback"
ROLLBACK_REGISTRATION_DIR="$ROLLBACK_HOME/.config/openshell/gateways/openshell"
ROLLBACK_ACTIVE_FILE="$ROLLBACK_HOME/.config/openshell/active_gateway"
mkdir -p "$ROLLBACK_REGISTRATION_DIR"
printf 'legacy-preimage\n' >"$ROLLBACK_REGISTRATION_DIR/metadata.json"
printf 'other-gateway\n' >"$ROLLBACK_ACTIVE_FILE"
: >"$TMP_DIR/calls.log"
MOCK_GATEWAY_LIST_JSON='[{"name":"openshell","endpoint":"http://127.0.0.1:17670","auth":"plaintext","source":"user","type":"local","is_remote":false,"active":false}]'
if env \
    HOME="$ROLLBACK_HOME" \
    PATH="$TMP_DIR/bin:/usr/bin:/bin" \
    MOCK_SYSTEMD_RUN_RC=0 \
    MOCK_STATUS_JSON="$GOOD_STATUS" \
    MOCK_INFO_OUTPUT="$GOOD_INFO" \
    MOCK_GATEWAY_LIST_JSON="$MOCK_GATEWAY_LIST_JSON" \
    MOCK_CALL_LOG="$TMP_DIR/calls.log" \
    MOCK_ADD_RC=1 \
    MOCK_SERVICE_ACTIVE_RC=1 \
    MOCK_SERVICE_ACTIVE_STATE=active \
    MOCK_SERVICE_UNIT_STATE=enabled \
    MOCK_REGISTRATION_DIR="$ROLLBACK_REGISTRATION_DIR" \
    MOCK_ACTIVE_GATEWAY_FILE="$ROLLBACK_ACTIVE_FILE" \
    OPENSHELL_GATEWAY_CONFIG_SOURCE="$PB_DIR/assets/openshell-gateway.toml" \
    OPENSHELL_GATEWAY_CONFIG_TARGET="$ROLLBACK_HOME/.config/openshell/gateway.toml" \
    bash "$CHECK" >/dev/null 2>&1; then
    echo "FAIL: failed mTLS registration returned success"
    exit 1
fi
if [[ -e "$ROLLBACK_HOME/.config/openshell/gateway.toml" ]]; then
    echo "FAIL: failed migration did not restore the absent config preimage"
    exit 1
fi
grep -qx 'legacy-preimage' "$ROLLBACK_REGISTRATION_DIR/metadata.json" || {
    echo "FAIL: failed migration did not restore exact registration metadata"
    exit 1
}
grep -qx 'other-gateway' "$ROLLBACK_ACTIVE_FILE" || {
    echo "FAIL: failed migration did not restore the prior active gateway"
    exit 1
}
echo "PASS: failed registration restored config, exact registration metadata, and active gateway"

# A timed-out registration mutation must still enter rollback, and rollback
# must remove persistent enablement before restoring an enabled-runtime unit.
RUNTIME_HOME="$TMP_DIR/home-runtime-rollback"
RUNTIME_REGISTRATION_DIR="$RUNTIME_HOME/.config/openshell/gateways/openshell"
RUNTIME_ACTIVE_FILE="$RUNTIME_HOME/.config/openshell/active_gateway"
mkdir -p "$RUNTIME_REGISTRATION_DIR"
printf 'runtime-legacy-preimage\n' >"$RUNTIME_REGISTRATION_DIR/metadata.json"
printf 'runtime-other-gateway\n' >"$RUNTIME_ACTIVE_FILE"
: >"$TMP_DIR/calls.log"
: >"$TMP_DIR/timeout.log"
MOCK_GATEWAY_LIST_JSON='[{"name":"openshell","endpoint":"http://127.0.0.1:17670","auth":"plaintext","source":"user","type":"local","is_remote":false,"active":false}]'
if env \
    HOME="$RUNTIME_HOME" \
    PATH="$TMP_DIR/bin:/usr/bin:/bin" \
    MOCK_SYSTEMD_RUN_RC=0 \
    MOCK_STATUS_JSON="$GOOD_STATUS" \
    MOCK_INFO_OUTPUT="$GOOD_INFO" \
    MOCK_GATEWAY_LIST_JSON="$MOCK_GATEWAY_LIST_JSON" \
    MOCK_TIMEOUT_FAIL_MATCH="gateway add https://127.0.0.1:17670" \
    MOCK_TIMEOUT_LOG="$TMP_DIR/timeout.log" \
    MOCK_CALL_LOG="$TMP_DIR/calls.log" \
    MOCK_SERVICE_ACTIVE_RC=1 \
    MOCK_SERVICE_ACTIVE_STATE=active \
    MOCK_SERVICE_UNIT_STATE=enabled-runtime \
    MOCK_REGISTRATION_DIR="$RUNTIME_REGISTRATION_DIR" \
    MOCK_ACTIVE_GATEWAY_FILE="$RUNTIME_ACTIVE_FILE" \
    OPENSHELL_GATEWAY_CONFIG_SOURCE="$PB_DIR/assets/openshell-gateway.toml" \
    OPENSHELL_GATEWAY_CONFIG_TARGET="$RUNTIME_HOME/.config/openshell/gateway.toml" \
    bash "$CHECK" >/dev/null 2>&1; then
    echo "FAIL: timed-out mTLS registration returned success"
    exit 1
fi
if [[ -e "$RUNTIME_HOME/.config/openshell/gateway.toml" ]]; then
    echo "FAIL: timed-out mutation did not restore the absent config preimage"
    exit 1
fi
grep -qx 'runtime-legacy-preimage' "$RUNTIME_REGISTRATION_DIR/metadata.json" || {
    echo "FAIL: timed-out mutation did not restore registration metadata"
    exit 1
}
grep -qx 'runtime-other-gateway' "$RUNTIME_ACTIVE_FILE" || {
    echo "FAIL: timed-out mutation did not restore the active gateway"
    exit 1
}
disable_line="$(grep -n -m1 -x -- '--user disable openshell-gateway.service' "$TMP_DIR/calls.log" | cut -d: -f1)"
runtime_line="$(grep -n -m1 -x -- '--user enable --runtime openshell-gateway.service' "$TMP_DIR/calls.log" | cut -d: -f1)"
if [[ -z "$disable_line" || -z "$runtime_line" ]] || \
   (( disable_line >= runtime_line )); then
    echo "FAIL: enabled-runtime rollback did not remove persistent enablement first"
    exit 1
fi
grep -Fq 'gateway add https://127.0.0.1:17670' "$TMP_DIR/timeout.log" || {
    echo "FAIL: registration mutation was not timeout-bounded"
    exit 1
}
echo "PASS: timed-out mutation rolled back and restored runtime-only enablement"

# Readiness has one overall deadline rather than multiplying the command
# timeout by every retry and every protected probe.
DEADLINE_HOME="$TMP_DIR/home-readiness-deadline"
mkdir -p "$DEADLINE_HOME"
: >"$TMP_DIR/calls.log"
deadline_started="$SECONDS"
if env \
    HOME="$DEADLINE_HOME" \
    PATH="$TMP_DIR/bin:/usr/bin:/bin" \
    MOCK_SYSTEMD_RUN_RC=0 \
    MOCK_STATUS_JSON="$GOOD_STATUS" \
    MOCK_INFO_OUTPUT="$GOOD_INFO" \
    MOCK_GATEWAY_LIST_JSON='[]' \
    MOCK_TIMEOUT_FAIL_MATCH="-g openshell status" \
    MOCK_CALL_LOG="$TMP_DIR/calls.log" \
    MOCK_SERVICE_ACTIVE_RC=1 \
    MOCK_SERVICE_ACTIVE_STATE=inactive \
    MOCK_SERVICE_UNIT_STATE=enabled \
    OPENSHELL_GATEWAY_READY_TIMEOUT_SECONDS=1 \
    OPENSHELL_GATEWAY_READY_PROBE_TIMEOUT_SECONDS=1 \
    OPENSHELL_GATEWAY_CONFIG_SOURCE="$PB_DIR/assets/openshell-gateway.toml" \
    OPENSHELL_GATEWAY_CONFIG_TARGET="$DEADLINE_HOME/.config/openshell/gateway.toml" \
    bash "$CHECK" >/dev/null 2>&1; then
    echo "FAIL: permanently failing readiness probe returned success"
    exit 1
fi
deadline_elapsed=$((SECONDS - deadline_started))
if (( deadline_elapsed > 6 )); then
    echo "FAIL: one-second readiness deadline took ${deadline_elapsed}s"
    exit 1
fi
if [[ -e "$DEADLINE_HOME/.config/openshell/gateway.toml" ]]; then
    echo "FAIL: readiness deadline failure did not roll back gateway config"
    exit 1
fi
echo "PASS: readiness failure honored one overall deadline and rolled back"

# Becoming healthy only in the post-deadline diagnostic must not turn a timed
# out transaction into success after the EXIT trap has rolled it back.
LATE_HOME="$TMP_DIR/home-late-readiness"
LATE_STATUS_COUNTER="$TMP_DIR/late-status-count"
mkdir -p "$LATE_HOME"
: >"$TMP_DIR/calls.log"
rm -f "$LATE_STATUS_COUNTER"
if env \
    HOME="$LATE_HOME" \
    PATH="$TMP_DIR/bin:/usr/bin:/bin" \
    MOCK_SYSTEMD_RUN_RC=0 \
    MOCK_STATUS_JSON="$GOOD_STATUS" \
    MOCK_STATUS_COUNTER_FILE="$LATE_STATUS_COUNTER" \
    MOCK_STATUS_FAILURE_COUNT=1 \
    MOCK_INFO_OUTPUT="$GOOD_INFO" \
    MOCK_GATEWAY_LIST_JSON='[]' \
    MOCK_CALL_LOG="$TMP_DIR/calls.log" \
    MOCK_SERVICE_ACTIVE_RC=1 \
    MOCK_SERVICE_ACTIVE_STATE=inactive \
    MOCK_SERVICE_UNIT_STATE=enabled \
    OPENSHELL_GATEWAY_READY_TIMEOUT_SECONDS=1 \
    OPENSHELL_GATEWAY_READY_PROBE_TIMEOUT_SECONDS=1 \
    OPENSHELL_GATEWAY_CONFIG_SOURCE="$PB_DIR/assets/openshell-gateway.toml" \
    OPENSHELL_GATEWAY_CONFIG_TARGET="$LATE_HOME/.config/openshell/gateway.toml" \
    bash "$CHECK" >/dev/null 2>&1; then
    echo "FAIL: post-deadline readiness returned success after rollback"
    exit 1
fi
if [[ ! -f "$LATE_STATUS_COUNTER" ]] || \
   [[ "$(cat "$LATE_STATUS_COUNTER")" -lt 2 ]]; then
    echo "FAIL: late-readiness fixture did not reach the diagnostic probe"
    exit 1
fi
if [[ -e "$LATE_HOME/.config/openshell/gateway.toml" ]]; then
    echo "FAIL: post-deadline readiness did not roll back gateway config"
    exit 1
fi
echo "PASS: post-deadline readiness remained a failure and rolled back"
