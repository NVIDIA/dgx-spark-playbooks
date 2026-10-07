#!/usr/bin/env bash
# Regression test for QA 6694759.
#
# A short-lived command passed to `openshell sandbox create` became the
# sandbox's canonical process. When it exited, OpenShell correctly marked the
# sandbox Completed/Error, leaving the rest of setup with an unusable sandbox.
# The setup must create a persistent scratch sandbox, detach, fail closed on
# create errors, serialize lifecycle mutations, validate its sandbox name, and
# reject terminal phases before upload.

set -u

PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"
SETUP="$PB_DIR/assets/scripts/setup_sandbox.sh"
FAILURES=0

fail() {
    echo "FAIL: $1"
    FAILURES=$((FAILURES + 1))
}

pass() {
    echo "PASS: $1"
}

[ -f "$SETUP" ] || { echo "FAIL: setup script not found: $SETUP"; exit 1; }

# Static contract: persistent scratch shell, detached invocation, and no
# obsolete interactive-session flags or short-lived canonical command.
if ! grep -Eq -- '^[[:space:]]+--detach[[:space:]]+9>&-;[[:space:]]*then' "$SETUP"; then
    fail "sandbox create does not use --detach with setup lock fd 9 closed"
elif grep -Eq -- '--keep|--no-tty|sandbox-ok' "$SETUP"; then
    fail "sandbox create still contains --keep, --no-tty, or sandbox-ok"
elif grep -Eq 'timeout[[:space:]]+[0-9]+[[:space:]]+openshell[[:space:]]+sandbox[[:space:]]+create' "$SETUP"; then
    fail "sandbox create still has an obsolete outer timeout"
elif grep -Eq 'SANDBOX_READY_(MAX_POLLS|POLL_INTERVAL)' "$SETUP"; then
    fail "readiness polling still exposes test-only environment knobs"
else
    pass "create uses --detach directly with lock fd 9 closed and no obsolete lifecycle flags"
fi

TMPROOT=$(mktemp -d "${TMPDIR:-/tmp}/sandbox-lifecycle.XXXXXX") || exit 1
trap 'rm -rf "$TMPROOT"' EXIT
mkdir -p "$TMPROOT/work/scripts" "$TMPROOT/bin" "$TMPROOT/home"

# Exercise setup through the lifecycle gate only; everything after Step 4 is
# unrelated to this regression and would require a live sandbox.
awk '/^# --- Step 4:/ { exit } { print }' "$SETUP" \
    > "$TMPROOT/work/scripts/setup_sandbox.sh"
chmod +x "$TMPROOT/work/scripts/setup_sandbox.sh"

cat > "$TMPROOT/bin/bash" <<'MOCK'
#!/bin/sh
# Gateway/policy helpers are outside this regression's lifecycle boundary.
exit 0
MOCK

cat > "$TMPROOT/bin/ip" <<'MOCK'
#!/bin/sh
echo '2: docker0: <BROADCAST,MULTICAST,UP> mtu 1500'
echo '    inet 172.17.0.1/16 scope global docker0'
MOCK

cat > "$TMPROOT/bin/grep" <<'MOCK'
#!/bin/sh
# macOS grep has no -P; emulate the setup script's two bridge-address probes.
if [ "${1:-}" = "-oP" ]; then
    case "${2:-}" in
        *inet*) cat >/dev/null; echo '172.17.0.1'; exit 0 ;;
        *via*) cat >/dev/null; echo '172.17.0.1'; exit 0 ;;
    esac
fi
exec /usr/bin/grep "$@"
MOCK

cat > "$TMPROOT/bin/curl" <<'MOCK'
#!/bin/sh
exit 0
MOCK

cat > "$TMPROOT/bin/ss" <<'MOCK'
#!/bin/sh
exit 1
MOCK

cat > "$TMPROOT/bin/flock" <<'MOCK'
#!/bin/sh
if [ "${SCENARIO:-}" = "lock-contention" ]; then
    exit 1
fi
exit 0
MOCK

for cmd in systemctl fuser sleep; do
    cat > "$TMPROOT/bin/$cmd" <<'MOCK'
#!/bin/sh
exit 0
MOCK
done

cat > "$TMPROOT/bin/openshell" <<'MOCK'
#!/bin/sh
printf '%s\n' "$*" >> "$MOCK_LOG"

if [ "${1:-}" = "--version" ]; then
    echo "${OPENSHELL_VERSION:-openshell 0.0.111}"
    exit 0
fi

case "${1:-}:${2:-}" in
    provider:list)
        echo 'ollama-local'
        ;;
    provider:create|inference:set|forward:stop)
        ;;
    forward:list)
        ;;
    sandbox:create)
        if ( : >&9 ) 2>/dev/null; then
            echo 'sandbox-create-fd9=open' >> "$MOCK_LOG"
        else
            echo 'sandbox-create-fd9=closed' >> "$MOCK_LOG"
        fi
        case "${SCENARIO:-}" in
            create-failure)
                exit 42
                ;;
            create-failure-cleanup-failure)
                : > "$MOCK_STATE/created"
                exit 42
                ;;
        esac
        : > "$MOCK_STATE/created"
        ;;
    sandbox:delete)
        case "${SCENARIO:-}" in
            delete-failure|create-failure-cleanup-failure)
                exit 43
                ;;
            async-delete)
                : > "$MOCK_STATE/old-deleted"
                : > "$MOCK_STATE/created"
                echo 0 > "$MOCK_STATE/delete-polls"
                ;;
            *)
                : > "$MOCK_STATE/old-deleted"
                rm -f "$MOCK_STATE/created"
                ;;
        esac
        ;;
    sandbox:list)
        echo 'NAME IMAGE PHASE'
        if [ -f "$MOCK_STATE/created" ]; then
            if [ -f "$MOCK_STATE/delete-polls" ]; then
                polls=$(cat "$MOCK_STATE/delete-polls")
                if [ "$polls" -ge 1 ]; then
                    rm -f "$MOCK_STATE/created" "$MOCK_STATE/delete-polls"
                    exit 0
                fi
                echo 1 > "$MOCK_STATE/delete-polls"
            fi
            echo "clinical-sandbox openclaw ${SANDBOX_PHASE:-Ready}"
        elif [ "${EXISTING_TARGET:-0}" = "1" ] && [ ! -f "$MOCK_STATE/old-deleted" ]; then
            echo 'clinical-sandbox openclaw Ready'
            echo 'unrelated-sandbox openclaw Ready'
        fi
        ;;
esac
exit 0
MOCK
chmod +x "$TMPROOT/bin/"*

cat > "$TMPROOT/bin/docker" <<'MOCK'
#!/bin/sh
if [ "${1:-}" = "info" ]; then echo '27.0.0'; fi
exit 0
MOCK

cat > "$TMPROOT/bin/node" <<'MOCK'
#!/bin/sh
echo 'v22.5.0'
MOCK

cat > "$TMPROOT/bin/df" <<'MOCK'
#!/bin/sh
echo 'Filesystem 1G-blocks Used Available Use% Mounted'
echo '/dev/mock 400G 100G 300G 25% /'
MOCK

cat > "$TMPROOT/bin/nvidia-smi" <<'MOCK'
#!/bin/sh
echo 'NVIDIA GB300'
MOCK
chmod +x "$TMPROOT/bin/"*

run_case() {
    name=$1
    scenario=$2
    phase=$3
    existing=${4:-0}
    sandbox_name=${5:-clinical-sandbox}
    openshell_version=${6:-openshell 0.0.111}
    state="$TMPROOT/state-$name"
    log="$TMPROOT/$name.log"
    out="$TMPROOT/$name.out"
    mkdir -p "$state"
    : > "$log"

    set +e
    env PATH="$TMPROOT/bin:/usr/bin:/bin" \
        HOME="$TMPROOT/home" \
        MOCK_LOG="$log" \
        MOCK_STATE="$state" \
        SCENARIO="$scenario" \
        SANDBOX_PHASE="$phase" \
        EXISTING_TARGET="$existing" \
        SANDBOX_NAME="$sandbox_name" \
        OPENSHELL_VERSION="$openshell_version" \
        /bin/bash "$TMPROOT/work/scripts/setup_sandbox.sh" >"$out" 2>&1
    rc=$?
    set -e

    CASE_RC=$rc
    CASE_LOG=$log
    CASE_OUT=$out
}

run_case ready success Ready
if [ "$CASE_RC" -eq 0 ] \
   && grep -q '^sandbox create .* --detach$' "$CASE_LOG" \
   && grep -q '^sandbox-create-fd9=closed$' "$CASE_LOG"; then
    pass "Ready sandbox proceeds after detached create"
else
    fail "Ready sandbox did not complete lifecycle gate with fd 9 closed (exit $CASE_RC)"
    sed -n '1,120p' "$CASE_OUT"
fi

for version_case in older malformed; do
    if [ "$version_case" = "older" ]; then
        version='openshell 0.0.110'
        expected='v0.0.111+ is required'
    else
        version='openshell 0.0.111.1'
        expected='Invalid OpenShell version output'
    fi
    run_case "setup-version-$version_case" success Ready 0 clinical-sandbox "$version"
    if [ "$CASE_RC" -ne 0 ] \
       && grep -q "$expected" "$CASE_OUT" \
       && [ "$(wc -l < "$CASE_LOG" | tr -d ' ')" -eq 1 ] \
       && grep -q '^--version$' "$CASE_LOG"; then
        pass "setup rejects $version_case OpenShell version before mutation"
    else
        fail "setup did not reject $version_case OpenShell version before mutation"
    fi
done

run_case create_failure create-failure Pending
if [ "$CASE_RC" -eq 42 ] && grep -q 'failed to create' "$CASE_OUT"; then
    pass "create failure is propagated without continuing"
else
    fail "create failure was not propagated (exit $CASE_RC, expected 42)"
fi

for invalid_name in '--all' 'Invalid_Name'; do
    run_case "invalid-name-${invalid_name#--}" success Ready 0 "$invalid_name"
    if [ "$CASE_RC" -eq 2 ] \
       && grep -q 'DNS-1123 label' "$CASE_OUT" \
       && [ ! -s "$CASE_LOG" ]; then
        pass "invalid sandbox name '$invalid_name' fails before OpenShell mutation"
    else
        fail "invalid sandbox name '$invalid_name' reached OpenShell or returned the wrong status"
    fi
done

run_case lock_contention lock-contention Ready
if [ "$CASE_RC" -eq 75 ] \
   && grep -q 'setup is already running' "$CASE_OUT" \
   && [ ! -s "$CASE_LOG" ]; then
    pass "setup-wide lock contention fails before OpenShell mutation"
else
    fail "setup-wide lock did not serialize lifecycle mutation"
fi

for terminal in Completed Error; do
    run_case "terminal-$terminal" success "$terminal"
    if [ "$CASE_RC" -ne 0 ] \
       && grep -q "terminal phase '$terminal'" "$CASE_OUT" \
       && grep -q '^sandbox delete clinical-sandbox$' "$CASE_LOG"; then
        pass "terminal phase $terminal fails promptly and cleans the named sandbox"
    else
        fail "terminal phase $terminal was not rejected and cleaned up"
    fi
done

run_case ready_timeout success Pending
if [ "$CASE_RC" -ne 0 ] \
   && grep -q 'did not reach Ready after 60 polls' "$CASE_OUT" \
   && grep -q '^sandbox delete clinical-sandbox$' "$CASE_LOG"; then
    pass "readiness timeout fails and cleans the named sandbox"
else
    fail "readiness timeout did not fail closed"
fi

run_case scoped_cleanup success Ready 1
if [ "$CASE_RC" -eq 0 ] \
   && grep -q '^sandbox delete clinical-sandbox$' "$CASE_LOG" \
   && ! grep -q '^sandbox delete unrelated-sandbox$' "$CASE_LOG"; then
    pass "cleanup deletes only the configured sandbox"
else
    fail "cleanup was not sandbox-scoped"
fi

run_case async_delete async-delete Ready 1
if [ "$CASE_RC" -eq 0 ] \
   && grep -q 'Sandbox deleted: clinical-sandbox' "$CASE_OUT" \
   && [ "$(grep -c '^sandbox list$' "$CASE_LOG")" -ge 3 ]; then
    pass "asynchronous deletion is polled until the sandbox is absent"
else
    fail "asynchronous sandbox deletion was not verified before create"
fi

run_case delete_failure delete-failure Ready 1
if [ "$CASE_RC" -ne 0 ] \
   && grep -q "Failed to delete sandbox 'clinical-sandbox'" "$CASE_OUT" \
   && ! grep -q '^sandbox create ' "$CASE_LOG"; then
    pass "delete failure is visible and prevents sandbox recreation"
else
    fail "delete failure was suppressed or setup continued"
fi

run_case rollback_failure create-failure-cleanup-failure Pending
if [ "$CASE_RC" -eq 42 ] \
   && grep -q 'Sandbox rollback failed after create exit 42' "$CASE_OUT"; then
    pass "failed rollback is visible while preserving the create exit code"
else
    fail "rollback failure hid or replaced the original create exit code"
fi

# The Makefile teardown path applies the same name boundary before invoking
# OpenShell. An option-like name must stop the target before any delete call.
teardown_log="$TMPROOT/teardown-invalid.log"
teardown_out="$TMPROOT/teardown-invalid.out"
: > "$teardown_log"
set +e
env PATH="$TMPROOT/bin:/usr/bin:/bin" \
    HOME="$TMPROOT/home" \
    MOCK_LOG="$teardown_log" \
    MOCK_STATE="$TMPROOT/state-teardown-invalid" \
    SANDBOX_NAME='--all' \
    /usr/bin/make -s -C "$PB_DIR/assets" teardown >"$teardown_out" 2>&1
teardown_rc=$?
set -e
if [ "$teardown_rc" -ne 0 ] \
   && grep -q 'DNS-1123 label' "$teardown_out" \
   && [ ! -s "$teardown_log" ]; then
    pass "make teardown rejects --all before any OpenShell call"
else
    fail "make teardown allowed an option-like sandbox name to reach OpenShell"
fi

# `make prereq` must enforce the OpenShell release that introduced the detached
# lifecycle relied on above. Other prerequisite commands are deterministic
# mocks so these cases exercise only the real Makefile version logic.
run_prereq_case() {
    name=$1
    version=$2
    out="$TMPROOT/prereq-$name.out"
    log="$TMPROOT/prereq-$name.log"
    : > "$log"
    set +e
    env PATH="$TMPROOT/bin:/usr/bin:/bin" \
        HOME="$TMPROOT/home" \
        MOCK_LOG="$log" \
        MOCK_STATE="$TMPROOT/state-prereq-$name" \
        OPENSHELL_VERSION="$version" \
        /usr/bin/make -s -C "$PB_DIR/assets" prereq >"$out" 2>&1
    rc=$?
    set -e
    CASE_RC=$rc
    CASE_OUT=$out
}

run_prereq_case minimum 'openshell 0.0.111'
if [ "$CASE_RC" -eq 0 ]; then
    pass "OpenShell 0.0.111 satisfies make prereq"
else
    fail "OpenShell 0.0.111 was rejected by make prereq"
fi

run_prereq_case newer 'openshell 0.1.0'
if [ "$CASE_RC" -eq 0 ]; then
    pass "newer OpenShell version satisfies make prereq"
else
    fail "newer OpenShell version was rejected by make prereq"
fi

run_prereq_case older 'openshell 0.0.110'
if [ "$CASE_RC" -ne 0 ] && grep -q 'need v0.0.111+' "$CASE_OUT"; then
    pass "older OpenShell version fails make prereq"
else
    fail "older OpenShell version did not fail make prereq"
fi

run_prereq_case invalid 'not-semver'
if [ "$CASE_RC" -ne 0 ] && grep -q 'invalid version output' "$CASE_OUT"; then
    pass "invalid OpenShell version fails make prereq"
else
    fail "invalid OpenShell version did not fail make prereq"
fi

run_prereq_case malformed 'openshell 0.0.111.1'
if [ "$CASE_RC" -ne 0 ] && grep -q 'invalid version output' "$CASE_OUT"; then
    pass "malformed OpenShell version fails make prereq"
else
    fail "malformed OpenShell version did not fail make prereq"
fi

if [ "$FAILURES" -ne 0 ]; then
    echo "FAIL: $FAILURES sandbox lifecycle assertion(s) failed"
    exit 1
fi

echo "PASS: QA 6694759 sandbox lifecycle regression checks all passed"
exit 0
