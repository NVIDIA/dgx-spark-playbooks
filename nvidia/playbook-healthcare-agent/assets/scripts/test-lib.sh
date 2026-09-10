#!/usr/bin/env bash
# test-lib.sh -- Shared utilities for the clinical-intelligence test suite.
# Source this file from test-all.sh:  source "$(dirname "$0")/test-lib.sh"

set -uo pipefail

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
SKIP_COUNT=0
TEST_LOG_DIR=""
VERBOSE="${VERBOSE:-false}"
SANDBOX_NAME="${SANDBOX_NAME:-clinical-sandbox}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

init_test_run() {
    local ts
    ts=$(date +%Y-%m-%d-%H%M%S)
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    TEST_LOG_DIR="${repo_root}/test-results/${ts}"
    mkdir -p "$TEST_LOG_DIR"
    ln -sfn "$TEST_LOG_DIR" "${repo_root}/test-results/latest"
    echo "Test logs: $TEST_LOG_DIR"
}

# Resolve the active gateway name for the ssh-proxy ProxyCommand.
# Precedence: OPENSHELL_GATEWAY env var → active gateway from `openshell status`
# → fallback to 'openshell'. Prevents "Unknown gateway 'openshell'" when the
# user previously ran the NemoClaw playbook (gateway registered as 'nemoclaw').
_gw_name() {
    if [ -n "${OPENSHELL_GATEWAY:-}" ]; then
        printf '%s' "$OPENSHELL_GATEWAY"
        return
    fi
    local name
    # Strip ANSI color codes first — `openshell status` colorizes the output,
    # putting a reset code between "Gateway:" and the name, which defeats the grep.
    name=$(openshell status 2>/dev/null \
        | sed "s/$(printf '\033')\[[0-9;]*[a-zA-Z]//g" \
        | grep -oE 'Gateway:[[:space:]]+[A-Za-z0-9_-]+' \
        | awk '{print $NF}' | head -1)
    printf '%s' "${name:-openshell}"
}
GW_NAME="${GW_NAME:-$(_gw_name)}"

_sandbox() {
    ssh -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR \
        -o ConnectTimeout=10 \
        -o "ProxyCommand=openshell ssh-proxy --gateway-name $GW_NAME --name $SANDBOX_NAME" \
        "sandbox@openshell-$SANDBOX_NAME" "$@"
}

_bridge_ip() {
    ip -4 addr show docker0 2>/dev/null | grep -oP 'inet \K[\d.]+' || echo "172.17.0.1"
}

# run_test TEST_ID NAME CMD ASSERTION HINT [EXTRA_ARGS...]
#
# Assertions: assert_exit_0, assert_exit_nonzero, assert_contains,
#             assert_not_contains, assert_equals, assert_numeric_gt,
#             assert_output_not_empty
#
# Severity: prefix assertion with "warn:" to log a warning instead of a failure.
#   e.g.  "warn:assert_contains" -- same check, WARN on mismatch instead of FAIL.
#   Warnings do NOT cause a non-zero exit code.
run_test() {
    local test_id="$1"
    local test_name="$2"
    local cmd="$3"
    local raw_assertion="$4"
    local hint="$5"
    shift 5
    local extra_args=("$@")

    local warn_only=false
    local assertion="$raw_assertion"
    if [[ "$assertion" == warn:* ]]; then
        warn_only=true
        assertion="${assertion#warn:}"
    fi

    local log_file="$TEST_LOG_DIR/${test_id}.log"
    local start_s
    start_s=$(date +%s)

    {
        echo "=== Test: $test_id $test_name ==="
        echo "Command: $cmd"
        echo "Assertion: $raw_assertion ${extra_args[*]:-}"
        echo "---"
    } > "$log_file"

    local output=""
    local exit_code=0
    output=$(eval "$cmd" 2>&1) || exit_code=$?

    {
        echo "Exit code: $exit_code"
        echo "Output:"
        echo "$output"
    } >> "$log_file"

    local end_s
    end_s=$(date +%s)
    local elapsed="$(( end_s - start_s ))s"

    local pass=false
    case "$assertion" in
        assert_exit_0)
            if [[ "$exit_code" -eq 0 ]]; then pass=true; fi
            ;;
        assert_exit_nonzero)
            if [[ "$exit_code" -ne 0 ]]; then pass=true; fi
            ;;
        assert_contains)
            local needle="${extra_args[0]:-}"
            if echo "$output" | grep -qi "$needle" 2>/dev/null; then pass=true; fi
            ;;
        assert_not_contains)
            local needle="${extra_args[0]:-}"
            if ! echo "$output" | grep -qi "$needle" 2>/dev/null; then pass=true; fi
            ;;
        assert_equals)
            local expected="${extra_args[0]:-}"
            if [[ "$(echo "$output" | tr -d '[:space:]')" == "$(echo "$expected" | tr -d '[:space:]')" ]]; then pass=true; fi
            ;;
        assert_numeric_gt)
            local threshold="${extra_args[0]:-0}"
            local val
            val=$(echo "$output" | tr -dc '0-9' | head -c 10)
            if [[ -n "$val" ]] && [[ "$val" -gt "$threshold" ]]; then pass=true; fi
            ;;
        assert_output_not_empty)
            if [[ -n "$(echo "$output" | tr -d '[:space:]')" ]]; then pass=true; fi
            ;;
        *)
            echo "  Unknown assertion: $assertion" >> "$log_file"
            ;;
    esac

    if $pass; then
        PASS_COUNT=$((PASS_COUNT + 1))
        printf "${GREEN}[PASS]${NC} %-6s %-45s ${CYAN}(%s)${NC}\n" "$test_id" "$test_name" "$elapsed"
        echo "Result: PASS" >> "$log_file"
    elif $warn_only; then
        WARN_COUNT=$((WARN_COUNT + 1))
        printf "${YELLOW}[WARN]${NC} %-6s %-45s ${CYAN}(%s)${NC}\n" "$test_id" "$test_name" "$elapsed"
        printf "       ${YELLOW}%s${NC}\n" "$hint"
        echo "Result: WARN" >> "$log_file"
        echo "Hint: $hint" >> "$log_file"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        printf "${RED}[FAIL]${NC} %-6s %-45s ${CYAN}(%s)${NC}\n" "$test_id" "$test_name" "$elapsed"
        printf "       ${YELLOW}Hint: %s${NC}\n" "$hint"
        echo "Result: FAIL" >> "$log_file"
        echo "Hint: $hint" >> "$log_file"
    fi

    if [[ "$VERBOSE" == "true" ]]; then
        echo "  Output: $(echo "$output" | head -5)"
        [[ $(echo "$output" | wc -l) -gt 5 ]] && echo "  ... (see $log_file)"
    fi
}

print_summary() {
    local total=$((PASS_COUNT + FAIL_COUNT + WARN_COUNT + SKIP_COUNT))
    echo ""
    echo "=============================="
    if (( WARN_COUNT > 0 )); then
        printf "${BOLD}Results: ${GREEN}%d passed${NC}, ${YELLOW}%d warnings${NC}, ${RED}%d failed${NC}, %d total\n" \
            "$PASS_COUNT" "$WARN_COUNT" "$FAIL_COUNT" "$total"
    else
        printf "${BOLD}Results: ${GREEN}%d passed${NC}, ${RED}%d failed${NC}, %d total\n" \
            "$PASS_COUNT" "$FAIL_COUNT" "$total"
    fi
    echo "=============================="

    if [[ -n "$TEST_LOG_DIR" ]]; then
        {
            echo "Total: $total"
            echo "Passed: $PASS_COUNT"
            echo "Warnings: $WARN_COUNT"
            echo "Failed: $FAIL_COUNT"
            echo ""
            if (( WARN_COUNT > 0 )); then
                echo "Warnings (non-blocking):"
                for f in "$TEST_LOG_DIR"/*.log; do
                    if grep -q "Result: WARN" "$f" 2>/dev/null; then
                        local tid thint
                        tid=$(basename "$f" .log)
                        thint=$(grep "^Hint:" "$f" | head -1 | sed 's/^Hint: //')
                        echo "  $tid -- $thint"
                    fi
                done
                echo ""
            fi
            if (( FAIL_COUNT > 0 )); then
                echo "Failed tests:"
                for f in "$TEST_LOG_DIR"/*.log; do
                    if grep -q "Result: FAIL" "$f" 2>/dev/null; then
                        local tid thint
                        tid=$(basename "$f" .log)
                        thint=$(grep "^Hint:" "$f" | head -1 | sed 's/^Hint: //')
                        echo "  $tid -- $thint"
                    fi
                done
            fi
        } > "$TEST_LOG_DIR/summary.txt"
        echo "Logs: $TEST_LOG_DIR/"
    fi

    (( FAIL_COUNT > 0 )) && return 1
    return 0
}
