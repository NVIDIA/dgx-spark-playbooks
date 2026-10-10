#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Focused tests for launch.sh. These tests mock mpirun and the NCCL test
# executable, so they validate argument construction and failure handling
# without requiring GPUs or a cluster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
LAUNCH="$(cd "$SCRIPT_DIR/../assets" && pwd -P)/launch.sh"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MOCK_BIN="$TMP_DIR/bin"
MOCK_HOME="$TMP_DIR/home"
mkdir -p "$MOCK_BIN" "$MOCK_HOME/nccl-tests/build"
mkdir -p "$TMP_DIR/custom"

cat >"$MOCK_BIN/mpirun" <<'EOF'
#!/bin/bash
printf '%s\n' "$@" >"${MOCK_MPIRUN_ARGS:?}"
if [ "${MOCK_MPIRUN_SLEEP:-0}" -gt 0 ]; then
    sleep "$MOCK_MPIRUN_SLEEP"
fi
exit "${MOCK_MPIRUN_STATUS:-0}"
EOF
chmod +x "$MOCK_BIN/mpirun"

cat >"$MOCK_BIN/timeout" <<'EOF'
#!/bin/bash
# Portable test double for GNU coreutils timeout. The production launcher
# requires the real timeout implementation; this double lets the argument
# and status mapping tests run on macOS too.
while [[ "$1" == --* ]]; do
    shift
done
shift # duration
if [ "${MOCK_TIMEOUT_STATUS:-0}" -ne 0 ]; then
    exit "$MOCK_TIMEOUT_STATUS"
fi
"$@"
EOF
chmod +x "$MOCK_BIN/timeout"

for binary in all_gather_perf all_reduce_perf; do
    printf '#!/bin/sh\nexit 0\n' >"$MOCK_HOME/nccl-tests/build/$binary"
    chmod +x "$MOCK_HOME/nccl-tests/build/$binary"
done
printf '#!/bin/sh\nexit 0\n' >"$TMP_DIR/custom/all_gather_perf"
chmod +x "$TMP_DIR/custom/all_gather_perf"

run_launch() {
    MOCK_MPIRUN_ARGS="$TMP_DIR/mpirun.args" \
        PATH="$MOCK_BIN:$PATH" \
        HOME="$MOCK_HOME" \
        MGMT_IFNAME=enP7s7 \
        BEGIN=8M END=64M FACTOR=2 \
        "$LAUNCH" "$@"
}

assert_contains() {
    local expected="$1"
    if ! rg -F -x -- "$expected" "$TMP_DIR/mpirun.args" >/dev/null; then
        echo "missing mocked mpirun argument: $expected" >&2
        sed -n '1,120p' "$TMP_DIR/mpirun.args" >&2
        exit 1
    fi
}

nodes=(node1 node2 node3 node4 node5 node6 node7 node8)
run_launch --topology switch "${nodes[@]}" >/dev/null
assert_contains "-np"
assert_contains "8"
assert_contains "-H"
assert_contains "node1:1,node2:1,node3:1,node4:1,node5:1,node6:1,node7:1,node8:1"
assert_contains "PATH=$MOCK_BIN:$PATH"
assert_contains "--prtemca"
assert_contains "plm_rsh_no_tree_spawn"
assert_contains "oob_tcp_if_include"
assert_contains "$MOCK_HOME/nccl-tests/build/all_gather_perf"

NCCL_TEST_BIN="$TMP_DIR/custom/all_gather_perf" \
    MOCK_MPIRUN_ARGS="$TMP_DIR/mpirun.args" \
    PATH="$MOCK_BIN:$PATH" \
    HOME="$MOCK_HOME" \
    MGMT_IFNAME=enP7s7 \
    "$LAUNCH" --topology switch node1 node2 >/dev/null
assert_contains "$TMP_DIR/custom/all_gather_perf"

run_launch --topology switch --collective all_reduce --interface fabric0 \
    node1 node2 node3 node4 node5 node6 >/dev/null
assert_contains "6"
assert_contains "PRTE_MCA_oob_tcp_if_include=fabric0"
assert_contains "UCX_NET_DEVICES=fabric0"
assert_contains "$MOCK_HOME/nccl-tests/build/all_reduce_perf"

rm -f "$TMP_DIR/dry-run.args"
PATH="$TMP_DIR/empty" HOME="$TMP_DIR/no-home" \
    MOCK_MPIRUN_ARGS="$TMP_DIR/dry-run.args" \
    "$LAUNCH" --dry-run --topology switch --collective all_reduce \
    --interface fabric0 node1 node2 >"$TMP_DIR/dry-run.out"
rg -F "Dry run: command not executed." "$TMP_DIR/dry-run.out" >/dev/null
[ ! -e "$TMP_DIR/dry-run.args" ] || {
    echo "dry-run invoked mpirun" >&2
    exit 1
}

if run_launch --topology switch node1 node1 >/dev/null 2>&1; then
    echo "duplicate node addresses were accepted" >&2
    exit 1
fi

if run_launch --topology switch node1 >/dev/null 2>&1; then
    echo "one-node switch configuration was accepted" >&2
    exit 1
fi

if run_launch --topology switch node1 node2 node3 node4 node5 node6 node7 node8 node9 >/dev/null 2>&1; then
    echo "nine-node switch configuration was accepted" >&2
    exit 1
fi

if run_launch --topology direct node1 node2 node3 >/dev/null 2>&1; then
    echo "three-node direct configuration was accepted" >&2
    exit 1
fi

if run_launch --topology switch --collective reduce node1 node2 >/dev/null 2>&1; then
    echo "invalid collective was accepted" >&2
    exit 1
fi

if run_launch --topology switch --interface "bad/name" node1 node2 >/dev/null 2>&1; then
    echo "invalid interface was accepted" >&2
    exit 1
fi

if run_launch --topology switch --timeout 0 node1 node2 >/dev/null 2>&1; then
    echo "zero timeout was accepted" >&2
    exit 1
fi

if MOCK_MPIRUN_ARGS="$TMP_DIR/mpirun.args" \
    MOCK_TIMEOUT_STATUS=124 \
    PATH="$MOCK_BIN:$PATH" \
    HOME="$MOCK_HOME" \
    "$LAUNCH" --topology switch --timeout 1 node1 node2 >/dev/null 2>"$TMP_DIR/timeout.err"; then
    echo "timed-out launch returned success" >&2
    exit 1
fi
rg -F "exceeded the 1s timeout" "$TMP_DIR/timeout.err" >/dev/null

echo "launch.sh argument and validation tests passed"
