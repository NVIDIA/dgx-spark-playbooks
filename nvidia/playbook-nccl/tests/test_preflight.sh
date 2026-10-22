#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# CPU-only tests for the read-only preflight. The SSH client is mocked so the
# tests validate aggregation and fail-closed behavior without remote hosts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
PREFLIGHT="$(cd "$SCRIPT_DIR/../assets" && pwd -P)/preflight.sh"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MOCK_BIN="$TMP_DIR/bin"
mkdir -p "$MOCK_BIN"

cat >"$MOCK_BIN/ssh" <<'EOF'
#!/bin/bash
node=""
skip_value=false
for arg in "$@"; do
    if [ "$skip_value" = true ]; then
        skip_value=false
        continue
    fi
    if [ "$arg" = "-o" ]; then
        skip_value=true
        continue
    fi
    if [[ "$arg" == -* ]]; then
        continue
    fi
    node="$arg"
    break
done

if [ "${MOCK_PREFLIGHT_FAIL_NODE:-}" = "$node" ]; then
    printf 'mocked SSH failure for %s\n' "$node" >&2
    exit 1
fi

printf 'arch=aarch64\tgpus=1\tdriver=595.84\tmpi=Open MPI v5\tstatus=pass\n'
EOF
chmod +x "$MOCK_BIN/ssh"

PATH="$MOCK_BIN:$PATH" NCCL_IB_HCA=mlx5_1:1,mlx5_3:1 \
    "$PREFLIGHT" --collective both --interface fabric0 \
    --test-binary /opt/nccl-tests/build/all_gather_perf node1 node2 \
    >"$TMP_DIR/pass.out"
rg -F "Preflight passed; no NCCL workload was started." "$TMP_DIR/pass.out" >/dev/null
rg -F $'node1\tarch=aarch64' "$TMP_DIR/pass.out" >/dev/null
rg -F $'node2\tarch=aarch64' "$TMP_DIR/pass.out" >/dev/null

if PATH="$MOCK_BIN:$PATH" "$PREFLIGHT" node1 node1 >/dev/null 2>&1; then
    echo "duplicate node addresses were accepted" >&2
    exit 1
fi

if MOCK_PREFLIGHT_FAIL_NODE=node2 PATH="$MOCK_BIN:$PATH" \
    "$PREFLIGHT" node1 node2 >"$TMP_DIR/fail.out" 2>&1; then
    echo "remote preflight failure returned success" >&2
    exit 1
fi
rg -F "Preflight failed; no NCCL workload was started." "$TMP_DIR/fail.out" >/dev/null

echo "preflight argument and fail-closed tests passed"
