#!/bin/bash
# Read-only prerequisite check for a multi-node NCCL launch.
#
# Usage: ./preflight.sh [options] <NODE_1> <NODE_2> [<NODE_3> ...]
#   --collective all_gather|all_reduce|both (default: both)
#   --interface <iface>       management/bootstrap interface
#   --test-binary <path>      exact NCCL test binary to check on every node
#
# This script does not install packages, change networking, start containers,
# or launch a collective. It checks the remote prerequisites that otherwise
# fail only after mpirun has started.
set -euo pipefail

MGMT_IFNAME="${MGMT_IFNAME:-enP7s7}"
COLLECTIVE="both"
TEST_BINARY="${NCCL_TEST_BIN:-}"
MPI_SSH_IDENTITY_FILE="${MPI_SSH_IDENTITY_FILE:-}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --collective)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "Error: --collective requires all_gather, all_reduce, or both." >&2
                exit 1
            fi
            COLLECTIVE="$2"; shift 2 ;;
        --collective=*) COLLECTIVE="${1#*=}"; shift ;;
        --interface)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "Error: --interface requires a value." >&2
                exit 1
            fi
            MGMT_IFNAME="$2"; shift 2 ;;
        --interface=*) MGMT_IFNAME="${1#*=}"; shift ;;
        --test-binary)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "Error: --test-binary requires a path." >&2
                exit 1
            fi
            TEST_BINARY="$2"; shift 2 ;;
        --test-binary=*) TEST_BINARY="${1#*=}"; shift ;;
        -*|--help|-h)
            echo "Usage: $0 [--collective all_gather|all_reduce|both] [--interface <iface>] [--test-binary <path>] <NODE_1> <NODE_2> [<NODE_3> ...]" >&2
            exit 1 ;;
        *) break ;;
    esac
done

case "$COLLECTIVE" in
    all_gather|all_reduce|both) ;;
    *) echo "Error: --collective must be all_gather, all_reduce, or both." >&2; exit 1 ;;
esac

if [ -z "$MGMT_IFNAME" ] || [[ ! "$MGMT_IFNAME" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
    echo "Error: interface must be a non-empty Linux interface name." >&2
    exit 1
fi

if [ -n "$TEST_BINARY" ] && [[ ! "$TEST_BINARY" =~ ^[A-Za-z0-9_./:+-]+$ ]]; then
    echo "Error: test binary path contains unsupported characters." >&2
    exit 1
fi

if [ -n "$MPI_SSH_IDENTITY_FILE" ] && [[ ! "$MPI_SSH_IDENTITY_FILE" =~ ^[A-Za-z0-9_./:-]+$ ]]; then
    echo "Error: MPI_SSH_IDENTITY_FILE contains unsupported characters." >&2
    exit 1
fi

NODES=("$@")
NP="${#NODES[@]}"
if [ "$NP" -lt 2 ] || [ "$NP" -gt 8 ]; then
    echo "Error: preflight requires 2-8 nodes (got $NP)." >&2
    exit 1
fi

seen_nodes="|"
for node in "${NODES[@]}"; do
    if [ -z "$node" ] || [[ ! "$node" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
        echo "Error: invalid node address '$node'." >&2
        exit 1
    fi
    if [[ "$seen_nodes" == *"|$node|"* ]]; then
        echo "Error: duplicate node address '$node'." >&2
        exit 1
    fi
    seen_nodes="${seen_nodes}${node}|"
done

command -v ssh >/dev/null || { echo "Error: ssh not found." >&2; exit 1; }

ssh_options=(
    -o BatchMode=yes
    -o ConnectTimeout=8
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=3
    -o UserKnownHostsFile=/dev/null
    -o StrictHostKeyChecking=no
)
if [ -n "$MPI_SSH_IDENTITY_FILE" ]; then
    ssh_options+=(-i "$MPI_SSH_IDENTITY_FILE")
fi

remote_check() {
    local node="$1"
    ssh "${ssh_options[@]}" "$node" bash -s -- \
        "$MGMT_IFNAME" "$COLLECTIVE" "${TEST_BINARY:-__DEFAULT_TEST_BINARY__}" "${NCCL_IB_HCA:-__NO_HCA__}" "${PATH:-}" <<'REMOTE'
set -u
iface="$1"
collective="$2"
test_binary="$3"
requested_hca="$4"
remote_path="$5"
[ "$test_binary" = "__DEFAULT_TEST_BINARY__" ] && test_binary=""
[ "$requested_hca" = "__NO_HCA__" ] && requested_hca=""
export PATH="$remote_path"
failures=()

command -v nvidia-smi >/dev/null || failures+=(nvidia-smi)
command -v mpirun >/dev/null || failures+=(mpirun)
command -v ip >/dev/null || failures+=(ip)
if ! command -v rdma >/dev/null && ! command -v ibdev2netdev >/dev/null; then
    failures+=(rdma-tools)
fi

if command -v nvidia-smi >/dev/null; then
    gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | awk 'NF {count++} END {print count+0}')"
    [ "$gpu_count" -gt 0 ] 2>/dev/null || failures+=(gpu)
else
    gpu_count="?"
fi

if command -v ip >/dev/null; then
    ip -o link show dev "$iface" >/dev/null 2>&1 || failures+=(interface)
else
    failures+=(interface)
fi

if [ -n "$test_binary" ]; then
    [ -x "$test_binary" ] || failures+=(test-binary)
else
    case "$collective" in
        all_gather) [ -x "$HOME/nccl-tests/build/all_gather_perf" ] || failures+=(all_gather_perf) ;;
        all_reduce) [ -x "$HOME/nccl-tests/build/all_reduce_perf" ] || failures+=(all_reduce_perf) ;;
        both)
            [ -x "$HOME/nccl-tests/build/all_gather_perf" ] || failures+=(all_gather_perf)
            [ -x "$HOME/nccl-tests/build/all_reduce_perf" ] || failures+=(all_reduce_perf) ;;
    esac
fi

if [ -n "$requested_hca" ]; then
    old_ifs="$IFS"
    IFS=,
    for hca in $requested_hca; do
        device="${hca%%:*}"
        [ -e "/sys/class/infiniband/$device" ] || failures+=("hca:$device")
    done
    IFS="$old_ifs"
fi

arch="$(uname -m 2>/dev/null || echo '?')"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
if command -v mpirun >/dev/null; then
    mpi="$(mpirun --version 2>/dev/null | head -1)"
else
    mpi="missing"
fi
printf 'arch=%s\tgpus=%s\tdriver=%s\tmpi=%s\tstatus=%s' \
    "$arch" "$gpu_count" "$driver" "$mpi" \
    "${failures[*]:-pass}"
[ "${#failures[@]}" -eq 0 ]
REMOTE
}

inter_node_check() {
    local source_node="$1"
    local target_node="$2"
    ssh "${ssh_options[@]}" "$source_node" bash -s -- \
        "$target_node" "${MPI_SSH_IDENTITY_FILE:-__DEFAULT_SSH_KEY__}" <<'REMOTE'
set -u
target_node="$1"
identity_file="$2"
ssh_args=(
    -o BatchMode=yes
    -o ConnectTimeout=8
    -o UserKnownHostsFile=/dev/null
    -o StrictHostKeyChecking=no
)
if [ "$identity_file" != "__DEFAULT_SSH_KEY__" ]; then
    ssh_args+=(-i "$identity_file")
fi
ssh "${ssh_args[@]}" "$target_node" true
REMOTE
}

echo "NCCL preflight | Nodes: $NP | Interface: $MGMT_IFNAME | Collective: $COLLECTIVE"
if [ -n "$TEST_BINARY" ]; then
    echo "Test binary override: $TEST_BINARY"
fi

failed=0
for node in "${NODES[@]}"; do
    printf '%s\t' "$node"
    set +e
    result="$(remote_check "$node" 2>&1)"
    status="$?"
    set -e
    printf '%s\n' "$result"
    if [ "$status" -ne 0 ]; then
        failed=1
    fi
done

for target_node in "${NODES[@]:1}"; do
    printf '%s -> %s\t' "${NODES[0]}" "$target_node"
    set +e
    inter_node_result="$(inter_node_check "${NODES[0]}" "$target_node" 2>&1)"
    inter_node_status="$?"
    set -e
    if [ "$inter_node_status" -eq 0 ]; then
        echo "inter-node SSH=pass"
    else
        printf 'inter-node SSH=failed: %s\n' "$inter_node_result"
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    echo "Preflight failed; no NCCL workload was started." >&2
    exit 1
fi
echo "Preflight passed; no NCCL workload was started."
