#!/bin/bash
# Run an NCCL collective across N nodes on multi-node capable hardware.
#
# Usage: ./launch.sh --topology direct|ring|switch [options] <NODE_1_IP> <NODE_2_IP> [<NODE_3_IP> ...]
#   Pass the MANAGEMENT IP of every node (the address you SSH to), Node 1 first.
#   -np and the mpirun host list are built from the number of IPs you pass.
#
#   --topology  direct (2 nodes), ring (3 nodes), or switch (2-8 nodes).
#               ring additionally sets NCCL_IB_SUBNET_AWARE_ROUTING=1 and
#               NCCL_NET_PLUGIN=none; direct and switch add nothing extra.
#   --collective all_gather|all_reduce (default: all_gather)
#   --interface <iface>       management/bootstrap interface
#   --timeout <seconds>       wall-clock limit (default: 300)
#   --dry-run                 print the resolved command without executing it
#
# MPI launches remote ranks from Node 1. Passwordless SSH must therefore work
# from Node 1 to every other node, not only from the operator's workstation.
# Set MPI_SSH_IDENTITY_FILE when that inter-node key is not a default SSH key.
#
# NCCL bootstrap runs over the management interface (default enP7s7 on validated
# multi-node capable hardware); collective data auto-routes over the RoCE ports
# (NCCL discovers them — no need to name them). Override the management
# interface with --interface <iface> or MGMT_IFNAME=<iface>. Override the buffer
# sweep with BEGIN/END/FACTOR (default 8M/64M/2).
#
# This mirrors Steps 4-5 of the manual playbook. Run it from Node 1.
set -e

MGMT_IFNAME="${MGMT_IFNAME:-enP7s7}"
COLLECTIVE="all_gather"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
DRY_RUN=false
MPI_SSH_IDENTITY_FILE="${MPI_SSH_IDENTITY_FILE:-}"

# Parse options (long options accept either "--option value" or
# "--option=value").
TOPOLOGY=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --topology)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "Error: --topology requires a value (direct|ring|switch)." >&2
                exit 1
            fi
            TOPOLOGY="$2"; shift 2 ;;
        --topology=*) TOPOLOGY="${1#*=}"; shift ;;
        --collective)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "Error: --collective requires a value (all_gather|all_reduce)." >&2
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
        --timeout)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "Error: --timeout requires a positive integer." >&2
                exit 1
            fi
            TIMEOUT_SECONDS="$2"; shift 2 ;;
        --timeout=*) TIMEOUT_SECONDS="${1#*=}"; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -*)
            echo "Error: unknown option '$1'." >&2
            echo "Usage: $0 --topology direct|ring|switch [options] <NODE_1_IP> <NODE_2_IP> [<NODE_3_IP> ...]" >&2
            exit 1 ;;
        *) break ;;
    esac
done

case "$TOPOLOGY" in
    direct|ring|switch) ;;
    *)
        echo "Usage: $0 --topology direct|ring|switch [options] <NODE_1_IP> <NODE_2_IP> [<NODE_3_IP> ...]" >&2
        echo "Pass one management IP per node (Node 1 first)." >&2
        exit 1 ;;
esac

case "$COLLECTIVE" in
    all_gather|all_reduce) ;;
    *)
        echo "Error: --collective must be all_gather or all_reduce (got '$COLLECTIVE')." >&2
        exit 1 ;;
esac

if [ -z "$MGMT_IFNAME" ] || [[ ! "$MGMT_IFNAME" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
    echo "Error: interface must be a non-empty Linux interface name." >&2
    exit 1
fi

if [[ ! "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: timeout must be a positive integer (got '$TIMEOUT_SECONDS')." >&2
    exit 1
fi

if [ -n "$MPI_SSH_IDENTITY_FILE" ] && [[ ! "$MPI_SSH_IDENTITY_FILE" =~ ^[A-Za-z0-9_./:-]+$ ]]; then
    echo "Error: MPI_SSH_IDENTITY_FILE contains unsupported characters." >&2
    exit 1
fi

if [ "$DRY_RUN" = false ]; then
    command -v mpirun >/dev/null || { echo "Error: mpirun not found." >&2; exit 1; }
    command -v timeout >/dev/null || { echo "Error: timeout not found; install coreutils or use a bounded runner." >&2; exit 1; }
fi

NODES=("$@")
NP="${#NODES[@]}"

# Validate the node count for the chosen topology: direct = exactly 2, ring =
# exactly 3 (only 3-node rings are officially supported), switch = 2-8.
case "$TOPOLOGY" in
    direct)
        [ "$NP" -eq 2 ] || { echo "Error: --topology direct requires exactly 2 nodes (got $NP)." >&2; exit 1; } ;;
    ring)
        [ "$NP" -eq 3 ] || { echo "Error: --topology ring requires exactly 3 nodes (got $NP)." >&2; exit 1; } ;;
    switch)
        [ "$NP" -ge 2 ] && [ "$NP" -le 8 ] || { echo "Error: --topology switch requires 2-8 nodes (got $NP)." >&2; exit 1; } ;;
esac

# Validate addresses before constructing the host list. Duplicate addresses can
# silently produce the requested rank count with fewer physical nodes.
seen_nodes="|"
for ip in "${NODES[@]}"; do
    if [ -z "$ip" ] || [[ ! "$ip" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
        echo "Error: invalid node address '$ip'." >&2
        exit 1
    fi
    if [[ "$seen_nodes" == *"|$ip|"* ]]; then
        echo "Error: duplicate node address '$ip'." >&2
        exit 1
    fi
    seen_nodes="${seen_nodes}${ip}|"
done

# Build the mpirun host list: one <address>:1 per node.
HOSTLIST=""
for ip in "${NODES[@]}"; do
    HOSTLIST="${HOSTLIST:+$HOSTLIST,}${ip}:1"
done

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MPI_HOME="${MPI_HOME:-/usr/lib/aarch64-linux-gnu/openmpi}"
export NCCL_HOME="${NCCL_HOME:-$HOME/nccl/build/}"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
# PRRTE otherwise advertises every local address. On multihomed DGX Spark
# hosts that can select an unreachable management address for the control
# connection, so constrain the MPI control plane to the selected interface.
export PRTE_MCA_oob_tcp_if_include="$MGMT_IFNAME"
export OMPI_MCA_oob_tcp_if_include="$MGMT_IFNAME"

# Ring-only extra env (subnet-aware RoCE routing + disable external net plugin).
ring_env=()
if [ "$TOPOLOGY" = "ring" ]; then
    ring_env=(-x NCCL_IB_SUBNET_AWARE_ROUTING=1 -x NCCL_NET_PLUGIN=none)
fi

BEGIN="${BEGIN:-8M}"
END="${END:-64M}"
FACTOR="${FACTOR:-2}"
NCCL_TEST_BIN="${NCCL_TEST_BIN:-$HOME/nccl-tests/build/${COLLECTIVE}_perf}"

if [ "$DRY_RUN" = false ] && [ ! -x "$NCCL_TEST_BIN" ]; then
    echo "Error: NCCL test binary is not executable: $NCCL_TEST_BIN" >&2
    exit 1
fi

mpirun_cmd=(
    mpirun -np "$NP" -H "$HOSTLIST"
    --mca plm_rsh_agent "ssh -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no${MPI_SSH_IDENTITY_FILE:+ -i $MPI_SSH_IDENTITY_FILE}"
    --prtemca plm_rsh_no_tree_spawn 1
    --prtemca oob_tcp_if_include "$MGMT_IFNAME"
    -x "PATH=$PATH"
    -x "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    -x "PRTE_MCA_oob_tcp_if_include=$MGMT_IFNAME"
    -x "OMPI_MCA_oob_tcp_if_include=$MGMT_IFNAME"
    -x "UCX_NET_DEVICES=$MGMT_IFNAME"
    -x "NCCL_SOCKET_IFNAME=$MGMT_IFNAME"
    -x "OMPI_MCA_btl_tcp_if_include=$MGMT_IFNAME"
    -x "NCCL_DEBUG=${NCCL_DEBUG:-INFO}"
    -x "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET}"
)
mpirun_cmd+=("${ring_env[@]}")
if [ -n "${NCCL_IB_HCA:-}" ]; then
    mpirun_cmd+=(-x "NCCL_IB_HCA=$NCCL_IB_HCA")
fi
mpirun_cmd+=("$NCCL_TEST_BIN" -b "$BEGIN" -e "$END" -f "$FACTOR")

echo "Topology: $TOPOLOGY | Nodes ($NP): ${NODES[*]} | Mgmt iface: $MGMT_IFNAME"
echo "Collective: ${COLLECTIVE}_perf | Message sizes: $BEGIN..$END (factor $FACTOR) | Timeout: ${TIMEOUT_SECONDS}s"
echo "Transport evidence: NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET}; inspect for 'Using network IB' or 'Using network Socket'."
printf 'Command:'
printf ' %q' "${mpirun_cmd[@]}"
printf '\n'

if [ "$DRY_RUN" = true ]; then
    echo "Dry run: command not executed."
    exit 0
fi

set +e
timeout --foreground --kill-after=10s "$TIMEOUT_SECONDS" "${mpirun_cmd[@]}"
status=$?
set -e

if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
    echo "Error: NCCL test exceeded the ${TIMEOUT_SECONDS}s timeout; mpirun was terminated." >&2
    exit 124
fi
exit "$status"
