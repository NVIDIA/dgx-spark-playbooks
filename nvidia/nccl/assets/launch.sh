#!/bin/bash
# Run the NCCL all_gather performance test across N DGX Spark nodes.
#
# Usage: ./launch.sh --topology direct|ring|switch <NODE_1_IP> <NODE_2_IP> [<NODE_3_IP> ...]
#   Pass the MANAGEMENT IP of every node (the address you SSH to), Node 1 first.
#   -np and the mpirun host list are built from the number of IPs you pass.
#
#   --topology  direct (2 nodes, cable), ring (3 nodes), or switch (4 nodes).
#               ring additionally sets NCCL_IB_SUBNET_AWARE_ROUTING=1 and
#               NCCL_NET_PLUGIN=none; direct and switch add nothing extra.
#
# NCCL bootstrap runs over the management interface (enP7s7); the collective
# data auto-routes over the CX-7 RoCE ports (NCCL discovers them — no need to
# name them). Override the management interface with MGMT_IFNAME=<iface>.
# Override the buffer sweep with BEGIN/END/FACTOR (default 16G/16G/2).
#
# This mirrors Steps 4-5 of the manual playbook. Run it from Node 1.
set -e

MGMT_IFNAME="${MGMT_IFNAME:-enP7s7}"

# Parse --topology (accepts "--topology ring" or "--topology=ring").
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
        -*)
            echo "Error: unknown option '$1'." >&2
            echo "Usage: $0 --topology direct|ring|switch <NODE_1_IP> <NODE_2_IP> [<NODE_3_IP> ...]" >&2
            exit 1 ;;
        *) break ;;
    esac
done

case "$TOPOLOGY" in
    direct|ring|switch) ;;
    *)
        echo "Usage: $0 --topology direct|ring|switch <NODE_1_IP> <NODE_2_IP> [<NODE_3_IP> ...]" >&2
        echo "Pass one management IP per node (Node 1 first)." >&2
        exit 1 ;;
esac

NODES=("$@")
NP="${#NODES[@]}"

# Validate the node count for the chosen topology: direct = exactly 2, ring =
# exactly 3 (only 3-node rings are officially supported), switch = 2 or more.
case "$TOPOLOGY" in
    direct)
        [ "$NP" -eq 2 ] || { echo "Error: --topology direct requires exactly 2 nodes (got $NP)." >&2; exit 1; } ;;
    ring)
        [ "$NP" -eq 3 ] || { echo "Error: --topology ring requires exactly 3 nodes (got $NP)." >&2; exit 1; } ;;
    switch)
        [ "$NP" -ge 2 ] || { echo "Error: --topology switch requires at least 2 nodes (got $NP)." >&2; exit 1; } ;;
esac

# Build the mpirun host list: one <IP>:1 per node.
HOSTLIST=""
for ip in "${NODES[@]}"; do
    HOSTLIST="${HOSTLIST:+$HOSTLIST,}${ip}:1"
done

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MPI_HOME="${MPI_HOME:-/usr/lib/aarch64-linux-gnu/openmpi}"
export NCCL_HOME="${NCCL_HOME:-$HOME/nccl/build/}"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"

# Ring-only extra env (subnet-aware RoCE routing + disable external net plugin).
ring_env=""
if [ "$TOPOLOGY" = "ring" ]; then
    ring_env="-x NCCL_IB_SUBNET_AWARE_ROUTING=1 -x NCCL_NET_PLUGIN=none"
fi

BEGIN="${BEGIN:-16G}"
END="${END:-16G}"
FACTOR="${FACTOR:-2}"

echo "Topology: $TOPOLOGY | Nodes ($NP): ${NODES[*]} | Mgmt iface: $MGMT_IFNAME"
echo "=== Running all_gather_perf across $NP nodes ==="
mpirun -np "$NP" -H "$HOSTLIST" \
    --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
    -x LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    -x UCX_NET_DEVICES="$MGMT_IFNAME" \
    -x NCCL_SOCKET_IFNAME="$MGMT_IFNAME" \
    -x OMPI_MCA_btl_tcp_if_include="$MGMT_IFNAME" \
    $ring_env \
    "$HOME/nccl-tests/build/all_gather_perf" -b "$BEGIN" -e "$END" -f "$FACTOR"
