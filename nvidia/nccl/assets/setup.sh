#!/bin/bash
# Build NCCL and the NCCL test suite on every DGX Spark node.
#
# Usage: ./setup.sh [WORKER_IP ...]
#   WORKER_IP ...  (optional) IP/hostname of each additional Spark (Node 2, 3, 4).
#                  The same build is run on each over SSH. Omit to build locally
#                  only. Works for any node count (2, 3, or 4 Sparks).
#
# This mirrors Steps 2-3 of the manual playbook. Run it from the node you use
# as the launcher (Node 1).
set -e

NCCL_VERSION="${NCCL_VERSION:-v2.30.7-1}"
case "$NCCL_VERSION" in
    *[!A-Za-z0-9._-]*)
        echo "Invalid NCCL_VERSION '$NCCL_VERSION' (allowed: letters, digits, . _ -)." >&2
        exit 1 ;;
esac

echo "=== Checking prerequisites ==="
command -v git >/dev/null || { echo "git not found"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
if [ "$#" -gt 0 ]; then
    command -v ssh >/dev/null || { echo "ssh not found (required for worker nodes)"; exit 1; }
fi

build_cmd="set -e; \
if [ -d \$HOME/nccl ]; then echo 'ERROR: ~/nccl already exists on this node. Remove it and retry.' >&2; exit 1; fi; \
if [ -d \$HOME/nccl-tests ]; then echo 'ERROR: ~/nccl-tests already exists on this node. Remove it and retry.' >&2; exit 1; fi; \
sudo -S -p '' sh -c 'apt-get update && apt-get install -y libopenmpi-dev' && \
git clone -b ${NCCL_VERSION} https://github.com/NVIDIA/nccl.git \$HOME/nccl/ && \
cd \$HOME/nccl/ && \
make -j\$(nproc) src.build NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121' && \
git clone https://github.com/NVIDIA/nccl-tests.git \$HOME/nccl-tests/ && \
cd \$HOME/nccl-tests/ && \
export CUDA_HOME='/usr/local/cuda' && \
export MPI_HOME='/usr/lib/aarch64-linux-gnu/openmpi' && \
export NCCL_HOME=\"\$HOME/nccl/build/\" && \
export LD_LIBRARY_PATH=\"\$NCCL_HOME/lib:\$CUDA_HOME/lib64/:\$MPI_HOME/lib:\$LD_LIBRARY_PATH\" && \
make -j\$(nproc) MPI=1"

# Collect a sudo password per node (local first, then each worker IP). Each
# password is piped only to its own node's `sudo -S`, so it never appears on a
# command line or in the build logs. Passwords may differ between nodes.
node_ips=("" "$@")
node_passwords=()
echo "=== Sudo access ==="
echo "Sudo is required to install the libopenmpi-dev build dependency on each node."
for ip in "${node_ips[@]}"; do
    if [ -z "$ip" ]; then
        read -r -s -p "Enter sudo password for this node (local): " pass
    else
        read -r -s -p "Enter sudo password for ${ip}: " pass
    fi
    echo
    node_passwords+=("$pass")
done

echo "=== Building NCCL ${NCCL_VERSION} on all nodes in parallel ==="
echo "(each node builds concurrently; output is streamed live, prefixed per node)"

pids=()
labels=()
statuses=()

# Launch a build in the background, streaming its output live with a per-node
# prefix so progress is visible. The build's real exit code is recorded to a
# status file (the pipeline's own exit code would be sed's, not the build's).
#   $1 = label, $2 = worker IP ("" for the local node), $3 = sudo password
launch_build() {
    local label="$1" ip="$2" pass="$3" status
    status="$(mktemp)"
    # Feed this node's sudo password to its `sudo -S` via stdin (one line).
    (
        if [ -z "$ip" ]; then
            printf '%s\n' "$pass" | sh -c "$build_cmd"
        else
            printf '%s\n' "$pass" | ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 "$USER@$ip" "$build_cmd"
        fi
        echo "$?" > "$status"
    ) 2>&1 | sed -u "s/^/[$label] /" &
    pids+=("$!")
    labels+=("$label")
    statuses+=("$status")
}

for idx in "${!node_ips[@]}"; do
    ip="${node_ips[$idx]}"
    if [ -z "$ip" ]; then
        launch_build "Node 1 (local)" "" "${node_passwords[$idx]}"
    else
        launch_build "Node $((idx + 1)) (${ip})" "$ip" "${node_passwords[$idx]}"
    fi
done

# Wait for every node and report per-node results (exit code from status file).
failed=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    rc="$(cat "${statuses[$i]}" 2>/dev/null || echo 1)"
    if [ "$rc" = "0" ]; then
        echo "=== ${labels[$i]}: build succeeded ==="
    else
        echo "=== ${labels[$i]}: build FAILED (see [${labels[$i]}] output above) ===" >&2
        failed=1
    fi
    rm -f "${statuses[$i]}"
done

if [ "$failed" -ne 0 ]; then
    echo "One or more nodes failed to build. See the output above." >&2
    exit 1
fi

echo "=== Setup complete on all nodes. ==="
echo "To run the communication test (pass one management IP per node):"
echo "  2 Sparks:  bash launch.sh --topology direct <NODE_1_IP> <NODE_2_IP>"
echo "  3 Sparks:  bash launch.sh --topology ring   <NODE_1_IP> <NODE_2_IP> <NODE_3_IP>"
echo "  4 Sparks:  bash launch.sh --topology switch <NODE_1_IP> <NODE_2_IP> <NODE_3_IP> <NODE_4_IP>"
