#!/usr/bin/env bash
# Generate a sandbox-policy.yaml with the correct Docker bridge IP for this machine.
# Usage: bash scripts/gen_sandbox_policy.sh [output_path]
#
# Auto-detects the docker0 bridge IP. Override with DOCKER_BRIDGE_IP env var.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$REPO_DIR/sandbox-policy.yaml"
OUTPUT="${1:-$REPO_DIR/sandbox-policy-local.yaml}"

if [ -n "${DOCKER_BRIDGE_IP:-}" ]; then
    BRIDGE_IP="$DOCKER_BRIDGE_IP"
else
    BRIDGE_IP=$(ip -4 addr show docker0 2>/dev/null | grep -oP 'inet \K[\d.]+' || true)
    if [ -z "$BRIDGE_IP" ]; then
        echo "ERROR: Could not detect docker0 IP. Set DOCKER_BRIDGE_IP manually." >&2
        exit 1
    fi
fi

echo "Docker bridge IP: $BRIDGE_IP"

sed -e "s|__DOCKER_BRIDGE_IP__|$BRIDGE_IP|g" \
    "$TEMPLATE" > "$OUTPUT"

echo "Policy written to: $OUTPUT"
