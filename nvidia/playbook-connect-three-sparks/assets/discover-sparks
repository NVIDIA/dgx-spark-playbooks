#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#!/bin/env bash

# discover-sparks
# Discover available systems using avahi-browse and generate MPI hosts file
# Searches all active interfaces automatically
#
# Usage: bash ./discover-sparks

set -euo pipefail

# Check if running as root or with sudo
if [[ $EUID -eq 0 ]] || [[ -n "${SUDO_USER:-}" ]]; then
    echo "Error: This script should not be run as root or with sudo"
    echo "Please run as a regular user"
    exit 1
fi

# Dynamically get interface names from ibdev2netdev output
# Use ibdev2netdev to list Infiniband devices and their network interfaces.
# The awk command searches for lines containing 'Up)' (i.e., interfaces that are up)
# and prints the 5th field, which is the interface name (e.g., enp1s0f0np0).
# The tr command removes any parentheses from the output.
INTERFACES=($(ibdev2netdev | awk '/Up\)/ {print $5}' | tr -d '()'))
if [ ${#INTERFACES[@]} -eq 0 ]; then
    echo "ERROR: No active interfaces found via ibdev2netdev."
    exit 1
fi

# Create temporary file for processing
TEMP_FILE=$(mktemp)
trap 'rm -f "$TEMP_FILE"' EXIT

# Check if avahi-browse is available
if ! command -v avahi-browse &> /dev/null; then
    echo "Error: avahi-browse not found. Please install avahi-utils package."
    exit 1
fi

# Run avahi-browse and filter for SSH services on specified interfaces
# -p: parseable output
# -r: resolve host names and addresses
# -f: terminate after dumping all entries available at startup
avahi_output=$(avahi-browse -p -r -f -t _ssh._tcp 2>/dev/null)

# Filter for both interfaces
found_services=false
for interface in "${INTERFACES[@]}"; do
    if echo "$avahi_output" | grep "$interface" >> "$TEMP_FILE"; then
        found_services=true
    fi
done

if [ "$found_services" = false ]; then
    echo "Warning: No services found on any specified interface"
    exit 0
fi

# Extract IPv4 addresses from the avahi-browse output
# Format: =;interface;IPv4;hostname\032service;description;local;fqdn;ip_address;port;
grep "^=" "$TEMP_FILE" | grep "IPv4" | while IFS=';' read -r prefix interface protocol hostname_service description local fqdn ip_address port rest; do
    # Clean up any trailing data
    clean_ip=$(echo "$ip_address" | sed 's/;.*$//')

    # Validate IP address format
    if [[ $clean_ip =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        echo "$clean_ip" >> "$TEMP_FILE.sorted"
        echo "Found: $clean_ip ($fqdn)"
    else
        echo "Warning: Invalid IP format: $clean_ip"
    fi
done

# Sort and remove duplicates
if [[ -s "$TEMP_FILE.sorted" ]]; then
    sort -u "$TEMP_FILE.sorted" -o "$TEMP_FILE.sorted"
else
    echo "No IPv4 addresses found."
    exit 1
fi

# Generate a shared SSH key if it doesn't exist
SHARED_KEY="$HOME/.ssh/id_ed25519_shared"
if [[ ! -f "$SHARED_KEY" ]]; then
    echo "Generating shared SSH key for all nodes..."
    ssh-keygen -t ed25519 -N "" -f "$SHARED_KEY" -q -C "shared-cluster-key"
fi

echo ""
echo "Setting up shared SSH access across all nodes..."
echo "You may be prompted for your password on each node."

# Ensure local .ssh directory exists with correct permissions
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

# Add shared public key to local authorized_keys
if ! grep -qF "$(cat "$SHARED_KEY.pub")" "$HOME/.ssh/authorized_keys" 2>/dev/null; then
    cat "$SHARED_KEY.pub" >> "$HOME/.ssh/authorized_keys"
    chmod 600 "$HOME/.ssh/authorized_keys"
    echo "  ✓ Added shared public key to local authorized_keys"
fi

# Distribute shared key to all remote nodes
while read -r node_ip; do
    if [[ -n "$node_ip" ]]; then
        echo "Configuring $node_ip..."

        # Copy shared key to remote node and set up authorized_keys
        if scp -o StrictHostKeyChecking=accept-new "$SHARED_KEY" "$SHARED_KEY.pub" "$USER@$node_ip:~/.ssh/" &>/dev/null; then
            ssh -n -o StrictHostKeyChecking=accept-new "$USER@$node_ip" "
                chmod 700 ~/.ssh
                chmod 600 ~/.ssh/id_ed25519_shared
                chmod 644 ~/.ssh/id_ed25519_shared.pub

                # Add shared public key to authorized_keys if not present
                if ! grep -qF \"\$(cat ~/.ssh/id_ed25519_shared.pub)\" ~/.ssh/authorized_keys 2>/dev/null; then
                    cat ~/.ssh/id_ed25519_shared.pub >> ~/.ssh/authorized_keys
                    chmod 600 ~/.ssh/authorized_keys
                fi

                # Create/update SSH config to use shared key by default
                if ! grep -q 'IdentityFile.*id_ed25519_shared' ~/.ssh/config 2>/dev/null; then
                    echo 'Host *' >> ~/.ssh/config
                    echo '    IdentityFile ~/.ssh/id_ed25519_shared' >> ~/.ssh/config
                    chmod 600 ~/.ssh/config
                fi
            " &>/dev/null

            echo "  ✓ Successfully configured $node_ip with shared key"
        else
            echo "  ✗ Failed to configure $node_ip"
        fi
    fi
done < "$TEMP_FILE.sorted"

# Update local SSH config to use shared key
if ! grep -q 'IdentityFile.*id_ed25519_shared' "$HOME/.ssh/config" 2>/dev/null; then
    touch "$HOME/.ssh/config"
    echo 'Host *' >> "$HOME/.ssh/config"
    echo '    IdentityFile ~/.ssh/id_ed25519_shared' >> "$HOME/.ssh/config"
    chmod 600 "$HOME/.ssh/config"
    echo "  ✓ Updated local SSH config to use shared key"
fi

echo ""
echo "Shared SSH setup complete!"
echo "All nodes can now SSH to each other using the shared key (id_ed25519_shared)."
