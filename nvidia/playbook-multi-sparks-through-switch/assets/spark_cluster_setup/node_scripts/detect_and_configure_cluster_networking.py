#!/usr/bin/env python3

#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import json
import socket
import struct
import threading
import time
import os
from collections import defaultdict

# Interfaces used for DISCOVERY (L2 broadcasts)
DISCOVERY_INTERFACES = ["enp1s0f0np0", "enp1s0f1np1"]

# Map discovery interfaces -> interfaces where we actually want to configure IPs
# Order matters: index 0 = lower host, index 1 = higher host (per node, per link)
DISCOVERY_TO_CONFIG_IFACES = {
    "enp1s0f0np0": ["enp1s0f0np0", "enP2p1s0f0np0"],
    "enp1s0f1np1": ["enp1s0f1np1", "enP2p1s0f1np1"],
}

# For switch mode, map discovery iface -> stable link index
SWITCH_IFACE_INDEX = {
    "enp1s0f0np0": 0,
    "enp1s0f1np1": 1,
}

# All interfaces that should appear in generated netplan (discovery + config)
ALL_INTERFACES = sorted(
    set(DISCOVERY_INTERFACES) | {iface for lst in DISCOVERY_TO_CONFIG_IFACES.values() for iface in lst}
)

ETHERTYPE = 0x88B5  # custom EtherType

# Discovery payload magic
DISCOVERY_MAGIC = b"TOPO_DISCOVER_2NODE_V2"
LISTEN_SECONDS = 20.0
SEND_INTERVAL = 0.5

# Primary mode settings
DEFAULT_PRIMARY_PORT = 9999
REPORT_TIMEOUT = 30  # seconds to wait for all nodes to report


def get_mac(iface: str) -> str:
    """Return MAC address string (lowercase) for the given interface."""
    path = f"/sys/class/net/{iface}/address"
    with open(path, "r") as f:
        return f.read().strip().lower()


def mac_str_to_bytes(mac_str: str) -> bytes:
    return bytes.fromhex(mac_str.replace(":", ""))


def mac_bytes_to_str(mac: bytes) -> str:
    return ":".join(f"{b:02x}" for b in mac)


def iface_link_up(iface: str) -> bool:
    """Check carrier == 1."""
    carrier_path = f"/sys/class/net/{iface}/carrier"
    try:
        with open(carrier_path, "r") as f:
            return f.read().strip() == "1"
    except FileNotFoundError:
        return False
    except Exception:
        return False


def create_socket(iface: str) -> socket.socket:
    """Create and bind raw AF_PACKET socket on given interface."""
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETHERTYPE))
    sock.bind((iface, 0))
    sock.settimeout(0.5)
    return sock


def get_local_machine_id() -> str:
    """
    Machine ID = lowest MAC across all DISCOVERY_INTERFACES that exist.
    This is used as a stable, machine-wide identifier.
    """
    macs = []
    for iface in DISCOVERY_INTERFACES:
        path = f"/sys/class/net/{iface}/address"
        if not os.path.exists(path):
            continue
        try:
            macs.append(get_mac(iface))
        except Exception:
            continue

    if not macs:
        raise RuntimeError("No MACs found on configured DISCOVERY_INTERFACES")

    return sorted(macs)[0]  # lowest MAC string


def sender_thread(iface: str, stop_event: threading.Event, machine_id_str: str):
    """
    Periodically send discovery frames on the given interface.

    Frame layout:
      dst_mac      (6)  = ff:ff:ff:ff:ff:ff
      src_mac      (6)  = this interface MAC
      ethertype    (2)  = 0x88B5
      payload:
        machine_id (6)  = lowest MAC of this machine
        magic      (N)  = DISCOVERY_MAGIC (padded by NIC/kernel)
    """
    try:
        src_mac_str = get_mac(iface)
        src_mac = mac_str_to_bytes(src_mac_str)
        machine_id_bytes = mac_str_to_bytes(machine_id_str)
        sock = create_socket(iface)
    except Exception as e:
        print(f"[{iface}] Sender error: {e}")
        return

    dst_mac = b"\xff\xff\xff\xff\xff\xff"
    ethertype_bytes = struct.pack("!H", ETHERTYPE)

    payload = machine_id_bytes + DISCOVERY_MAGIC
    frame = dst_mac + src_mac + ethertype_bytes + payload

    print(f"[{iface}] Sender started (local MAC {src_mac_str}, machine_id {machine_id_str})")
    try:
        while not stop_event.is_set():
            try:
                sock.send(frame)
            except Exception as e:
                print(f"[{iface}] Send error: {e}")
                break
            time.sleep(SEND_INTERVAL)
    finally:
        sock.close()
        print(f"[{iface}] Sender stopped")


def listener_thread(iface: str,
                    stop_event: threading.Event,
                    local_mac: str,
                    local_machine_id: str,
                    neighbors_info: dict):
    """
    Listen for discovery frames and record neighbor machine and NIC info.

    neighbors_info[iface] will contain entries:
      (neighbor_machine_id_str, neighbor_nic_mac_str)
    """
    try:
        sock = create_socket(iface)
    except Exception as e:
        print(f"[{iface}] Listener error: {e}")
        return

    local_mac_bytes = mac_str_to_bytes(local_mac)

    print(f"[{iface}] Listener started (local MAC {local_mac}, machine_id {local_machine_id})")
    try:
        while not stop_event.is_set():
            try:
                frame, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[{iface}] Recv error: {e}")
                break

            if len(frame) < 14 + 6 + len(DISCOVERY_MAGIC):
                continue

            src_mac = frame[6:12]
            ethertype = struct.unpack("!H", frame[12:14])[0]
            payload = frame[14:]

            if ethertype != ETHERTYPE:
                continue

            # payload: machine_id(6) + magic
            machine_id_bytes = payload[:6]
            magic = payload[6:]

            if not magic.startswith(DISCOVERY_MAGIC):
                continue

            src_mac_str = mac_bytes_to_str(src_mac)
            machine_id_str = mac_bytes_to_str(machine_id_bytes)

            # Ignore our own frames (by src NIC MAC)
            if src_mac == local_mac_bytes:
                print(f"[{iface}] Saw our own DISCOVER from {src_mac_str}, ignoring")
                continue

            print(f"[{iface}] Saw DISCOVER from neighbor NIC {src_mac_str}, "
                  f"neighbor machine_id {machine_id_str}")
            neighbors_info[iface].add((machine_id_str, src_mac_str))
    finally:
        sock.close()
        print(f"[{iface}] Listener stopped")


def build_netplan_yaml(iface_to_ip: dict) -> str:
    """
    Return the netplan YAML as a string.

    - Includes ALL_INTERFACES.
    - Sets dhcp4: false on all of them.
    - Adds addresses only for those present in iface_to_ip.
    """
    lines = [
        "network:",
        "  version: 2",
        "  ethernets:",
    ]
    for iface in ALL_INTERFACES:
        lines.append(f"    {iface}:")
        lines.append("      dhcp4: false")
        if iface in iface_to_ip:
            ip_cidr = iface_to_ip[iface]
            lines.append("      addresses:")
            lines.append(f"        - {ip_cidr}")
    lines.append("")
    return "\n".join(lines)


def mac_str_to_int(mac_str: str) -> int:
    return int(mac_str.replace(":", ""), 16)


# ---------- IP assignment helpers ----------

def ip_for_2node_link(link_index: int, node_id: int, local_index_in_pair: int) -> str:
    """
    /24 scheme with 4 hosts per link (2 per node).

    For each link_index:
      network = 192.168.link_index.0/24
      hosts .1 .. .4 used for the two nodes (2 endpoints each).

    Node 1:
      local_index_in_pair = 0 -> .1
      local_index_in_pair = 1 -> .2

    Node 2:
      local_index_in_pair = 0 -> .3
      local_index_in_pair = 1 -> .4
    """
    host = 1 + (0 if node_id == 1 else 2) + local_index_in_pair
    return f"192.168.{link_index}.{host}/24"

def ip_for_3node_ring_link(link_index: int, node_id: int, local_index_in_pair: int) -> str:
    """
    /24 scheme for 3-node ring topology.

    For each node_id:
      network = 192.168.third_octet.node_id/24
      third_octet = link_index * 2 + local_index_in_pair

    Node 1:
      192.168.[0, 1].1/24 -> Node 2
      192.168.[2, 3].1/24 -> Node 3

    Node 2:
      192.168.[4, 5].1/24 -> Node 3
      192.168.[0, 1].2/24 -> Node 1

    Node 3:
      192.168.[2, 3].2/24 -> Node 1
      192.168.[4, 5].2/24 -> Node 2
    """
    return f"192.168.{link_index * 2 + local_index_in_pair}.{node_id}/24"

def ip_for_switch_link(link_index: int, node_index: int, local_index_in_pair: int) -> str:
    """
    /24 scheme for N-node switch topology.

    For each link_index:
      network = 192.168.link_index.0/24
      host = 10 + node_index * 2 + local_index_in_pair

    node_index is 0-based index in sorted cluster_machine_ids.
    local_index_in_pair is 0 for discovery iface, 1 for paired iface.
    """
    base_octet3 = link_index  # 192.168.<link_index>.X
    host = 10 + node_index * 2 + local_index_in_pair
    return f"192.168.{base_octet3}.{host}/24"

# ---------- Main topology logic ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Network topology discovery script. Run on all nodes to discover neighbors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run normal discovery on a worker node:
  sudo python3 detect_all.py

  # Run as primary to collect topology from all nodes and print diagram:
  sudo python3 detect_all.py --primary

  # Run on worker node and report to primary at specific address:
  sudo python3 detect_all.py --report-to 10.0.0.1:9999

  # Primary with custom port: 
  sudo python3 detect_all.py --primary --port 8888
"""
    )
    parser.add_argument(
        "--primary",
        action="store_true",
        help="Run as primary node: collect reports from all nodes and print topology diagram"
    )
    parser.add_argument(
        "--report-to",
        type=str,
        metavar="HOST:PORT",
        help="Report topology to primary node at HOST:PORT after discovery"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PRIMARY_PORT,
        help=f"Port for primary to listen on (default: {DEFAULT_PRIMARY_PORT})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=REPORT_TIMEOUT,
        help=f"Seconds to wait for node reports in primary mode (default: {REPORT_TIMEOUT})"
    )
    parser.add_argument(
        "--apply-netplan-yaml",
        action="store_true",
        default=False,
        help="Apply netplan YAML"
    )
    return parser.parse_args()

def apply_netplan_yaml(netplan_yaml) -> bool:
    netplan_path = "/etc/netplan/40-cx7.yaml"
    try:
        print(f"Applying netplan YAML")

        with open(netplan_path, "w") as f:
            f.write(netplan_yaml)

        os.chmod(netplan_path, 0o600)

        ret = os.system("netplan apply")
        if ret != 0:
            raise Exception(f"netplan apply failed: {ret}")

    except Exception as e:
        print(f"Failed to apply netplan YAML on node: {e}")
        return False

    return True

def main() -> bool:
    args = parse_args()
    
    if os.geteuid() != 0:
        print("ERROR: This script must be run as root (raw sockets).")
        return False

    # Determine which discovery interfaces are physically up
    active_ifaces = []
    for iface in DISCOVERY_INTERFACES:
        if not os.path.exists(f"/sys/class/net/{iface}/address"):
            print(f"[{iface}] Interface does not exist, skipping")
            continue
        if iface_link_up(iface):
            active_ifaces.append(iface)
        else:
            print(f"[{iface}] Link down, skipping")

    if not active_ifaces:
        print("No active discovery interfaces among:", DISCOVERY_INTERFACES)
        return False

    print(f"Active discovery interfaces: {active_ifaces}")

    # Compute local machine_id (lowest MAC across all discovery interfaces)
    try:
        local_machine_id = get_local_machine_id()
    except Exception as e:
        print(f"Could not compute local machine_id: {e}")
        return False

    print(f"Local MACHINE_ID (lowest MAC): {local_machine_id}")

    # Only support up to 2 active ports for now
    if len(active_ifaces) > 2:
        print("More than 2 active discovery interfaces detected; this script currently "
              "supports only up to 2 active ports.")
        return False

    # Pre-read local MACs for active discovery interfaces
    local_macs = {}
    for iface in active_ifaces:
        try:
            local_macs[iface] = get_mac(iface)
        except Exception as e:
            print(f"[{iface}] Cannot get MAC, skipping: {e}")

    stop_event = threading.Event()
    neighbors_info = defaultdict(set)  # iface -> set of (machine_id_str, nic_mac_str)

    threads = []

    # Start listeners first
    for iface in active_ifaces:
        t = threading.Thread(
            target=listener_thread,
            args=(iface, stop_event, local_macs[iface], local_machine_id, neighbors_info),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Start senders
    for iface in active_ifaces:
        t = threading.Thread(
            target=sender_thread,
            args=(iface, stop_event, local_machine_id),
            daemon=True,
        )
        t.start()
        threads.append(t)

    print(f"\nListening and broadcasting for {LISTEN_SECONDS} seconds...")
    time.sleep(LISTEN_SECONDS)
    stop_event.set()

    for t in threads:
        t.join(timeout=1.0)

    # Summarize neighbors
    print("\nNeighbor summary per discovery interface:")
    all_neighbor_machines = set()
    all_neighbor_nics = set()
    machines_per_iface = {}

    for iface in active_ifaces:
        entries = neighbors_info[iface]
        machines_here = {m for (m, n) in entries}
        nics_here = {n for (m, n) in entries}
        machines_per_iface[iface] = machines_here
        all_neighbor_machines |= machines_here
        all_neighbor_nics |= nics_here
        print(f"  {iface}:")
        print(f"    Neighbor MACHINE_IDs: {machines_here}")
        print(f"    Neighbor NIC MACs:    {nics_here}")

    print(f"\nAll neighbor MACHINE_IDs (excluding self): {all_neighbor_machines}")
    print(f"All neighbor NIC MACs:                    {all_neighbor_nics}")

    # Include self in the cluster view
    cluster_machine_ids = sorted(all_neighbor_machines | {local_machine_id})
    num_machines = len(cluster_machine_ids)
    print(f"\nCluster MACHINE_IDs (including self): {cluster_machine_ids}")
    print(f"Detected {num_machines} machines total in this broadcast domain")

    if num_machines == 1:
        print("No neighbors detected on any active discovery interface. Aborting.")
        return False

    active_ifaces_with_neighbor = [
        iface for iface in active_ifaces if neighbors_info[iface]
    ]
    if not active_ifaces_with_neighbor:
        print("Active discovery interfaces exist but no neighbor traffic seen. Aborting.")
        return False

    # Point-to-point pattern: each iface sees exactly one neighbor machine
    p2p_per_iface = all(
        len(machines_per_iface[iface]) == 1
        for iface in active_ifaces_with_neighbor
    )
    # Union of neighbor machines across ifaces
    union_neighbors = set()
    for iface in active_ifaces_with_neighbor:
        union_neighbors |= machines_per_iface[iface]

    # ---- Topology classification ----
    mode = None

    if num_machines == 2:
        mode = "2node"
        topo_desc = "2-node (direct or dual-link)"
    elif num_machines == 3:
        if p2p_per_iface and len(union_neighbors) == 2:
            mode = "ring3"
            topo_desc = "3-node ring topology"
        else:
            mode = "switch"
            topo_desc = "3-node switch-like topology"
    else:  # num_machines >= 4
        if p2p_per_iface and len(union_neighbors) == 2:
            print("\nDetected a ring/line-style point-to-point topology with 4 or more machines.")
            print("This configuration is NOT supported by this script yet. Aborting.")
            return False
        else:
            mode = "switch"
            topo_desc = f"{num_machines}-node switch-like topology"

    print(f"\nTopology classification (from this node's perspective): {topo_desc}")

    # ---- Role within cluster ----
    if mode == "2node":
        # Exactly one other machine_id
        other_id = [m for m in cluster_machine_ids if m != local_machine_id][0]
        local_id_int = mac_str_to_int(local_machine_id)
        other_id_int = mac_str_to_int(other_id)
        node_id = 1 if local_id_int < other_id_int else 2
        print(f"\n2-node mode: this node is Node {node_id}")
    elif mode in ("switch", "ring3"):
        node_index = cluster_machine_ids.index(local_machine_id)
        print(f"\nCluster index: this node has node_index={node_index} "
              f"in cluster_machine_ids")

    # ---- Build link entries for p2p-style modes (2node / ring3) ----
    link_entries = []  # list of (link_id_tuple, discovery_iface, neighbor_machine_id)

    if mode in ("2node", "ring3"):
        for iface in active_ifaces_with_neighbor:
            entries = list(neighbors_info[iface])
            if not entries:
                continue
            # pick the first neighbor for link-id purposes
            neighbor_machine, neighbor_nic_mac = entries[0]
            local_nic_mac = local_macs[iface]
            link_id = tuple(sorted([local_nic_mac, neighbor_nic_mac]))
            link_entries.append((link_id, iface, neighbor_machine))

        # Sort links by link_id to make ordering deterministic (for 2-node case)
        link_entries.sort(key=lambda x: x[0])

    # ---- IP assignment ----
    iface_to_ip = {}

    if mode == "switch":
        # Each discovery interface maps to a fixed subnet (based on its name)
        node_index = cluster_machine_ids.index(local_machine_id)
        for discover_iface in active_ifaces_with_neighbor:
            config_ifaces = DISCOVERY_TO_CONFIG_IFACES.get(discover_iface, [])
            if not config_ifaces:
                print(f"[{discover_iface}] No mapped config interfaces; skipping IP assignment for this link")
                continue
            link_index = SWITCH_IFACE_INDEX.get(discover_iface, 0)
            for local_idx, cfg_iface in enumerate(config_ifaces):
                ip_cidr = ip_for_switch_link(link_index, node_index, local_idx)
                iface_to_ip[cfg_iface] = ip_cidr
            print(
                f"Switch link for iface {discover_iface}: link_index={link_index}, "
                f"config_ifaces={config_ifaces}"
            )

    elif mode == "2node":
        # Same node_id across all links
        for link_index, (link_id, discover_iface, neighbor_machine) in enumerate(link_entries):
            config_ifaces = DISCOVERY_TO_CONFIG_IFACES.get(discover_iface, [])
            if not config_ifaces:
                print(f"[{discover_iface}] No mapped config interfaces; skipping IP assignment for this link")
                continue
            for local_idx, cfg_iface in enumerate(config_ifaces):
                ip_cidr = ip_for_2node_link(link_index, node_id, local_idx)
                iface_to_ip[cfg_iface] = ip_cidr
            print(
                f"2-node link {link_index}: discover_iface {discover_iface} "
                f"-> config_ifaces {config_ifaces}, link_id {link_id}"
            )

    elif mode == "ring3":
        # We need a stable link_index per pair of machines across the 3-node ring.
        # Build all possible pairs of machine indices and sort them.
        n = 3
        pair_list = []
        for i in range(n):
            for j in range(i + 1, n):
                pair_list.append((i, j))
        pair_list.sort()  # deterministic: e.g. (0,1),(0,2),(1,2)

        local_idx_node = cluster_machine_ids.index(local_machine_id)

        for (link_id, discover_iface, neighbor_machine) in link_entries:
            config_ifaces = DISCOVERY_TO_CONFIG_IFACES.get(discover_iface, [])
            if not config_ifaces:
                print(f"[{discover_iface}] No mapped config interfaces; skipping IP assignment for this link")
                continue

            neighbor_idx_node = cluster_machine_ids.index(neighbor_machine)
            pair = (min(local_idx_node, neighbor_idx_node),
                    max(local_idx_node, neighbor_idx_node))
            try:
                link_index = pair_list.index(pair)
            except ValueError:
                print(f"[{discover_iface}] Could not find pair {pair} in pair_list {pair_list}, skipping")
                continue

            # Per link, decide Node 1 vs Node 2 based on machine_id ordering
            node_id_link = 1 if local_machine_id < neighbor_machine else 2

            for local_idx, cfg_iface in enumerate(config_ifaces):
                ip_cidr = ip_for_3node_ring_link(link_index, node_id_link, local_idx)
                iface_to_ip[cfg_iface] = ip_cidr

            print(
                f"3-node ring link {link_index}: discover_iface {discover_iface} "
                f"neighbors {local_machine_id} <-> {neighbor_machine}, "
                f"config_ifaces {config_ifaces}, link_id {link_id}"
            )

    if not iface_to_ip:
        print("No config interfaces to assign IPs to. Aborting.")
        return False

    # print netplan YAML for ALL_INTERFACES
    yaml = build_netplan_yaml(iface_to_ip)
    print("\n--- Netplan YAML ---\n")
    print(yaml)
    print("--- End Netplan YAML ---\n")
    if args.apply_netplan_yaml:
        if not apply_netplan_yaml(yaml):
            return False
    else:
        print("No changes were made to /etc/netplan and netplan was NOT applied.")
    return True


if __name__ == "__main__":
    ret = main()
    if not ret:
        print("Failed to configure cluster networking. Aborting.")
        exit(1)
    else:
        print("Successfully configured cluster networking.")
        exit(0)
