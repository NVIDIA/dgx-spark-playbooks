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

import logging
import paramiko
import argparse
import time
import json
from scp import SCPClient
import threading
import sys
from pathlib import Path
import os
import subprocess
import re
from ipaddress import ip_address as ip_addr_obj, ip_network

logging.getLogger("paramiko").setLevel(logging.CRITICAL)
logging.getLogger("paramiko.transport").setLevel(logging.CRITICAL)

# Default paths
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "spark_config.json"
SHARED_KEY = Path.home() / ".ssh" / "id_ed25519_shared"
SSH_DIR = Path.home() / ".ssh"
AUTHORIZED_KEYS = SSH_DIR / "authorized_keys"
SSH_CONFIG = SSH_DIR / "config"
IDENTITY_LINE = "IdentityFile ~/.ssh/id_ed25519_shared"

NETWORK_SETUP_SCRIPT_NAME = "detect_and_configure_cluster_networking.py"
NETWORK_SETUP_SCRIPT = SCRIPT_DIR / "node_scripts" / NETWORK_SETUP_SCRIPT_NAME

IP_PREFIX = "192.168.100."
LAST_OCTET_START = 10
SUBNET_SIZE = 24

MIN_NCCL_TEST_BW = 21.875 # 175 Gbps
MIN_NCCL_TEST_BW_RING = 10 # 80 Gbps

NCCL_ENV = """export CUDA_HOME="/usr/local/cuda" && export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi" && export NCCL_HOME="$HOME/nccl_spark_cluster/build/" && export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH" """

class ExceptionThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exc = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self.exc = e

    def join(self, timeout=None):
        super().join(timeout)
        if self.exc:
            raise self.exc

def create_ssh_client(server, port, user, password, timeout=10):
    """Creates a Paramiko SSH client and connects."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    time.sleep(1)
    client.connect(server, port, user, password, timeout=timeout)
    return client

def paramiko_run_command_with_output(ssh_client, cmd):
    _, stdout, stderr = ssh_client.exec_command(cmd)

    output = ""
    error = ""
    while not stdout.channel.exit_status_ready():
        if stdout.channel.recv_ready():
            out = stdout.channel.recv(1024).decode('utf-8')
            output += out
        if stderr.channel.recv_ready():
            error_out = stderr.channel.recv(1024).decode('utf-8')
            error += error_out
        time.sleep(0.1)

    # After the loop finishes, there might be remaining output
    # Read the rest of the output
    remaining_output = stdout.read().decode('utf-8')
    output += remaining_output
    remaining_error = stderr.read().decode('utf-8')
    if remaining_error:
        error += remaining_error

    exit_code = stdout.channel.recv_exit_status()
    return exit_code, output, error

def paramiko_run_command(ssh_client, cmd):
    _, stdout, stderr = ssh_client.exec_command(cmd)

    while not stdout.channel.exit_status_ready():
        if stdout.channel.recv_ready():
            stdout.channel.recv(1024).decode('utf-8')
        if stderr.channel.recv_ready():
            stderr.channel.recv(1024).decode('utf-8')
        time.sleep(0.1)

    # After the loop finishes, there might be remaining output
    # Read the rest of the output
    stdout.read().decode('utf-8')
    stderr.read().decode('utf-8')

    # Get the exit status
    exit_code = stdout.channel.recv_exit_status()
    return exit_code


def _paramiko_run_sudo_impl(ssh_client, password, cmd, capture_output):
    """Run a command with sudo, feeding password via stdin when needed.
    Password is never put on the command line, so it won't appear in ps or /proc.
    """
    if password:
        full_cmd = "sudo -S " + cmd
        stdin, stdout, stderr = ssh_client.exec_command(full_cmd)
        stdin.write(password + "\n")
        stdin.channel.shutdown_write()
    else:
        full_cmd = "sudo -n " + cmd
        stdin, stdout, stderr = ssh_client.exec_command(full_cmd)

    output = ""
    error = ""
    while not stdout.channel.exit_status_ready():
        if stdout.channel.recv_ready():
            out = stdout.channel.recv(1024).decode('utf-8')
            if capture_output:
                output += out
        if stderr.channel.recv_ready():
            err = stderr.channel.recv(1024).decode('utf-8')
            if capture_output:
                error += err
        time.sleep(0.1)

    output += stdout.read().decode('utf-8')
    err_rest = stderr.read().decode('utf-8')
    if err_rest:
        error += err_rest
    exit_code = stdout.channel.recv_exit_status()
    return exit_code, output, error


def paramiko_run_sudo_command(ssh_client, password, cmd):
    """Run a command with sudo (password via stdin if provided). Returns exit code only."""
    exit_code, _, _ = _paramiko_run_sudo_impl(ssh_client, password, cmd, capture_output=False)
    return exit_code


def paramiko_run_sudo_command_with_output(ssh_client, password, cmd):
    """Run a command with sudo (password via stdin if provided). Returns (exit_code, output, error)."""
    return _paramiko_run_sudo_impl(ssh_client, password, cmd, capture_output=True)


def ssh_client_active(ssh):
    return bool(ssh and ssh.get_transport() and ssh.get_transport().is_active())

def close_ssh_session(ssh):
    if ssh_client_active(ssh):
        ssh.close()

def setup_nccl_deps(node, ring_topology):
    """Setup NCCL dependencies on the node."""
    ssh = None
    try:
        ssh = create_ssh_client(node["ip_address"], node["port"], node["user"], node["password"])
        if not ssh.get_transport().is_active():
            raise Exception(f"Could not establish a session to node {node["ip_address"]}. Check the credentials and try again.")

        print(f"Updating apt on node {node["ip_address"]}...")
        paramiko_run_sudo_command(ssh, node["password"], "apt update")

        print(f"Installing libopenmpi-dev on node {node["ip_address"]}...")
        exit_code, output, error = paramiko_run_sudo_command_with_output(ssh, node["password"], "apt install -y libopenmpi-dev")
        if exit_code:
            raise Exception(f"Failed to install libopenmpi-dev on node {node["ip_address"]}: output:{output} error:{error}")

        print(f"Cloning NCCL repo on node {node["ip_address"]}...")
        if ring_topology:
            cmd = """rm -rf ~/nccl_spark_cluster/ && git clone -b dgxspark-3node-ring https://github.com/zyang-dev/nccl.git ~/nccl_spark_cluster/"""
        else:
            cmd = """rm -rf ~/nccl_spark_cluster/ && git clone -b v2.28.9-1 https://github.com/NVIDIA/nccl.git ~/nccl_spark_cluster/"""
        exit_code, output, error = paramiko_run_command_with_output(ssh, cmd)
        if exit_code:
            raise Exception(f"Failed to clone NCCL repo on node {node['ip_address']}: output:{output} error:{error}")

        print(f"Building NCCL on node {node["ip_address"]}...")
        cmd = """cd ~/nccl_spark_cluster/ && make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" """
        exit_code, output, error = paramiko_run_command_with_output(ssh, cmd)
        if exit_code:
            raise Exception(f"Failed to build NCCL on node {node['ip_address']}: output:{output} error:{error}")

        print(f"Cloning NCCL tests repo on node {node["ip_address"]}...")
        cmd = """rm -rf ~/nccl-tests_spark_cluster/ && git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests_spark_cluster/"""
        exit_code, output, error = paramiko_run_command_with_output(ssh, cmd)
        if exit_code:
            raise Exception(f"Failed to clone NCCL tests repo on node {node['ip_address']}: output:{output} error:{error}")

        print(f"Building NCCL tests on node {node["ip_address"]}...")
        cmd = """cd ~/nccl-tests_spark_cluster/ && %s && make MPI=1 -j8 """ % NCCL_ENV
        exit_code, output, error = paramiko_run_command_with_output(ssh, cmd)
        if exit_code:
            raise Exception(f"Failed to build NCCL tests on node {node['ip_address']}: {error}")
        
        print(f"Successfully setup NCCL dependencies on node {node['ip_address']}")
        close_ssh_session(ssh)
    except Exception as e:
        close_ssh_session(ssh)
        raise Exception(f"Failed to setup NCCL dependencies on node {node["ip_address"]}:\n{e}")

def run_nccl_test(nodes_info, ring_topology):
    """Runs the NCCL test."""

    threads = []
    for i, node in enumerate(nodes_info):
        t = ExceptionThread(target=setup_nccl_deps, args=(node, ring_topology,))
        threads.append(t)
        t.start()

    for t in threads:
        try:
            t.join()
        except Exception as e:
            print(f"An error occurred when running NCCL setup on nodes:\n{e}")
            return False

    print(f"Successfully setup NCCL dependencies on all nodes...")

    print(f"Running NCCL test...")
    
    # Generate the mpirun command
    host_list = ",".join(f"{node['ip_address']}:1" for node in nodes_info)
    ring_topology_specific_env = "-x NCCL_IB_MERGE_NICS=0 -x NCCL_NET_PLUGIN=none " if ring_topology else ""
    mpirun_cmd = (
        f"{NCCL_ENV} && mpirun -np {len(nodes_info)} -H {host_list} "
        '--mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" '
        "-x LD_LIBRARY_PATH=$LD_LIBRARY_PATH "
        "-x UCX_NET_DEVICES=enP7s7 "
        "-x NCCL_SOCKET_IFNAME=enP7s7 "
        "-x OMPI_MCA_btl_tcp_if_include=enP7s7 "
        "-x NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1 "
        "-x NCCL_IB_SUBNET_AWARE_ROUTING=1 "
        f"{ring_topology_specific_env}"
        "$HOME/nccl-tests_spark_cluster/build/all_gather_perf -b 16G -e 16G -f 2"
    )

    # Run command on the primary node (first node in the list)
    node0 = nodes_info[0]
    ssh = create_ssh_client(node0["ip_address"], node0["port"], node0["user"], node0["password"])
    if not ssh.get_transport().is_active():
        print(f"Could not establish a session to node {node0}. Check the credentials and try again.")
        return False

    print(f"NCCL test command: {mpirun_cmd}")
    exit_code, output, error = paramiko_run_command_with_output(ssh, mpirun_cmd)
    if exit_code:
        print(f"Failed to run NCCL test on node {node0["ip_address"]}: output:{output} error:{error}")
        close_ssh_session(ssh)
        return False
    
    # Extract the "Avg bus bandwidth" value from the NCCL test output
    avg_bus_bw = None
    # The output could potentially be multiline (as it is command output)
    # We need to search for a line matching "# Avg bus bandwidth    : value"
    for line in output.splitlines():
        m = re.match(r"# Avg bus bandwidth\s*:\s*([0-9.]+)", line.strip())
        if m:
            avg_bus_bw = float(m.group(1))
            print(f"Avg bus bandwidth from NCCL test: {avg_bus_bw} GB/s")
            break

    if avg_bus_bw is None:
        print("WARNING: Failed to extract Avg bus bandwidth from NCCL test output.")
    else:
        # If the average bus bandwidth is less then throw a warning
        if (ring_topology and avg_bus_bw < MIN_NCCL_TEST_BW_RING) or (not ring_topology and avg_bus_bw < MIN_NCCL_TEST_BW):
            print("WARNING: NCCL Test bandwidth is less than expected. Stop any GPU workloads on the nodes and try NCCL test again using the NCCL test command above.")
        else:
            print(f"NCCL test BW is as expected")

    close_ssh_session(ssh)
    return True

def ensure_ssh_dir():
    """Ensure ~/.ssh exists with mode 0700."""
    SSH_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(SSH_DIR, 0o700)

def generate_shared_key():
    """Generate shared ed25519 key if it does not exist."""
    if SHARED_KEY.exists():
        return
    ensure_ssh_dir()
    print("Generating shared SSH key for all nodes...")
    os.system(f"ssh-keygen -t ed25519 -N '' -f {SHARED_KEY} -q -C 'shared-cluster-key' > /dev/null 2>&1")
    if not SHARED_KEY.exists():
        raise Exception("Failed to generate shared SSH key.")

def add_pubkey_to_authorized_keys():
    """Add shared public key to local authorized_keys if not present."""
    ensure_ssh_dir()
    pub_content = (SHARED_KEY.with_suffix(".pub")).read_text()
    if AUTHORIZED_KEYS.exists():
        current = AUTHORIZED_KEYS.read_text()
        if pub_content.strip() in current:
            return
    with open(AUTHORIZED_KEYS, "a") as f:
        f.write(pub_content)
    os.chmod(AUTHORIZED_KEYS, 0o600)
    print("Added shared public key to local authorized_keys")

def update_local_ssh_config():
    """Add IdentityFile for shared key to local SSH config if missing."""
    ensure_ssh_dir()
    if SSH_CONFIG.exists():
        content = SSH_CONFIG.read_text()
        if "id_ed25519_shared" in content:
            return
    with open(SSH_CONFIG, "a") as f:
        f.write("Host *\n")
        f.write(f"    {IDENTITY_LINE}\n")
    os.chmod(SSH_CONFIG, 0o600)
    print("Updated local SSH config to use shared key")

def configure_node_ssh_keys(node) -> bool:
    """Copy shared key to node and set up authorized_keys and SSH config."""
    ssh = None
    try:
        ssh = create_ssh_client(node["ip_address"], node["port"], node["user"], node["password"])
        if not ssh.get_transport().is_active():
            raise Exception(f"Could not establish a session to node {node["ip_address"]}. Check the credentials and try again.")

        # Resolve remote home (e.g. /home/nvidia or /root)
        _, stdout, _ = ssh.exec_command("echo $HOME")
        home = stdout.read().decode().strip() or f"/home/{node["user"]}"
        remote_ssh = f"{home}/.ssh"

        exit_code, output, error = paramiko_run_command_with_output(ssh, f"mkdir -p {remote_ssh} && chmod 700 {remote_ssh}")
        if exit_code:
            raise Exception(f"Failed to create remote SSH directory on node {node["ip_address"]}: output:{output} error:{error}")

        with SCPClient(ssh.get_transport()) as scp:
            scp.put(str(SHARED_KEY), f"{remote_ssh}/id_ed25519_shared")
            scp.put(str(SHARED_KEY.with_suffix(".pub")), f"{remote_ssh}/id_ed25519_shared.pub")

        # Set key permissions and add to authorized_keys
        exit_code, output, error = paramiko_run_command_with_output(ssh, f"chmod 600 {remote_ssh}/id_ed25519_shared")
        if exit_code:
            raise Exception(f"Failed to set permissions on {remote_ssh}/id_ed25519_shared: output:{output} error:{error}")

        exit_code, output, error = paramiko_run_command_with_output(ssh, f"chmod 644 {remote_ssh}/id_ed25519_shared.pub")
        if exit_code:
            raise Exception(f"Failed to set permissions on {remote_ssh}/id_ed25519_shared.pub: output:{output} error:{error}")

        pub_line = (SHARED_KEY.with_suffix(".pub")).read_text().strip()
        pub_escaped = pub_line.replace("'", "'\"'\"'")
        exit_code, output, error = paramiko_run_command_with_output(
            ssh,
            f"grep -qF '{pub_escaped}' {remote_ssh}/authorized_keys 2>/dev/null || "
            f"echo '{pub_escaped}' >> {remote_ssh}/authorized_keys",
        )
        if exit_code:
            raise Exception(f"Failed to add {SHARED_KEY.with_suffix(".pub")} to authorized_keys: output:{output} error:{error}")

        exit_code, output, error = paramiko_run_command_with_output(ssh, f"chmod 600 {remote_ssh}/authorized_keys")
        if exit_code:
            raise Exception(f"Failed to set permissions on {remote_ssh}/authorized_keys: output:{output} error:{error}")

        exit_code, output, error = paramiko_run_command_with_output(
            ssh,
            f"grep -q 'IdentityFile.*id_ed25519_shared' {remote_ssh}/config 2>/dev/null || "
            f"(echo 'Host *' >> {remote_ssh}/config && echo '    {IDENTITY_LINE}' >> {remote_ssh}/config)",
        )
        if exit_code:
            raise Exception(f"Failed to add {IDENTITY_LINE} to config: output:{output} error:{error}")

        exit_code, output, error = paramiko_run_command_with_output(ssh, f"chmod 600 {remote_ssh}/config")
        if exit_code:
            raise Exception(f"Failed to set permissions on {remote_ssh}/config: output:{output} error:{error}")

        print(f"Successfully configured {node["ip_address"]} with shared key")
        return True
    except Exception as e:
        print(f"  ✗ Failed to configure {node["ip_address"]}:\n{e}")
        return False
    finally:
        close_ssh_session(ssh)

def check_and_get_up_cx7_interfaces(node_info):
    ssh = None
    up_ifaces = []
    try:
        node = node_info[0]
        ssh = create_ssh_client(node["ip_address"], node["port"], node["user"], node["password"])
        if not ssh.get_transport().is_active():
            raise Exception(f"Could not establish a session to node {node["ip_address"]}. Check the credentials and try again.")

        if_groups = [["enp1s0f0np0", "enP2p1s0f0np0"], ["enp1s0f1np1", "enP2p1s0f1np1"]]
        for if_group in if_groups:
            up_count = 0
            for if_name in if_group:
                cmd = r"""ip link show %s | grep -c "state UP" """ % if_name
                exit_code = paramiko_run_command(ssh, cmd)
                if not exit_code:
                    up_count+=1

            if up_count == len(if_group):
                # found the if_group which has UP interfaces
                up_ifaces.extend(if_group)
        close_ssh_session(ssh)
        if not len(up_ifaces):
            print(f"ERROR: CX7 interfaces on {node["ip_address"]} are not UP")
            return []

        print(f"Found UP CX7 interfaces {up_ifaces} on {node["ip_address"]}. Checking other nodes...")
        for node in node_info[1:]:
            ssh = create_ssh_client(node["ip_address"], node["port"], node["user"], node["password"])
            if not ssh.get_transport().is_active():
                raise Exception(f"Could not establish a session to node {node["ip_address"]}. Check the credentials and try again.")

            for if_group in if_groups:
                up_count = 0
                for if_name in if_group:
                    cmd = r"""ip link show %s | grep -c "state UP" """ % if_name
                    exit_code = paramiko_run_command(ssh, cmd)
                    if not exit_code:
                        if if_name not in up_ifaces:
                            raise Exception(f"ERROR: CX7 interface {if_name} on {node["ip_address"]} is UP which is not in {up_ifaces}. Make sure the same CX7 port(s) on each node are connected and try again.")
                    else:
                        if if_name in up_ifaces:
                            raise Exception(f"ERROR: CX7 interface {if_name} on {node["ip_address"]} is DOWN. {up_ifaces} are expected to be UP. Make sure the same CX7 port(s) on each node are connected and try again.")

            close_ssh_session(ssh)

    except Exception as e:
        close_ssh_session(ssh)
        raise Exception(f"ERROR: An error occurred when checking UP CX7 interfaces:\n{e}")

    return up_ifaces

def check_interface_link_speed(nodes_info, interfaces):
    """Checks the link speed of the interfaces."""
    ssh = None
    try:
        for node in nodes_info:
            ssh = create_ssh_client(node["ip_address"], node["port"], node["user"], node["password"])
            if not ssh.get_transport().is_active():
                raise Exception(f"Could not establish a session to node {node["ip_address"]}. Check the credentials and try again.")

            for iface in interfaces:
                cmd = """ethtool %s | grep -i speed | awk '-F: ' '{print $2}' """ % iface
                exit_code, output, error = paramiko_run_sudo_command_with_output(ssh, node["password"], cmd)
                if exit_code:
                    print(f"ERROR: Failed to check link speed on {iface} on node {node["ip_address"]}: {error}")
                    close_ssh_session(ssh)
                    return False

                speed = output.strip()
                if "200000" not in speed:
                    print(f"ERROR: Link speed on {iface} on node {node["ip_address"]} is not 200Gbps.")
                    print("Check the following:\n"
                            "- QSFP cable should be compatible and rated at least for 200Gbps.\n"
                            "- If running with a switch then check the switch port speed.\n"
                            "- With a switch, sometimes auto-negotiation may not negotiate 200Gbps, in which case set the link speed manually on the switch ports.\n")
                    close_ssh_session(ssh)
                    return False

            close_ssh_session(ssh)

    except Exception as e:
        close_ssh_session(ssh)
        raise Exception(f"Failed to check link speed:\n{e}")

    return True

def scp_put_file_with_ssh(client, local_file, remote_file) -> bool:
    if not local_file or not remote_file:
        print("ERROR: Local file or remote file not specified")
        return False

    try:
        with SCPClient(client.get_transport()) as scp:
            scp.put(local_file, remote_file)
    except Exception as e:
        print(f"scp_put_file: An error occurred:\n{e}")
        return False
    
    return True

def copy_network_setup_script_to_nodes(nodes_info) -> bool:
    """Copies the detect_and_configure_cluster_networking.py script to the nodes and runs it in threads."""
    ssh = None
    try:
        for node in nodes_info:
            ssh = create_ssh_client(node["ip_address"], node["port"], node["user"], node["password"])
            if not ssh.get_transport().is_active():
                print(f"Could not establish a session to node {node["ip_address"]}. Check the credentials and try again.")
                return False

            if not scp_put_file_with_ssh(ssh, NETWORK_SETUP_SCRIPT, f"~/{NETWORK_SETUP_SCRIPT_NAME}"):
                raise Exception(f"Failed to copy {NETWORK_SETUP_SCRIPT_NAME} to node {node["ip_address"]}")

            close_ssh_session(ssh)
    except Exception as e:
        close_ssh_session(ssh)
        print(f"Failed to copy {NETWORK_SETUP_SCRIPT_NAME}:\n{e}")
        return False

    return True

def run_network_setup_script(node, cmd):
    """Runs the network setup script on the node. cmd is the command to run under sudo (no 'sudo' prefix)."""
    ssh = None
    try:
        ssh = create_ssh_client(node["ip_address"], node["port"], node["user"], node["password"])
        if not ssh.get_transport().is_active():
            raise Exception(f"Could not establish a session to node {node["ip_address"]}.")

        exit_code, output, error = paramiko_run_sudo_command_with_output(ssh, node["password"], cmd)
        if exit_code:
            raise Exception(f"Failed to run network setup script on node {node["ip_address"]}: output:{output} error:{error}\n")

        close_ssh_session(ssh)

    except Exception as e:
        close_ssh_session(ssh)
        raise Exception(f"Failed to run network setup script on node {node["ip_address"]}:\n{e}")

def run_network_setup_scripts_on_nodes(nodes_info):
    """Runs the network setup scripts on the nodes in threads."""

    threads = []
    ret = True
    for i, node in enumerate(nodes_info):
        cmd = f"python3 ~/{NETWORK_SETUP_SCRIPT_NAME} --apply-netplan-yaml"
        if i == 0:
            cmd = cmd + " --primary"
        t = ExceptionThread(target=run_network_setup_script, args=(node, cmd))
        threads.append(t)
        t.start()

    for t in threads:
        try:
            t.join()
        except Exception as e:
            print(f"An error occurred when running network setup on nodes:\n{e}")
            ret = False

    return ret

def verify_ip_addresses(nodes_info, up_interfaces) -> bool:
    """Verifies that the IP addresses are assigned to the interfaces."""
    ssh = None
    try:
        nodes_to_ip_cidrs = {}
        all_nodes_ip_addresses = []
        for node in nodes_info:
            ssh = create_ssh_client(node["ip_address"], node["port"], node["user"], node["password"])
            if not ssh.get_transport().is_active():
                raise Exception(f"Could not establish a session to node {node["ip_address"]}. Check the credentials and try again.")

            nodes_to_ip_cidrs[node["ip_address"]] = []
            for iface in up_interfaces:
                cmd = """ip addr show %s | grep -w inet | awk '{print $2}'""" % iface
                exit_code, output, error = paramiko_run_command_with_output(ssh, cmd)
                if exit_code:
                    raise Exception(f"ERROR: Failed to verify IP address on {iface} on node {node["ip_address"]}")

                ip_addresses = output.strip().split("\n")
                if len(ip_addresses) != 1:
                    raise Exception(f"ERROR: Zero or multiple IP addresses found on node {node["ip_address"]}, {iface}: {ip_addresses}")

                if len(ip_addresses[0]) == 0:
                    raise Exception(f"ERROR: No IP address found on node {node["ip_address"]}, {iface}")

                # Parse CIDR (e.g. 192.168.1.1/24) for uniqueness check by IP only
                ip_parts = [a.split("/")[0] for a in ip_addresses]
                if set(all_nodes_ip_addresses).intersection(ip_parts):
                    raise Exception(f"ERROR: IP address {ip_addresses} on node {node["ip_address"]}, {iface} is already assigned to another node.")

                print(f"IP address on node {node["ip_address"]}, {iface}: {ip_addresses}")
                all_nodes_ip_addresses.extend(ip_parts)
                nodes_to_ip_cidrs[node["ip_address"]].extend(ip_addresses)

            close_ssh_session(ssh)

        print(f"Running cluster connectivity test...")

        for node in nodes_info:
            # Run mesh ping test between all nodes in the cluster
            ssh = create_ssh_client(node["ip_address"], node["port"], node["user"], node["password"])
            if not ssh.get_transport().is_active():
                raise Exception(f"Could not establish a session to node {node["ip_address"]}. Check the credentials and try again.")

            node_subnets = list(set([ip_network(cidr, strict=False) for cidr in nodes_to_ip_cidrs[node["ip_address"]]]))

            for ip_address in all_nodes_ip_addresses:
                # Check if the ip_address is in one of the node's subnets
                if not any([ip_addr_obj(ip_address) in s for s in node_subnets]):
                    continue

                cmd = f"ping -c 1 {ip_address} > /dev/null 2>&1"
                exit_code = paramiko_run_command(ssh, cmd)
                if exit_code:
                    raise Exception(f"Failed to run ping test from node {node["ip_address"]} to node {ip_address}")

            close_ssh_session(ssh)

        print(f"Cluster connectivity test completed successfully.")

    except Exception as e:
        close_ssh_session(ssh)
        print(f"Failed to verify IP addresses:\n{e}")
        return False

    return True

def configure_ssh_keys_on_nodes(nodes_info) -> bool:
    """Configures the ssh keys on the nodes."""

    print("Generating shared SSH key for all nodes...")
    generate_shared_key()

    print("Setting up shared SSH access across all nodes...")
    add_pubkey_to_authorized_keys()

    for node in nodes_info:
        print(f"Configuring shared SSH key on node {node["ip_address"]}...")
        if not configure_node_ssh_keys(node):
            return False

    update_local_ssh_config()

    print("Shared SSH keys configured successfully.")

    return True

def handle_cluster_setup(config) -> tuple[bool, bool]:
    """Handles the cluster network setup."""
    try:
        nodes_info = config.get("nodes_info", None)
        if not nodes_info:
            print("ERROR: Nodes information not found.")
            return False, False

        print(f"Checking UP CX7 interfaces...")
        up_interfaces = check_and_get_up_cx7_interfaces(nodes_info)
        if not up_interfaces:
            print("ERROR: Failed to check UP CX7 interfaces. Check the QSFP cable connection and try again.")
            return False, False

        print(f"Checking CX7 interface link speed...")
        if not check_interface_link_speed(nodes_info, up_interfaces):
            return False, False

        print(f"Copying network setup scripts on nodes...")
        # Copy the detect_and_configure_cluster_networking.py script to the nodes and run it in threads
        if not copy_network_setup_script_to_nodes(nodes_info):
            return False, False

        print(f"Running network setup scripts on nodes...")
        if not run_network_setup_scripts_on_nodes(nodes_info):
            print("ERROR: Failed to run network setup scripts on nodes. Check the QSFP cable connections and the nodes config in the json file and try again.")
            return False, False

        # Verify that the IP addresses are assigned to the interfaces
        max_retries = 5
        retries = max_retries
        while retries > 0:
            wait_secs = (max_retries - retries + 1) * 10
            print(f"Waiting for {wait_secs} seconds before checking IP addresses")
            time.sleep(wait_secs)
            if not verify_ip_addresses(nodes_info, up_interfaces):
                print(f"ERROR: Failed to verify IP addresses on nodes. ({retries - 1} retries left)...")
                retries -= 1
                continue
            break

        if retries == 0:
            print("ERROR: Failed to verify IP addresses on nodes. Check the QSFP cable connections and the nodes config in the json file and try again.")
            return False, False

        # Configure ssh keys across nodes
        if not configure_ssh_keys_on_nodes(nodes_info):
            print("ERROR: Failed to configure ssh keys on nodes. Please check the configuration and try again.")
            return False, False

        ring_topology = (len(nodes_info) == 3 and len(up_interfaces) == 4)

    except Exception as e:
        print(f"ERROR: An error occurred when handling cluster setup:\n{e}")
        return False, False

    return True, ring_topology

def validate_config(config):
    """Validates the configuration."""

    if not config.get("nodes_info", None):
        print("ERROR: Nodes information not found.")
        return False

    if len(config.get("nodes_info")) < 2 or len(config.get("nodes_info")) > 4:
        print("ERROR: Cluster can not contain less than 2 or more than 4 nodes. Please check the configuration and try again.")
        return False

    cmd = """ip a | grep -w inet | awk -F"inet |/" '{print $2}' """
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if result.returncode:
        print(f"ERROR: Failed to check IP addresses on current machine: {result.stderr}")
        return False
    else:
        ip_addresses = result.stdout.strip().split("\n")

    print(f"Checking connectivity and permissions...")
    nodes_valid = True
    current_node_in_cluster = False
    for node in config.get("nodes_info", []):
        if not node.get("ip_address", None):
            print("ERROR: IP address not found for node.")
            return False
        if not node.get("user", None):
            print("ERROR: User not found for node.")
            return False
        if not node.get("port", None):
            # Default port is 22
            node["port"] = 22
        if not node.get("password", None):
            # Default password is empty
            node["password"] = ""

        ssh = None
        try:
            ssh = create_ssh_client(node["ip_address"], node["port"], node["user"], node["password"])
            if not ssh.get_transport().is_active():
                print(f"ERROR: Could not establish a session to node {node["ip_address"]}. Check the credentials and try again.")
                return False
        except Exception as e:
            print(f"ERROR: Could not establish a session to node {node["ip_address"]}, check the credentials in the config file and try again: {e}")
            return False

        if node["password"] == "":
            # No password is provided, so we need to validate the ssh key
            exit_code = paramiko_run_sudo_command(ssh, "", "true")
        else:
            exit_code = paramiko_run_sudo_command(ssh, node["password"], "true")
        if exit_code:
            print(f"ERROR: Failed to check sudo access on node {node["ip_address"]}. If password is not specified then make sure that user has sudo access without password.")
            nodes_valid = False
            close_ssh_session(ssh)
            break

        if node["ip_address"] in ip_addresses:
            current_node_in_cluster = True

        close_ssh_session(ssh)

    if not nodes_valid:
        return False

    if not current_node_in_cluster:
        print("ERROR: This script must be run on a node in the cluster.")
        return False

    return True

def validate_environment():
    """Validates the environment."""

    # Check if the script is being run directly instead of via the spark_cluster_setup.sh shell script
    # We expect an environment variable ONLY set by the shell wrapper (e.g. SPARK_CLUSTER_SETUP_WRAPPER=1)
    if os.environ.get("SPARK_CLUSTER_SETUP_WRAPPER") != "1":
        print("ERROR: Please run this script via the spark_cluster_setup.sh shell script, not directly.")
        return False

    # Check if we are running inside a virtual environment
    if sys.prefix == sys.base_prefix:
        print("ERROR: Please run this script inside a Python virtual environment (venv) with requirements installed.")
        return False

    # Check if /etc/dgx-release exists and contains the expected DGX Spark markers
    try:
        with open("/etc/dgx-release", "r") as f:
            content = f.read()
        # Look for DGX_NAME="DGX Spark" and DGX_PRETTY_NAME="NVIDIA DGX Spark"
        if 'DGX_NAME="DGX Spark"' not in content or 'DGX_PRETTY_NAME="NVIDIA DGX Spark"' not in content:
            print("ERROR: This script must be run on a DGX Spark.")
            return False
    except FileNotFoundError:
        print("ERROR: /etc/dgx-release not found. This is not a DGX Spark environment.")
        return False

    return True


def main():
    """Main function to setup the Spark cluster."""
    parser = argparse.ArgumentParser(description="Setup the Spark cluster.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file.")

    args = parser.parse_args()
    config = args.config
    with open(config, "r") as f:
        config = json.load(f)

    if not config:
        print("ERROR: Configuration file not found.")
        return

    try:
        # Validate env
        print(f"Validating environment...")
        if not validate_environment():
            return

        # Validate the config
        print(f"Validating configuration...")
        if not validate_config(config):
            return

        print("Setting up Spark cluster...")
        ret, ring_topology = handle_cluster_setup(config)
        if not ret:
            return

        print("Spark cluster setup completed successfully.")

        print("Running NCCL test...")
        if not run_nccl_test(config.get("nodes_info", []), ring_topology):
            return
        print("NCCL test completed.")

    except Exception as e:
        print(f"ERROR: An error occurred when running Spark cluster setup:\n{e}")

if __name__ == "__main__":
    main()
