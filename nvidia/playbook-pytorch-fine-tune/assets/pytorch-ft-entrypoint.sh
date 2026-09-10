#!/bin/env bash
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

set -e

SSH_PORT="${SSH_PORT:-2233}"

# Install and configure OpenSSH server
apt-get update && \
    apt-get install -y openssh-server && \
    mkdir -p /var/run/sshd

ls -lha /tmp/.ssh
cp -R /tmp/.ssh /root/
ls -lha /root/.ssh
chown -R $USER: /root/.ssh
chmod 700 /root/.ssh
chmod 600 /root/.ssh/*
if compgen -G "/root/.ssh/*.pub" > /dev/null; then
    chmod 644 /root/.ssh/*.pub
fi


# Allow root login and key-based auth, move port to 2233
sed -i.bak \
    -e 's/^#\?\s*PermitRootLogin\s.*/PermitRootLogin yes/' \
    -e 's/^#\?\s*PubkeyAuthentication\s.*/PubkeyAuthentication yes/' \
    -e 's/^#\?\s*Port\s\+22\s*$/Port '$SSH_PORT'/' \
    /etc/ssh/sshd_config

# Set root password
echo "root:root" | chpasswd

# Configure SSH client for root to disable host key checks within *
echo -e '\nHost *\n    StrictHostKeyChecking no\n    Port '$SSH_PORT'\n    UserKnownHostsFile=/dev/null' > /etc/ssh/ssh_config.d/pyt-ft.conf && \
    chmod 600 /etc/ssh/ssh_config.d/pyt-ft.conf

# Fix login session for container
sed 's@session\\s*required\\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd


# Start SSHD
echo "Starting SSH"
exec /usr/sbin/sshd -D
sshd_rc = $?
echo "Failed to start SSHD, rc $sshd_rc"
exit $sshd_rc
