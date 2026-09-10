#!/bin/bash
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
#
# Clear buffer cache and restart Ollama to fix unified memory detection
# This script addresses the issue where Ollama can't see full GPU memory
# due to buffer cache not being reclaimable in unified memory systems
#

set -e

echo "🧹 Clearing system buffer cache..."
echo "Current memory status:"
free -h

echo "Stopping Ollama container..."
docker compose -f /home/nvidia/txt2kg/txt2kg/deploy/services/ollama/docker-compose.yml down

echo "Clearing buffer cache..."
sudo sync
sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'

echo "Memory status after cache clear:"
free -h

echo "Restarting Ollama container..."
docker compose -f /home/nvidia/txt2kg/txt2kg/deploy/services/ollama/docker-compose.yml up -d

echo "Waiting for Ollama to start..."
sleep 10

echo "Checking GPU memory detection..."
timeout 30 bash -c 'while ! docker logs ollama-server 2>&1 | grep -q "inference compute"; do sleep 1; done'
docker logs ollama-server 2>&1 | grep "inference compute" | tail -1

echo "✅ Ollama restarted with cleared cache"
