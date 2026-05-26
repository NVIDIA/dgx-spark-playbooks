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

# Stop Remote GPU Rendering Services

echo "🛑 Stopping Remote GPU Rendering Services"
echo "========================================="

# Function to stop service by PID file
stop_service() {
    local service_name=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $service_name (PID: $pid)..."
            kill "$pid"
            
            # Wait for graceful shutdown
            local count=0
            while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force killing $service_name..."
                kill -9 "$pid"
            fi
            
            echo "  ✓ $service_name stopped"
        else
            echo "  $service_name was not running"
        fi
        rm -f "$pid_file"
    else
        echo "  No PID file found for $service_name"
    fi
}

# Stop services
stop_service "Remote GPU Renderer" "logs/remotegpurenderer_pid.txt"
stop_service "Service Monitor" "logs/monitor_pid.txt"

# Stop any remaining processes on the service ports
echo ""
echo "🔍 Checking for remaining processes on service ports..."

ports=(8082)
for port in "${ports[@]}"; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Killing process on port $port..."
        kill $(lsof -t -i:$port) 2>/dev/null || true
    fi
done

echo ""
echo "✅ All Remote GPU Rendering Services stopped"
echo ""
echo "📁 Log files are preserved in logs/ directory:"
echo "  - logs/remote_gpu_rendering.log"
echo ""
echo "To restart services, run: ./start_remote_gpu_services.sh" 