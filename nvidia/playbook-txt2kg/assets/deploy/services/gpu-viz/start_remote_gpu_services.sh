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

# Start Remote GPU Rendering Services
# This script starts the custom remote GPU rendering service as an alternative to PyGraphistry cloud

echo "🚀 Starting Remote GPU Rendering Services"
echo "========================================="

# Check if we're in a RAPIDS/cuGraph environment
if python -c "import cudf, cugraph" 2>/dev/null; then
    echo "✓ RAPIDS/cuGraph environment detected"
    GPU_AVAILABLE=true
else
    echo "⚠ RAPIDS/cuGraph not available - will use CPU fallback"
    GPU_AVAILABLE=false
fi

# Check if Redis is available (optional for session storage)
if command -v redis-server >/dev/null 2>&1; then
    echo "✓ Redis available for session storage"
    
    # Start Redis if not running
    if ! pgrep -x "redis-server" > /dev/null; then
        echo "Starting Redis server..."
        redis-server --daemonize yes --port 6379
        sleep 2
    fi
else
    echo "⚠ Redis not available - using in-memory session storage"
fi

# Set environment variables
export REDIS_HOST=${REDIS_HOST:-localhost}
export REDIS_PORT=${REDIS_PORT:-6379}

# Create log directory
mkdir -p logs

echo ""
echo "🎯 Service Configuration:"
echo "  GPU Processing: $GPU_AVAILABLE"
echo "  Session Storage: ${REDIS_HOST:-memory}:${REDIS_PORT:-N/A}"
echo "  Service Port: 8082"
echo ""

# Function to start service with proper error handling
start_service() {
    local service_name=$1
    local script_path=$2
    local port=$3
    local log_file=$4
    
    echo "Starting $service_name on port $port..."
    
    # Kill existing process if running
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; then
        echo "  Killing existing process on port $port"
        kill $(lsof -t -i:$port) 2>/dev/null || true
        sleep 2
    fi
    
    # Start the service
    python $script_path > logs/$log_file 2>&1 &
    local pid=$!
    
    # Wait a moment and check if it started successfully
    sleep 3
    if kill -0 $pid 2>/dev/null; then
        echo "  ✓ $service_name started successfully (PID: $pid)"
        echo $pid > logs/${service_name,,}_pid.txt
        return 0
    else
        echo "  ✗ Failed to start $service_name"
        echo "  Check logs/$log_file for details"
        return 1
    fi
}

# Start Remote GPU Rendering Service
echo "📊 Starting Remote GPU Rendering Service..."
start_service "RemoteGPURenderer" "remote_gpu_rendering_service.py" 8082 "remote_gpu_rendering.log"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Remote GPU Rendering Service is ready!"
    echo ""
    echo "🎯 Available endpoints:"
    echo "  Process graph:        POST http://localhost:8082/api/render"
    echo "  Iframe visualization: GET  http://localhost:8082/embed/{session_id}"
    echo "  Session status:       GET  http://localhost:8082/api/session/{session_id}"
    echo "  Real-time updates:    WS   ws://localhost:8082/ws/{session_id}"
    echo "  Health check:         GET  http://localhost:8082/api/health"
    echo ""
    echo "📋 Usage examples:"
    echo ""
    echo "  # Test health check"
    echo "  curl http://localhost:8082/api/health"
    echo ""
    echo "  # Process a sample graph"
    echo "  curl -X POST http://localhost:8082/api/render \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{"
    echo "      \"graph_data\": {"
    echo "        \"nodes\": [{\"id\": \"1\", \"name\": \"Node 1\"}, {\"id\": \"2\", \"name\": \"Node 2\"}],"
    echo "        \"links\": [{\"source\": \"1\", \"target\": \"2\", \"name\": \"edge_1_2\"}]"
    echo "      },"
    echo "      \"layout_algorithm\": \"force_atlas2\","
    echo "      \"clustering_algorithm\": \"leiden\","
    echo "      \"compute_centrality\": true,"
    echo "      \"render_quality\": \"high\","
    echo "      \"interactive_mode\": true"
    echo "    }'"
    echo ""
    echo "📁 Logs are available in:"
    echo "  Remote GPU Rendering: logs/remote_gpu_rendering.log"
    echo ""
    echo "🛠️ Integration with frontend:"
    echo "  import { RemoteGPUViewer } from '@/components/remote-gpu-viewer'"
    echo "  <RemoteGPUViewer"
    echo "    graphData={graphData}"
    echo "    remoteServiceUrl=\"http://localhost:8082\""
    echo "    onError={(err) => console.error(err)}"
    echo "  />"
    echo ""
    echo "⚡ Performance tips:"
    echo "  - Use 'ultra' quality for 1M+ node graphs"
    echo "  - Enable Redis for production session storage"
    echo "  - Run on GPU server for maximum performance"
    echo "  - Use iframe embedding to isolate visualization"
    echo ""
    
    # Start a simple monitoring script
    echo "🔍 Starting service monitor..."
    monitor_services() {
        while true; do
            sleep 30
            
            # Check if services are still running
            if [ -f logs/remotegpurenderer_pid.txt ]; then
                pid=$(cat logs/remotegpurenderer_pid.txt)
                if ! kill -0 $pid 2>/dev/null; then
                    echo "$(date): Remote GPU Rendering Service died, restarting..."
                    start_service "RemoteGPURenderer" "remote_gpu_rendering_service.py" 8082 "remote_gpu_rendering.log"
                fi
            fi
        done
    }
    
    # Run monitor in background
    monitor_services &
    echo $! > logs/monitor_pid.txt
    
    echo "✅ All services started and monitoring enabled!"
    echo ""
    echo "To stop all services, run: ./stop_remote_gpu_services.sh"
    echo "To view logs in real-time: tail -f logs/remote_gpu_rendering.log"
    
else
    echo ""
    echo "❌ Failed to start Remote GPU Rendering Service"
    echo "Check the logs for details and ensure dependencies are installed:"
    echo "  - FastAPI: pip install fastapi uvicorn"
    echo "  - RAPIDS (optional): conda install -c rapidsai cudf cugraph"
    echo "  - Redis (optional): sudo apt-get install redis-server"
    exit 1
fi 