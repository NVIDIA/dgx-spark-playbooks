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

# vLLM Llama3 8B Benchmark Runner
# Uses NVIDIA vLLM container for optimal performance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_URL="http://localhost:8001"
RUNS=3
MAX_TOKENS=512
OUTPUT_FILE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  🚀 vLLM Llama3 8B Benchmark Suite${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -u, --url URL          vLLM service URL (default: http://localhost:8001)"
    echo "  -r, --runs NUMBER      Number of runs per prompt (default: 3)"
    echo "  -t, --max-tokens NUM   Maximum tokens to generate (default: 512)"
    echo "  -o, --output FILE      Output file for detailed results (JSON)"
    echo "  -d, --docker           Run using Docker Compose"
    echo "  -s, --start-service    Start vLLM service first"
    echo "  -h, --health-check     Only run health check"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run basic benchmark"
    echo "  $0 --docker --start-service          # Start service and run benchmark in Docker"
    echo "  $0 -r 5 -t 1024 -o results.json     # Custom settings with output file"
    echo "  $0 --health-check                    # Check if service is running"
}

check_dependencies() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 is required but not installed${NC}"
        exit 1
    fi
    
    if ! python3 -c "import aiohttp, asyncio" &> /dev/null; then
        echo -e "${YELLOW}⚠️  Installing required Python packages...${NC}"
        pip3 install aiohttp asyncio
    fi
}

check_nvidia_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is required but not installed${NC}"
        exit 1
    fi
    
    if ! docker info | grep -q "nvidia"; then
        echo -e "${YELLOW}⚠️  NVIDIA Docker runtime not detected. Make sure nvidia-container-toolkit is installed${NC}"
    fi
}

start_vllm_service() {
    echo -e "${BLUE}🚀 Starting vLLM Llama3 8B service...${NC}"
    
    cd "$SCRIPT_DIR"
    docker-compose -f docker-compose.llama3-8b.yml up -d vllm-llama3-8b
    
    echo -e "${YELLOW}⏳ Waiting for model to load (this may take several minutes)...${NC}"
    
    # Wait for service to be healthy
    local max_attempts=60  # 10 minutes
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$VLLM_URL/v1/models" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ vLLM service is ready!${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}⏳ Attempt $attempt/$max_attempts - waiting for service...${NC}"
        sleep 10
        ((attempt++))
    done
    
    echo -e "${RED}❌ vLLM service failed to start within timeout${NC}"
    echo -e "${YELLOW}📋 Checking service logs:${NC}"
    docker-compose -f docker-compose.llama3-8b.yml logs vllm-llama3-8b
    exit 1
}

run_benchmark() {
    local cmd_args=("--url" "$VLLM_URL" "--runs" "$RUNS" "--max-tokens" "$MAX_TOKENS")
    
    if [ -n "$OUTPUT_FILE" ]; then
        cmd_args+=("--output" "$OUTPUT_FILE")
    fi
    
    if [ "$HEALTH_CHECK_ONLY" = true ]; then
        cmd_args+=("--health-check-only")
    fi
    
    echo -e "${BLUE}🧪 Running vLLM Llama3 8B benchmark...${NC}"
    echo -e "${BLUE}URL: $VLLM_URL${NC}"
    echo -e "${BLUE}Runs per prompt: $RUNS${NC}"
    echo -e "${BLUE}Max tokens: $MAX_TOKENS${NC}"
    
    if [ "$USE_DOCKER" = true ]; then
        # Run benchmark in Docker
        cd "$SCRIPT_DIR"
        docker-compose -f docker-compose.llama3-8b.yml run --rm vllm-benchmark \
            python /app/vllm_llama3_benchmark.py "${cmd_args[@]}"
    else
        # Run benchmark locally
        python3 "$SCRIPT_DIR/vllm_llama3_benchmark.py" "${cmd_args[@]}"
    fi
}

# Parse command line arguments
USE_DOCKER=false
START_SERVICE=false
HEALTH_CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--url)
            VLLM_URL="$2"
            shift 2
            ;;
        -r|--runs)
            RUNS="$2"
            shift 2
            ;;
        -t|--max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -d|--docker)
            USE_DOCKER=true
            shift
            ;;
        -s|--start-service)
            START_SERVICE=true
            shift
            ;;
        -h|--health-check)
            HEALTH_CHECK_ONLY=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Main execution
print_header

if [ "$USE_DOCKER" = true ]; then
    check_nvidia_docker
    
    if [ "$START_SERVICE" = true ]; then
        start_vllm_service
    fi
    
    run_benchmark
else
    check_dependencies
    
    if [ "$START_SERVICE" = true ]; then
        echo -e "${YELLOW}⚠️  --start-service only works with --docker flag${NC}"
        exit 1
    fi
    
    run_benchmark
fi

echo -e "${GREEN}✅ Benchmark completed successfully!${NC}"

if [ -n "$OUTPUT_FILE" ] && [ -f "$OUTPUT_FILE" ]; then
    echo -e "${BLUE}📊 Detailed results saved to: $OUTPUT_FILE${NC}"
fi
