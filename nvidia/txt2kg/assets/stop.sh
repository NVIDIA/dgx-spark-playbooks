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

# Stop script for txt2kg project

# Check which Docker Compose version is available
DOCKER_COMPOSE_CMD=""
if docker compose version &> /dev/null; then
  DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
  DOCKER_COMPOSE_CMD="docker-compose"
else
  echo "Error: Neither 'docker compose' nor 'docker-compose' is available"
  exit 1
fi

# Parse command line arguments
USE_VLLM=false
USE_VECTOR_SEARCH=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --vllm)
      USE_VLLM=true
      shift
      ;;
    --vector-search)
      USE_VECTOR_SEARCH=true
      shift
      ;;
    --help|-h)
      echo "Usage: ./stop.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --vllm            Stop vLLM stack (use if you started with --vllm)"
      echo "  --vector-search   Include vector search services"
      echo "  --help, -h        Show this help message"
      echo ""
      echo "Note: Use the same flags you used with ./start.sh"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './stop.sh --help' for usage information"
      exit 1
      ;;
  esac
done

# Select compose file
COMPOSE_DIR="$(pwd)/deploy/compose"
PROFILES=""

if [ "$USE_VLLM" = true ]; then
  COMPOSE_FILE="$COMPOSE_DIR/docker-compose.vllm.yml"
else
  COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"
fi

CMD="$DOCKER_COMPOSE_CMD -f $COMPOSE_FILE"

if [ "$USE_VECTOR_SEARCH" = true ]; then
  PROFILES="--profile vector-search"
fi

echo "Stopping txt2kg services..."
cd $(dirname "$0")
eval "$CMD $PROFILES down"

echo ""
echo "All services stopped."
echo "To start again, run: ./start.sh"
