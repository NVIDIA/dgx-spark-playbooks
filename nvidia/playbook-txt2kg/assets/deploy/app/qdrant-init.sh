#!/bin/sh
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

# Script to initialize Qdrant collection at container startup
echo "Initializing Qdrant collection..."

# Wait for the Qdrant service to become available
echo "Waiting for Qdrant service to start..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
  if curl -s http://qdrant:6333/healthz > /dev/null; then
    echo "Qdrant service is up!"
    break
  fi
  echo "Waiting for Qdrant service (attempt $attempt/$max_attempts)..."
  attempt=$((attempt + 1))
  sleep 2
done

if [ $attempt -gt $max_attempts ]; then
  echo "Timed out waiting for Qdrant service"
  exit 1
fi

# Check if collection already exists
echo "Checking if collection 'entity-embeddings' exists..."
COLLECTION_EXISTS=$(curl -s http://qdrant:6333/collections/entity-embeddings | grep -c '"status":"ok"' || echo "0")

if [ "$COLLECTION_EXISTS" -gt "0" ]; then
  echo "Collection 'entity-embeddings' already exists, skipping creation"
else
  # Create the collection
  echo "Creating collection 'entity-embeddings'..."
  curl -X PUT "http://qdrant:6333/collections/entity-embeddings" \
    -H "Content-Type: application/json" \
    -d '{
      "vectors": {
        "size": 384,
        "distance": "Cosine"
      }
    }'

  if [ $? -eq 0 ]; then
    echo "✅ Collection 'entity-embeddings' created successfully"
  else
    echo "❌ Failed to create collection"
    exit 1
  fi
fi

echo "Qdrant initialization complete"
