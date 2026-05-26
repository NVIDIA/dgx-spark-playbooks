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
set -e

# Wait for ArangoDB to be ready
echo "Waiting for ArangoDB to start..."
until curl --silent --fail http://localhost:8529/_api/version > /dev/null; do
  echo "ArangoDB is unavailable - sleeping"
  sleep 1
done

echo "ArangoDB is up - executing initialization script"

# Run the database creation script
arangosh \
  --server.endpoint tcp://127.0.0.1:8529 \
  --server.authentication false \
  --javascript.execute /docker-entrypoint-initdb.d/create-database.js

echo "Initialization completed" 