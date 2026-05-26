//
// SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
/**
 * Triple interface representing a knowledge graph edge
 */
export interface Triple {
  subject: string
  predicate: string
  object: string
  confidence?: number
  usedFallback?: boolean
}

// Add this interface to the file
export interface VectorDBStats {
  nodes: number;
  relationships: number;
  source: string;
  httpHealthy?: boolean;
} 