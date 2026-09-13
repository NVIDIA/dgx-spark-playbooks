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
import { NextRequest, NextResponse } from 'next/server';
import { QdrantService } from '@/lib/qdrant';

/**
 * Clear all data from the Qdrant vector database
 * POST /api/vector-db/clear
 */
export async function POST() {
  // Get the Qdrant service instance
  const qdrantService = QdrantService.getInstance();
  
  // Clear all vectors from the database
  const deleteSuccess = await qdrantService.deleteAllEntities();
  
  // Get updated stats after clearing
  const stats = await qdrantService.getStats();
  
  // Return response based on operation success
  return NextResponse.json({
    success: deleteSuccess,
    message: deleteSuccess 
      ? 'Successfully cleared all data from Qdrant vector database'
      : 'Failed to clear Qdrant database - service may not be available',
    totalVectorCount: stats.totalVectorCount || 0,
    httpHealthy: stats.httpHealthy || false
  });
}

