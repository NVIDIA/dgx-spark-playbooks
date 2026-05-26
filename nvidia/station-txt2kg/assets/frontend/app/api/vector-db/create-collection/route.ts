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
import { NextResponse } from 'next/server';
import { QdrantService } from '@/lib/qdrant';

/**
 * Create Qdrant collection API endpoint
 * POST /api/vector-db/create-collection
 */
export async function POST() {
  try {
    // Get the Qdrant service instance
    const qdrantService = QdrantService.getInstance();
    
    // Force re-initialization to create the collection
    (qdrantService as any).initialized = false;
    await qdrantService.initialize();
    
    // Check if initialization was successful by getting stats
    const stats = await qdrantService.getStats();
    
    return NextResponse.json({
      success: true,
      message: 'Qdrant collection created successfully',
      httpHealthy: stats.httpHealthy || false
    });
  } catch (error) {
    console.error('Error creating Qdrant collection:', error);
    
    return NextResponse.json(
      { 
        success: false,
        error: `Failed to create Qdrant collection: ${error instanceof Error ? error.message : String(error)}`
      },
      { status: 500 }
    );
  }
}

