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
 * Get Qdrant vector database stats
 */
export async function GET() {
  try {
    // Initialize Qdrant service
    const qdrantService = QdrantService.getInstance();
    
    // We can now directly call getStats() which handles initialization and error recovery
    const stats = await qdrantService.getStats();
    
    return NextResponse.json({
      ...stats,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error getting Qdrant stats:', error);
    
    // Return a successful response with error information
    // This prevents the UI from breaking when Qdrant is unavailable
    let errorMessage = error instanceof Error ? error.message : String(error);
    
    // More specific error message for 404 errors
    if (errorMessage.includes('404')) {
      errorMessage = 'Qdrant server returned 404. The server may not be running or the collection does not exist.';
    }
    
    return NextResponse.json(
      { 
        error: `Failed to get Qdrant stats: ${errorMessage}`,
        totalVectorCount: 0,
        source: 'error',
        httpHealthy: false,
        timestamp: new Date().toISOString()
      },
      { status: 200 } // Use 200 instead of 500 to avoid UI errors
    );
  }
}

