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
import { RemoteBackendService } from '@/lib/remote-backend';

/**
 * API endpoint for enhanced RAG query with LangChain features
 * POST /api/enhanced-query
 */
export async function POST(req: NextRequest) {
  try {
    // Parse request body
    const body = await req.json();
    const { query, kNeighbors, fanout, numHops, topK, queryMode, useTraditional } = body;

    if (!query || typeof query !== 'string') {
      return NextResponse.json({ error: 'Query is required' }, { status: 400 });
    }

    // Initialize the backend service
    const backend = RemoteBackendService.getInstance();
    
    // Prepare parameters with defaults
    const params = {
      kNeighbors: kNeighbors || 4096,
      fanout: fanout || 400,
      numHops: numHops || 2,
      topK: topK || 5
    };
    
    console.log(`Enhanced RAG query: "${query}" with params:`, params);
    console.log(`Query mode: ${queryMode}, useTraditional: ${useTraditional}`);
    
    // Determine search method - if traditional is specified, use that
    const shouldUseTraditional = useTraditional || (queryMode === 'traditional');
    
    if (shouldUseTraditional) {
      console.log('Using traditional search for enhanced query');
      // Call the regular query method with traditional flag
      const relevantTriples = await backend.query(
        query,
        params.kNeighbors,
        params.fanout,
        params.numHops,
        { 
          topk: params.topK, 
          topk_e: params.topK, 
          cost_e: 0.5, 
          num_clusters: 2 
        },
        true // Use traditional search
      );
      
      // Return the results
      return NextResponse.json({
        relevantTriples,
        count: relevantTriples.length,
        metadata: {
          searchType: 'traditional'
        },
        success: true
      });
    }

    // Use the enhanced query with metadata for vector search
    const { relevantTriples, queryMetadata } = await backend.enhancedQuery(
      query,
      params.kNeighbors,
      params.fanout,
      params.numHops,
      { 
        topk: params.topK, 
        topk_e: params.topK, 
        cost_e: 0.5, 
        num_clusters: 2 
      }
    );

    // Return the results
    return NextResponse.json({
      relevantTriples,
      count: relevantTriples.length,
      metadata: queryMetadata,
      success: true
    });
  } catch (error) {
    console.error('Error in enhanced RAG query:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: `Failed to execute enhanced query: ${errorMessage}` },
      { status: 500 }
    );
  }
} 