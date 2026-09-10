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
import backendService from '@/lib/backend-service';
import { getGraphDbType } from '../settings/route';

/**
 * API endpoint for LLM-enhanced graph query
 * This retrieves triples using graph search and generates an answer using LLM
 * Makes traditional graph search comparable to RAG for fair benchmarking
 * POST /api/graph-query-llm
 */
export async function POST(request: NextRequest) {
  try {
    const { query, topK = 5, useTraditional = true, llmModel, llmProvider } = await request.json();
    
    if (!query) {
      return NextResponse.json({ error: 'Query is required' }, { status: 400 });
    }
    
    // Initialize backend if needed with the selected graph DB type
    if (!backendService.isInitialized) {
      const graphDbType = getGraphDbType();
      console.log(`Initializing backend with graph DB type: ${graphDbType}`);
      await backendService.initialize(graphDbType);
    }
    
    console.log(`Graph query with LLM: "${query}", topK=${topK}, traditional=${useTraditional}, model=${llmModel || 'default'}, provider=${llmProvider || 'default'}`);
    
    // Query the backend with LLM enhancement
    const result = await backendService.queryWithLLM(query, topK, useTraditional, llmModel, llmProvider);

    // DEBUG: Log first triple in API route to verify depth/pathLength
    if (result.triples && result.triples.length > 0) {
      console.log('API route - first triple:', JSON.stringify(result.triples[0], null, 2));
    }

    // Return results
    return NextResponse.json({
      query,
      answer: result.answer,
      triples: result.triples,
      count: result.count,
      message: `Retrieved ${result.count} triples and generated answer using ${useTraditional ? 'traditional' : 'vector'} graph search + LLM`
    });
  } catch (error) {
    console.error('Error in graph query with LLM:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}

