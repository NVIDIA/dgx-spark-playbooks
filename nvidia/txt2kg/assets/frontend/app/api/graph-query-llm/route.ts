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

