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
import RAGService from '@/lib/rag';

/**
 * API endpoint for RAG-based question answering
 * Uses Qdrant for document retrieval and LangChain for generation
 * POST /api/rag-query
 */
export async function POST(req: NextRequest) {
  try {
    // Parse request body
    const body = await req.json();
    const { query, topK = 5 } = body;

    if (!query || typeof query !== 'string') {
      return NextResponse.json({ error: 'Query is required' }, { status: 400 });
    }

    // Initialize the RAG service
    const ragService = RAGService;
    await ragService.initialize();
    
    console.log(`Processing Pure RAG query: "${query}" with topK=${topK}`);

    // Retrieve documents and generate answer
    const result = await ragService.retrievalQA(query, topK);
    
    // Check if this is a fallback response
    const isGeneralKnowledgeFallback = result.answer.startsWith('[Note: No specific information was found');

    console.log(`✅ Pure RAG query completed. Retrieved ${result.documentCount} document chunks`);

    // Return the results
    return NextResponse.json({
      answer: result.answer,
      documentCount: result.documentCount,
      usedFallback: isGeneralKnowledgeFallback,
      success: true
    });
  } catch (error) {
    console.error('Error in RAG query:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: `Failed to execute RAG query: ${errorMessage}` },
      { status: 500 }
    );
  }
} 