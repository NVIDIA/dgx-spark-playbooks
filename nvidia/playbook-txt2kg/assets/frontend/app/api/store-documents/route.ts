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
 * API endpoint for storing documents in the RAG system
 * POST /api/store-documents
 */
export async function POST(req: NextRequest) {
  try {
    // Parse request body
    const body = await req.json();
    const { documents, metadata } = body;

    if (!documents || !Array.isArray(documents) || documents.length === 0) {
      return NextResponse.json({ error: 'Documents array is required' }, { status: 400 });
    }

    // Validate that all documents are strings
    const isValid = documents.every(doc => typeof doc === 'string' && doc.trim().length > 0);
    if (!isValid) {
      return NextResponse.json({ 
        error: 'All documents must be non-empty strings' 
      }, { status: 400 });
    }

    // Initialize the RAG service
    const ragService = RAGService;
    await ragService.initialize();
    
    console.log(`Storing ${documents.length} documents in RAG system`);

    // Store the documents
    await ragService.storeDocuments(documents, metadata);

    // Return success
    return NextResponse.json({
      success: true,
      count: documents.length,
      message: `Successfully stored ${documents.length} documents`
    });
  } catch (error) {
    console.error('Error storing documents:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: `Failed to store documents: ${errorMessage}` },
      { status: 500 }
    );
  }
} 