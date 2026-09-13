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
import { EmbeddingsService } from '@/lib/embeddings';
import type { Triple } from '@/types/graph';
import { BackendService } from '@/lib/backend-service';
import { getGraphDbType } from '../settings/route';

/**
 * API endpoint for processing documents with LangChain, generating embeddings,
 * and storing in the knowledge graph
 * POST /api/process-document
 */
export async function POST(req: NextRequest) {
  try {
    // Parse request body
    const body = await req.json();
    const { 
      text, 
      filename, 
      triples, 
      useLangChain, 
      useGraphTransformer,
      systemPrompt,
      extractionPrompt,
      graphTransformerPrompt
    } = body;

    if (!text || typeof text !== 'string') {
      return NextResponse.json({ error: 'Text is required' }, { status: 400 });
    }

    if (!triples || !Array.isArray(triples)) {
      return NextResponse.json({ error: 'Triples are required' }, { status: 400 });
    }

    // Initialize services
    const backendService = RemoteBackendService.getInstance();
    const embeddingsService = EmbeddingsService.getInstance();

    console.log(`🔍 API: Processing document "${filename || 'unnamed'}" (${text.length} chars)`);
    console.log(`🔍 API: Processing ${triples.length} triples`);
    console.log(`🔍 API: Using LangChain for triple extraction: ${useLangChain ? 'Yes' : 'No'}`);
    console.log(`🔍 API: First few triples:`, triples.slice(0, 3));
    if (useLangChain) {
      console.log(`Using LLMGraphTransformer: ${useGraphTransformer ? 'Yes' : 'No'}`);
    }
    
    // Log if custom prompts are being used
    if (systemPrompt || extractionPrompt || graphTransformerPrompt) {
      console.log('Using custom prompts for extraction');
      if (systemPrompt) console.log('Custom system prompt provided');
      if (extractionPrompt) console.log('Custom extraction prompt provided');
      if (graphTransformerPrompt) console.log('Custom graph transformer prompt provided');
    }

    // Filter triples to ensure they are valid
    const validTriples = triples.filter((triple: any) => {
      return (
        triple &&
        typeof triple.subject === 'string' && triple.subject.trim() !== '' &&
        typeof triple.predicate === 'string' && triple.predicate.trim() !== '' &&
        typeof triple.object === 'string' && triple.object.trim() !== ''
      );
    }) as Triple[];

    console.log(`Found ${validTriples.length} valid triples`);

    // If useLangChain flag is set, we'll extract triples using the LangChain route
    let triplesForProcessing = validTriples;
    
    if (useLangChain && !filename?.toLowerCase().endsWith('.csv')) {
      try {
        console.log('Using LangChain for native triple extraction...');
        // Use absolute URL with origin from request to fix URL parsing error
        const baseUrl = new URL(req.url).origin;
        console.log(`Using base URL: ${baseUrl} for LangChain API call`);
        
        // Call the extract-triples endpoint with useLangChain flag and custom prompts
        const requestBody: any = { 
          text, 
          useLangChain: true,
          useGraphTransformer
        };
        
        // Add custom prompts if available
        if (systemPrompt) requestBody.systemPrompt = systemPrompt;
        if (extractionPrompt) requestBody.extractionPrompt = extractionPrompt;
        if (graphTransformerPrompt) requestBody.graphTransformerPrompt = graphTransformerPrompt;
        
        const langchainResponse = await fetch(`${baseUrl}/api/extract-triples`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody)
        });
        
        if (!langchainResponse.ok) {
          const errorText = await langchainResponse.text();
          console.error(`LangChain API error: ${langchainResponse.status} ${langchainResponse.statusText}`, errorText);
          throw new Error(`LangChain extraction failed: ${langchainResponse.statusText} (${langchainResponse.status})`);
        }
        
        const langchainResult = await langchainResponse.json();
        if (langchainResult.triples && Array.isArray(langchainResult.triples) && langchainResult.triples.length > 0) {
          console.log(`Successfully extracted ${langchainResult.triples.length} triples using LangChain${useGraphTransformer ? ' with GraphTransformer' : ''}`);
          triplesForProcessing = langchainResult.triples;
        } else {
          console.warn('LangChain extraction returned no triples, falling back to provided triples');
        }
      } catch (langchainError) {
        console.error('Error using LangChain for triple extraction:', langchainError);
        console.log('Falling back to provided triples');
      }
    }

    // Check if this is a CSV file - if so, skip processing
    const isCSVFile = filename && filename.toLowerCase().endsWith('.csv');
    const isJSONFile = filename && filename.toLowerCase().endsWith('.json');
    
    if (isCSVFile) {
      console.log('CSV file detected, skipping text processor');
      // NOTE: Neo4j storage is no longer done automatically
      // This is now handled manually through the "Store in Graph DB" button in the UI
    } else if (isJSONFile) {
      console.log('JSON file detected, processed as unstructured text document - embeddings can be generated manually via the UI');
      // NOTE: Automatic embeddings generation has been disabled for JSON files.
      // Embeddings are now generated only when explicitly requested through the "Generate Embeddings" button in the UI.
    } else {
      // Regular text processing flow - no automatic embeddings generation
      console.log('Document processed successfully - embeddings can be generated manually via the UI');
      // NOTE: Automatic embeddings generation has been disabled.
      // Embeddings are now generated only when explicitly requested through the "Generate Embeddings" button in the UI.
    }

    // Return success response
    return NextResponse.json({
      success: true,
      message: 'Document processed successfully',
      tripleCount: triplesForProcessing.length,
      triples: triplesForProcessing,
      documentName: filename || 'unnamed',
      langchainUsed: useLangChain,
      graphTransformerUsed: useGraphTransformer,
      customPromptsUsed: !!(systemPrompt || extractionPrompt || graphTransformerPrompt),
      graphDbType: getGraphDbType()
    });
  } catch (error) {
    console.error('Error processing document:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: `Failed to process document: ${errorMessage}` },
      { status: 500 }
    );
  }
} 