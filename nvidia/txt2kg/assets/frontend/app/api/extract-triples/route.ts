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
import { processDocument, TextProcessor } from '@/lib/text-processor';
import { llmService } from '@/lib/llm-service';

// Configure route for dynamic operations and long-running requests
export const dynamic = 'force-dynamic';
export const maxDuration = 1800; // 30 minutes for large model processing

/**
 * API endpoint for extracting triples from text using the LangChain-based pipeline
 * POST /api/extract-triples
 */
export async function POST(req: NextRequest) {
  const startTime = Date.now();
  console.log(`[${new Date().toISOString()}] extract-triples: Request received`);
  
  try {
    // Parse request body
    const body = await req.json();
    console.log(`[${new Date().toISOString()}] extract-triples: Body parsed, text length: ${body.text?.length || 0}`);
    const { 
      text, 
      useLangChain = false, 
      useGraphTransformer = false,
      systemPrompt,
      extractionPrompt,
      graphTransformerPrompt,
      llmProvider,
      ollamaModel,
      ollamaBaseUrl,
      vllmModel,
      vllmBaseUrl,
      nvidiaModel
    } = body;

    if (!text || typeof text !== 'string') {
      return NextResponse.json({ error: 'Text is required' }, { status: 400 });
    }

    // If Ollama is specified, call llmService directly (avoid internal fetch timeout)
    if (llmProvider === 'ollama') {
      console.log(`[${new Date().toISOString()}] extract-triples: Processing with Ollama model: ${ollamaModel || 'llama3.1:8b'}`);
      const llmStartTime = Date.now();
      
      try {
        const model = ollamaModel || 'llama3.1:8b';
        const messages = [
          {
            role: 'system' as const,
            content: 'You are a knowledge graph builder. Extract subject-predicate-object triples from text and return them as a JSON array.'
          },
          {
            role: 'user' as const,
            content: `Extract triples from this text:\n\n${text}`
          }
        ];

        console.log(`[${new Date().toISOString()}] extract-triples: Calling llmService.generateOllamaCompletion directly`);
        const response = await llmService.generateOllamaCompletion(
          model,
          messages,
          { temperature: 0.1, maxTokens: 8192 }
        );

        const llmDuration = ((Date.now() - llmStartTime) / 1000).toFixed(2);
        console.log(`[${new Date().toISOString()}] extract-triples: LLM completion received after ${llmDuration}s, response length: ${response?.length || 0}`);

        // Parse the response to extract triples
        let triples = [];
        try {
          const jsonMatch = response.match(/\[[\s\S]*\]/);
          if (jsonMatch) {
            triples = JSON.parse(jsonMatch[0]);
          } else {
            // Fallback parser
            triples = parseTriplesFallback(response);
          }
        } catch (parseError) {
          console.warn('Failed to parse JSON response, using fallback parser:', parseError);
          triples = parseTriplesFallback(response);
        }

        const totalDuration = ((Date.now() - llmStartTime) / 1000).toFixed(2);
        console.log(`[${new Date().toISOString()}] extract-triples: Returning ${triples.length} triples, total duration: ${totalDuration}s`);

        return NextResponse.json({
          triples: triples.map((triple) => ({
            ...triple,
            confidence: 0.8,
            metadata: {
              entityTypes: [],
              source: text.substring(0, 100) + '...',
              context: text.substring(0, 200) + '...',
              extractionMethod: 'ollama',
              model: model
            }
          })),
          count: triples.length,
          success: true,
          method: 'ollama',
          model: model
        });
      } catch (llmError) {
        const llmDuration = ((Date.now() - llmStartTime) / 1000).toFixed(2);
        console.error(`[${new Date().toISOString()}] extract-triples: Ollama processing failed after ${llmDuration}s:`, llmError);
        throw llmError;
      }
    }

    // If vLLM is specified, use the vLLM API endpoint
    if (llmProvider === 'vllm') {
      const vllmResponse = await fetch(`${req.nextUrl.origin}/api/vllm`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          model: vllmModel || process.env.VLLM_MODEL || 'nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8',
          temperature: 0.1,
          maxTokens: 4096  // Reduced to leave room for input tokens in context
        })
      });

      if (!vllmResponse.ok) {
        throw new Error(`vLLM API error: ${vllmResponse.statusText}`);
      }

      const vllmResult = await vllmResponse.json();
      return NextResponse.json(vllmResult);
    }

    // Configure TextProcessor for the specified LLM provider
    const processor = TextProcessor.getInstance();
    if (llmProvider && ['ollama', 'nvidia', 'vllm'].includes(llmProvider)) {
      processor.setLLMProvider(llmProvider as 'ollama' | 'nvidia' | 'vllm', {
        ollamaModel: ollamaModel,
        ollamaBaseUrl: ollamaBaseUrl,
        vllmModel: vllmModel,
        vllmBaseUrl: vllmBaseUrl,
        nvidiaModel: nvidiaModel
      });
    }

    // Process the text to extract triples using either default pipeline or LangChain transformer
    // When both useLangChain and useGraphTransformer are true, use the GraphTransformer
    // When only useLangChain is true, use the default LangChain pipeline
    // Pass custom prompts if provided
    const options = {
      systemPrompt,
      extractionPrompt,
      graphTransformerPrompt
    };
    
    const triples = await processDocument(text, useLangChain, useGraphTransformer, options);

    // Return the extracted triples
    return NextResponse.json({
      triples,
      count: triples.length,
      success: true,
      method: useGraphTransformer 
        ? 'langchain_graphtransformer' 
        : useLangChain 
          ? 'langchain_default' 
          : 'standard_pipeline',
      llmProvider: processor.getLLMProvider(),
      customPromptUsed: !!(systemPrompt || extractionPrompt || graphTransformerPrompt)
    });
  } catch (error) {
    console.error('Error in triple extraction:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: `Failed to extract triples: ${errorMessage}` },
      { status: 500 }
    );
  }
}

// Helper function to parse triples from text when JSON parsing fails
function parseTriplesFallback(text: string): Array<{subject: string, predicate: string, object: string}> {
  const triples = [];
  const lines = text.split('\n');
  
  for (const line of lines) {
    // Look for patterns like "Subject - Predicate - Object" or similar
    const tripleMatch = line.match(/^[\s\-\*\d\.]*(.+?)\s*[\-\|]\s*(.+?)\s*[\-\|]\s*(.+)$/);
    if (tripleMatch) {
      triples.push({
        subject: tripleMatch[1].trim(),
        predicate: tripleMatch[2].trim(),
        object: tripleMatch[3].trim()
      });
    }
    
    // Also look for JSON-like objects in the text
    const jsonObjectMatch = line.match(/\{\s*"subject"\s*:\s*"([^"]+)"\s*,\s*"predicate"\s*:\s*"([^"]+)"\s*,\s*"object"\s*:\s*"([^"]+)"\s*\}/);
    if (jsonObjectMatch) {
      triples.push({
        subject: jsonObjectMatch[1],
        predicate: jsonObjectMatch[2],
        object: jsonObjectMatch[3]
      });
    }
  }
  
  return triples;
}

