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
import { llmService } from '@/lib/llm-service';

// Configure route for dynamic operations and long-running requests
export const dynamic = 'force-dynamic';
export const maxDuration = 1800; // 30 minutes for large model processing

/**
 * API endpoint for Ollama-specific operations
 * GET /api/ollama - Test connection and list available models
 * POST /api/ollama/extract-triples - Extract triples using Ollama model
 */

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const action = searchParams.get('action');

    if (action === 'test-connection') {
      const result = await llmService.testOllamaConnection();
      return NextResponse.json(result);
    }

    // Default: test connection and return models
    const result = await llmService.testOllamaConnection();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error in Ollama API:', error);
    return NextResponse.json(
      { 
        error: 'Failed to connect to Ollama server',
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  const startTime = Date.now();
  console.log(`[${new Date().toISOString()}] /api/ollama: POST request received`);
  
  try {
    const { text, model = 'qwen3:1.7b', temperature = 0.1, maxTokens = 4096 } = await req.json();
    console.log(`[${new Date().toISOString()}] /api/ollama: Parsed body - model: ${model}, text length: ${text?.length || 0}, maxTokens: ${maxTokens}`);

    if (!text || typeof text !== 'string') {
      return NextResponse.json({ error: 'Text is required' }, { status: 400 });
    }

    // Use the LLM service to generate completion with Ollama
    const messages = [
      {
        role: 'system' as const,
        content: `You are a knowledge graph builder that extracts structured information from text.
Extract subject-predicate-object triples from the following text.

Guidelines:
- Extract only factual triples present in the text
- Normalize entity names to their canonical form
- Return results in JSON format as an array of objects with "subject", "predicate", "object" fields
- Each triple should represent a clear relationship between two entities
- Focus on the most important relationships in the text`
      },
      {
        role: 'user' as const,
        content: `Extract triples from this text:\n\n${text}`
      }
    ];

    console.log(`[${new Date().toISOString()}] /api/ollama: Calling llmService.generateOllamaCompletion with model: ${model}`);
    const llmStartTime = Date.now();
    
    const response = await llmService.generateOllamaCompletion(
      model,
      messages,
      { temperature, maxTokens }
    );
    
    const llmDuration = ((Date.now() - llmStartTime) / 1000).toFixed(2);
    console.log(`[${new Date().toISOString()}] /api/ollama: LLM completion received after ${llmDuration}s, response length: ${response?.length || 0}`);

    // Parse the response to extract triples
    let triples = [];
    try {
      // Try to parse as JSON first
      const jsonMatch = response.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        triples = JSON.parse(jsonMatch[0]);
      } else {
        // Fallback: parse line by line
        triples = parseTriplesFallback(response);
      }
    } catch (parseError) {
      console.warn('Failed to parse JSON response, using fallback parser:', parseError);
      triples = parseTriplesFallback(response);
    }

    const totalDuration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`[${new Date().toISOString()}] /api/ollama: Returning ${triples.length} triples, total duration: ${totalDuration}s`);
    
    return NextResponse.json({
      triples: triples.map((triple, index) => ({
        ...triple,
        confidence: 0.8, // Default confidence for Ollama extractions
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
  } catch (error) {
    const totalDuration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.error(`[${new Date().toISOString()}] /api/ollama: Error after ${totalDuration}s:`, error);
    return NextResponse.json(
      { 
        error: 'Failed to extract triples with Ollama',
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}

// Fallback parser for when JSON parsing fails
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
