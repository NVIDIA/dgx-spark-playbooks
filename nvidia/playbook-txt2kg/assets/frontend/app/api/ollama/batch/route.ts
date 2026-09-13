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
import { llmService, LLMMessage } from '@/lib/llm-service';

/**
 * API endpoint for batch Ollama operations
 * POST /api/ollama/batch - Process multiple texts in batch for triple extraction
 */

interface BatchTripleRequest {
  texts: string[];
  model?: string;
  temperature?: number;
  maxTokens?: number;
  concurrency?: number;
}

export async function POST(req: NextRequest) {
  try {
    const { 
      texts, 
      model = 'qwen3:1.7b', 
      temperature = 0.1, 
      maxTokens = 8192,
      concurrency = 5
    }: BatchTripleRequest = await req.json();

    if (!texts || !Array.isArray(texts) || texts.length === 0) {
      return NextResponse.json({ 
        error: 'Texts array is required and must not be empty' 
      }, { status: 400 });
    }

    if (texts.length > 100) {
      return NextResponse.json({ 
        error: 'Batch size limited to 100 texts maximum' 
      }, { status: 400 });
    }

    // Validate all texts are strings
    const invalidTexts = texts.filter(text => !text || typeof text !== 'string');
    if (invalidTexts.length > 0) {
      return NextResponse.json({ 
        error: `Invalid texts found at indices: ${texts.map((text, i) => 
          (!text || typeof text !== 'string') ? i : null
        ).filter(i => i !== null).join(', ')}` 
      }, { status: 400 });
    }

    console.log(`Starting batch triple extraction for ${texts.length} texts using model ${model}`);

    // Create system prompt for triple extraction
    const systemPrompt = `You are a knowledge graph builder that extracts structured information from text.
Extract subject-predicate-object triples from the following text.

Guidelines:
- Extract only factual triples present in the text
- Normalize entity names to their canonical form
- Return results in JSON format as an array of objects with "subject", "predicate", "object" fields
- Each triple should represent a clear relationship between two entities
- Focus on the most important relationships in the text`;

    // Prepare batch messages
    const messagesBatch: LLMMessage[][] = texts.map(text => [
      {
        role: 'system' as const,
        content: systemPrompt
      },
      {
        role: 'user' as const,
        content: `Extract triples from this text:\n\n${text}`
      }
    ]);

    // Process batch with Ollama
    const batchResult = await llmService.generateOllamaBatchCompletion(
      model,
      messagesBatch,
      { temperature, maxTokens, concurrency }
    );

    // Parse responses to extract triples
    const processedResults = batchResult.results.map((response, index) => {
      let triples = [];
      
      if (response) {
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
          console.warn(`Failed to parse response for text ${index}:`, parseError);
          triples = parseTriplesFallback(response);
        }
      }

      return {
        textIndex: index,
        originalText: texts[index].substring(0, 200) + (texts[index].length > 200 ? '...' : ''),
        triples: triples.map((triple: any, tripleIndex: number) => ({
          ...triple,
          confidence: 0.8, // Default confidence for Ollama extractions
          metadata: {
            entityTypes: [],
            source: texts[index].substring(0, 100) + '...',
            context: texts[index].substring(0, 200) + '...',
            extractionMethod: 'ollama_batch',
            model: model,
            textIndex: index,
            tripleIndex: tripleIndex
          }
        })),
        tripleCount: triples.length,
        success: !batchResult.errors.some(error => error.index === index)
      };
    });

    // Calculate summary statistics
    const totalTriples = processedResults.reduce((sum, result) => sum + result.tripleCount, 0);
    const successfulTexts = processedResults.filter(result => result.success).length;

    return NextResponse.json({
      results: processedResults,
      summary: {
        totalTexts: texts.length,
        successfulTexts: successfulTexts,
        failedTexts: batchResult.errors.length,
        totalTriples: totalTriples,
        averageTriples: successfulTexts > 0 ? (totalTriples / successfulTexts).toFixed(2) : 0
      },
      batchInfo: {
        model: model,
        concurrency: concurrency,
        processingTime: Date.now(), // Could be enhanced with actual timing
        method: 'ollama_batch'
      },
      errors: batchResult.errors,
      success: true
    });
  } catch (error) {
    console.error('Error in Ollama batch triple extraction:', error);
    return NextResponse.json(
      { 
        error: 'Failed to process batch triple extraction with Ollama',
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}

// Fallback parser for when JSON parsing fails (reused from single endpoint)
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
