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
import { LLMService } from '@/lib/llm-service';

const llmService = LLMService.getInstance();

/**
 * Test vLLM connection and list available models
 * GET /api/vllm?action=test-connection
 */
export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const action = searchParams.get('action');

  if (action === 'test-connection') {
    try {
      const vllmBaseUrl = process.env.VLLM_BASE_URL || 'http://localhost:8001/v1';
      
      // Test connection to vLLM service using built-in /v1/models endpoint
      const response = await fetch(`${vllmBaseUrl}/models`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(10000),
      });

      if (!response.ok) {
        throw new Error(`vLLM service returned ${response.status}: ${response.statusText}`);
      }

      const healthData = { 
        status: "healthy", 
        service: "vllm",
        note: "Using vLLM's built-in OpenAI API server"
      };
      
      // Get available models (reuse the response from health check)
      const modelsData = await response.json();
      const models = modelsData.data?.map((model: any) => model.id) || [];

      return NextResponse.json({
        connected: true,
        health: healthData,
        models: models,
        baseUrl: vllmBaseUrl
      });

    } catch (error) {
      console.error('vLLM connection test failed:', error);
      return NextResponse.json(
        { 
          connected: false, 
          error: error instanceof Error ? error.message : String(error),
          baseUrl: process.env.VLLM_BASE_URL || 'http://localhost:8001/v1'
        },
        { status: 503 }
      );
    }
  }

  return NextResponse.json(
    { error: 'Invalid action parameter' },
    { status: 400 }
  );
}

/**
 * Extract triples using vLLM
 * POST /api/vllm
 */
export async function POST(req: NextRequest) {
  try {
    const { text, model = process.env.VLLM_MODEL || 'nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8', temperature = 0.1, maxTokens = 1024 } = await req.json();

    if (!text || typeof text !== 'string') {
      return NextResponse.json({ error: 'Text is required' }, { status: 400 });
    }

    // Use the LLM service to generate completion with vLLM
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

    // Use LLMService for direct chat completion via vLLM's OpenAI API
    const response = await llmService.generateVllmCompletion(
      model,
      messages,
      { temperature, maxTokens }
    );

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

    return NextResponse.json({
      triples: triples,
      model: model,
      provider: 'vllm',
      rawResponse: response
    });

  } catch (error) {
    console.error('Error in vLLM triple extraction:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}

/**
 * Fallback parser for extracting triples from text response
 */
function parseTriplesFallback(text: string): Array<{ subject: string; predicate: string; object: string }> {
  const triples = [];
  const lines = text.split('\n');

  for (const line of lines) {
    const trimmedLine = line.trim();
    if (!trimmedLine || trimmedLine.startsWith('#') || trimmedLine.startsWith('//')) {
      continue;
    }

    // Try to parse different formats
    if (trimmedLine.includes(' -> ')) {
      const parts = trimmedLine.split(' -> ');
      if (parts.length >= 3) {
        triples.push({
          subject: parts[0].trim(),
          predicate: parts[1].trim(),
          object: parts[2].trim()
        });
      }
    } else if (trimmedLine.includes('|')) {
      const parts = trimmedLine.split('|');
      if (parts.length >= 3) {
        triples.push({
          subject: parts[0].trim(),
          predicate: parts[1].trim(),
          object: parts[2].trim()
        });
      }
    } else if (trimmedLine.includes(',')) {
      // Try comma-separated format: "subject, predicate, object"
      const parts = trimmedLine.split(',');
      if (parts.length >= 3) {
        triples.push({
          subject: parts[0].trim().replace(/['"]/g, ''),
          predicate: parts[1].trim().replace(/['"]/g, ''),
          object: parts[2].trim().replace(/['"]/g, '')
        });
      }
    }
  }

  return triples;
}
