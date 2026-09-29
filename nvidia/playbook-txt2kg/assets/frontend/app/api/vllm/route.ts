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

type Triple = {
  subject: string;
  predicate: string;
  object: string;
};

const triplesResponseFormat = {
  type: 'json_schema',
  json_schema: {
    name: 'knowledge_graph_triples',
    strict: true,
    schema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        triples: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              subject: { type: 'string' },
              predicate: { type: 'string' },
              object: { type: 'string' }
            },
            required: ['subject', 'predicate', 'object']
          }
        }
      },
      required: ['triples']
    }
  }
};

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
    const { text, model = process.env.VLLM_MODEL || 'nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8', temperature = 0, maxTokens = 4096 } = await req.json();

    if (!text || typeof text !== 'string') {
      return NextResponse.json({ error: 'Text is required' }, { status: 400 });
    }

    const isNemotronReasoningModel = /nemotron/i.test(model);

    // Use the LLM service to generate completion with vLLM
    const messages = [
      {
        role: 'system' as const,
        content: isNemotronReasoningModel
          ? 'detailed thinking off'
          : 'You are a knowledge graph builder. Return only valid JSON matching the requested schema.'
      },
      {
        role: 'user' as const,
        content: `Extract subject-predicate-object triples from the text below.

Guidelines:
- Extract only factual triples present in the text.
- Normalize entity names to their canonical form.
- Each triple must represent a clear relationship between two entities.
- Focus on the most important relationships in the text.
- Return only a JSON object with this shape: {"triples":[{"subject":"...","predicate":"...","object":"..."}]}.
- Do not include explanations, markdown, or reasoning text.

Text:
${text}`
      }
    ];

    // Use LLMService for direct chat completion via vLLM's OpenAI API
    const response = await llmService.generateVllmCompletion(
      model,
      messages,
      {
        temperature,
        maxTokens,
        topP: 1,
        responseFormat: triplesResponseFormat
      }
    );

    const triples = parseTriplesResponse(response);

    return NextResponse.json({
      triples: triples,
      model: model,
      provider: 'vllm',
      rawResponse: JSON.stringify({ triples })
    });

  } catch (error) {
    console.error('Error in vLLM triple extraction:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 502 }
    );
  }
}

function parseTriplesResponse(response: string): Triple[] {
  const parsed = parseJsonPayload(response);
  const triples = Array.isArray(parsed) ? parsed : parsed?.triples;

  if (!Array.isArray(triples)) {
    throw new Error('vLLM returned JSON, but it did not contain a triples array.');
  }

  const validTriples = triples
    .map((triple: unknown) => normalizeTriple(triple))
    .filter((triple): triple is Triple => triple !== null);

  if (validTriples.length === 0 && triples.length > 0) {
    throw new Error('vLLM returned triples, but none had valid subject, predicate, and object fields.');
  }

  return validTriples;
}

function parseJsonPayload(response: string): any {
  const trimmedResponse = stripMarkdownFence(response.trim());
  const objectMatch = trimmedResponse.match(/\{[\s\S]*\}/);
  const arrayMatch = trimmedResponse.match(/\[[\s\S]*\]/);
  const candidates = [
    trimmedResponse,
    objectMatch?.[0],
    arrayMatch?.[0]
  ].filter((candidate): candidate is string => typeof candidate === 'string');

  for (const candidate of candidates) {
    try {
      return JSON.parse(candidate);
    } catch {
      // Try the next candidate before failing closed.
    }
  }

  throw new Error('vLLM did not return valid JSON triples. Refusing to fallback-parse free-form text to avoid storing reasoning output as graph entities.');
}

function stripMarkdownFence(text: string): string {
  return text
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();
}

function normalizeTriple(triple: unknown): Triple | null {
  if (!triple || typeof triple !== 'object') {
    return null;
  }

  const candidate = triple as Record<string, unknown>;
  const subject = normalizeTripleField(candidate.subject);
  const predicate = normalizeTripleField(candidate.predicate);
  const object = normalizeTripleField(candidate.object);

  if (!subject || !predicate || !object) {
    return null;
  }

  return { subject, predicate, object };
}

function normalizeTripleField(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null;
  }

  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}
