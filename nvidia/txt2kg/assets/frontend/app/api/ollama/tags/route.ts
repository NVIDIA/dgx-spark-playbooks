//
// SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
import { NextResponse } from 'next/server';

/**
 * Fetch available models from Ollama
 * GET /api/ollama/tags
 */
export async function GET() {
  const ollamaUrl = process.env.OLLAMA_BASE_URL || 'http://ollama:11434/v1';
  // Convert /v1 URL to base URL for tags endpoint
  const baseUrl = ollamaUrl.replace('/v1', '');
  
  try {
    const response = await fetch(`${baseUrl}/api/tags`, {
      signal: AbortSignal.timeout(5000),
    });
    
    if (!response.ok) {
      return NextResponse.json({ models: [] }, { status: 200 });
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    // Return empty models array if Ollama is not available
    return NextResponse.json({ models: [] }, { status: 200 });
  }
}

