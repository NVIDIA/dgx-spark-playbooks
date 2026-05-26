//
// SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
import { NextResponse } from 'next/server';

/**
 * Fetch available models from vLLM
 * GET /api/vllm/models
 */
export async function GET() {
  const vllmUrl = process.env.VLLM_BASE_URL || 'http://vllm:8001/v1';
  
  try {
    const response = await fetch(`${vllmUrl}/models`, {
      signal: AbortSignal.timeout(5000),
    });
    
    if (!response.ok) {
      return NextResponse.json({ models: [] }, { status: 200 });
    }
    
    const data = await response.json();
    
    // vLLM returns OpenAI-compatible format: { data: [{ id: "model-name", ... }] }
    if (data.data && Array.isArray(data.data)) {
      const models = data.data.map((model: any) => ({
        id: model.id,
        name: model.id,
      }));
      return NextResponse.json({ models });
    }
    
    return NextResponse.json({ models: [] });
  } catch (error) {
    // Return empty models array if vLLM is not available
    return NextResponse.json({ models: [] }, { status: 200 });
  }
}

