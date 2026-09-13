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
import { NextResponse } from "next/server";

export async function GET() {
  // Only return the necessary configuration data
  return NextResponse.json({
    nvidiaApiKey: process.env.NVIDIA_API_KEY || null,
    // xaiApiKey removed - integration has been removed
    ollamaBaseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1',
    ollamaModel: process.env.OLLAMA_MODEL || 'qwen3:1.7b',
    vllmBaseUrl: process.env.VLLM_BASE_URL || 'http://localhost:8001/v1',
    vllmModel: process.env.VLLM_MODEL || 'meta-llama/Llama-3.2-3B-Instruct',
    // Add other config values as needed
  });
} 