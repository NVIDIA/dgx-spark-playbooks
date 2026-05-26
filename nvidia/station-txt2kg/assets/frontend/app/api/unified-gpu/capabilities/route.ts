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
import { NextRequest, NextResponse } from 'next/server'

const UNIFIED_GPU_SERVICE_URL = process.env.UNIFIED_GPU_SERVICE_URL || 'http://localhost:8080'

export async function GET(request: NextRequest) {
  try {
    // Forward the request to the unified GPU service
    const response = await fetch(`${UNIFIED_GPU_SERVICE_URL}/api/capabilities`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Unified GPU service error:', errorText)
      return NextResponse.json(
        { 
          error: 'Unified GPU service error', 
          details: errorText 
        },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
    
  } catch (error) {
    console.error('Error forwarding to unified GPU service:', error)
    return NextResponse.json(
      { 
        error: 'Failed to connect to unified GPU service',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    )
  }
} 