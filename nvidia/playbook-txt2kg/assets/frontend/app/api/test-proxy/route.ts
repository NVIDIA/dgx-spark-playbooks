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

// Simple test endpoint to verify proxy connectivity
const REMOTE_WEBGPU_SERVICE_URL = process.env.REMOTE_WEBGPU_SERVICE_URL || 'http://txt2kg-remote-webgpu:8083'

export async function GET() {
  try {
    console.log(`Testing connection to: ${REMOTE_WEBGPU_SERVICE_URL}`)
    
    const response = await fetch(`${REMOTE_WEBGPU_SERVICE_URL}/health`)
    
    if (!response.ok) {
      throw new Error(`Service responded with ${response.status}: ${response.statusText}`)
    }
    
    const data = await response.json()
    
    return NextResponse.json({
      success: true,
      service_url: REMOTE_WEBGPU_SERVICE_URL,
      service_response: data
    })
    
  } catch (error) {
    console.error('Proxy test failed:', error)
    return NextResponse.json({
      success: false,
      error: String(error),
      service_url: REMOTE_WEBGPU_SERVICE_URL
    }, { status: 500 })
  }
}
