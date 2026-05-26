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

// Proxy route specifically for WebRTC streaming frames
// This handles binary image data streaming from the remote WebGPU service

const REMOTE_WEBGPU_SERVICE_URL = process.env.REMOTE_WEBGPU_SERVICE_URL || 'http://txt2kg-remote-webgpu:8083'

export async function GET(
  request: NextRequest,
  { params }: { params: { sessionId: string } }
) {
  try {
    const sessionId = params.sessionId
    const searchParams = request.nextUrl.searchParams.toString()
    const url = `${REMOTE_WEBGPU_SERVICE_URL}/api/stream/${sessionId}${searchParams ? `?${searchParams}` : ''}`
    
    console.log(`Proxying WebRTC stream request to: ${url}`)
    
    const response = await fetch(url, {
      method: 'GET',
    })
    
    if (!response.ok) {
      throw new Error(`Remote WebGPU service responded with ${response.status}: ${response.statusText}`)
    }
    
    // Get the image data as array buffer
    const imageBuffer = await response.arrayBuffer()
    const contentType = response.headers.get('content-type') || 'image/png'
    
    // Return the image with proper headers
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
      },
    })
    
  } catch (error) {
    console.error('WebRTC stream proxy error:', error)
    return NextResponse.json(
      { error: 'Failed to stream from remote WebGPU service', details: String(error) },
      { status: 500 }
    )
  }
}
