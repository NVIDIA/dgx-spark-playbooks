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

// Global flag to track if embeddings generation should be stopped
let shouldStopEmbeddings = false

// Function to check if embeddings generation should stop
export function getShouldStopEmbeddings(): boolean {
  return shouldStopEmbeddings
}

// Function to reset the stop flag
export function resetStopEmbeddings(): void {
  shouldStopEmbeddings = false
}

// Function to set the stop flag
export function setStopEmbeddings(): void {
  shouldStopEmbeddings = true
}

export async function POST(request: NextRequest) {
  try {
    console.log('Stop embeddings generation request received')
    
    // Set the global flag to stop embeddings generation
    shouldStopEmbeddings = true
    
    return NextResponse.json({ 
      success: true, 
      message: 'Embeddings generation stop signal sent' 
    })
  } catch (error) {
    console.error('Error stopping embeddings generation:', error)
    return NextResponse.json(
      { error: 'Failed to stop embeddings generation' },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  // Allow checking the current stop status
  return NextResponse.json({ 
    shouldStop: shouldStopEmbeddings 
  })
}
