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
import remoteBackend from '@/lib/remote-backend';
import type { Triple } from '@/types/graph';
import { getGraphDbType } from '../settings/route';

/**
 * Remote backend API that provides endpoints for creating and querying a knowledge graph
 * using the selected graph database, Qdrant, and SentenceTransformer
 */

/**
 * Create a backend from triples
 */
export async function POST(request: NextRequest) {
  try {
    const { triples } = await request.json();
    
    if (!triples || !Array.isArray(triples) || triples.length === 0) {
      return NextResponse.json(
        { error: 'Triples are required and must be a non-empty array' }, 
        { status: 400 }
      );
    }
    
    // Initialize backend with the selected graph database type
    if (!remoteBackend.isInitialized()) {
      const graphDbType = getGraphDbType();
      console.log(`Initializing backend with graph DB type: ${graphDbType}`);
      await remoteBackend.initialize(graphDbType);
    }
    
    // Create backend from triples
    await remoteBackend.createBackendFromTriples(triples);
    
    return NextResponse.json({
      success: true,
      message: `Created backend successfully with ${triples.length} triples`,
      graphDbType: getGraphDbType()
    });
  } catch (error) {
    console.error('Error creating backend from triples:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}

/**
 * Query the backend with a given query text
 */
export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const query = url.searchParams.get('query');
    
    if (!query) {
      return NextResponse.json({ error: 'Query parameter is required' }, { status: 400 });
    }
    
    // Parse optional parameters with fallbacks
    const kNeighbors = parseInt(url.searchParams.get('kNeighbors') || '4096', 10);
    const fanout = parseInt(url.searchParams.get('fanout') || '400', 10);
    const numHops = parseInt(url.searchParams.get('numHops') || '2', 10);
    
    // Initialize backend with the selected graph database type
    if (!remoteBackend.isInitialized()) {
      const graphDbType = getGraphDbType();
      console.log(`Initializing backend with graph DB type: ${graphDbType}`);
      await remoteBackend.initialize(graphDbType);
    }
    
    // Query the backend
    const relevantTriples = await remoteBackend.query(query, kNeighbors, fanout, numHops);
    
    return NextResponse.json({
      query,
      triples: relevantTriples,
      count: relevantTriples.length,
      parameters: {
        kNeighbors,
        fanout,
        numHops
      },
      graphDbType: getGraphDbType()
    });
  } catch (error) {
    console.error('Error querying backend:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
} 