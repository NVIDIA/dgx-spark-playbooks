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
import { Neo4jService } from '@/lib/neo4j';

// Initialize Neo4j service
const neo4jService = Neo4jService.getInstance();

// Initialize connection on first request
let isInitialized = false;

/**
 * Initialize Neo4j connection if not already initialized
 * @param request Optional request containing connection parameters
 */
function ensureConnection(request?: NextRequest) {
  try {
    let uri = process.env.NEO4J_URI;
    let username = process.env.NEO4J_USER;
    let password = process.env.NEO4J_PASSWORD;

    // Override with URL parameters if provided
    if (request) {
      const params = request.nextUrl.searchParams;
      if (params.has('url')) uri = params.get('url') as string;
      if (params.has('username')) username = params.get('username') as string;
      if (params.has('password')) password = params.get('password') as string;
    }

    // Connect to Neo4j instance
    neo4jService.initialize(uri, username, password);
    isInitialized = true;
  } catch (error) {
    console.error('Failed to initialize Neo4j connection:', error);
    throw error;
  }
}

/**
 * Legacy Neo4j endpoint - redirects to the new graph-db endpoint
 * @deprecated Use /api/graph-db instead with type=neo4j
 */
export async function GET(request: NextRequest) {
  console.log('Redirecting from deprecated /api/neo4j to /api/graph-db?type=neo4j');
  
  // Create the new URL with the same query parameters
  const url = new URL(request.url);
  const newUrl = new URL('/api/graph-db', url.origin);
  
  // Copy all query parameters
  url.searchParams.forEach((value, key) => {
    newUrl.searchParams.append(key, value);
  });
  
  // Add Neo4j type parameter if not present
  if (!newUrl.searchParams.has('type')) {
    newUrl.searchParams.append('type', 'neo4j');
  }
  
  // Return a redirect response
  return NextResponse.redirect(newUrl);
}

/**
 * Legacy Neo4j POST endpoint - redirects to the new graph-db endpoint with a type parameter
 * @deprecated Use /api/graph-db instead with type=neo4j
 */
export async function POST(request: NextRequest) {
  console.log('Redirecting from deprecated /api/neo4j to /api/graph-db?type=neo4j');
  
  // Create the new URL with the neo4j type parameter
  const url = new URL(request.url);
  const newUrl = new URL('/api/graph-db', url.origin);
  
  // Copy all query parameters
  url.searchParams.forEach((value, key) => {
    newUrl.searchParams.append(key, value);
  });
  
  // Add Neo4j type parameter if not present
  if (!newUrl.searchParams.has('type')) {
    newUrl.searchParams.append('type', 'neo4j');
  }
  
  // Clone the request with the new URL
  const newRequest = new Request(newUrl, {
    method: request.method,
    headers: request.headers,
    body: request.body,
    cache: request.cache,
    credentials: request.credentials,
    integrity: request.integrity,
    keepalive: request.keepalive,
    mode: request.mode,
    redirect: request.redirect,
    referrer: request.referrer,
    referrerPolicy: request.referrerPolicy,
    signal: request.signal,
    duplex: 'half',
  } as RequestInit);
  
  // Fetch from the new endpoint
  const response = await fetch(newRequest);
  
  // Return the response
  return response;
}