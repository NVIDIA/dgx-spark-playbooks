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
import neo4jService from '@/lib/neo4j';

/**
 * Simple endpoint to directly add a query log with a high count
 */
export async function GET(request: NextRequest) {
  try {
    // Get the query text from URL params or use a default
    const query = request.nextUrl.searchParams.get('query') || 'How does machine learning work?';
    const count = parseInt(request.nextUrl.searchParams.get('count') || '20');
    
    // Initialize Neo4j
    if (!neo4jService.isInitialized()) {
      neo4jService.initialize();
    }
    
    // Execute direct Cypher query to create a query log with a high count
    const session = neo4jService.getSession();
    
    try {
      const cypher = `
        MERGE (q:QueryLog {query: $query})
        ON CREATE SET 
          q.firstQueried = datetime(),
          q.count = $count
        ON MATCH SET 
          q.lastQueried = datetime(),
          q.count = $count
        
        CREATE (e:QueryExecution {
          timestamp: datetime(),
          queryMode: 'traditional',
          executionTimeMs: 0,
          relevanceScore: 0,
          precision: 0,
          recall: 0,
          resultCount: 0
        })
        
        CREATE (q)-[:HAS_EXECUTION]->(e)
        
        RETURN q.query as query, q.count as count
      `;
      
      const result = await session.run(cypher, { 
        query, 
        count 
      });
      
      const addedQuery = result.records.length > 0 ? {
        query: result.records[0].get('query'),
        count: result.records[0].get('count').toNumber()
      } : null;
      
      // Also add a few more queries
      if (count >= 10) {
        await session.run(cypher, { 
          query: 'What are the applications of artificial intelligence?', 
          count: count - 4 
        });
        
        await session.run(cypher, { 
          query: 'Explain the principles of deep learning', 
          count: count - 8 
        });
      }
      
      // Get the current logs to verify
      const logs = await neo4jService.getQueryLogs(5);
      
      return NextResponse.json({
        success: true,
        message: `Added query log for "${query}" with count ${count}`,
        addedQuery,
        logs
      });
    } finally {
      session.close();
    }
  } catch (error) {
    console.error('Error adding query log:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : String(error)
    }, { status: 500 });
  }
} 