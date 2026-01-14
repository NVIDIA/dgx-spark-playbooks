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
import remoteBackendInstance from '@/lib/remote-backend';
import { getGraphDbService } from '@/lib/graph-db-util';
import { getGraphDbType } from '../settings/route';
import { QdrantService } from '@/lib/qdrant';
import RAGService from '@/lib/rag';
import queryLoggerService, { QueryLogSummary } from '@/lib/query-logger';

/**
 * Metrics API that provides performance statistics about the RAG system
 */
export async function GET(request: NextRequest) {
  try {
    // Initialize services with the correct graph database type
    const graphDbType = getGraphDbType();
    const graphDbService = getGraphDbService(graphDbType);
    const qdrantService = QdrantService.getInstance();
    
    // Initialize graph database if needed
    if (!graphDbService.isInitialized()) {
      if (graphDbType === 'arangodb') {
        await graphDbService.initialize(
          process.env.ARANGODB_URL,
          process.env.ARANGODB_DB,
          process.env.ARANGODB_USER,
          process.env.ARANGODB_PASSWORD
        );
      } else if (graphDbType === 'neo4j') {
        graphDbService.initialize(
          process.env.NEO4J_URI,
          process.env.NEO4J_USER || process.env.NEO4J_USERNAME,
          process.env.NEO4J_PASSWORD
        );
      }
    }
    
    // Get graph stats from the active graph database
    const graphData = await graphDbService.getGraphData();
    
    // Get unique entities (nodes)
    const uniqueEntities = new Set<string>();
    graphData.nodes.forEach((node: any) => uniqueEntities.add(node.name));
    
    // Get total triples (relationships)
    const totalTriples = graphData.relationships.length;
    
    // Get vector stats from Qdrant if available
    let vectorStats = {
      totalVectors: 0,
      avgQueryTime: 0,
      avgRelevanceScore: 0
    };
    
    try {
      await qdrantService.initialize();
      const stats = await qdrantService.getStats();
      
      vectorStats = {
        totalVectors: stats.totalVectorCount || 0,
        avgQueryTime: stats.averageQueryTime || 0,
        avgRelevanceScore: stats.averageRelevanceScore || 0
      };
    } catch (error) {
      console.warn('Could not fetch Qdrant stats:', error);
    }

    // Get real query logs instead of mock data
    let queryLogs: QueryLogSummary[] = [];
    let precision = 0;
    let recall = 0;
    let f1Score = 0;
    let avgQueryTime = vectorStats.avgQueryTime || 0;
    let avgRelevance = 0;
    let queryTimesByMode: Record<string, number> = {};

    // Get query logs from file-based logger instead of Neo4j
    try {
      // Initialize query logger if needed
      if (!queryLoggerService.isInitialized()) {
        await queryLoggerService.initialize();
      }
      
      // Get the logs
      console.log('Getting query logs from file');
      queryLogs = await queryLoggerService.getQueryLogs(25);
      console.log(`Found ${queryLogs.length} query logs from file-based logger`);
      
      // Calculate metrics from the query logs
      if (queryLogs.length > 0) {
        // Calculate metrics from logs with actual data
        const logsWithMetrics = queryLogs.filter(log =>
          log.metrics.avgPrecision > 0 ||
          log.metrics.avgRecall > 0 ||
          log.metrics.avgExecutionTimeMs > 0
        );

        const logsWithRelevance = queryLogs.filter(log => log.metrics.avgRelevanceScore > 0);

        if (logsWithMetrics.length > 0) {
          precision = logsWithMetrics.reduce((sum, log) => sum + (log.metrics.avgPrecision || 0), 0) / logsWithMetrics.length;
          recall = logsWithMetrics.reduce((sum, log) => sum + (log.metrics.avgRecall || 0), 0) / logsWithMetrics.length;
          avgQueryTime = logsWithMetrics.reduce((sum, log) => sum + (log.metrics.avgExecutionTimeMs || 0), 0) / logsWithMetrics.length;
          f1Score = precision > 0 && recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
        }

        if (logsWithRelevance.length > 0) {
          avgRelevance = logsWithRelevance.reduce((sum, log) => sum + (log.metrics.avgRelevanceScore || 0), 0) / logsWithRelevance.length;
        }
      }

      // Calculate per-mode query times
      const logsByMode = queryLogs.reduce((acc, log) => {
        const mode = log.queryMode || 'traditional';
        if (!acc[mode]) acc[mode] = [];
        acc[mode].push(log);
        return acc;
      }, {} as Record<string, typeof queryLogs>);

      Object.entries(logsByMode).forEach(([mode, logs]) => {
        const logsWithTime = logs.filter(log => log.metrics.avgExecutionTimeMs > 0);
        if (logsWithTime.length > 0) {
          queryTimesByMode[mode] = logsWithTime.reduce((sum, log) =>
            sum + log.metrics.avgExecutionTimeMs, 0
          ) / logsWithTime.length;
        }
      });
    } catch (error) {
      console.warn('Error getting query logs from file:', error);
      // Keep values at 0 instead of using defaults
    }
    
    // Get top queries from real logs
    const topQueries = queryLogs.length > 0 
      ? queryLogs
          .sort((a, b) => b.count - a.count)
          .slice(0, 5)
          .map(log => ({ 
            query: log.query, 
            count: log.count 
          }))
      : [];
    
    // Aggregate metrics
    const metrics = {
      totalTriples,
      totalEntities: uniqueEntities.size,
      avgQueryTime,
      avgRelevance: avgRelevance || vectorStats.avgRelevanceScore || 0, // Use query log relevance score, fallback to vector stats
      precision,
      recall,
      f1Score,
      topQueries,
      queryTimesByMode: queryTimesByMode || {},
      // Add metadata about query logs
      queryLogStats: {
        totalQueryLogs: queryLogs.length,
        totalExecutions: queryLogs.reduce((sum, log) => sum + log.executionCount, 0),
        lastQueriedAt: queryLogs.length > 0 ? queryLogs[0].lastQueried : null
      }
    };
    
    return NextResponse.json(metrics);
  } catch (error) {
    console.error('Error fetching metrics:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}

/**
 * Function to calculate precision and recall has been replaced by real data from query logs
 */ 