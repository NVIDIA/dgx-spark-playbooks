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
import fs from 'fs';
import path from 'path';
import { promises as fsPromises } from 'fs';

export interface QueryLogEntry {
  query: string;
  queryMode: 'traditional' | 'vector-search' | 'pure-rag';
  timestamp: string;
  metrics: {
    executionTimeMs: number;
    relevanceScore?: number;
    precision?: number;
    recall?: number;
    resultCount: number;
  };
}

export interface QueryLogSummary {
  query: string;
  queryMode: 'traditional' | 'vector-search' | 'pure-rag';
  count: number;
  firstQueried: string;
  lastQueried: string;
  metrics: {
    avgExecutionTimeMs: number;
    avgRelevanceScore: number;
    avgPrecision: number;
    avgRecall: number;
    avgResultCount: number;
  };
  executionCount: number;
}

/**
 * Service for logging queries to a file
 */
export class QueryLoggerService {
  private static instance: QueryLoggerService;
  private logFilePath: string;
  private initialized: boolean = false;
  
  private constructor() {
    // Default path is in the data directory of the project
    this.logFilePath = path.join(process.cwd(), 'data', 'query-logs.json');
  }
  
  /**
   * Get the singleton instance of the QueryLoggerService
   */
  public static getInstance(): QueryLoggerService {
    if (!QueryLoggerService.instance) {
      QueryLoggerService.instance = new QueryLoggerService();
    }
    return QueryLoggerService.instance;
  }

  /**
   * Initialize the logger
   * @param customPath Optional custom path for log file
   */
  public async initialize(customPath?: string): Promise<void> {
    try {
      if (customPath) {
        this.logFilePath = customPath;
      }

      // Ensure the directory exists
      const dir = path.dirname(this.logFilePath);
      await fsPromises.mkdir(dir, { recursive: true });
      
      // Check if file exists, create it if it doesn't
      if (!fs.existsSync(this.logFilePath)) {
        await fsPromises.writeFile(this.logFilePath, JSON.stringify([]));
        console.log(`[QueryLogger] Initialized empty log file at ${this.logFilePath}`);
      } else {
        console.log(`[QueryLogger] Using existing log file at ${this.logFilePath}`);
      }
      
      this.initialized = true;
    } catch (error) {
      console.error('[QueryLogger] Error initializing logger:', error);
      throw error;
    }
  }

  /**
   * Log a RAG query with its performance metrics
   */
  public async logQuery(
    query: string,
    queryMode: 'traditional' | 'vector-search' | 'pure-rag',
    metrics: {
      executionTimeMs: number;
      relevanceScore?: number;
      precision?: number;
      recall?: number;
      resultCount: number;
    }
  ): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }

    console.log(`[QueryLogger] Logging query: "${query}" (${queryMode})`);
    
    try {
      // Read existing logs
      const existingLogsRaw = await fsPromises.readFile(this.logFilePath, 'utf-8');
      const existingLogs: QueryLogEntry[] = JSON.parse(existingLogsRaw || '[]');
      
      // Add new log entry
      const newEntry: QueryLogEntry = {
        query,
        queryMode,
        timestamp: new Date().toISOString(),
        metrics
      };
      
      existingLogs.push(newEntry);
      
      // Write updated logs back to file
      await fsPromises.writeFile(this.logFilePath, JSON.stringify(existingLogs, null, 2));
      console.log(`[QueryLogger] Query logged successfully to file`);
    } catch (error) {
      console.error('[QueryLogger] Error logging query to file:', error);
      // Non-critical error, so just log it but don't throw
    }
  }

  /**
   * Get query logs with performance metrics
   * @param limit Maximum number of query logs to return
   * @returns Promise resolving to an array of query logs
   */
  public async getQueryLogs(limit: number = 100): Promise<QueryLogSummary[]> {
    if (!this.initialized) {
      await this.initialize();
    }

    console.log(`[QueryLogger] Getting query logs with limit: ${limit}`);
    
    try {
      // Read logs from file
      const logsRaw = await fsPromises.readFile(this.logFilePath, 'utf-8');
      const logs: QueryLogEntry[] = JSON.parse(logsRaw || '[]');
      
      if (logs.length === 0) {
        console.log('[QueryLogger] No query logs found');
        return [];
      }

      // Group logs by query AND queryMode
      const querySummaries = new Map<string, {
        query: string;
        queryMode: 'traditional' | 'vector-search' | 'pure-rag';
        count: number;
        timestamps: string[];
        executionTimes: number[];
        relevanceScores: number[];
        precisions: number[];
        recalls: number[];
        resultCounts: number[];
      }>();

      logs.forEach(entry => {
        // Use composite key: query + mode
        const key = `${entry.query}|||${entry.queryMode}`;
        const existing = querySummaries.get(key) || {
          query: entry.query,
          queryMode: entry.queryMode,
          count: 0,
          timestamps: [],
          executionTimes: [],
          relevanceScores: [],
          precisions: [],
          recalls: [],
          resultCounts: []
        };

        existing.count++;
        existing.timestamps.push(entry.timestamp);
        existing.executionTimes.push(entry.metrics.executionTimeMs);
        if (entry.metrics.relevanceScore !== undefined) existing.relevanceScores.push(entry.metrics.relevanceScore);
        if (entry.metrics.precision !== undefined) existing.precisions.push(entry.metrics.precision);
        if (entry.metrics.recall !== undefined) existing.recalls.push(entry.metrics.recall);
        existing.resultCounts.push(entry.metrics.resultCount);

        querySummaries.set(key, existing);
      });

      // Convert to array and format
      const result: QueryLogSummary[] = Array.from(querySummaries.values()).map(summary => ({
        query: summary.query,
        queryMode: summary.queryMode,
        count: summary.count,
        firstQueried: summary.timestamps[0],
        lastQueried: summary.timestamps[summary.timestamps.length - 1],
        metrics: {
          avgExecutionTimeMs: this.calculateAverage(summary.executionTimes),
          avgRelevanceScore: this.calculateAverage(summary.relevanceScores),
          avgPrecision: this.calculateAverage(summary.precisions),
          avgRecall: this.calculateAverage(summary.recalls),
          avgResultCount: this.calculateAverage(summary.resultCounts)
        },
        executionCount: summary.count
      }));

      // Sort by count (descending) and limit
      return result
        .sort((a, b) => b.count - a.count)
        .slice(0, limit);
    } catch (error) {
      console.error('[QueryLogger] Error getting query logs:', error);
      throw error;
    }
  }

  /**
   * Calculate average of an array of numbers
   * @param values Array of numbers
   * @returns Average value or 0 if array is empty
   */
  private calculateAverage(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  /**
   * Check if the logger is initialized
   */
  public isInitialized(): boolean {
    return this.initialized;
  }
}

// Create and export singleton instance
const queryLoggerService = QueryLoggerService.getInstance();
export default queryLoggerService; 