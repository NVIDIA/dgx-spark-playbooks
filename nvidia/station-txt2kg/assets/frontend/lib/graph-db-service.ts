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
import { Neo4jService } from './neo4j';
import { ArangoDBService } from './arangodb';
import type { Triple } from '@/types/graph';

export type GraphDBType = 'neo4j' | 'arangodb';

/**
 * GraphDBService that provides a unified interface for different graph database implementations
 */
export class GraphDBService {
  private neo4jService: Neo4jService;
  private arangoDBService: ArangoDBService;
  private activeDBType: GraphDBType | null = null; // Set at runtime, not build time
  private static instance: GraphDBService;

  private constructor() {
    this.neo4jService = Neo4jService.getInstance();
    this.arangoDBService = ArangoDBService.getInstance();
  }

  /**
   * Get the active DB type, reading from env at runtime if not set
   */
  private getActiveDBType(): GraphDBType {
    if (this.activeDBType === null) {
      this.activeDBType = (process.env.GRAPH_DB_TYPE as GraphDBType) || 'arangodb';
      console.log(`[GraphDBService] Initialized activeDBType at runtime: ${this.activeDBType}`);
    }
    return this.activeDBType;
  }

  /**
   * Get the singleton instance of GraphDBService
   */
  public static getInstance(): GraphDBService {
    if (!GraphDBService.instance) {
      GraphDBService.instance = new GraphDBService();
    }
    return GraphDBService.instance;
  }

  /**
   * Initialize the graph database with the specified type
   * @param dbType - Type of graph database to use (defaults to GRAPH_DB_TYPE env var)
   * @param uri - Connection URL
   * @param username - Database username
   * @param password - Database password
   */
  public async initialize(dbType?: GraphDBType, uri?: string, username?: string, password?: string): Promise<void> {
    const graphDbType = dbType || (process.env.GRAPH_DB_TYPE as GraphDBType) || 'arangodb';
    this.activeDBType = graphDbType;
    
    try {
      if (graphDbType === 'neo4j') {
        this.neo4jService.initialize(uri, username, password);
        console.log('Neo4j initialized successfully');
      } else if (graphDbType === 'arangodb') {
        await this.arangoDBService.initialize(uri, undefined, username, password);
        console.log('ArangoDB initialized successfully');
      }
    } catch (error) {
      console.error(`Failed to initialize ${graphDbType}:`, error);
      throw error;
    }
  }

  /**
   * Set the active graph database type
   */
  public setDBType(dbType: GraphDBType): void {
    this.activeDBType = dbType;
  }

  /**
   * Get the active graph database type
   */
  public getDBType(): GraphDBType {
    return this.getActiveDBType();
  }

  /**
   * Check if the active database is initialized
   */
  public isInitialized(): boolean {
    if (this.getActiveDBType() === 'neo4j') {
      return this.neo4jService.isInitialized();
    } else {
      return this.arangoDBService.isInitialized();
    }
  }

  /**
   * Import triples into the active graph database
   */
  public async importTriples(triples: { subject: string; predicate: string; object: string }[]): Promise<void> {
    if (this.getActiveDBType() === 'neo4j') {
      await this.neo4jService.importTriples(triples);
    } else {
      await this.arangoDBService.importTriples(triples);
    }
  }

  /**
   * Get graph data from the active graph database
   */
  public async getGraphData(): Promise<{ 
    nodes: Array<{ 
      id: string; 
      labels: string[]; 
      [key: string]: any 
    }>; 
    relationships: Array<{ 
      id: string; 
      source: string; 
      target: string; 
      type: string; 
      [key: string]: any 
    }>; 
  }> {
    if (this.getActiveDBType() === 'neo4j') {
      return await this.neo4jService.getGraphData();
    } else {
      return await this.arangoDBService.getGraphData();
    }
  }

  /**
   * Log query information and metrics
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
    if (this.getActiveDBType() === 'neo4j') {
      await this.neo4jService.logQuery(query, queryMode, metrics);
    } else {
      await this.arangoDBService.logQuery(query, queryMode, metrics);
    }
  }

  /**
   * Get query logs from the active graph database
   */
  public async getQueryLogs(limit: number = 100): Promise<any[]> {
    if (this.getActiveDBType() === 'neo4j') {
      return await this.neo4jService.getQueryLogs(limit);
    } else {
      return await this.arangoDBService.getQueryLogs(limit);
    }
  }

  /**
   * Close the connection to the active graph database
   */
  public async close(): Promise<void> {
    if (this.getActiveDBType() === 'neo4j') {
      this.neo4jService.close();
    } else {
      this.arangoDBService.close();
    }
  }

  /**
   * Get info about the active graph database driver
   */
  public getDriverInfo(): Record<string, any> {
    if (this.getActiveDBType() === 'neo4j') {
      return this.neo4jService.getDriverInfo();
    } else {
      return this.arangoDBService.getDriverInfo();
    }
  }

  /**
   * Perform graph traversal using native database capabilities
   * Only available for ArangoDB
   */
  public async graphTraversal(
    keywords: string[],
    maxDepth: number = 2,
    maxResults: number = 100
  ): Promise<Array<{
    subject: string;
    predicate: string;
    object: string;
    confidence: number;
    depth?: number;
  }>> {
    if (this.getActiveDBType() === 'arangodb') {
      return await this.arangoDBService.graphTraversal(keywords, maxDepth, maxResults);
    } else {
      // Neo4j doesn't have this method yet, return empty array
      console.warn('graphTraversal is only available for ArangoDB');
      return [];
    }
  }

  /**
   * Clear all data from the active graph database
   */
  public async clearDatabase(): Promise<void> {
    if (this.getActiveDBType() === 'neo4j') {
      // TODO: Implement Neo4j clear database functionality
      throw new Error('Clear database functionality not implemented for Neo4j');
    } else {
      await this.arangoDBService.clearDatabase();
    }
  }
} 