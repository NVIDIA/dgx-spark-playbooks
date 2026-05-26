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
import neo4j, { Driver, Session, Record as Neo4jRecord, Config } from 'neo4j-driver';

/**
 * Neo4j service for database operations
 * Provides methods to connect to and interact with a Neo4j database
 */
export class Neo4jService {
  private driver: Driver | null = null;
  private static instance: Neo4jService;

  private constructor() {}

  /**
   * Get the singleton instance of Neo4jService
   */
  public static getInstance(): Neo4jService {
    if (!Neo4jService.instance) {
      Neo4jService.instance = new Neo4jService();
    }
    return Neo4jService.instance;
  }

  /**
   * Initialize the Neo4j driver connection
   * @param uri - Neo4j connection URI (defaults to NEO4J_URI env var or 'bolt://localhost:7687')
   * @param username - Neo4j username (optional if auth is disabled)
   * @param password - Neo4j password (optional if auth is disabled)
   */
  public initialize(uri?: string, username?: string, password?: string): void {
    // Use provided URI, or environment variable, or default to localhost
    const connectionUri = uri || process.env.NEO4J_URI || 'bolt://localhost:7687';
    try {
      // Default configuration for Neo4j driver
      const config: Config = {
        maxConnectionPoolSize: 100,
        connectionAcquisitionTimeout: 60000, // 60 seconds
        connectionTimeout: 30000, // 30 seconds
        disableLosslessIntegers: true, // Convert Neo4j integers to JavaScript numbers
        logging: {
          level: 'info',
          logger: (level: string, message: string) => {
            switch (level) {
              case 'error':
                console.error(message);
                break;
              case 'warn':
                console.warn(message);
                break;
              case 'info':
                console.info(message);
                break;
              default:
                console.log(message);
            }
          }
        }
      };
      
      // Validate the URI scheme - Neo4j only supports bolt://, neo4j://, and neo4j+s:// schemes
      if (connectionUri && !connectionUri.startsWith('bolt://') && 
          !connectionUri.startsWith('neo4j://') && 
          !connectionUri.startsWith('neo4j+s://')) {
        throw new Error(`Invalid Neo4j URI scheme: ${connectionUri}. Must use bolt://, neo4j://, or neo4j+s:// protocol.`);
      }
      
      // If authentication is provided, use it; otherwise connect without auth
      if (username && password) {
        this.driver = neo4j.driver(connectionUri, neo4j.auth.basic(username, password), config);
      } else {
        this.driver = neo4j.driver(connectionUri, neo4j.auth.basic('neo4j', ''), config);
      }
      console.log('Neo4j driver initialized successfully');
    } catch (error) {
      console.error('Failed to initialize Neo4j driver:', error);
      throw error;
    }
  }

  /**
   * Check if the driver is initialized
   */
  public isInitialized(): boolean {
    return this.driver !== null;
  }

  /**
   * Get a session for database operations
   */
  public getSession(): Session {
    if (!this.driver) {
      throw new Error('Neo4j driver not initialized. Call initialize() first.');
    }
    return this.driver.session();
  }

  /**
   * Close the Neo4j driver connection
   */
  public close(): void {
    if (this.driver) {
      this.driver.close();
      this.driver = null;
      console.log('Neo4j driver closed');
    }
  }

  /**
   * Execute a Cypher query
   * @param cypher - Cypher query string
   * @param params - Parameters for the query
   * @returns Promise resolving to query results
   */
  public async executeQuery(cypher: string, params: Record<string, any> = {}): Promise<Neo4jRecord[]> {
    if (!this.driver) {
      throw new Error('Neo4j driver not initialized. Call initialize() first.');
    }

    const session = this.getSession();
    try {
      const result = await session.run(cypher, params);
      return result.records;
    } catch (error) {
      console.error('Error executing Neo4j query:', error);
      throw error;
    } finally {
      session.close();
    }
  }

  /**
   * Create a node in the graph database
   * @param label - Node label
   * @param properties - Node properties
   * @returns Promise resolving to the created node
   */
  public async createNode(label: string, properties: Record<string, any>): Promise<Neo4jRecord[]> {
    const cypher = `CREATE (n:${label} $props) RETURN n`;
    return this.executeQuery(cypher, { props: properties });
  }

  /**
   * Create a relationship between two nodes
   * @param startNodeId - ID of the start node
   * @param endNodeId - ID of the end node
   * @param relationType - Type of relationship
   * @param properties - Relationship properties
   * @returns Promise resolving to the created relationship
   */
  public async createRelationship(
    startNodeId: string | number,
    endNodeId: string | number,
    relationType: string,
    properties: Record<string, any> = {}
  ): Promise<Neo4jRecord[]> {
    const cypher = `
      MATCH (a), (b)
      WHERE ID(a) = $startNodeId AND ID(b) = $endNodeId
      CREATE (a)-[r:${relationType} $props]->(b)
      RETURN r
    `;
    return this.executeQuery(cypher, {
      startNodeId,
      endNodeId,
      props: properties,
    });
  }

  /**
   * Import triples (subject, predicate, object) into the graph database
   * @param triples - Array of triples to import
   * @returns Promise resolving when import is complete
   */
  public async importTriples(triples: { subject: string; predicate: string; object: string }[]): Promise<void> {
    if (!this.driver) {
      throw new Error('Neo4j driver not initialized. Call initialize() first.');
    }

    // First transaction for schema operations
    const schemaSession = this.getSession();
    try {
      // Create indices for faster lookups if they don't exist
      await schemaSession.run('CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)');
      await schemaSession.run('CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.type)');
      console.log('Schema indices verified/created');
    } catch (error) {
      console.error('Error creating schema indices:', error);
      throw error;
    } finally {
      schemaSession.close();
    }

    // Second transaction for data operations
    const dataSession = this.getSession();
    let txc = null;
    try {
      // Use transactions for better performance and atomicity
      txc = dataSession.beginTransaction();
      
      for (const triple of triples) {
        // Normalize triple values to avoid case-sensitivity duplicates
        const normalizedSubject = triple.subject.trim();
        const normalizedPredicate = triple.predicate.trim();
        const normalizedObject = triple.object.trim();
        
        // Skip empty or invalid triples
        if (!normalizedSubject || !normalizedPredicate || !normalizedObject) {
          console.warn('Skipping invalid triple:', triple);
          continue;
        }
        
        // Create or merge subject and object nodes
        await txc.run(
          'MERGE (s:Entity {name: $subject}) MERGE (o:Entity {name: $object})',
          { subject: normalizedSubject, object: normalizedObject }
        );
        
        // Check if relationship already exists to prevent duplicates
        const checkQuery = `
          MATCH (s:Entity {name: $subject})-[r]->(o:Entity {name: $object})
          WHERE type(r) = 'RELATIONSHIP' AND r.type = $relType
          RETURN count(r) > 0 AS exists
        `;
        
        const checkResult = await txc.run(checkQuery, {
          subject: normalizedSubject,
          object: normalizedObject,
          relType: normalizedPredicate
        });
        
        const relationshipExists = checkResult.records[0]?.get('exists') || false;
        
        // Only create relationship if it doesn't already exist
        if (!relationshipExists) {
          // Use a generic relationship type and store the actual predicate as a property
          await txc.run(
            'MATCH (s:Entity {name: $subject}), (o:Entity {name: $object}) ' +
            'CREATE (s)-[r:RELATIONSHIP {type: $predicate}]->(o)',
            { 
              subject: normalizedSubject, 
              object: normalizedObject,
              predicate: normalizedPredicate
            }
          );
        }
      }
      
      await txc.commit();
      console.log(`Successfully imported ${triples.length} triples into Neo4j`);
    } catch (error) {
      console.error('Error importing triples into Neo4j:', error);
      // Rollback the transaction if an error occurs
      if (txc) {
        await txc.rollback();
      }
      throw error;
    } finally {
      dataSession.close();
    }
  }

  /**
   * Get all nodes and relationships from the database
   * @returns Promise resolving to nodes and relationships
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
    if (!this.driver) {
      throw new Error('Neo4j driver not initialized. Call initialize() first.');
    }

    const session = this.getSession();
    try {
      // Get all nodes
      const nodesResult = await session.run('MATCH (n) RETURN n');
      const nodes = nodesResult.records.map(record => {
        const node = record.get('n');
        return {
          id: node.identity.toString(),
          ...node.properties,
          labels: node.labels
        };
      });

      // Get all relationships
      const relsResult = await session.run(
        'MATCH ()-[r]->() RETURN r, startNode(r) as source, endNode(r) as target'
      );
      const relationships = relsResult.records.map(record => {
        const rel = record.get('r');
        const source = record.get('source');
        const target = record.get('target');
        return {
          id: rel.identity.toString(),
          source: source.identity.toString(),
          target: target.identity.toString(),
          // Use the type property if available, otherwise use the relationship type
          type: rel.properties.type || rel.type,
          ...rel.properties
        };
      });

      return { nodes, relationships };
    } catch (error) {
      console.error('Error fetching graph data from Neo4j:', error);
      throw error;
    } finally {
      session.close();
    }
  }

  /**
   * Log a RAG query with its performance metrics
   * @param query The user's query string
   * @param queryMode The query mode used (traditional, vector-search, pure-rag)
   * @param metrics Performance metrics for the query
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
    if (!this.driver) {
      console.error('Neo4j driver not initialized for logQuery. Attempting to initialize...');
      this.initialize();
      if (!this.driver) {
        console.error('Failed to initialize Neo4j driver for logQuery');
        throw new Error('Neo4j driver not initialized. Call initialize() first.');
      }
    }

    console.log(`[Neo4j] Logging query: "${query}" (${queryMode})`);
    console.log(`[Neo4j] Query metrics:`, JSON.stringify(metrics));

    const session = this.getSession();
    try {
      // Create a QueryLog node with timestamp and metrics
      const cypher = `
        MERGE (q:QueryLog {query: $query})
        ON CREATE SET 
          q.firstQueried = datetime(),
          q.count = 1
        ON MATCH SET 
          q.lastQueried = datetime(),
          q.count = q.count + 1
        
        CREATE (e:QueryExecution {
          timestamp: datetime(),
          queryMode: $queryMode,
          executionTimeMs: $executionTimeMs,
          relevanceScore: $relevanceScore,
          precision: $precision,
          recall: $recall,
          resultCount: $resultCount
        })
        
        CREATE (q)-[:HAS_EXECUTION]->(e)
        
        RETURN q, e
      `;

      console.log(`[Neo4j] Executing Cypher query for logQuery`);
      
      const result = await session.run(cypher, {
        query,
        queryMode,
        executionTimeMs: metrics.executionTimeMs,
        relevanceScore: metrics.relevanceScore || null,
        precision: metrics.precision || null,
        recall: metrics.recall || null,
        resultCount: metrics.resultCount
      });
      
      console.log(`[Neo4j] Query logged successfully. Summary: created ${result.summary.counters.updates().nodesCreated} nodes, ${result.summary.counters.updates().relationshipsCreated} relationships`);
    } catch (error) {
      console.error('[Neo4j] Error logging query:', error);
      // Non-critical error, so just log it but don't throw
    } finally {
      session.close();
    }
  }

  /**
   * Get query logs with performance metrics
   * @param limit Maximum number of query logs to return
   * @returns Promise resolving to an array of query logs
   */
  public async getQueryLogs(limit: number = 100): Promise<any[]> {
    if (!this.driver) {
      console.error('Neo4j driver not initialized for getQueryLogs. Attempting to initialize...');
      this.initialize();
      if (!this.driver) {
        console.error('Failed to initialize Neo4j driver for getQueryLogs');
        throw new Error('Neo4j driver not initialized. Call initialize() first.');
      }
    }

    console.log(`[Neo4j] Getting query logs with limit: ${limit}`);

    const session = this.getSession();
    try {
      // Get queries with their execution metrics, ordered by count
      const cypher = `
        MATCH (q:QueryLog)-[:HAS_EXECUTION]->(e:QueryExecution)
        WITH q, collect(e) as executions
        RETURN 
          q.query as query, 
          q.count as count,
          q.firstQueried as firstQueried,
          q.lastQueried as lastQueried,
          avg(e.executionTimeMs) as avgExecutionTimeMs,
          avg(e.relevanceScore) as avgRelevanceScore,
          avg(e.precision) as avgPrecision,
          avg(e.recall) as avgRecall,
          avg(e.resultCount) as avgResultCount,
          count(e) as executionCount
        ORDER BY q.count DESC
        LIMIT $limit
      `;

      console.log(`[Neo4j] Executing Cypher query for getQueryLogs`);
      
      const result = await session.run(cypher, { limit });
      
      console.log(`[Neo4j] Retrieved ${result.records.length} query logs`);

      if (result.records.length === 0) {
        console.log('[Neo4j] No query logs found. Checking if QueryLog nodes exist...');
        
        // Check if QueryLog nodes exist at all
        const checkNodesQuery = `MATCH (q:QueryLog) RETURN count(q) as count`;
        const checkNodesResult = await session.run(checkNodesQuery);
        const nodeCount = checkNodesResult.records[0].get('count').toNumber();
        
        console.log(`[Neo4j] Found ${nodeCount} QueryLog nodes`);
        
        if (nodeCount > 0) {
          console.log('[Neo4j] Checking if relationships exist...');
          const checkRelsQuery = `MATCH (q:QueryLog)-[r:HAS_EXECUTION]->() RETURN count(r) as count`;
          const checkRelsResult = await session.run(checkRelsQuery);
          const relCount = checkRelsResult.records[0].get('count').toNumber();
          
          console.log(`[Neo4j] Found ${relCount} HAS_EXECUTION relationships`);
        }
      }
      
      return result.records.map(record => {
        const mappedRecord = {
          query: record.get('query'),
          count: record.get('count').toNumber(),
          firstQueried: record.get('firstQueried'),
          lastQueried: record.get('lastQueried'),
          metrics: {
            avgExecutionTimeMs: record.get('avgExecutionTimeMs'),
            avgRelevanceScore: record.get('avgRelevanceScore'),
            avgPrecision: record.get('avgPrecision'),
            avgRecall: record.get('avgRecall'),
            avgResultCount: record.get('avgResultCount')
          },
          executionCount: record.get('executionCount').toNumber()
        };
        
        return mappedRecord;
      });
    } catch (error) {
      console.error('[Neo4j] Error getting query logs:', error);
      throw error;
    } finally {
      session.close();
    }
  }

  /**
   * Get information about the driver connection
   * @returns Object with connection info
   */
  public getDriverInfo(): Record<string, any> {
    if (!this.driver) {
      return { 
        connected: false,
        message: 'Driver not initialized'
      };
    }
    
    try {
      // Get connection URI to return (strip password if present)
      const connectionInfo = (this.driver as any)._connectionProvider?._connectionPool?._address || 'Unknown';
      return {
        connected: true,
        connectionInfo: String(connectionInfo),
      };
    } catch (error) {
      console.error('Error getting driver info:', error);
      return {
        connected: true,
        error: 'Could not extract driver details'
      };
    }
  }

  /**
   * Creates a test query log entry for debugging
   * Useful for debugging
   * @param query The query text
   * @returns Promise that resolves when the operation is complete
   */
  public async createTestQueryLog(query: string): Promise<void> {
    return this.logQuery(
      query,
      'traditional',
      {
        executionTimeMs: 0,
        relevanceScore: 0,
        precision: 0,
        recall: 0,
        resultCount: 0
      }
    );
  }
}

export default Neo4jService.getInstance();