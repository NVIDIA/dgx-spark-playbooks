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
import { Database, aql } from 'arangojs';
import { createHash } from 'crypto';

/**
 * ArangoDB service for database operations
 * Provides methods to connect to and interact with an ArangoDB database
 */
export class ArangoDBService {
  private db: Database | null = null;
  private static instance: ArangoDBService;
  private collectionName: string = 'entities';
  private edgeCollectionName: string = 'relationships';
  private documentsCollectionName: string = 'processed_documents';

  private constructor() {}

  /**
   * Generate a deterministic _key from input string using MD5 hash
   * Uses Node.js built-in crypto module - truncated to 16 chars for compact keys
   * @param input - String to hash
   * @returns Hex-encoded hash string (16 chars, safe for ArangoDB _key)
   */
  private generateKey(input: string): string {
    return createHash('md5').update(input).digest('hex').slice(0, 16);
  }

  /**
   * Generate a deterministic _key for an entity based on its name
   * @param name - Entity name
   * @returns Deterministic _key string
   */
  private generateEntityKey(name: string): string {
    return this.generateKey(name.toLowerCase().trim());
  }

  /**
   * Generate a deterministic _key for an edge based on its endpoints and type
   * @param fromKey - Source entity _key
   * @param toKey - Target entity _key
   * @param relationType - Relationship type/predicate
   * @returns Deterministic _key string
   */
  private generateEdgeKey(fromKey: string, toKey: string, relationType: string): string {
    return this.generateKey(`${fromKey}|${relationType.toLowerCase().trim()}|${toKey}`);
  }

  /**
   * Get the singleton instance of ArangoDBService
   */
  public static getInstance(): ArangoDBService {
    if (!ArangoDBService.instance) {
      ArangoDBService.instance = new ArangoDBService();
    }
    return ArangoDBService.instance;
  }

  /**
   * Initialize the ArangoDB connection
   * @param url - ArangoDB connection URL (defaults to ARANGODB_URL env var or 'http://localhost:8529')
   * @param databaseName - ArangoDB database name (defaults to ARANGODB_DB env var or 'txt2kg')
   * @param username - ArangoDB username (defaults to ARANGODB_USER env var or 'root')
   * @param password - ArangoDB password (defaults to ARANGODB_PASSWORD env var or '')
   */
  public async initialize(url?: string, databaseName?: string, username?: string, password?: string): Promise<void> {
    // Use provided values, or environment variables, or defaults
    const connectionUrl = url || process.env.ARANGODB_URL || 'http://localhost:8529';
    const dbName = databaseName || process.env.ARANGODB_DB || 'txt2kg';
    const user = username || process.env.ARANGODB_USER || 'root';
    const pass = password || process.env.ARANGODB_PASSWORD || '';

    try {
      // Initialize the database connection
      this.db = new Database({
        url: connectionUrl,
        databaseName: dbName,
        auth: { username: user, password: pass },
      });

      // Check if database exists, create if it doesn't
      const dbExists = await this.db.exists();
      if (!dbExists) {
        console.log(`Database ${dbName} does not exist, creating it...`);
        await this.db.createDatabase(dbName);
        this.db.useDatabase(dbName);
      }

      // Check if collections exist, create if they don't
      const collections = await this.db.listCollections();
      const collectionNames = collections.map(c => c.name);

      // Create entity collection if it doesn't exist
      if (!collectionNames.includes(this.collectionName)) {
        await this.db.createCollection(this.collectionName);
        await this.db.collection(this.collectionName).ensureIndex({
          name: 'inverted_index',
          type: 'inverted',
          fields: ['name'],
          analyzer: 'text_en'
        });
        await this.db.createView(`${this.collectionName}_view`, {
          type: 'search-alias',
          indexes: [
            {
              collection: this.collectionName,
              index: 'inverted_index'
            }
          ]
        });
      }

      // Create edge collection if it doesn't exist
      if (!collectionNames.includes(this.edgeCollectionName)) {
        await this.db.createEdgeCollection(this.edgeCollectionName);
        await this.db.collection(this.edgeCollectionName).ensureIndex({
          name: 'inverted_index',
          type: 'inverted',
          fields: ['type'],
          analyzer: 'text_en'
        });
        await this.db.createView(`${this.edgeCollectionName}_view`, {
          type: 'search-alias',
          indexes: [
            {
              collection: this.edgeCollectionName,
              index: 'inverted_index'
            }
          ]
        });
      }

      // Create documents collection if it doesn't exist
      if (!collectionNames.includes(this.documentsCollectionName)) {
        await this.db.createCollection(this.documentsCollectionName);
      }

      console.log('ArangoDB initialized successfully');
    } catch (error) {
      console.error('Failed to initialize ArangoDB:', error);
      throw error;
    }
  }

  /**
   * Check if the database connection is initialized
   */
  public isInitialized(): boolean {
    return this.db !== null;
  }

  /**
   * Close the ArangoDB connection
   */
  public close(): void {
    if (this.db) {
      this.db = null;
      console.log('ArangoDB connection closed');
    }
  }

  /**
   * Execute an AQL query
   * @param query - AQL query string
   * @param bindVars - Parameters for the query
   * @returns Promise resolving to query results
   */
  public async executeQuery(query: string, bindVars: Record<string, any> = {}): Promise<any[]> {
    if (!this.db) {
      throw new Error('ArangoDB connection not initialized. Call initialize() first.');
    }

    try {
      const cursor = await this.db.query(query, bindVars);
      return await cursor.all();
    } catch (error) {
      console.error('Error executing ArangoDB query:', error);
      throw error;
    }
  }

  /**
   * Create a node in the graph database
   * @param properties - Node properties
   * @returns Promise resolving to the created node
   */
  public async createNode(properties: Record<string, any>): Promise<any> {
    if (!this.db) {
      throw new Error('ArangoDB connection not initialized. Call initialize() first.');
    }

    try {
      const collection = this.db.collection(this.collectionName);
      const doc = { ...properties, _key: this.generateEntityKey(properties.name) }
      return await collection.save(doc, { overwriteMode: 'update' });
    } catch (error) {
      console.error('Error creating node in ArangoDB:', error);
      throw error;
    }
  }

  /**
   * Create a relationship between two nodes
   * @param fromKey - Key of the start node
   * @param toKey - Key of the end node
   * @param relationType - Type of relationship
   * @param properties - Relationship properties
   * @returns Promise resolving to the created relationship
   */
  public async createRelationship(
    fromKey: string,
    toKey: string,
    relationType: string,
    properties: Record<string, any> = {}
  ): Promise<any> {
    if (!this.db) {
      throw new Error('ArangoDB connection not initialized. Call initialize() first.');
    }

    try {
      const edgeCollection = this.db.collection(this.edgeCollectionName);
      const edgeData = {
        _key: this.generateEdgeKey(fromKey, toKey, relationType),
        _from: `${this.collectionName}/${fromKey}`,
        _to: `${this.collectionName}/${toKey}`,
        type: relationType,
        ...properties
      };
      return await edgeCollection.save(edgeData, { overwriteMode: 'update' });
    } catch (error) {
      console.error('Error creating relationship in ArangoDB:', error);
      throw error;
    }
  }

  /**
   * Import triples (subject, predicate, object) into the graph database
   * Batches inserts every 1000 documents by default
   * @param triples - Array of triples to import
   * @param batchSize - Number of documents to insert per batch (default: 1000)
   * @returns Promise resolving when import is complete
   */
  public async importTriples(
    triples: { subject: string; predicate: string; object: string }[],
    batchSize: number = 1000
  ): Promise<void> {
    if (!this.db) {
      throw new Error('ArangoDB connection not initialized. Call initialize() first.');
    }

    let entityBatch: Array<{ _key: string; name: string }> = [];
    let edgeBatch: Array<{ _key: string; _from: string; _to: string; type: string }> = [];

    const importEntities = async () => {
      if (entityBatch.length === 0) return;
      await this.db!.collection(this.collectionName).saveAll(entityBatch, { overwriteMode: 'ignore' });
      console.log(`[ArangoDB] Imported ${entityBatch.length} entities`);
      entityBatch = [];
    };

    const importEdges = async () => {
      if (edgeBatch.length === 0) return;
      await this.db!.collection(this.edgeCollectionName).saveAll(edgeBatch, { overwriteMode: 'ignore' });
      console.log(`[ArangoDB] Imported ${edgeBatch.length} edges`);
      edgeBatch = [];
    };

    try {
      for (const triple of triples) {
        const normalizedSubject = triple.subject.trim();
        const normalizedPredicate = triple.predicate.trim();
        const normalizedObject = triple.object.trim();

        if (!normalizedSubject || !normalizedPredicate || !normalizedObject) {
          console.warn('Skipping invalid triple:', triple);
          continue;
        }

        const subjectKey = this.generateEntityKey(normalizedSubject);
        const objectKey = this.generateEntityKey(normalizedObject);
        const edgeKey = this.generateEdgeKey(subjectKey, objectKey, normalizedPredicate);

        entityBatch.push({ _key: subjectKey, name: normalizedSubject });
        entityBatch.push({ _key: objectKey, name: normalizedObject });

        edgeBatch.push({
          _key: edgeKey,
          _from: `${this.collectionName}/${subjectKey}`,
          _to: `${this.collectionName}/${objectKey}`,
          type: normalizedPredicate
        });

        if (entityBatch.length >= batchSize) await importEntities();
        if (edgeBatch.length >= batchSize) await importEdges();
      }

      // Flush remaining
      await importEntities();
      await importEdges();

      console.log(`Successfully imported ${triples.length} triples into ArangoDB`);
    } catch (error) {
      console.error('Error importing triples into ArangoDB:', error);
      throw error;
    }
  }

  /**
   * Check if a document has already been processed and stored in ArangoDB
   * @param documentName - Name of the document to check
   * @returns Promise resolving to true if document exists, false otherwise
   */
  public async isDocumentProcessed(documentName: string): Promise<boolean> {
    if (!this.db) {
      throw new Error('ArangoDB connection not initialized. Call initialize() first.');
    }

    const collection = this.db.collection(this.documentsCollectionName);
    const key = this.generateKey(documentName.trim());
    return await collection.documentExists(key);
  }

  /**
   * Mark a document as processed in ArangoDB
   * @param documentName - Name of the document
   * @param tripleCount - Number of triples stored for this document
   * @returns Promise resolving when the document is marked as processed
   */
  public async markDocumentAsProcessed(documentName: string, tripleCount: number): Promise<void> {
    if (!this.db) {
      throw new Error('ArangoDB connection not initialized. Call initialize() first.');
    }

    try {
      const collection = this.db.collection(this.documentsCollectionName);
      const doc = {
        _key: this.generateKey(documentName.trim()),
        documentName,
        tripleCount,
        processedAt: new Date().toISOString()
      };

      await collection.save(doc, { overwriteMode: 'replace' });
      console.log(`Marked document "${documentName}" as processed with ${tripleCount} triples`);
    } catch (error) {
      console.error('Error marking document as processed:', error);
      throw error;
    }
  }

  /**
   * Get all processed documents from ArangoDB
   * @returns Promise resolving to array of processed document names
   */
  public async getProcessedDocuments(): Promise<string[]> {
    if (!this.db) {
      throw new Error('ArangoDB connection not initialized. Call initialize() first.');
    }

    try {
      const documents = await this.executeQuery(
        `FOR d IN ${this.documentsCollectionName} RETURN d.documentName`
      );
      return documents;
    } catch (error) {
      console.error('Error getting processed documents:', error);
      return [];
    }
  }

  /**
   * Get graph data in a format compatible with the existing application
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
    if (!this.db) {
      throw new Error('ArangoDB connection not initialized. Call initialize() first.');
    }

    try {
      // Get all entities (nodes)
      const entities = await this.executeQuery(
        `FOR e IN ${this.collectionName} RETURN e`
      );

      // Get all relationships (edges)
      const relationships = await this.executeQuery(
        `FOR r IN ${this.edgeCollectionName} RETURN r`
      );

      // Format nodes in a way compatible with the application
      const nodes = entities.map(entity => ({
        id: entity._key,
        labels: ['Entity'],
        name: entity.name,
        ...entity
      }));

      // Format relationships in a way compatible with the application
      const formattedRelationships = relationships.map(rel => {
        const source = rel._from.split('/')[1];
        const target = rel._to.split('/')[1];

        return {
          id: rel._key,
          source,
          target,
          type: rel.type,
          ...rel
        };
      });

      return {
        nodes,
        relationships: formattedRelationships
      };
    } catch (error) {
      console.error('Error getting graph data from ArangoDB:', error);
      throw error;
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
    if (!this.db) {
      throw new Error('ArangoDB connection not initialized. Call initialize() first.');
    }

    try {
      // Create a queryLogs collection if it doesn't exist
      const collections = await this.db.listCollections();
      const collectionNames = collections.map(c => c.name);

      if (!collectionNames.includes('queryLogs')) {
        await this.db.createCollection('queryLogs');
      }

      // Store query log
      const queryLog = {
        query,
        queryMode,
        metrics,
        timestamp: new Date().toISOString()
      };

      await this.db.collection('queryLogs').save(queryLog);
    } catch (error) {
      console.error('Error logging query to ArangoDB:', error);
      // We don't want to throw here as query logging is non-critical
      console.error('Query logging failed but continuing execution');
    }
  }

  /**
   * Get query logs
   * @param limit - Maximum number of logs to retrieve
   * @returns Promise resolving to query logs
   */
  public async getQueryLogs(limit: number = 100): Promise<any[]> {
    if (!this.db) {
      throw new Error('ArangoDB connection not initialized. Call initialize() first.');
    }

    try {
      // Check if queryLogs collection exists
      const collections = await this.db.listCollections();
      const collectionNames = collections.map(c => c.name);

      if (!collectionNames.includes('queryLogs')) {
        return [];
      }

      // Get logs sorted by timestamp
      const logs = await this.executeQuery(
        `FOR l IN queryLogs SORT l.timestamp DESC LIMIT @limit RETURN l`,
        { limit }
      );

      return logs;
    } catch (error) {
      console.error('Error getting query logs from ArangoDB:', error);
      return [];
    }
  }

  /**
   * Perform graph traversal to find relevant triples using ArangoDB's native text search and graph capabilities
   * Uses inverted indexes with BM25 scoring for efficient keyword matching
   * @param keywords - Array of keywords to search for
   * @param maxDepth - Maximum traversal depth (default: 2)
   * @param maxResults - Maximum number of results to return (default: 100)
   * @param maxSeeds - Maximum number of seed nodes/edges from text search (default: 50)
   * @returns Promise resolving to array of triples with relevance scores
   */
  public async graphTraversal(
    keywords: string[],
    maxDepth: number = 2,
    maxResults: number = 100,
    maxSeeds: number = 50
  ): Promise<Array<{
    subject: string;
    predicate: string;
    object: string;
    confidence: number;
    depth?: number;
  }>> {
    console.log(`[ArangoDB] graphTraversal called with keywords: ${keywords.join(', ')}`);

    if (!this.db) {
      throw new Error('ArangoDB connection not initialized. Call initialize() first.');
    }

    try {
      // Build case-insensitive keyword matching conditions
      const keywordConditions = keywords
        .filter(kw => kw.length > 2)  // Filter short words
        .map(kw => kw.toLowerCase());

      if (keywordConditions.length === 0) {
        return [];
      }

      const query = `
        // 1. Tokenize keywords using the same analyzer as the index
        LET keywords_merged = CONCAT_SEPARATOR(" ", @keywords)
        LET keywords_tokens = TOKENS(keywords_merged, "text_en")

        // 2. Match for entity.name
        LET seedNodes = (
          FOR vertex IN ${this.collectionName}_view
            SEARCH ANALYZER(vertex.name IN keywords_tokens, "text_en")
            LET score = BM25(vertex)
            SORT score DESC
            LIMIT @maxSeeds
            RETURN { vertex, score }
        )

        // 3. Match for relationship.type
        LET seedEdges = (
          FOR edge IN ${this.edgeCollectionName}_view
            SEARCH ANALYZER(edge.type IN keywords_tokens, "text_en")
            LET score = BM25(edge)
            SORT score DESC
            LIMIT @maxSeeds
            RETURN { edge, score }
        )

        // 4. Normalize scores
        LET maxNodeScore = MAX(seedNodes[*].score) || 1
        LET maxEdgeScore = MAX(seedEdges[*].score) || 1

        // 5. Traverse from seedNodes up to maxDepth
        LET traversalResults = (
          FOR seed IN seedNodes
            FOR v, e, p IN 1..@maxDepth ANY seed.vertex ${this.edgeCollectionName}
              OPTIONS { uniqueVertices: 'path', bfs: true }

              LET subjectEntity = DOCUMENT(e._from)
              LET objectEntity = DOCUMENT(e._to)
              LET depth = LENGTH(p.edges) - 1

              // Depth penalty: closer to seed = higher score
              LET depthPenalty = 1.0 / (1.0 + depth * 0.2)

              // Normalize seed score and apply depth penalty
              LET normalizedSeedScore = seed.score / maxNodeScore
              LET confidence = normalizedSeedScore * depthPenalty

              RETURN {
                subject: subjectEntity.name,
                predicate: e.type,
                object: objectEntity.name,
                confidence: confidence,
                depth: depth,
                _edgeId: e._id,
                pathLength: LENGTH(p.edges)
              }
        )

        // 6. Collect triples from seedEdges (direct hits)
        LET edgeResults = (
          FOR seed IN seedEdges
            LET subjectEntity = DOCUMENT(seed.edge._from)
            LET objectEntity = DOCUMENT(seed.edge._to)

            // Direct edge matches get a boost (depth 0)
            LET normalizedScore = seed.score / maxEdgeScore

            RETURN {
              subject: subjectEntity.name,
              predicate: seed.edge.type,
              object: objectEntity.name,
              confidence: normalizedScore * 1.2, // Boost direct edge matches
              depth: 0,
              _edgeId: seed.edge._id,
              pathLength: 1
            }
        )

        // 7. Combine traversalResults and edgeResults
        LET combinedResults = APPEND(traversalResults, edgeResults)

        // 8. Remove duplicates by edge ID and sort by confidence
        LET uniqueResults = (
          FOR result IN combinedResults
            COLLECT edgeId = result._edgeId INTO groups
            LET best = FIRST(
              FOR g IN groups
                SORT g.result.confidence DESC
                RETURN g.result
            )
            RETURN best
        )

        // 9. Sort by confidence and limit results
        FOR result IN uniqueResults
          FILTER result != null
          SORT result.confidence DESC, result.depth ASC
          LIMIT @maxResults
          RETURN {
            subject: result.subject,
            predicate: result.predicate,
            object: result.object,
            confidence: result.confidence,
            depth: result.depth,
            pathLength: result.pathLength
          }
      `;

      console.log(`[ArangoDB] Executing query with ${keywordConditions.length} keywords`);

      const results = await this.executeQuery(query, {
        keywords: keywordConditions,
        maxDepth,
        maxResults,
        maxSeeds
      });

      console.log(`[ArangoDB] Found ${results.length} triples for keywords: ${keywords.join(', ')}`);

      // Log top 10 results with confidence scores
      if (results.length > 0) {
        console.log('[ArangoDB] Top 10 triples by confidence:');
        results.slice(0, 10).forEach((triple: any, idx: number) => {
          const pathInfo = triple.pathLength ? ` path=${triple.pathLength}` : '';
          console.log(`  ${idx + 1}. [conf=${triple.confidence?.toFixed(3)}] ${triple.subject} -> ${triple.predicate} -> ${triple.object} (depth=${triple.depth}${pathInfo})`);
        });
      } else {
        console.log('[ArangoDB] No triples found!');
      }

      return results;
    } catch (error) {
      console.error('Error performing graph traversal in ArangoDB:', error);
      throw error;
    }
  }

  /**
   * Get basic info about the ArangoDB connection
   */
  public getDriverInfo(): Record<string, any> {
    if (!this.db) {
      return { status: 'not connected' };
    }

    return {
      status: 'connected',
      url: this.db.url,
      database: this.db.name
    };
  }

  /**
   * Clear all data from the graph database
   * @returns Promise resolving when all collections are cleared
   */
  public async clearDatabase(): Promise<void> {
    if (!this.db) {
      throw new Error('ArangoDB connection not initialized. Call initialize() first.');
    }

    try {
      // Truncate the entities collection (nodes)
      await this.db.collection(this.collectionName).truncate();

      // Truncate the relationships collection (edges)
      await this.db.collection(this.edgeCollectionName).truncate();

      // Also clear query logs if they exist
      const collections = await this.db.listCollections();
      const collectionNames = collections.map(c => c.name);

      if (collectionNames.includes('queryLogs')) {
        await this.db.collection('queryLogs').truncate();
      }

      console.log('ArangoDB database cleared successfully');
    } catch (error) {
      console.error('Error clearing ArangoDB database:', error);
      throw error;
    }
  }
}
