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
/**
 * Qdrant service for vector embeddings
 */
import { Document } from "@langchain/core/documents";
import { randomUUID } from "crypto";

// Helper function to generate deterministic UUID from string
function stringToUUID(str: string): string {
  // Create a simple hash-based UUID v4
  const hash = str.split('').reduce((acc, char) => {
    return ((acc << 5) - acc) + char.charCodeAt(0) | 0;
  }, 0);

  // Generate a deterministic UUID from the hash
  const hex = Math.abs(hash).toString(16).padStart(32, '0').substring(0, 32);
  return `${hex.substring(0, 8)}-${hex.substring(8, 12)}-4${hex.substring(13, 16)}-${hex.substring(16, 20)}-${hex.substring(20, 32)}`;
}

// Define types for Qdrant requests and responses
interface QdrantPoint {
  id: string | number;
  vector: number[];
  payload?: Record<string, any>;
}

interface QdrantQueryResponse {
  result: Array<{
    id: string | number;
    score: number;
    payload?: Record<string, any>;
  }>;
}

// Define interface for document search results
export interface DocumentSearchResult {
  id: string;
  score: number;
  metadata?: Record<string, any>;
}

export class QdrantService {
  private dimension: number = 384; // Dimension for MiniLM-L6-v2
  private static instance: QdrantService;
  private initialized: boolean = false;
  private collectionName: string = 'entity-embeddings';
  private hostUrl: string;
  private isInitializing = false;

  private constructor() {
    // Get environment variables with defaults
    const qdrantUrl = process.env.QDRANT_URL || 'http://localhost:6333';
    this.hostUrl = qdrantUrl;

    console.log(`Initializing Qdrant service with host: ${this.hostUrl}`);
  }

  /**
   * Get singleton instance
   */
  public static getInstance(): QdrantService {
    if (!QdrantService.instance) {
      QdrantService.instance = new QdrantService();
    }
    return QdrantService.instance;
  }

  /**
   * Check if the service is initialized
   */
  public isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Make a request to the Qdrant API
   */
  private async makeRequest(endpoint: string, method: string = 'GET', body?: any): Promise<any> {
    try {
      const url = endpoint.startsWith('http') ? endpoint : `${this.hostUrl}${endpoint}`;

      console.log(`Making Qdrant request to: ${url}`);

      const options: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
        }
      };

      if (body) {
        options.body = JSON.stringify(body);
      }

      const response = await fetch(url, options);

      if (!response.ok) {
        const errorText = await response.text();
        console.log(`Qdrant API error (${response.status}) for ${url}: ${errorText}`);
        return null;
      }

      // For HEAD requests or empty responses
      if (method === 'HEAD' || response.headers.get('content-length') === '0') {
        return { status: response.status };
      }

      return await response.json();
    } catch (error) {
      console.log(`Error in Qdrant API request to ${endpoint} - request failed`);
      return null;
    }
  }

  /**
   * Check if the Qdrant server is up and running
   */
  private isQdrantRunningCheck = false;

  public async isQdrantRunning(): Promise<boolean> {
    // Prevent concurrent checks that could cause loops
    if (this.isQdrantRunningCheck) {
      console.log('Already checking if Qdrant is running, returning true to break cycle');
      return true;
    }

    this.isQdrantRunningCheck = true;

    try {
      // Check Qdrant health endpoint
      const response = await fetch(`${this.hostUrl}/healthz`, {
        method: 'GET'
      });

      if (response.ok) {
        console.log(`Qdrant server is up and healthy`);
        this.isQdrantRunningCheck = false;
        return true;
      }

      console.log('Qdrant health check failed - server might not be running');
      this.isQdrantRunningCheck = false;
      return false;
    } catch (error) {
      console.log('Error checking Qdrant server health - server appears to be down');
      this.isQdrantRunningCheck = false;
      return false;
    }
  }

  /**
   * Initialize Qdrant and create collection if needed
   */
  public async initialize(forceCreateCollection: boolean = false): Promise<void> {
    if ((this.initialized && !forceCreateCollection) || this.isInitializing) {
      return;
    }

    this.isInitializing = true;

    try {
      console.log('Qdrant service initializing...');

      // Check if Qdrant server is running
      const isRunning = await this.isQdrantRunning();
      if (!isRunning) {
        console.log('Qdrant server does not appear to be running. Please ensure it is started in Docker.');
        this.isInitializing = false;
        return;
      }

      // Check if collection exists
      const collectionInfo = await this.makeRequest(`/collections/${this.collectionName}`, 'GET');

      if (!collectionInfo || collectionInfo.status === 'error') {
        // Create collection
        console.log(`Creating Qdrant collection: ${this.collectionName}`);
        const createResult = await this.makeRequest(`/collections/${this.collectionName}`, 'PUT', {
          vectors: {
            size: this.dimension,
            distance: 'Cosine'
          }
        });

        if (createResult && createResult.result === true) {
          console.log(`Created Qdrant collection with ${this.dimension} dimensions`);
          this.initialized = true;
        } else {
          console.log('Failed to create Qdrant collection - continuing without initialization');
        }
      } else {
        // Collection exists
        const vectorCount = collectionInfo.result?.points_count || 0;
        console.log(`Connected to Qdrant collection with ${vectorCount} vectors`);
        this.initialized = true;
      }

      this.isInitializing = false;
      console.log('Qdrant service initialization completed');
    } catch (error) {
      console.log('Error during Qdrant service initialization - continuing without connection');
      this.isInitializing = false;
    }
  }

  /**
   * Store embeddings for entities
   */
  public async storeEmbeddings(
    entityEmbeddings: Map<string, number[]>,
    textContentMap?: Map<string, string>
  ): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      console.log('Qdrant not available - skipping embedding storage');
      return;
    }

    try {
      const points: QdrantPoint[] = [];

      // Convert to Qdrant point format
      for (const [entityName, embedding] of entityEmbeddings.entries()) {
        const point: QdrantPoint = {
          id: stringToUUID(entityName), // Convert string ID to UUID
          vector: embedding,
          payload: {
            originalId: entityName, // Store original ID in payload for retrieval
            text: textContentMap?.get(entityName) || entityName,
            type: 'entity'
          }
        };
        points.push(point);
      }

      // Use batching for efficient upserts
      const batchSize = 100;
      for (let i = 0; i < points.length; i += batchSize) {
        const batch = points.slice(i, i + batchSize);

        const success = await this.upsertVectors(batch);
        if (success) {
          console.log(`Upserted batch ${Math.floor(i/batchSize) + 1} of ${Math.ceil(points.length/batchSize)}`);
        } else {
          console.log(`Failed to upsert batch ${Math.floor(i/batchSize) + 1} - continuing`);
        }
      }

      console.log(`Completed embedding storage attempt for ${points.length} embeddings`);
    } catch (error) {
      console.log('Error storing embeddings - continuing without storage');
    }
  }

  /**
   * Upsert vectors to Qdrant
   * @param points Array of points to upsert
   * @param collectionName Optional collection name (defaults to entity-embeddings)
   */
  public async upsertVectors(points: QdrantPoint[], collectionName?: string): Promise<boolean> {
    const targetCollection = collectionName || this.collectionName;
    
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      console.log('Qdrant not available - skipping vector upsert');
      return false;
    }

    try {
      console.log(`Upserting ${points.length} vectors to Qdrant collection: ${targetCollection}`);

      const response = await this.makeRequest(`/collections/${targetCollection}/points`, 'PUT', {
        points: points
      });

      if (!response || response.status === 'error') {
        console.log(`Qdrant upsert failed`);
        return false;
      }

      console.log(`Successfully upserted ${points.length} vectors`);
      return true;
    } catch (error) {
      console.log('Error upserting vectors to Qdrant - continuing without storage');
      return false;
    }
  }

  /**
   * Store embeddings with metadata
   */
  public async storeEmbeddingsWithMetadata(
    embeddings: Map<string, number[]>,
    textContent: Map<string, string>,
    metadata: Map<string, any>
  ): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      console.log('Qdrant not available - skipping embedding storage with metadata');
      return;
    }

    try {
      const points: QdrantPoint[] = [];

      // Convert to Qdrant point format
      for (const [key, embedding] of embeddings.entries()) {
        const point: QdrantPoint = {
          id: stringToUUID(key), // Convert string ID to UUID
          vector: embedding,
          payload: {
            originalId: key, // Store original ID in payload for retrieval
            text: textContent.get(key) || '',
            ...metadata.get(key) || {}
          }
        };
        points.push(point);
      }

      // Use batching for efficient upserts
      const batchSize = 100;
      for (let i = 0; i < points.length; i += batchSize) {
        const batch = points.slice(i, i + batchSize);

        const success = await this.upsertVectors(batch);
        if (success) {
          console.log(`Upserted batch ${Math.floor(i/batchSize) + 1} of ${Math.ceil(points.length/batchSize)}`);
        } else {
          console.log(`Failed to upsert batch ${Math.floor(i/batchSize) + 1} - continuing`);
        }
      }

      console.log(`Completed embedding storage attempt for ${points.length} embeddings with metadata`);
    } catch (error) {
      console.log('Error storing embeddings with metadata - continuing without storage');
    }
  }

  /**
   * Find similar entities to a query embedding
   */
  public async findSimilarEntitiesWithMetadata(
    embedding: number[],
    limit: number = 10
  ): Promise<{ entities: string[], metadata: Map<string, any> }> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      console.log('Qdrant not available - returning empty results');
      return { entities: [], metadata: new Map() };
    }

    try {
      const queryResponse = await this.queryVectors(embedding, limit, true);

      if (!queryResponse) {
        return { entities: [], metadata: new Map() };
      }

      // Extract entities and metadata, using originalId from payload
      const entities = queryResponse.result.map(match =>
        match.payload?.originalId || String(match.id)
      );
      const metadataMap = new Map<string, any>();

      queryResponse.result.forEach(match => {
        const originalId = match.payload?.originalId || String(match.id);
        metadataMap.set(originalId, {
          ...match.payload,
          score: match.score
        });
      });

      return { entities, metadata: metadataMap };
    } catch (error) {
      console.log('Error finding similar entities - returning empty results');
      return { entities: [], metadata: new Map() };
    }
  }

  /**
   * Query vectors in Qdrant
   */
  private async queryVectors(
    vector: number[],
    limit: number = 10,
    withPayload: boolean = false
  ): Promise<QdrantQueryResponse | null> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      console.log('Qdrant not available - cannot query vectors');
      return null;
    }

    try {
      const response = await this.makeRequest(`/collections/${this.collectionName}/points/query`, 'POST', {
        query: vector,
        limit: limit,
        with_payload: withPayload
      });

      if (!response || response.status === 'error') {
        console.log(`Qdrant query failed`);
        return null;
      }

      return response;
    } catch (error) {
      console.log('Error querying vectors from Qdrant - returning null');
      return null;
    }
  }

  /**
   * Find similar entities to a query embedding
   */
  public async findSimilarEntities(queryEmbedding: number[], topK: number = 10): Promise<string[]> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      console.log('Qdrant not available - returning empty entity list');
      return [];
    }

    try {
      const queryResponse = await this.queryVectors(queryEmbedding, topK, true);
      if (!queryResponse) {
        return [];
      }
      return queryResponse.result.map(match =>
        match.payload?.originalId || String(match.id)
      );
    } catch (error) {
      console.log('Error finding similar entities - returning empty list');
      return [];
    }
  }

  /**
   * Get all entities in the collection (up to limit)
   */
  public async getAllEntities(limit: number = 1000): Promise<string[]> {
    if (!this.initialized) {
      await this.initialize();
    }

    try {
      // Use scroll API to get points
      // We'll use scroll API to get points
      const response = await this.makeRequest(`/collections/${this.collectionName}/points/scroll`, 'POST', {
        limit: limit,
        with_payload: false,
        with_vector: false
      });

      if (!response || !response.result || !response.result.points) {
        return [];
      }

      return response.result.points.map((point: any) => String(point.id));
    } catch (error) {
      console.error('Error getting all entities:', error);
      return [];
    }
  }

  /**
   * Delete entities from the collection
   */
  public async deleteEntities(entityIds: string[]): Promise<boolean> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      console.log('Qdrant not available - cannot delete entities');
      return false;
    }

    try {
      console.log(`Deleting ${entityIds.length} entities from Qdrant`);

      const response = await this.makeRequest(`/collections/${this.collectionName}/points/delete`, 'POST', {
        points: entityIds
      });

      if (!response || response.status === 'error') {
        console.log(`Qdrant delete failed`);
        return false;
      }

      console.log(`Successfully deleted ${entityIds.length} entities`);
      return true;
    } catch (error) {
      console.log('Error deleting entities from Qdrant - operation failed');
      return false;
    }
  }

  /**
   * Get collection statistics from Qdrant
   */
  public async getStats(): Promise<any> {
    try {
      console.log('Getting stats from Qdrant...');
      const response = await this.makeRequest(`/collections/${this.collectionName}`, 'GET');

      if (response && response.result) {
        const stats = response.result;
        console.log('Successfully retrieved stats from Qdrant');
        return {
          totalVectorCount: stats.points_count || 0,
          indexedVectorCount: stats.indexed_vectors_count || 0,
          status: stats.status || 'unknown',
          optimizerStatus: stats.optimizer_status || 'unknown',
          vectorSize: stats.config?.params?.vectors?.size || this.dimension,
          distance: stats.config?.params?.vectors?.distance || 'Cosine',
          source: 'qdrant',
          httpHealthy: true,
          url: this.hostUrl
        };
      } else {
        console.log(`Qdrant stats request failed`);
        return {
          totalVectorCount: 0,
          source: 'error',
          httpHealthy: false,
          error: 'Failed to get stats'
        };
      }
    } catch (error) {
      console.log('Qdrant connection failed - server may not be running');
      return {
        totalVectorCount: 0,
        source: 'error',
        httpHealthy: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Delete all entities in the collection
   */
  public async deleteAllEntities(): Promise<boolean> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      console.log('Qdrant not available - cannot delete all entities');
      return false;
    }

    try {
      console.log('Deleting all entities from Qdrant');

      // Delete the entire collection and recreate it
      const deleteResult = await this.makeRequest(`/collections/${this.collectionName}`, 'DELETE');

      if (!deleteResult || deleteResult.status === 'error') {
        console.log(`Qdrant delete collection failed`);
        return false;
      }

      // Recreate the collection
      await this.initialize(true);

      console.log('Successfully deleted all entities from Qdrant');
      return true;
    } catch (error) {
      console.log('Error deleting all entities from Qdrant - operation failed');
      return false;
    }
  }

  /**
   * Find similar documents to a query embedding
   * @param queryEmbedding Query embedding vector
   * @param topK Number of results to return
   * @returns Promise resolving to array of document search results
   */
  public async findSimilarDocuments(queryEmbedding: number[], topK: number = 10): Promise<DocumentSearchResult[]> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      console.log('Qdrant not available - returning empty document results');
      return [];
    }

    try {
      const queryResponse = await this.queryVectors(queryEmbedding, topK, true);
      if (!queryResponse) {
        return [];
      }
      return queryResponse.result.map(match => ({
        id: match.payload?.originalId || String(match.id),
        score: match.score,
        metadata: match.payload
      }));
    } catch (error) {
      console.log('Error finding similar documents - returning empty results');
      return [];
    }
  }

  /**
   * Store document chunks for Pure RAG
   * @param documents Array of document text chunks
   * @param embeddings Array of embeddings corresponding to the documents
   * @param metadata Optional metadata for each document
   */
  public async storeDocumentChunks(
    documents: string[],
    embeddings: number[][],
    metadata?: Record<string, any>[]
  ): Promise<void> {
    const documentCollection = 'document-embeddings';
    
    try {
      console.log(`Storing ${documents.length} document chunks in collection: ${documentCollection}`);

      // Ensure the document collection exists
      const collectionInfo = await this.makeRequest(`/collections/${documentCollection}`, 'GET');
      if (!collectionInfo || collectionInfo.status === 'error') {
        console.log(`Creating Qdrant collection: ${documentCollection}`);
        await this.makeRequest(`/collections/${documentCollection}`, 'PUT', {
          vectors: {
            size: this.dimension,
            distance: 'Cosine'
          }
        });
      }

      const points: QdrantPoint[] = [];

      // Convert to Qdrant point format
      for (let i = 0; i < documents.length; i++) {
        const docId = metadata?.[i]?.id || `doc_${randomUUID()}`;
        const point: QdrantPoint = {
          id: stringToUUID(docId),
          vector: embeddings[i],
          payload: {
            originalId: docId,
            text: documents[i],
            type: 'document',
            ...(metadata?.[i] || {})
          }
        };
        points.push(point);
      }

      // Use batching for efficient upserts
      const batchSize = 100;
      for (let i = 0; i < points.length; i += batchSize) {
        const batch = points.slice(i, i + batchSize);
        const success = await this.upsertVectors(batch, documentCollection);
        if (success) {
          console.log(`Upserted document batch ${Math.floor(i/batchSize) + 1} of ${Math.ceil(points.length/batchSize)}`);
        } else {
          console.log(`Failed to upsert document batch ${Math.floor(i/batchSize) + 1} - continuing`);
        }
      }

      console.log(`✅ Completed storing ${points.length} document chunks`);
    } catch (error) {
      console.log('Error storing document chunks - continuing without storage');
    }
  }
}
