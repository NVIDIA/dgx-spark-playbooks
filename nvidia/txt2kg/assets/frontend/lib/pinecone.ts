/**
 * Pinecone service for vector embeddings
 * Uses direct API calls for Pinecone local server
 */
import { Document } from "@langchain/core/documents";

// Define types for Pinecone requests and responses
interface PineconeRecord {
  id: string;
  values: number[];
  metadata?: Record<string, any>;
}

interface PineconeQueryResponse {
  matches: Array<{
    id: string;
    score: number;
    metadata?: Record<string, any>;
  }>;
}

// Define interface for document search results
export interface DocumentSearchResult {
  id: string;
  score: number;
  metadata?: Record<string, any>;
}

export class PineconeService {
  private dimension: number = 384; // Dimension for MiniLM-L6-v2
  private static instance: PineconeService;
  private initialized: boolean = false;
  private indexName: string = 'entity-embeddings';
  private namespace: string = ''; // Empty string is the default namespace for pinecone-index
  private hostUrl: string;
  private apiKey: string;
  private isInitializing = false;
  
  private constructor() {
    // Get environment variables with defaults
    const host = process.env.PINECONE_HOST || 'localhost';
    const port = process.env.PINECONE_PORT || '5081'; // Default to 5081 for pinecone-index
    const apiKey = process.env.PINECONE_API_KEY || 'pclocal';
    
    this.hostUrl = `http://${host}:${port}`;
    this.apiKey = apiKey;
    
    console.log(`Initializing Pinecone service with host: ${this.hostUrl}`);
  }
  
  /**
   * Get singleton instance
   */
  public static getInstance(): PineconeService {
    if (!PineconeService.instance) {
      PineconeService.instance = new PineconeService();
    }
    return PineconeService.instance;
  }
  
  /**
   * Check if the service is initialized
   */
  public isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Make a request to the Pinecone API
   */
  private async makeRequest(endpoint: string, method: string = 'GET', body?: any): Promise<any> {
    try {
      // For the Docker container setup, Pinecone is a separate service
      const url = endpoint.startsWith('http') ? endpoint : `${this.hostUrl}${endpoint}`;
      
      console.log(`Making Pinecone request to: ${url}`);
      
      const options: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
          'Api-Key': this.apiKey,
        }
      };
      
      if (body) {
        options.body = JSON.stringify(body);
      }
      
      const response = await fetch(url, options);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.log(`Pinecone API error (${response.status}) for ${url}: ${errorText}`);
        return null;
      }
      
      // For HEAD requests or empty responses
      if (method === 'HEAD' || response.headers.get('content-length') === '0') {
        return { status: response.status };
      }
      
      return await response.json();
    } catch (error) {
      console.log(`Error in Pinecone API request to ${endpoint} - request failed`);
      return null;
    }
  }

  /**
   * Check if the Pinecone server is up and running
   */
  private isPineconeRunningCheck = false;
  
  public async isPineconeRunning(): Promise<boolean> {
    // Prevent concurrent checks that could cause loops
    if (this.isPineconeRunningCheck) {
      console.log('Already checking if Pinecone is running, returning true to break cycle');
      return true;
    }
    
    this.isPineconeRunningCheck = true;
    
    try {
      // In Docker Compose setup, Pinecone might be available but need time to initialize
      // Try with simplified health checks first
      try {
        // This is a simplified test to see if the server responds at all
        const response = await fetch(this.hostUrl, {
          method: 'GET',
          headers: {
            'Api-Key': this.apiKey
          }
        });
        
        // Even a 404 is fine - it means the server is responding
        if (response.status >= 200 && response.status < 500) {
          console.log(`Pinecone server is up (basic connectivity check)`);
          this.isPineconeRunningCheck = false;
          return true;
        }
      } catch (e) {
        console.log(`Basic connectivity check failed:`, e);
      }
      
      // Try multiple health check endpoints
      const healthEndpoints = [
        '/health',
        '/ready',
        '/',
        '/version',
        '/list_indexes',
        '/collections',
        '/status'
      ];
      
      for (const endpoint of healthEndpoints) {
        try {
          // Try with direct fetch without going through makeRequest
          const response = await fetch(`${this.hostUrl}${endpoint}`, {
            method: 'GET',
            headers: {
              'Api-Key': this.apiKey
            }
          });
          
          // Even a 404 might be fine - at least the server is responding
          if (response.status >= 200 && response.status < 500) {
            console.log(`Pinecone server is up (checked with ${endpoint}, status: ${response.status})`);
            this.isPineconeRunningCheck = false;
            return true;
          }
        } catch (e) {
          console.log(`Health check failed for ${endpoint}:`, e);
        }
      }
      
      console.log('All Pinecone health checks failed - server might not be running');
      this.isPineconeRunningCheck = false;
      return false;
    } catch (error) {
      console.log('Error checking Pinecone server health - server appears to be down');
      this.isPineconeRunningCheck = false;
      return false;
    }
  }
  
  /**
   * Initialize Pinecone and create index if needed
   */
  public async initialize(forceCreateIndex: boolean = false): Promise<void> {
    if ((this.initialized && !forceCreateIndex) || this.isInitializing) {
      return;
    }
    
    this.isInitializing = true;
    
    try {
      console.log('Pinecone service initializing...');
      
      // Check if Pinecone server is running
      const isRunning = await this.isPineconeRunning();
      if (!isRunning) {
        console.log('Pinecone server does not appear to be running. Please ensure it is started in Docker.');
        this.isInitializing = false;
        return; // Don't throw, just return
      }
      
      // Check if we can access the index by getting stats
      const stats = await this.getStats();
      if (stats.httpHealthy) {
        console.log(`Connected to Pinecone index with ${stats.totalVectorCount} vectors`);
        this.initialized = true;
      } else {
        console.log('Failed to access Pinecone index - continuing without initialization');
      }
      
      this.isInitializing = false;
      console.log('Pinecone service initialization completed');
    } catch (error) {
      console.log('Error during Pinecone service initialization - continuing without connection');
      this.isInitializing = false;
      // Don't throw error, just log and continue
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

    // If still not initialized after attempt, skip storage
    if (!this.initialized) {
      console.log('Pinecone not available - skipping embedding storage');
      return;
    }

    try {
      const records: PineconeRecord[] = [];
      
      // Convert to Pinecone vector format
      for (const [entityName, embedding] of entityEmbeddings.entries()) {
        const record: PineconeRecord = {
          id: entityName,
          values: embedding,
          metadata: {
            text: textContentMap?.get(entityName) || entityName,
            type: 'entity'
          }
        };
        records.push(record);
      }
      
      // Use batching for efficient upserts
      const batchSize = 100;
      for (let i = 0; i < records.length; i += batchSize) {
        const batch = records.slice(i, i + batchSize);
        
        const success = await this.upsertVectors(batch);
        if (success) {
          console.log(`Upserted batch ${Math.floor(i/batchSize) + 1} of ${Math.ceil(records.length/batchSize)}`);
        } else {
          console.log(`Failed to upsert batch ${Math.floor(i/batchSize) + 1} - continuing`);
        }
      }
      
      console.log(`Completed embedding storage attempt for ${records.length} embeddings`);
    } catch (error) {
      console.log('Error storing embeddings - continuing without storage');
    }
  }
  
  /**
   * Upsert vectors to Pinecone
   */
  public async upsertVectors(vectors: PineconeRecord[]): Promise<boolean> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      console.log('Pinecone not available - skipping vector upsert');
      return false;
    }

    try {
      console.log(`Upserting ${vectors.length} vectors to Pinecone`);
      
      const response = await fetch(`${this.hostUrl}/vectors/upsert`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Api-Key': this.apiKey
        },
        body: JSON.stringify({
          vectors: vectors,
          namespace: this.namespace
        })
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.log(`Pinecone upsert failed: ${response.status} - ${errorText}`);
        return false;
      }
      
      console.log(`Successfully upserted ${vectors.length} vectors`);
      return true;
    } catch (error) {
      console.log('Error upserting vectors to Pinecone - continuing without storage');
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
      console.log('Pinecone not available - skipping embedding storage with metadata');
      return;
    }

    try {
      const records: PineconeRecord[] = [];
      
      // Convert to Pinecone vector format
      for (const [key, embedding] of embeddings.entries()) {
        const record: PineconeRecord = {
          id: key,
          values: embedding,
          metadata: {
            text: textContent.get(key) || '',
            ...metadata.get(key) || {}
          }
        };
        records.push(record);
      }
      
      // Use batching for efficient upserts
      const batchSize = 100;
      for (let i = 0; i < records.length; i += batchSize) {
        const batch = records.slice(i, i + batchSize);
        
        const success = await this.upsertVectors(batch);
        if (success) {
          console.log(`Upserted batch ${Math.floor(i/batchSize) + 1} of ${Math.ceil(records.length/batchSize)}`);
        } else {
          console.log(`Failed to upsert batch ${Math.floor(i/batchSize) + 1} - continuing`);
        }
      }
      
      console.log(`Completed embedding storage attempt for ${records.length} embeddings with metadata`);
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
      console.log('Pinecone not available - returning empty results');
      return { entities: [], metadata: new Map() };
    }

    try {
      const queryResponse = await this.queryVectors(embedding, limit, true);
      
      if (!queryResponse) {
        return { entities: [], metadata: new Map() };
      }
      
      // Extract entities and metadata
      const entities = queryResponse.matches.map(match => match.id);
      const metadataMap = new Map<string, any>();
      
      queryResponse.matches.forEach(match => {
        metadataMap.set(match.id, {
          ...match.metadata,
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
   * Query vectors in Pinecone
   */
  private async queryVectors(
    vector: number[], 
    topK: number = 10, 
    includeMetadata: boolean = false
  ): Promise<PineconeQueryResponse | null> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      console.log('Pinecone not available - cannot query vectors');
      return null;
    }

    try {
      const response = await fetch(`${this.hostUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Api-Key': this.apiKey
        },
        body: JSON.stringify({
          vector: vector,
          topK: topK,
          includeMetadata: includeMetadata,
          namespace: this.namespace
        })
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.log(`Pinecone query failed: ${response.status} - ${errorText}`);
        return null;
      }
      
      return await response.json();
    } catch (error) {
      console.log('Error querying vectors from Pinecone - returning null');
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
      console.log('Pinecone not available - returning empty entity list');
      return [];
    }

    try {
      const queryResponse = await this.queryVectors(queryEmbedding, topK, false);
      if (!queryResponse) {
        return [];
      }
      return queryResponse.matches.map(match => match.id);
    } catch (error) {
      console.log('Error finding similar entities - returning empty list');
      return [];
    }
  }
  
  /**
   * Get all entities in the index (up to limit)
   */
  public async getAllEntities(limit: number = 1000): Promise<string[]> {
    if (!this.initialized) {
      await this.initialize();
    }

    try {
      // Create a dummy query that will match all vectors
      const dummyVector = Array(this.dimension).fill(0);
      const queryResponse = await this.queryVectors(dummyVector, limit, false);
      return queryResponse.matches.map(match => match.id);
    } catch (error) {
      console.error('Error getting all entities:', error);
      return [];
    }
  }
  
  /**
   * Delete entities from the index
   */
  public async deleteEntities(entityIds: string[]): Promise<boolean> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      console.log('Pinecone not available - cannot delete entities');
      return false;
    }

    try {
      console.log(`Deleting ${entityIds.length} entities from Pinecone`);
      
      const response = await fetch(`${this.hostUrl}/vectors/delete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Api-Key': this.apiKey
        },
        body: JSON.stringify({
          ids: entityIds,
          namespace: this.namespace
        })
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.log(`Pinecone delete failed: ${response.status} - ${errorText}`);
        return false;
      }
      
      console.log(`Successfully deleted ${entityIds.length} entities`);
      return true;
    } catch (error) {
      console.log('Error deleting entities from Pinecone - operation failed');
      return false;
    }
  }
  
  /**
   * Get index statistics from Pinecone
   */
  public async getStats(): Promise<any> {
    try {
      // Try direct HTTP requests to the describe_index_stats endpoint
      try {
        console.log('Getting stats from Pinecone...');
        const response = await fetch(`${this.hostUrl}/describe_index_stats`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Api-Key': this.apiKey
          }
        });
        
        if (response.ok) {
          const statsData = await response.json();
          console.log('Successfully retrieved stats from Pinecone');
          return {
            totalVectorCount: statsData.totalVectorCount || 0,
            namespaces: statsData.namespaces || {},
            source: 'direct-http',
            httpHealthy: true
          };
        } else {
          console.log(`Pinecone stats request failed with status: ${response.status}`);
          return {
            totalVectorCount: 0,
            source: 'error',
            httpHealthy: false,
            error: `Failed to get stats: ${response.status}`
          };
        }
      } catch (error) {
        console.log('Pinecone connection failed - server may not be running');
        return {
          totalVectorCount: 0,
          source: 'error',
          httpHealthy: false,
          error: error instanceof Error ? error.message : String(error)
        };
      }
    } catch (error) {
      console.log('Error accessing Pinecone service');
      return {
        totalVectorCount: 0,
        source: 'error',
        httpHealthy: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
  
  /**
   * Delete all entities in the index
   */
  public async deleteAllEntities(): Promise<boolean> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      console.log('Pinecone not available - cannot delete all entities');
      return false;
    }

    try {
      console.log('Deleting all entities from Pinecone');
      
      const response = await fetch(`${this.hostUrl}/vectors/delete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Api-Key': this.apiKey
        },
        body: JSON.stringify({
          deleteAll: true,
          namespace: this.namespace
        })
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.log(`Pinecone delete all failed: ${response.status} - ${errorText}`);
        return false;
      }
      
      console.log('Successfully deleted all entities from Pinecone');
      return true;
    } catch (error) {
      console.log('Error deleting all entities from Pinecone - operation failed');
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
      console.log('Pinecone not available - returning empty document results');
      return [];
    }

    try {
      const queryResponse = await this.queryVectors(queryEmbedding, topK, true);
      if (!queryResponse) {
        return [];
      }
      return queryResponse.matches.map(match => ({
        id: match.id,
        score: match.score,
        metadata: match.metadata
      }));
    } catch (error) {
      console.log('Error finding similar documents - returning empty results');
      return [];
    }
  }
}