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
import axios from 'axios';
import { GraphDBService, GraphDBType } from './graph-db-service';
import { QdrantService } from './qdrant';
import { getGraphDbService } from './graph-db-util';
import type { Triple } from '@/types/graph';

/**
 * Backend service that combines graph database for storage and Qdrant for embeddings
 * 
 * Two distinct modes:
 * 1. Knowledge Graph Mode: Stores triples in graph DB + entity names in 'entity-embeddings' collection
 * 2. Pure RAG Mode: Stores document chunks in 'document-embeddings' collection (via RAGService)
 * 
 * Use processTriples() for knowledge graph ingestion
 * Use storeDocumentChunks() for Pure RAG document ingestion
 */
export class BackendService {
  private graphDBService: GraphDBService;
  private qdrantService: QdrantService;
  private sentenceTransformerUrl: string = 'http://sentence-transformers:80';
  private modelName: string = 'all-MiniLM-L6-v2';
  private static instance: BackendService;
  private initialized: boolean = false;
  private activeGraphDbType: GraphDBType | null = null; // Set at runtime, not build time
  
  private getRuntimeGraphDbType(): GraphDBType {
    if (this.activeGraphDbType === null) {
      this.activeGraphDbType = (process.env.GRAPH_DB_TYPE as GraphDBType) || 'arangodb';
      console.log(`[BackendService] Initialized activeGraphDbType at runtime: ${this.activeGraphDbType}`);
    }
    return this.activeGraphDbType;
  }
  
  private constructor() {
    this.graphDBService = GraphDBService.getInstance();
    this.qdrantService = QdrantService.getInstance();
    
    // Use environment variables if available
    if (process.env.SENTENCE_TRANSFORMER_URL) {
      this.sentenceTransformerUrl = process.env.SENTENCE_TRANSFORMER_URL;
    }
    if (process.env.MODEL_NAME) {
      this.modelName = process.env.MODEL_NAME;
    }
  }
  
  /**
   * Get the singleton instance of BackendService
   */
  public static getInstance(): BackendService {
    if (!BackendService.instance) {
      BackendService.instance = new BackendService();
    }
    return BackendService.instance;
  }
  
  /**
   * Initialize the backend services
   * @param graphDbType - Type of graph database to use (defaults to GRAPH_DB_TYPE env var)
   */
  public async initialize(graphDbType?: GraphDBType): Promise<void> {
    const dbType = graphDbType || (process.env.GRAPH_DB_TYPE as GraphDBType) || 'arangodb';
    this.activeGraphDbType = dbType;
    
    // Initialize Graph Database
    if (!this.graphDBService.isInitialized()) {
      try {
        // Get the appropriate service based on type
        const graphDbService = getGraphDbService(dbType);
        
        // Try to get settings from server settings API first
        let serverSettings: Record<string, string> = {};
        try {
          const response = await fetch('/api/settings');
          if (response.ok) {
            const data = await response.json();
            serverSettings = data.settings || {};
            console.log('Successfully loaded settings from server API');
          }
        } catch (error) {
          console.log('Failed to load settings from server API, falling back to environment variables:', error);
        }
        
        if (dbType === 'neo4j') {
          // Get Neo4j credentials from server settings first, then fallback to environment
          const uri = serverSettings.neo4j_url || process.env.NEO4J_URI;
          const username = serverSettings.neo4j_user || process.env.NEO4J_USER || process.env.NEO4J_USERNAME;
          const password = serverSettings.neo4j_password || process.env.NEO4J_PASSWORD;
          
          console.log(`Using Neo4j URI: ${uri}`);
          await this.graphDBService.initialize('neo4j', uri, username, password);
        } else {
          // Prioritize environment variables over server settings for Docker deployments
          const url = process.env.ARANGODB_URL || serverSettings.arango_url || 'http://localhost:8529';
          const dbName = process.env.ARANGODB_DB || serverSettings.arango_db || 'txt2kg';
          const username = process.env.ARANGODB_USER || serverSettings.arango_user;
          const password = process.env.ARANGODB_PASSWORD || serverSettings.arango_password;
          
          console.log(`Using ArangoDB URL: ${url}`);
          console.log(`Using ArangoDB database: ${dbName}`);
          await this.graphDBService.initialize('arangodb', url, username, password);
        }
        console.log(`${dbType} initialized successfully in backend service`);
      } catch (error) {
        console.error(`Failed to initialize ${dbType} in backend service:`, error);
        if (process.env.NODE_ENV === 'development') {
          console.log('Development mode: Continuing despite graph database initialization error');
        } else {
          throw new Error('Graph database service initialization failed');
        }
      }
    }
    
    // Initialize Qdrant
    if (!this.qdrantService.isInitialized()) {
      await this.qdrantService.initialize();
    }
    
    // Check if sentence-transformer service is available
    try {
      // Remove the check skip in development mode
      const response = await axios.get(`${this.sentenceTransformerUrl}/health`);
      console.log(`Connected to SentenceTransformer service: ${response.data.model}`);
      this.initialized = true;
    } catch (error) {
      console.error(`Failed to connect to sentence-transformer service: ${error}`);
      if (process.env.NODE_ENV === 'development') {
        console.log('Development mode: Continuing despite sentence transformer error');
        this.initialized = true;
      } else {
        throw new Error('Sentence transformer service is not available');
      }
    }
  }
  
  /**
   * Check if the backend is initialized
   */
  public get isInitialized(): boolean {
    return this.initialized && this.graphDBService.isInitialized();
  }
  
  /**
   * Get the active graph database type
   */
  public getGraphDbType(): GraphDBType {
    return this.getRuntimeGraphDbType();
  }
  
  /**
   * Generate embeddings using the sentence-transformer service
   */
  private async generateEmbeddings(texts: string[]): Promise<number[][]> {
    try {
      const response = await axios.post(`${this.sentenceTransformerUrl}/embed`, {
        texts,
        batch_size: 32
      });
      
      return response.data.embeddings;
    } catch (error) {
      console.error(`Error generating embeddings: ${error}`);
      throw new Error('Failed to generate embeddings');
    }
  }

  /**
   * Convert our triple format to database format
   */
  private convertTriples(triples: Triple[]): { subject: string; predicate: string; object: string }[] {
    return triples.map(triple => ({
      subject: triple.subject,
      predicate: triple.predicate,
      object: triple.object
    }));
  }
  
  /**
   * Process and store triples in graph database and embeddings in Qdrant
   */
  public async processTriples(triples: Triple[]): Promise<void> {
    // Preprocess triples: lowercase and remove duplicates
    const processedTriples = triples.map(triple => ({
      subject: triple.subject.toLowerCase(),
      predicate: triple.predicate.toLowerCase(),
      object: triple.object.toLowerCase()
    }));
    
    // Remove duplicate triples
    const uniqueTriples = Array.from(
      new Map(processedTriples.map(triple => [JSON.stringify(triple), triple])).values()
    );
    
    console.log(`Processed ${triples.length} triples, removed ${triples.length - uniqueTriples.length} duplicates`);
    
    // Store triples in graph database
    console.log(`Storing triples in ${this.activeGraphDbType} database`);
    await this.graphDBService.importTriples(this.convertTriples(uniqueTriples));
    
    // Extract unique entities from triples
    const entities = new Set<string>();
    for (const triple of uniqueTriples) {
      entities.add(triple.subject); // subject
      entities.add(triple.object); // object
    }
    
    // Generate embeddings for entities in batches
    const entityList = Array.from(entities);
    const batchSize = 256;
    const entityEmbeddings = new Map<string, number[]>();
    const textContent = new Map<string, string>(); // Map for text content
    
    console.log(`Generating embeddings for ${entityList.length} entities`);
    
    for (let i = 0; i < entityList.length; i += batchSize) {
      const batch = entityList.slice(i, i + batchSize);
      console.log(`Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(entityList.length/batchSize)}`);
      
      const embeddings = await this.generateEmbeddings(batch);
      
      // Store in maps
      for (let j = 0; j < batch.length; j++) {
        entityEmbeddings.set(batch[j], embeddings[j]);
        textContent.set(batch[j], batch[j]); // Store the entity name as text content
      }
    }
    
    // Store embeddings and text content in Qdrant
    await this.qdrantService.storeEmbeddings(entityEmbeddings, textContent);
    
    console.log(`Backend processing complete: ${uniqueTriples.length} triples and ${entityList.length} entities stored using ${this.activeGraphDbType}`);
  }
  
  /**
   * Perform a traditional query using direct pattern matching on the graph
   * This bypasses the vector embeddings and uses text matching
   */
  public async queryTraditional(queryText: string): Promise<Triple[]> {
    console.log(`Performing traditional graph query: "${queryText}"`);

    // Extract keywords from query
    const keywords = this.extractKeywords(queryText);
    console.log(`Extracted keywords: ${keywords.join(', ')}`);

    // Filter out stop words
    const filteredKeywords = keywords.filter(kw => !this.isStopWord(kw));

    // If using ArangoDB, use its native graph traversal capabilities
    if (this.getRuntimeGraphDbType() === 'arangodb') {
      console.log(`Using ArangoDB native graph traversal for keywords: ${filteredKeywords.join(', ')}`);

      try {
        const results = await this.graphDBService.graphTraversal(filteredKeywords, 2, 100);
        console.log(`ArangoDB graph traversal found ${results.length} relevant triples`);

        // Log top 10 results with confidence scores for debugging
        console.log('Top 10 triples by confidence:');
        results.slice(0, 10).forEach((triple, idx) => {
          console.log(`  ${idx + 1}. [${triple.confidence.toFixed(3)}] ${triple.subject} -> ${triple.predicate} -> ${triple.object} (depth: ${triple.depth})`);
        });

        return results;
      } catch (error) {
        console.error('Error using ArangoDB graph traversal, falling back to traditional method:', error);
        // Fall through to traditional method if ArangoDB traversal fails
      }
    }

    // Fallback to traditional keyword matching for Neo4j or if ArangoDB traversal fails
    console.log(`Using fallback keyword-based search`);

    // Get graph data from graph database
    const graphData = await this.graphDBService.getGraphData();
    console.log(`Retrieved graph from ${this.activeGraphDbType} with ${graphData.nodes.length} nodes and ${graphData.relationships.length} relationships`);

    // Create a map of node IDs to names
    const nodeIdToName = new Map<string, string>();
    for (const node of graphData.nodes) {
      nodeIdToName.set(node.id, node.name);
    }

    // Find matching nodes based on keywords
    const matchingNodeIds = new Set<string>();
    for (const node of graphData.nodes) {
      for (const keyword of filteredKeywords) {
        // Simple text matching - convert to lowercase for case-insensitive matching
        if (node.name.toLowerCase().includes(keyword.toLowerCase())) {
          matchingNodeIds.add(node.id);
          break;
        }
      }
    }

    console.log(`Found ${matchingNodeIds.size} nodes matching keywords directly`);

    // Find relationships where either subject or object matches
    const relevantTriples: Triple[] = [];

    for (const rel of graphData.relationships) {
      // Check if either end of the relationship matches our search
      const isSourceMatching = matchingNodeIds.has(rel.source);
      const isTargetMatching = matchingNodeIds.has(rel.target);

      if (isSourceMatching || isTargetMatching) {
        const sourceName = nodeIdToName.get(rel.source);
        const targetName = nodeIdToName.get(rel.target);

        if (sourceName && targetName) {
          // Check if the relationship type matches keywords
          let matchesRelationship = false;
          for (const keyword of filteredKeywords) {
            if (rel.type.toLowerCase().includes(keyword.toLowerCase())) {
              matchesRelationship = true;
              break;
            }
          }

          // Higher relevance to relationships that match the query directly
          const relevance = (isSourceMatching ? 1 : 0) +
                           (isTargetMatching ? 1 : 0) +
                           (matchesRelationship ? 2 : 0);

          if (relevance > 0) {
            relevantTriples.push({
              subject: sourceName,
              predicate: rel.type,
              object: targetName,
              confidence: relevance / 4.0  // Scale from 0 to 1
            });
          }
        }
      }
    }

    // Sort by confidence (highest first)
    relevantTriples.sort((a, b) =>
      (b.confidence || 0) - (a.confidence || 0)
    );

    // Return all relevant triples, sorted by relevance
    console.log(`Found ${relevantTriples.length} relevant triples with traditional search`);
    return relevantTriples;
  }
  
  /**
   * Extract keywords from query text
   */
  private extractKeywords(text: string): string[] {
    return text.toLowerCase()
      .replace(/[.,?!;:()]/g, ' ')  // Remove punctuation
      .split(/\s+/)                  // Split by whitespace
      .filter(word => word.length > 2); // Filter out very short words
  }
  
  /**
   * Check if a word is a common stop word
   */
  private isStopWord(word: string): boolean {
    const stopWords = new Set([
      'the', 'and', 'are', 'for', 'was', 'with', 
      'how', 'what', 'why', 'who', 'when', 'which',
      'many', 'much', 'from', 'have', 'has', 'had',
      'that', 'this', 'these', 'those', 'they', 'their'
    ]);
    return stopWords.has(word.toLowerCase());
  }
  
  /**
   * Query the backend for relevant information
   */
  public async query(
    queryText: string, 
    kNeighbors: number = 4096, 
    fanout: number = 400, 
    numHops: number = 2,
    useTraditional: boolean = false
  ): Promise<Triple[]> {
    console.log(`Querying backend with database type: ${this.activeGraphDbType}, useTraditional: ${useTraditional}`);
    
    // If using traditional search, bypass the vector embeddings
    if (useTraditional) {
      return this.queryTraditional(queryText);
    }
    
    // Generate embedding for query
    const queryEmbedding = (await this.generateEmbeddings([queryText]))[0];
    
    // Find nearest neighbors using Qdrant
    const seedNodes = await this.qdrantService.findSimilarEntities(queryEmbedding, kNeighbors);
    console.log(`Found ${seedNodes.length} seed nodes for query: "${queryText}"`);
    
    // Get graph data from graph database
    const graphData = await this.graphDBService.getGraphData();
    console.log(`Retrieved graph from ${this.activeGraphDbType} with ${graphData.nodes.length} nodes and ${graphData.relationships.length} relationships`);
    
    // Build adjacency map for neighborhood exploration
    const adjacencyMap = new Map<string, string[]>();
    
    // Map Neo4j IDs to entity names
    const nodeIdToName = new Map<string, string>();
    for (const node of graphData.nodes) {
      nodeIdToName.set(node.id, node.name);
      adjacencyMap.set(node.name, []);
    }
    
    // Build adjacency lists
    for (const rel of graphData.relationships) {
      const sourceName = nodeIdToName.get(rel.source);
      const targetName = nodeIdToName.get(rel.target);
      
      if (sourceName && targetName) {
        const neighbors = adjacencyMap.get(sourceName) || [];
        neighbors.push(targetName);
        adjacencyMap.set(sourceName, neighbors);
      }
    }
    
    // Perform multi-hop exploration
    const visitedNodes = new Set<string>(seedNodes);
    const nodesToExplore = [...seedNodes];
    
    for (let hop = 0; hop < numHops; hop++) {
      const currentNodes = [...nodesToExplore];
      nodesToExplore.length = 0; // Clear the array
      
      for (const node of currentNodes) {
        const neighbors = adjacencyMap.get(node) || [];
        const limitedNeighbors = neighbors.slice(0, fanout);
        
        for (const neighbor of limitedNeighbors) {
          if (!visitedNodes.has(neighbor)) {
            visitedNodes.add(neighbor);
            nodesToExplore.push(neighbor);
          }
        }
      }
      
      console.log(`Hop ${hop+1}: Explored ${currentNodes.length} nodes, found ${nodesToExplore.length} new neighbors`);
    }
    
    // Extract relevant triples
    const relevantTriples: Triple[] = [];
    
    for (const rel of graphData.relationships) {
      const sourceName = nodeIdToName.get(rel.source);
      const targetName = nodeIdToName.get(rel.target);
      
      if (sourceName && targetName && 
         (visitedNodes.has(sourceName) || visitedNodes.has(targetName))) {
        // Include relationship type from metadata
        const predicate = rel.type === 'RELATIONSHIP' ? rel.type : rel.type;
        relevantTriples.push({
          subject: sourceName,
          predicate: predicate,
          object: targetName
        });
      }
    }
    
    // Apply local filtering (simplified version of PCST algorithm)
    // Just return top N triples for simplicity
    const topK = 5; // topk parameter from the Python example
    
    console.log(`Found ${relevantTriples.length} relevant triples, returning top ${topK * 5}`);
    return relevantTriples.slice(0, topK * 5);
  }
  
  /**
   * Query with LLM enhancement: retrieve triples and use LLM to generate answer
   * This makes traditional graph search comparable to RAG by adding LLM generation
   * @param queryText - The user's question
   * @param topK - Number of top triples to use as context (default 5)
   * @param useTraditional - Whether to use traditional (keyword-based) or vector search
   * @param llmModel - Optional LLM model to use (defaults to environment variable)
   * @param llmProvider - Optional LLM provider (ollama, nvidia, etc.)
   * @returns Generated answer from LLM based on retrieved triples
   */
  public async queryWithLLM(
    queryText: string,
    topK: number = 5,
    useTraditional: boolean = true,
    llmModel?: string,
    llmProvider?: string
  ): Promise<{ answer: string; triples: Triple[]; count: number }> {
    console.log(`Querying with LLM enhancement: "${queryText}", topK=${topK}, traditional=${useTraditional}`);
    
    // Step 1: Retrieve relevant triples using graph search
    const allTriples = await this.query(queryText, 4096, 400, 2, useTraditional);
    
    // Step 2: Take top K triples for context
    const topTriples = allTriples.slice(0, topK);
    console.log(`Using top ${topTriples.length} triples as context for LLM`);

    // DEBUG: Log first triple to verify depth/pathLength are present
    if (topTriples.length > 0) {
      console.log('First triple structure:', JSON.stringify(topTriples[0], null, 2));
    }
    
    if (topTriples.length === 0) {
      return {
        answer: "I couldn't find any relevant information in the knowledge graph to answer this question.",
        triples: [],
        count: 0
      };
    }
    
    // Step 3: Format triples as natural language context
    const context = topTriples
      .map(triple => {
        // Convert triple to natural language
        const predicate = triple.predicate
          .replace(/_/g, ' ')
          .replace(/-/g, ' ')
          .toLowerCase();
        return `${triple.subject} ${predicate} ${triple.object}`;
      })
      .join('. ');
    
    // Step 4: Use LLM to generate answer from context
    try {
      // Simplified prompt to work better with NVIDIA Nemotron's natural reasoning format
      const prompt = `Answer the question based on the following context from the knowledge graph.

Context:
${context}

Question: ${queryText}

Answer:`;

      // Determine LLM endpoint and model based on provider
      const finalProvider = llmProvider || 'ollama';
      const finalModel = llmModel || process.env.OLLAMA_MODEL || 'llama3.1:8b';

      console.log(`Using LLM: provider=${finalProvider}, model=${finalModel}`);

      let response;

      if (finalProvider === 'nvidia') {
        // Use NVIDIA API
        const nvidiaApiKey = process.env.NVIDIA_API_KEY;
        if (!nvidiaApiKey) {
          throw new Error('NVIDIA_API_KEY is required for NVIDIA provider. Please set the NVIDIA_API_KEY environment variable.');
        }

        const nvidiaUrl = 'https://integrate.api.nvidia.com/v1';

        // Note: NVIDIA API doesn't support streaming in axios, so we'll use non-streaming
        // and format the thinking content into <think> tags manually
        response = await axios.post(`${nvidiaUrl}/chat/completions`, {
          model: finalModel,
          messages: [
            {
              role: 'system',
              content: '/think'  // Special NVIDIA API command to activate thinking mode
            },
            {
              role: 'user',
              content: prompt
            }
          ],
          temperature: 0.2,
          max_tokens: 4096,
          top_p: 0.95,
          frequency_penalty: 0,
          presence_penalty: 0,
          stream: false,  // We need non-streaming to get thinking tokens
          // NVIDIA-specific thinking token parameters
          min_thinking_tokens: 1024,
          max_thinking_tokens: 2048
        }, {
          headers: {
            'Authorization': `Bearer ${nvidiaApiKey}`,
            'Content-Type': 'application/json'
          },
          timeout: 120000  // 120 second timeout
        });
      } else {
        // Use Ollama (default)
        const ollamaUrl = process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1';

        response = await axios.post(`${ollamaUrl}/chat/completions`, {
          model: finalModel,
          messages: [
            {
              role: 'system',
              content: 'You are a knowledgeable research assistant specializing in biomedical and scientific literature. Provide accurate, well-structured answers based on the provided context. Maintain a professional yet accessible tone, and clearly indicate when information is limited or uncertain.'
            },
            {
              role: 'user',
              content: prompt
            }
          ],
          temperature: 0.2,  // Lower for more factual, consistent responses
          max_tokens: 800    // Increased for more comprehensive answers
        });
      }

      // Extract answer and reasoning (if using NVIDIA with thinking tokens)
      const messageData = response.data.choices[0].message;
      let answer = messageData.content || '';

      // Check if NVIDIA API returned reasoning_content (thinking tokens)
      if (finalProvider === 'nvidia' && messageData.reasoning_content) {
        // Format with <think> tags for UI parsing
        answer = `<think>\n${messageData.reasoning_content}\n</think>\n\n${answer}`;
        console.log('Formatted response with thinking content');
      }

      // DEBUG: Log triples before returning to verify they still have depth/pathLength
      console.log('Returning triples (first one):', JSON.stringify(topTriples[0], null, 2));

      return {
        answer,
        triples: topTriples,
        count: topTriples.length
      };
    } catch (error) {
      console.error('Error calling LLM for answer generation:', error);
      // Fallback: return triples without LLM enhancement
      return {
        answer: `Found ${topTriples.length} relevant triples:\n\n${context}`,
        triples: topTriples,
        count: topTriples.length
      };
    }
  }

  /**
   * Store document chunks for Pure RAG (separate from entity embeddings)
   * This stores full text chunks rather than just entity names
   * @param documents Array of document text chunks
   * @param metadata Optional metadata for each document
   */
  public async storeDocumentChunks(
    documents: string[],
    metadata?: Record<string, any>[]
  ): Promise<void> {
    console.log(`Storing ${documents.length} document chunks for Pure RAG`);

    // Generate embeddings for document chunks
    const embeddings = await this.generateEmbeddings(documents);
    
    // Store in Qdrant document-embeddings collection
    await this.qdrantService.storeDocumentChunks(documents, embeddings, metadata);
    
    console.log(`✅ Stored ${documents.length} document chunks in document-embeddings collection`);
  }

  /**
   * Close connections to backend services
   */
  public async close(): Promise<void> {
    if (this.graphDBService.isInitialized()) {
      this.graphDBService.close();
    }
    console.log('Backend service closed');
  }
}

export default BackendService.getInstance(); 