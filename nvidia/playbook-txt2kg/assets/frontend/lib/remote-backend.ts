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
import { GraphDBService, GraphDBType } from './graph-db-service';
import { QdrantService } from './qdrant';
import { EmbeddingsService } from './embeddings';
import { TextProcessor } from './text-processor';
import type { Triple } from '@/types/graph';

/**
 * Remote backend implementation that uses a graph database for storage,
 * Qdrant for vector embeddings, and SentenceTransformer for generating embeddings.
 * Follows the implementation in PyTorch Geometric's txt2kg.py
 * Enhanced with LangChain text processing for better extraction
 */
export class RemoteBackendService {
  private graphDBService: GraphDBService;
  private qdrantService: QdrantService;
  private embeddingsService: EmbeddingsService;
  private textProcessor: TextProcessor;
  private initialized: boolean = false;
  private static instance: RemoteBackendService;

  private constructor() {
    this.graphDBService = GraphDBService.getInstance();
    this.qdrantService = QdrantService.getInstance();
    this.embeddingsService = EmbeddingsService.getInstance();
    this.textProcessor = TextProcessor.getInstance();
  }

  /**
   * Get the singleton instance of RemoteBackendService
   */
  public static getInstance(): RemoteBackendService {
    if (!RemoteBackendService.instance) {
      RemoteBackendService.instance = new RemoteBackendService();
    }
    return RemoteBackendService.instance;
  }

  /**
   * Check if the backend is initialized
   */
  public isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Initialize the remote backend with all required services
   * @param graphDbType - Type of graph database to use (defaults to GRAPH_DB_TYPE env var)
   */
  public async initialize(graphDbType?: GraphDBType): Promise<void> {
    const dbType = graphDbType || (process.env.GRAPH_DB_TYPE as GraphDBType) || 'arangodb';
    console.log(`Initializing remote backend with ${dbType}...`);
    
    // Initialize Graph Database
    await this.graphDBService.initialize(dbType);
    console.log(`${dbType} service initialized`);
    
    // Initialize Qdrant
    await this.qdrantService.initialize();
    console.log('Qdrant service initialized');
    
    // Initialize Embeddings service
    await this.embeddingsService.initialize();
    console.log('Embeddings service initialized');
    
    this.initialized = true;
    console.log('Remote backend initialized successfully');
  }

  /**
   * Process raw text document and extract triples for the knowledge graph
   * @param text Raw text to process
   * @returns Extracted triples with metadata
   */
  public async processText(text: string): Promise<Array<Triple & { confidence: number, metadata: any }>> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    console.log(`Processing text document of length ${text.length}`);
    
    // Use the LangChain-based text processor to extract triples
    const triples = await this.textProcessor.processText(text);
    console.log(`Extracted ${triples.length} triples with metadata`);
    
    return triples;
  }

  /**
   * Create backend from raw text
   * @param text Raw text to process
   */
  public async createBackendFromText(text: string): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Extract triples with metadata using the text processor
    const processedTriples = await this.processText(text);
    
    // Convert to simple triples for storage
    const simpleTriples = processedTriples.map(triple => ({
      subject: triple.subject,
      predicate: triple.predicate,
      object: triple.object
    }));
    
    // Store triples in graph database
    await this.storeTriplesToNeo4j(simpleTriples);
    
    // Extract entities and generate embeddings
    const entities = this.extractEntitiesFromTriples(simpleTriples);
    console.log(`Extracted ${entities.length} unique entities from triples`);
    
    // Generate embeddings for entities
    const embeddings = await this.embeddingsService.encode(entities);
    console.log(`Generated embeddings for ${embeddings.length} entities`);
    
    // Create entity-embedding map with metadata
    const entityEmbeddings = new Map<string, number[]>();
    const textContent = new Map<string, string>();
    const entityMetadata = new Map<string, any>();
    
    // Process each entity
    for (let i = 0; i < entities.length; i++) {
      const entity = entities[i];
      entityEmbeddings.set(entity, embeddings[i]);
      textContent.set(entity, entity);
      
      // Collect metadata for each entity from processed triples
      const entityData: any = {
        types: [] as string[],
        contexts: [] as string[]
      };
      
      // Find all triples that mention this entity
      for (const triple of processedTriples) {
        // Check subject
        if (triple.subject.toLowerCase() === entity.toLowerCase()) {
          // Add subject type if available
          if (triple.metadata?.entityTypes?.[0] && !entityData.types.includes(triple.metadata.entityTypes[0])) {
            entityData.types.push(triple.metadata.entityTypes[0]);
          }
          
          // Add context if available
          if (triple.metadata?.context && !entityData.contexts.includes(triple.metadata.context)) {
            entityData.contexts.push(triple.metadata.context);
          }
        }
        
        // Check object
        if (triple.object.toLowerCase() === entity.toLowerCase()) {
          // Add object type if available
          if (triple.metadata?.entityTypes?.[1] && !entityData.types.includes(triple.metadata.entityTypes[1])) {
            entityData.types.push(triple.metadata.entityTypes[1]);
          }
          
          // Add context if available
          if (triple.metadata?.context && !entityData.contexts.includes(triple.metadata.context)) {
            entityData.contexts.push(triple.metadata.context);
          }
        }
      }
      
      entityMetadata.set(entity, entityData);
    }
    
    // Store embeddings and metadata in Qdrant
    await this.qdrantService.storeEmbeddingsWithMetadata(entityEmbeddings, textContent, entityMetadata);
    console.log('Stored embeddings with metadata in Qdrant');
    
    console.log('Backend created successfully from text');
  }

  /**
   * Create backend from triples
   * @param triples Array of triples to create backend from
   */
  public async createBackendFromTriples(triples: Triple[]): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Store triples in graph database
    await this.storeTriplesToNeo4j(triples);
    
    // Extract entities and generate embeddings
    const entities = this.extractEntitiesFromTriples(triples);
    console.log(`Extracted ${entities.length} unique entities from triples`);
    
    // Generate embeddings for entities
    const embeddings = await this.embeddingsService.encode(entities);
    console.log(`Generated embeddings for ${embeddings.length} entities`);
    
    // Create entity-embedding map with simple metadata
    const entityEmbeddings = new Map<string, number[]>();
    const textContent = new Map<string, string>();
    const entityMetadata = new Map<string, any>();
    
    // Process each entity
    for (let i = 0; i < entities.length; i++) {
      const entity = entities[i];
      entityEmbeddings.set(entity, embeddings[i]);
      textContent.set(entity, entity);
      
      // Simple metadata for triples
      entityMetadata.set(entity, {
        types: [],
        contexts: []
      });
    }
    
    // Store embeddings and metadata in Qdrant
    await this.qdrantService.storeEmbeddingsWithMetadata(entityEmbeddings, textContent, entityMetadata);
    console.log('Stored embeddings with metadata in Qdrant');
    
    console.log('Backend created successfully from triples');
  }

  /**
   * Store triples in graph database
   * @param triples - Array of triples to store
   */
  public async storeTriplesToNeo4j(triples: Triple[]): Promise<void> {
    // Triples are already in the correct format for graph database
    await this.graphDBService.importTriples(triples);
  }

  /**
   * Extract unique entities from triples
   */
  private extractEntitiesFromTriples(triples: Triple[]): string[] {
    const entitySet = new Set<string>();
    
    for (const triple of triples) {
      entitySet.add(triple.subject); // subject
      entitySet.add(triple.object); // object
    }
    
    return Array.from(entitySet);
  }

  /**
   * Query the backend
   * @param query Query text
   * @param kNeighbors Number of KNN neighbors to retrieve
   * @param fanout Number of neighbors for each node in neighborhood sampling
   * @param numHops Number of hops for neighborhood sampling
   * @param filterParams Parameters for local filtering
   * @param useTraditional Whether to use traditional search (direct pattern matching)
   */
  public async query(
    query: string,
    kNeighbors: number = 4096,
    fanout: number = 400,
    numHops: number = 2,
    filterParams: { topk: number, topk_e: number, cost_e: number, num_clusters: number } = 
      { topk: 5, topk_e: 5, cost_e: 0.5, num_clusters: 2 },
    useTraditional: boolean = false
  ): Promise<Triple[]> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    console.log(`Querying backend with: "${query}"`);
    console.log(`Parameters: kNeighbors=${kNeighbors}, fanout=${fanout}, numHops=${numHops}, useTraditional=${useTraditional}`);
    
    // Use traditional search if specified (direct pattern matching)
    if (useTraditional) {
      return this.queryTraditional(query);
    }
    
    // Step 1: Generate embedding for query
    const queryEmbedding = (await this.embeddingsService.encode([query]))[0];
    
    // Step 2: Find nearest neighbors using Qdrant
    const seedNodes = await this.qdrantService.findSimilarEntities(queryEmbedding, kNeighbors);
    console.log(`Found ${seedNodes.length} seed nodes using KNN`);
    
    // Step 3: Retrieve graph data from graph database
    const graphData = await this.graphDBService.getGraphData();
    console.log(`Retrieved graph with ${graphData.nodes.length} nodes and ${graphData.relationships.length} relationships`);
    
    // Step 4: Build adjacency map for neighborhood sampling
    const adjacencyMap = this.buildAdjacencyMap(graphData);
    
    // Step 5: Perform neighborhood sampling
    const subgraphNodes = this.performNeighborhoodSampling(seedNodes, adjacencyMap, fanout, numHops);
    console.log(`Neighborhood sampling found ${subgraphNodes.size} nodes`);
    
    // Step 6: Extract relevant triples
    const relevantTriples = this.extractRelevantTriples(graphData, subgraphNodes);
    console.log(`Extracted ${relevantTriples.length} relevant triples`);
    
    // Step 7: Apply local filtering
    const filteredTriples = this.applyLocalFiltering(relevantTriples, filterParams);
    console.log(`Applied local filtering, returned ${filteredTriples.length} triples`);
    
    return filteredTriples;
  }

  /**
   * Perform a traditional query using direct pattern matching on the graph
   * This bypasses the vector embeddings and uses text matching
   */
  private async queryTraditional(queryText: string): Promise<Triple[]> {
    console.log(`Performing traditional graph query: "${queryText}"`);
    
    // Get graph data from graph database
    const graphData = await this.graphDBService.getGraphData();
    console.log(`Retrieved graph with ${graphData.nodes.length} nodes and ${graphData.relationships.length} relationships`);
    
    // Create a map of node IDs to names
    const nodeIdToName = new Map<string, string>();
    for (const node of graphData.nodes) {
      nodeIdToName.set(node.id, node.name);
    }
    
    // Extract keywords from query
    const keywords = this.extractKeywords(queryText);
    console.log(`Extracted keywords: ${keywords.join(', ')}`);
    
    // Find matching nodes based on keywords
    const matchingNodeIds = new Set<string>();
    for (const node of graphData.nodes) {
      for (const keyword of keywords) {
        // Skip common words
        if (this.isStopWord(keyword)) continue;
        
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
          for (const keyword of keywords) {
            if (this.isStopWord(keyword)) continue;
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
   * Build adjacency map from graph data
   */
  private buildAdjacencyMap(graphData: any): Map<string, string[]> {
    const adjacencyMap = new Map<string, string[]>();
    const nodeIdToName = new Map<string, string>();
    
    // Map node IDs to names
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
    
    return adjacencyMap;
  }

  /**
   * Perform neighborhood sampling starting from seed nodes
   */
  private performNeighborhoodSampling(
    seedNodes: string[],
    adjacencyMap: Map<string, string[]>,
    fanout: number,
    numHops: number
  ): Set<string> {
    const visitedNodes = new Set<string>(seedNodes);
    let nodesToExplore = [...seedNodes];
    
    for (let hop = 0; hop < numHops; hop++) {
      const currentNodes = [...nodesToExplore];
      nodesToExplore = [];
      
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
    
    return visitedNodes;
  }

  /**
   * Extract relevant triples from graph data based on subgraph nodes
   */
  private extractRelevantTriples(graphData: any, subgraphNodes: Set<string>): Triple[] {
    const relevantTriples: Triple[] = [];
    const nodeIdToName = new Map<string, string>();
    
    // Map node IDs to names
    for (const node of graphData.nodes) {
      nodeIdToName.set(node.id, node.name);
    }
    
    // Extract relevant relationships
    for (const rel of graphData.relationships) {
      const sourceName = nodeIdToName.get(rel.source);
      const targetName = nodeIdToName.get(rel.target);
      
      if (sourceName && targetName && 
          (subgraphNodes.has(sourceName) || subgraphNodes.has(targetName))) {
        // Get relationship type from metadata
        const predicate = rel.type === 'RELATIONSHIP' && rel.type ? rel.type : rel.type;
        relevantTriples.push({ subject: sourceName, predicate: predicate, object: targetName });
      }
    }
    
    return relevantTriples;
  }

  /**
   * Apply local filtering to triples (simplified PCST algorithm)
   */
  private applyLocalFiltering(
    triples: Triple[],
    params: { topk: number, topk_e: number, cost_e: number, num_clusters: number }
  ): Triple[] {
    // For simplicity, just return top N triples
    // A full implementation would use the Prize-Collecting Steiner Tree algorithm
    const totalResultSize = params.topk * params.topk_e * params.num_clusters;
    return triples.slice(0, totalResultSize);
  }

  /**
   * Enhanced query method that uses entity metadata for better retrieval
   * @param query Query text
   * @param kNeighbors Number of KNN neighbors to retrieve
   * @param fanout Number of neighbors for each node in neighborhood sampling
   * @param numHops Number of hops for neighborhood sampling
   * @param filterParams Parameters for local filtering
   */
  public async enhancedQuery(
    query: string,
    kNeighbors: number = 4096,
    fanout: number = 400,
    numHops: number = 2,
    filterParams: { topk: number, topk_e: number, cost_e: number, num_clusters: number } = 
      { topk: 5, topk_e: 5, cost_e: 0.5, num_clusters: 2 }
  ): Promise<{ relevantTriples: Triple[], queryMetadata: any }> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    console.log(`Enhanced query with: "${query}"`);
    
    // Step 1: Generate embedding for query
    const queryEmbedding = (await this.embeddingsService.encode([query]))[0];
    
    // Step 2: Find nearest neighbors using Qdrant with metadata
    const { entities: seedNodes, metadata: seedMetadata } = 
      await this.qdrantService.findSimilarEntitiesWithMetadata(queryEmbedding, kNeighbors);
    console.log(`Found ${seedNodes.length} seed nodes using KNN with metadata`);
    
    // Step 3: Retrieve graph data from graph database
    const graphData = await this.graphDBService.getGraphData();
    console.log(`Retrieved graph with ${graphData.nodes.length} nodes and ${graphData.relationships.length} relationships`);
    
    // Step 4: Build adjacency map for neighborhood sampling
    const adjacencyMap = this.buildAdjacencyMap(graphData);
    
    // Step 5: Perform enhanced neighborhood sampling with metadata weighting
    const { subgraphNodes, nodeScores } = this.performEnhancedSampling(seedNodes, seedMetadata, adjacencyMap, fanout, numHops);
    console.log(`Enhanced sampling found ${subgraphNodes.size} nodes`);
    
    // Step 6: Extract relevant triples with scores
    const scoredTriples = this.extractRelevantTriplesWithScores(graphData, subgraphNodes, nodeScores);
    console.log(`Extracted ${scoredTriples.length} relevant triples with scores`);
    
    // Step 7: Apply improved local filtering using metadata
    const filteredTriples = this.applyImprovedFiltering(scoredTriples, filterParams);
    console.log(`Applied improved filtering, returned ${filteredTriples.length} triples`);
    
    // Collect query metadata for analysis and debugging
    const queryMetadata = {
      entityMatches: seedNodes.length,
      topEntityScores: Object.fromEntries(
        seedNodes.slice(0, 5).map((node: string, i: number) => [node, nodeScores.get(node) || 0])
      ),
      retrievalStats: {
        initialTriples: scoredTriples.length,
        finalTriples: filteredTriples.length,
        avgScore: scoredTriples.reduce((sum, t) => sum + (t.score || 0), 0) / scoredTriples.length
      }
    };
    
    return { 
      relevantTriples: filteredTriples,
      queryMetadata
    };
  }

  /**
   * Perform enhanced neighborhood sampling with metadata weighting
   * @private
   */
  private performEnhancedSampling(
    seedNodes: string[],
    seedMetadata: Map<string, any>,
    adjacencyMap: Map<string, string[]>,
    fanout: number,
    numHops: number
  ): { subgraphNodes: Set<string>, nodeScores: Map<string, number> } {
    const visitedNodes = new Set<string>(seedNodes);
    const nodeScores = new Map<string, number>();
    
    // Initialize scores for seed nodes based on metadata relevance
    for (let i = 0; i < seedNodes.length; i++) {
      const node = seedNodes[i];
      const metadata = seedMetadata.get(node);
      
      // Base score is inversely proportional to position in results
      let score = 1.0 - (i / seedNodes.length);
      
      // Boost score based on metadata if available
      if (metadata) {
        // Boost if node has types
        if (metadata.types && metadata.types.length > 0) {
          score += 0.2;
        }
        
        // Boost if node has rich context
        if (metadata.contexts && metadata.contexts.length > 0) {
          score += 0.1 * Math.min(metadata.contexts.length, 3);
        }
      }
      
      nodeScores.set(node, score);
    }
    
    let currentLayer = [...seedNodes];
    
    // BFS traversal with hop-dependent score decay
    for (let hop = 0; hop < numHops; hop++) {
      const nextLayer: string[] = [];
      const hopDecay = 0.7 ** hop; // Score decays with distance
      
      for (const node of currentLayer) {
        const nodeScore = nodeScores.get(node) || 0;
        const neighbors = adjacencyMap.get(node) || [];
        
        // Sort neighbors by any existing scores
        const scoredNeighbors = neighbors.map(neighbor => ({
          neighbor,
          score: nodeScores.get(neighbor) || 0
        }));
        
        // Sort by score (if any have scores) and take top 'fanout'
        scoredNeighbors.sort((a, b) => b.score - a.score);
        const limitedNeighbors = scoredNeighbors.slice(0, fanout);
        
        for (const { neighbor } of limitedNeighbors) {
          // Propagate score to neighbor with decay
          const propagatedScore = nodeScore * hopDecay;
          const currentScore = nodeScores.get(neighbor) || 0;
          
          // Update score if propagated score is higher
          if (propagatedScore > currentScore) {
            nodeScores.set(neighbor, propagatedScore);
          }
          
          if (!visitedNodes.has(neighbor)) {
            visitedNodes.add(neighbor);
            nextLayer.push(neighbor);
          }
        }
      }
      
      currentLayer = nextLayer;
    }
    
    return { subgraphNodes: visitedNodes, nodeScores };
  }

  /**
   * Extract relevant triples with relevance scores
   * @private
   */
  private extractRelevantTriplesWithScores(
    graphData: any, 
    subgraphNodes: Set<string>,
    nodeScores: Map<string, number>
  ): Array<Triple & { score?: number }> {
    const nodeIdToName = new Map<string, string>();
    const nodeNameToId = new Map<string, string>();
    
    // Map node IDs to names
    for (const node of graphData.nodes) {
      nodeIdToName.set(node.id, node.name);
      nodeNameToId.set(node.name, node.id);
    }
    
    const relevantTriples: Array<Triple & { score?: number }> = [];
    
    // Extract triples involving subgraph nodes and compute scores
    for (const rel of graphData.relationships) {
      const sourceNode = nodeIdToName.get(rel.source);
      const targetNode = nodeIdToName.get(rel.target);
      
      if (sourceNode && targetNode && 
          subgraphNodes.has(sourceNode) && 
          subgraphNodes.has(targetNode)) {
        
        // Calculate triple score based on endpoint nodes
        const sourceScore = nodeScores.get(sourceNode) || 0;
        const targetScore = nodeScores.get(targetNode) || 0;
        const tripleScore = (sourceScore + targetScore) / 2;
        
        relevantTriples.push({
          subject: sourceNode,
          predicate: rel.type,
          object: targetNode,
          score: tripleScore
        });
      }
    }
    
    // Sort by score
    relevantTriples.sort((a, b) => (b.score || 0) - (a.score || 0));
    
    return relevantTriples;
  }

  /**
   * Improved filtering with metadata awareness
   * @private
   */
  private applyImprovedFiltering(
    scoredTriples: Array<Triple & { score?: number }>,
    filterParams: { topk: number, topk_e: number, cost_e: number, num_clusters: number }
  ): Triple[] {
    // For now, simply take the top-k by score
    // In a future improvement, implement a proper prize-collecting Steiner tree algorithm
    return scoredTriples
      .slice(0, filterParams.topk)
      .map(({ subject, predicate, object }) => ({ subject, predicate, object }));
  }

  /**
   * Close connections to services
   */
  public async close(): Promise<void> {
    if (this.graphDBService.isInitialized()) {
      this.graphDBService.close();
    }
    
    this.initialized = false;
    console.log('Remote backend closed');
  }
}

export default RemoteBackendService.getInstance(); 