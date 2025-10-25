import axios from 'axios';
import { GraphDBService, GraphDBType } from './graph-db-service';
import { QdrantService } from './qdrant';
import { getGraphDbService } from './graph-db-util';
import type { Triple } from '@/types/graph';

/**
 * Backend service that combines graph database for storage and Qdrant for embeddings
 */
export class BackendService {
  private graphDBService: GraphDBService;
  private pineconeService: QdrantService;
  private sentenceTransformerUrl: string = 'http://sentence-transformers:80';
  private modelName: string = 'all-MiniLM-L6-v2';
  private static instance: BackendService;
  private initialized: boolean = false;
  private activeGraphDbType: GraphDBType = 'arangodb';
  
  private constructor() {
    this.graphDBService = GraphDBService.getInstance();
    this.pineconeService = QdrantService.getInstance();
    
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
   * @param graphDbType - Type of graph database to use (neo4j or arangodb)
   */
  public async initialize(graphDbType: GraphDBType = 'arangodb'): Promise<void> {
    this.activeGraphDbType = graphDbType;
    
    // Initialize Graph Database
    if (!this.graphDBService.isInitialized()) {
      try {
        // Get the appropriate service based on type
        const graphDbService = getGraphDbService(graphDbType);
        
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
        
        if (graphDbType === 'neo4j') {
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
        console.log(`${graphDbType} initialized successfully in backend service`);
      } catch (error) {
        console.error(`Failed to initialize ${graphDbType} in backend service:`, error);
        if (process.env.NODE_ENV === 'development') {
          console.log('Development mode: Continuing despite graph database initialization error');
        } else {
          throw new Error('Graph database service initialization failed');
        }
      }
    }
    
    // Initialize Pinecone
    if (!this.pineconeService.isInitialized()) {
      await this.pineconeService.initialize();
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
    return this.activeGraphDbType;
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
   * Process and store triples in graph database and embeddings in Pinecone
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
    
    // Store embeddings and text content in Pinecone
    await this.pineconeService.storeEmbeddings(entityEmbeddings, textContent);
    
    console.log(`Backend processing complete: ${uniqueTriples.length} triples and ${entityList.length} entities stored using ${this.activeGraphDbType}`);
  }
  
  /**
   * Perform a traditional query using direct pattern matching on the graph
   * This bypasses the vector embeddings and uses text matching
   */
  public async queryTraditional(queryText: string): Promise<Triple[]> {
    console.log(`Performing traditional graph query: "${queryText}"`);
    
    // Get graph data from graph database
    const graphData = await this.graphDBService.getGraphData();
    console.log(`Retrieved graph from ${this.activeGraphDbType} with ${graphData.nodes.length} nodes and ${graphData.relationships.length} relationships`);
    
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
    
    // Find nearest neighbors using Pinecone
    const seedNodes = await this.pineconeService.findSimilarEntities(queryEmbedding, kNeighbors);
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
      // Research-backed prompt structure for KG-enhanced RAG
      // Based on: "Retrieval-Augmented Generation with Knowledge Graphs" (Lewis et al.)
      // and "Enhancing LLMs with Structured Knowledge" (Pan et al.)
      const prompt = `You are a knowledgeable research assistant with access to a curated scientific knowledge base.

Context: ${context}

Based on the information provided above, please answer the following question. Be direct and informative, synthesizing the key facts into a coherent response. Draw reasonable scientific inferences when appropriate, but clearly distinguish between what is directly stated and what is inferred. If critical information is missing, note this briefly.

Question: ${queryText}

Answer:`;

      // Determine LLM endpoint and model
      const finalProvider = llmProvider || 'ollama';
      const ollamaUrl = process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1';
      const finalModel = llmModel || process.env.OLLAMA_MODEL || 'llama3.1:8b';
      
      console.log(`Using LLM: provider=${finalProvider}, model=${finalModel}`);
      
      const response = await axios.post(`${ollamaUrl}/chat/completions`, {
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
      
      const answer = response.data.choices[0].message.content;
      
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