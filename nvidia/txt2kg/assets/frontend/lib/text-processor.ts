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
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { StructuredOutputParser } from "langchain/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import { Triple } from "@/types/graph";
import axios from "axios";

import { ChatOpenAI } from "@langchain/openai";
import { Document } from "langchain/document";
import { LLMGraphTransformer } from "@langchain/community/experimental/graph_transformers/llm";
import { BaseLanguageModel } from "@langchain/core/language_models/base";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { langChainService } from "./langchain-service";
import { getShouldStopProcessing, resetStopProcessing } from "@/app/api/stop-processing/route";

// Define a type for sentence with embedding
export interface SentenceEmbedding {
  sentence: string;
  embedding: number[];
  metadata?: {
    index: number;
    documentId?: string;
    context?: string;
  };
}

// Define interfaces for graph document types
interface NodeType {
  id: string;
  type: string;
  properties?: Record<string, any>;
}

interface RelationshipType {
  source: NodeType;
  target: NodeType;
  type: string;
  properties?: Record<string, any>;
}

interface GraphDocument {
  nodes: NodeType[];
  relationships: RelationshipType[];
}

interface LLMGraphTransformerOptions {
  llm: BaseLanguageModel;
  allowedNodes?: string[];
  allowedRelationships?: string[];
  nodeProperties?: string[];
}

// Add new interface for prompt options
interface PromptOptions {
  systemPrompt?: string;
  extractionPrompt?: string;
  graphTransformerPrompt?: string;
}

/**
 * Text processing pipeline using LangChain.js that:
 * 1. Chunks documents into optimal sizes
 * 2. Extracts entities and relationships
 * 3. Performs metadata enrichment
 * 4. Outputs structured triples for the knowledge graph
 */
export class TextProcessor {
  private static instance: TextProcessor;
  private sentenceTransformerUrl: string;
  private modelName: string;
  private llm: ChatOpenAI | null = null;
  private tripleParser: StructuredOutputParser<any> | null = null;
  private extractionTemplate: PromptTemplate | null = null;
  private selectedLLMProvider: 'ollama' | 'nvidia' | 'vllm' = 'ollama';
  private ollamaModel: string = 'llama3.1:8b';
  private ollamaBaseUrl: string = 'http://localhost:11434/v1';
  private vllmModel: string = 'meta-llama/Llama-3.2-3B-Instruct';
  private vllmBaseUrl: string = 'http://localhost:8001/v1';
  private nvidiaModel: string = 'nvidia/llama-3.3-nemotron-super-49b-v1.5'; // Default NVIDIA model

  private constructor() {
    this.sentenceTransformerUrl = process.env.SENTENCE_TRANSFORMER_URL || "http://localhost:8000";
    this.modelName = process.env.MODEL_NAME || "all-MiniLM-L6-v2";
    
    // Check for Ollama configuration
    this.ollamaBaseUrl = process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1';
    this.ollamaModel = process.env.OLLAMA_MODEL || 'llama3.1:8b';
    
    // Check for vLLM configuration
    this.vllmBaseUrl = process.env.VLLM_BASE_URL || 'http://localhost:8001/v1';
    this.vllmModel = process.env.VLLM_MODEL || 'meta-llama/Llama-3.2-3B-Instruct';
    
    // Determine which LLM provider to use based on configuration
    // Priority: vLLM > NVIDIA > Ollama
    if (process.env.VLLM_BASE_URL) {
      this.selectedLLMProvider = 'vllm';
    } else if (process.env.NVIDIA_API_KEY) {
      this.selectedLLMProvider = 'nvidia';
    } else {
      // Default to Ollama (no API key required)
      this.selectedLLMProvider = 'ollama';
    }
  }

  /**
   * Get the singleton instance of TextProcessor
   */
  public static getInstance(): TextProcessor {
    if (!TextProcessor.instance) {
      TextProcessor.instance = new TextProcessor();
    }
    return TextProcessor.instance;
  }

  /**
   * Initialize the TextProcessor with the required components
   */
  public async initialize(): Promise<void> {
    // Only require API keys for specific providers, Ollama works without API keys
    if (this.selectedLLMProvider === 'nvidia' && !process.env.NVIDIA_API_KEY) {
      throw new Error("NVIDIA API key is required when using NVIDIA provider. Please set NVIDIA_API_KEY in your environment variables.");
    }

    // Initialize LLM based on selected provider
    switch (this.selectedLLMProvider) {
      
      case 'ollama':
        try {
          this.llm = await langChainService.getOllamaModel(this.ollamaModel, {
            temperature: 0.1,
            maxTokens: 8192,
            baseURL: this.ollamaBaseUrl
          });
        } catch (error) {
          console.error('Failed to initialize Ollama model:', error);
          throw new Error(`Failed to initialize Ollama model: ${error instanceof Error ? error.message : String(error)}`);
        }
        break;
      
      case 'nvidia':
        try {
          // For NVIDIA, we'll use direct OpenAI client instead of LangChain
          // This is handled in processText method
          this.llm = null; // Set to null, will be handled differently
        } catch (error) {
          console.error('Failed to initialize NVIDIA model:', error);
          throw new Error(`Failed to initialize NVIDIA model: ${error instanceof Error ? error.message : String(error)}`);
        }
        break;
      
      case 'vllm':
        try {
          this.llm = await langChainService.getVllmModel(this.vllmModel, {
            temperature: 0.1,
            maxTokens: 8192,
            baseURL: this.vllmBaseUrl
          });
        } catch (error) {
          console.error('Failed to initialize vLLM model:', error);
          throw new Error(`Failed to initialize vLLM model: ${error instanceof Error ? error.message : String(error)}`);
        }
        break;
    }

    // Initialize Triple Parser
    this.tripleParser = StructuredOutputParser.fromZodSchema(
      z.array(
        z.object({
          subject: z.string().describe("The subject entity of the triple"),
          predicate: z.string().describe("The relation/predicate connecting subject and object"),
          object: z.string().describe("The object entity of the triple"),
          confidence: z.number().min(0).max(1).describe("Confidence score between 0 and 1"),
          metadata: z.object({
            entityTypes: z.array(z.string()).describe("Entity types for subject and object"),
            source: z.string().describe("The source text this triple was extracted from"),
            context: z.string().describe("Surrounding context for the triple")
          }).describe("Additional metadata about the triple")
        })
      ).describe("Array of knowledge graph triples extracted from the text")
    );

    // Initialize Extraction Template
    const templateString = `
      You are a knowledge graph builder that extracts structured information from text.
      Extract subject-predicate-object triples from the following text.
      
      Guidelines:
      - Extract only factual triples present in the text
      - Normalize entity names to their canonical form
      - Assign appropriate confidence scores (0-1)
      - Include entity types in metadata
      - For each triple, include a brief context from the source text
      
      Text: {text}
      
      {format_instructions}
    `;

    this.extractionTemplate = PromptTemplate.fromTemplate(templateString);
  }

  /**
   * Process text to extract structured triples
   * @param text Text to process
   * @returns Array of triples with metadata
   */
  public async processText(text: string): Promise<Array<Triple & { confidence: number, metadata: any }>> {
    if (!this.llm || !this.tripleParser || !this.extractionTemplate) {
      await this.initialize();
    }

    // For NVIDIA, use direct OpenAI client
    if (this.selectedLLMProvider === 'nvidia') {
      return await this.processTextWithNvidiaAPI(text);
    }

    // Ensure we have an LLM to extract triples for non-NVIDIA providers
    if (!this.llm) {
      const providerMessage = this.selectedLLMProvider === 'ollama'
        ? "Ollama server connection failed. Please ensure Ollama is running and accessible."
        : "LLM configuration error";
      throw new Error(`LLM configuration error: ${providerMessage}`);
    }

    // Step 1: Chunk the text into manageable pieces
    const chunks = await this.chunkText(text);
    console.log(`Split text into ${chunks.length} chunks`);

    // Step 2: Process chunks in parallel with controlled concurrency
    // DGX Spark has unified memory, so we can prefetch batches into GPU before processing
    const concurrency = this.selectedLLMProvider === 'ollama' ? 4 : 2; // Higher concurrency for local Ollama
    const allTriples: Array<Triple & { confidence: number, metadata: any }> = [];
    
    console.log(`Processing with concurrency: ${concurrency} (provider: ${this.selectedLLMProvider})`);
    
    // Helper function to process a single chunk
    const processChunk = async (chunk: string, index: number) => {
      // Check if processing should be stopped
      if (getShouldStopProcessing()) {
        console.log(`Processing stopped by user at chunk ${index + 1}/${chunks.length}`);
        resetStopProcessing();
        throw new Error('Processing stopped by user');
      }
      
      console.log(`Processing chunk ${index + 1}/${chunks.length} (${chunk.length} chars)`);
      
      // Format the prompt with the chunk and parser instructions
      const formatInstructions = this.tripleParser!.getFormatInstructions();
      const prompt = await this.extractionTemplate!.format({
        text: chunk,
        format_instructions: formatInstructions
      });

      // Extract triples using the LLM
      const response = await this.llm!.invoke(prompt);
      const responseText = response.content as string;
      const parsedTriples = await this.tripleParser!.parse(responseText);
      
      return parsedTriples;
    };

    // Process chunks in batches with controlled concurrency
    for (let i = 0; i < chunks.length; i += concurrency) {
      const batch = chunks.slice(i, i + concurrency);
      const batchIndices = Array.from({ length: batch.length }, (_, idx) => i + idx);
      
      console.log(`Processing batch ${Math.floor(i / concurrency) + 1}/${Math.ceil(chunks.length / concurrency)} (${batch.length} chunks in parallel)`);
      
      try {
        // Process batch in parallel - GPU can prefetch next chunks while processing current ones
        const results = await Promise.all(
          batch.map((chunk, idx) => processChunk(chunk, batchIndices[idx]))
        );
        
        // Flatten and add to all triples
        results.forEach((triples: Array<Triple & { confidence: number, metadata: any }>) => {
          allTriples.push(...triples);
        });
      } catch (error) {
        console.error(`Error processing batch:`, error);
        // Continue with next batch instead of failing completely
        if (error instanceof Error && error.message === 'Processing stopped by user') {
          throw error;
        }
      }
    }

    // Step 3: Post-process to remove duplicates and normalize
    const processedTriples = this.postProcessTriples(allTriples);
    console.log(`Extracted ${processedTriples.length} unique triples after post-processing`);

    return processedTriples;
  }

  /**
   * Process text using NVIDIA API directly with OpenAI client (bypasses LangChain)
   * @param text Text to process
   * @returns Array of triples with metadata
   */
  private async processTextWithNvidiaAPI(text: string): Promise<Array<Triple & { confidence: number, metadata: any }>> {
    const apiKey = process.env.NVIDIA_API_KEY;
    if (!apiKey) {
      throw new Error('NVIDIA_API_KEY is required but not set');
    }

    // Initialize parser if needed
    if (!this.tripleParser) {
      await this.initialize();
    }

    // Step 1: Chunk the text
    const chunks = await this.chunkText(text);
    console.log(`Split text into ${chunks.length} chunks`);

    // Step 2: Process each chunk
    const allTriples: Array<Triple & { confidence: number, metadata: any }> = [];
    
    for (let i = 0; i < chunks.length; i++) {
      // Check if processing should be stopped
      if (getShouldStopProcessing()) {
        console.log(`Processing stopped by user at chunk ${i + 1}/${chunks.length}`);
        resetStopProcessing();
        throw new Error('Processing stopped by user');
      }
      
      const chunk = chunks[i];
      console.log(`Processing chunk ${i + 1}/${chunks.length} (${chunk.length} chars)`);
      
      try {
        // Create the prompt
        const formatInstructions = this.tripleParser!.getFormatInstructions();
        const prompt = `You are a knowledge graph builder that extracts structured information from text.
Extract subject-predicate-object triples from the following text.

Guidelines:
- Extract only factual triples present in the text
- Normalize entity names to their canonical form
- Assign appropriate confidence scores (0-1)
- Include entity types in metadata
- For each triple, include a brief context from the source text

Text: ${chunk}

${formatInstructions}`;

        // Call NVIDIA API directly using fetch
        console.log(`🖥️ Calling NVIDIA API with model: ${this.nvidiaModel}`);
        const response = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
          },
          body: JSON.stringify({
            model: this.nvidiaModel, // Use the configured model
            messages: [
              {
                role: 'user',
                content: prompt
              }
            ],
            temperature: 0.1,
            max_tokens: 4096,  // Reduced to leave room for input tokens in context
            top_p: 0.95
          })
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`NVIDIA API error (${response.status}): ${errorText}`);
        }

        const data = await response.json();
        const responseText = data.choices[0].message.content;
        
        // Parse the response
        const parsedTriples = await this.tripleParser!.parse(responseText);
        allTriples.push(...parsedTriples);
        
      } catch (error) {
        console.error(`Error processing chunk ${i + 1}:`, error);
        throw error; // Re-throw to see the actual error
      }
    }

    // Step 3: Post-process
    const processedTriples = this.postProcessTriples(allTriples);
    console.log(`Extracted ${processedTriples.length} unique triples after post-processing`);

    return processedTriples;
  }

  /**
   * Split text into chunks of appropriate size
   * @param text Text to split
   * @returns Array of text chunks
   */
  private async chunkText(text: string): Promise<string[]> {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 64000,         // Increased chunk size for Llama 70B models (16K tokens)
      chunkOverlap: 1000,       // Increased overlap to maintain context
      separators: ["\n\n", "\n", ". ", " ", ""],  // Preferred split locations
    });

    return await splitter.splitText(text);
  }

  /**
   * Split text into sentence-level chunks
   * @param text Text to split into sentences
   * @returns Array of sentences
   */
  public async splitIntoSentences(text: string): Promise<string[]> {
    const sentenceSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,         // Maximum sentence length (very long to ensure sentences aren't split)
      chunkOverlap: 0,         // No overlap for sentences
      separators: [". ", "! ", "? ", "\n", "\t"],  // Sentence endings and paragraph breaks
    });

    // First split by paragraphs, then by sentence delimiters
    const paragraphs = text.split(/\n{2,}/);  // Split on double newlines for paragraphs
    const sentences: string[] = [];

    for (const paragraph of paragraphs) {
      if (paragraph.trim().length === 0) continue;

      // Further split by sentence delimiters
      const paragraphSentences = await sentenceSplitter.splitText(paragraph);
      sentences.push(...paragraphSentences);
    }

    // Clean up sentences
    return sentences
      .map(s => s.trim())
      .filter(s => s.length >= 10);  // Filter out very short sentences
  }

  /**
   * Generate embeddings using local Sentence Transformer service
   * @param texts Array of texts to embed
   * @returns Array of embeddings
   */
  private async generateEmbeddings(texts: string[]): Promise<number[][]> {
    try {
      console.log(`Generating embeddings for ${texts.length} texts using local Sentence Transformer service`);
      
      // Use the sentence-transformers service defined in docker-compose
      const response = await axios.post(`${this.sentenceTransformerUrl}/embed`, {
        texts: texts,
        model: this.modelName
      });
      
      if (response.status !== 200) {
        throw new Error(`Failed to generate embeddings: ${response.statusText}`);
      }
      
      return response.data.embeddings;
    } catch (error) {
      console.error('Error generating embeddings with Sentence Transformer:', error);
      throw new Error(`Failed to generate embeddings: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Generate embeddings for an array of sentences
   * @param sentences Array of sentences to embed
   * @param documentId Optional document identifier for metadata
   * @returns Array of sentence embeddings
   */
  public async generateSentenceEmbeddings(
    sentences: string[], 
    documentId?: string
  ): Promise<SentenceEmbedding[]> {
    console.log(`Generating embeddings for ${sentences.length} sentences`);

    // Generate embeddings using the local Sentence Transformer service
    const embeddings = await this.generateEmbeddings(sentences);

    // Map embeddings to sentences with metadata
    return sentences.map((sentence, i) => ({
      sentence,
      embedding: embeddings[i],
      metadata: {
        index: i,
        documentId: documentId || undefined,
        context: this.getSentenceContext(sentences, i),
      }
    }));
  }

  /**
   * Get surrounding context for a sentence
   * @private
   */
  private getSentenceContext(sentences: string[], index: number): string {
    // Get previous and next sentence as context if available
    const previousSentence = index > 0 ? sentences[index - 1] : '';
    const nextSentence = index < sentences.length - 1 ? sentences[index + 1] : '';
    
    // Create a context window with up to 3 sentences
    let context = sentences[index];
    
    if (previousSentence) {
      context = previousSentence + ' ' + context;
    }
    
    if (nextSentence) {
      context = context + ' ' + nextSentence;
    }
    
    return context;
  }

  /**
   * Post-process extracted triples to remove duplicates and normalize
   * @param triples Array of raw triples
   * @returns Array of processed triples
   */
  private postProcessTriples(
    triples: Array<Triple & { confidence: number, metadata: any }>
  ): Array<Triple & { confidence: number, metadata: any }> {
    // Convert to lowercase for comparison
    const normalizedTriples = triples.map(triple => ({
      ...triple,
      subject: triple.subject.toLowerCase().trim(),
      predicate: triple.predicate.toLowerCase().trim(),
      object: triple.object.toLowerCase().trim()
    }));

    // Remove duplicates using a Map with string key
    const tripleMap = new Map<string, Triple & { confidence: number, metadata: any }>();
    
    for (const triple of normalizedTriples) {
      const key = `${triple.subject}|${triple.predicate}|${triple.object}`;
      
      // If triple exists, keep the one with higher confidence
      if (tripleMap.has(key)) {
        const existingTriple = tripleMap.get(key)!;
        if (triple.confidence > existingTriple.confidence) {
          tripleMap.set(key, triple);
        }
      } else {
        tripleMap.set(key, triple);
      }
    }

    // Filter out low confidence triples
    return Array.from(tripleMap.values())
      .filter(triple => triple.confidence >= 0.6) // Only keep reasonably confident triples
      .sort((a, b) => b.confidence - a.confidence); // Sort by confidence (highest first)
  }

  // Make LLM accessible for the LLMGraphTransformer
  public getLLM(): ChatOpenAI | null {
    return this.llm;
  }

  /**
   * Set the LLM provider to use for triple extraction
   */
  public setLLMProvider(provider: 'ollama' | 'nvidia' | 'vllm', options?: { 
    ollamaModel?: string; 
    ollamaBaseUrl?: string;
    vllmModel?: string;
    vllmBaseUrl?: string;
    nvidiaModel?: string;
  }): void {
    this.selectedLLMProvider = provider;
    if (provider === 'ollama') {
      this.ollamaModel = options?.ollamaModel || this.ollamaModel;
      this.ollamaBaseUrl = options?.ollamaBaseUrl || this.ollamaBaseUrl;
    } else if (provider === 'nvidia') {
      this.nvidiaModel = options?.nvidiaModel || this.nvidiaModel;
      console.log(`🖥️ TextProcessor: NVIDIA model set to: ${this.nvidiaModel}`);
    } else if (provider === 'vllm') {
      this.vllmModel = options?.vllmModel || this.vllmModel;
      this.vllmBaseUrl = options?.vllmBaseUrl || this.vllmBaseUrl;
    }
    // Reset the LLM so it gets re-initialized with the new provider
    this.llm = null;
  }

  /**
   * Get the current LLM provider
   */
  public getLLMProvider(): 'ollama' | 'nvidia' | 'vllm' {
    return this.selectedLLMProvider;
  }

  /**
   * Process text to extract structured triples with a custom prompt template
   * @param text Text to process
   * @param customPrompt Custom prompt template to use instead of the default
   * @returns Array of triples with metadata
   */
  public async processTextWithCustomPrompt(text: string, customPrompt: string): Promise<Array<Triple & { confidence: number, metadata: any }>> {
    if (!this.llm || !this.tripleParser) {
      await this.initialize();
    }

    // Ensure we have an LLM to extract triples
    if (!this.llm) {
      throw new Error("LLM is not initialized. Please ensure your selected provider is properly configured.");
    }

    // Step 1: Chunk the text into manageable pieces
    const chunks = await this.chunkText(text);
    console.log(`Split text into ${chunks.length} chunks`);

    // Step 2: Process each chunk to extract triples with the custom prompt
    const allTriples: Array<Triple & { confidence: number, metadata: any }> = [];
    
    // Create a custom prompt template
    const customTemplate = PromptTemplate.fromTemplate(customPrompt);
    
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      console.log(`Processing chunk ${i + 1}/${chunks.length} (${chunk.length} chars) with custom prompt`);
      
      try {
        // Format the prompt with the chunk and parser instructions
        const formatInstructions = this.tripleParser!.getFormatInstructions();
        const prompt = await customTemplate.format({
          text: chunk,
          format_instructions: formatInstructions
        });

        // Extract triples using the LLM
        const response = await this.llm!.invoke(prompt);
        const responseText = response.content as string;
        const parsedTriples = await this.tripleParser!.parse(responseText);
        
        allTriples.push(...parsedTriples);
      } catch (error) {
        console.error(`Error processing chunk ${i + 1} with custom prompt:`, error);
      }
    }

    // Step 3: Post-process to remove duplicates and normalize
    const processedTriples = this.postProcessTriples(allTriples);
    console.log(`Extracted ${processedTriples.length} unique triples after post-processing with custom prompt`);

    return processedTriples;
  }

  /**
   * Process text to extract structured triples with a custom system prompt
   * This is used for direct LLM invocation without LangChain
   * @param text Text to process
   * @param customSystemPrompt Custom system prompt to use
   * @returns Array of triples with metadata
   */
  public async processTextWithCustomSystemPrompt(text: string, customSystemPrompt: string): Promise<Array<Triple & { confidence: number, metadata: any }>> {
    if (!this.llm) {
      await this.initialize();
    }

    // Ensure we have an LLM to extract triples
    if (!this.llm) {
      throw new Error("LLM is not initialized. Please ensure your selected provider is properly configured.");
    }

    // Step 1: Chunk the text into manageable pieces
    const chunks = await this.chunkText(text);
    console.log(`Split text into ${chunks.length} chunks for processing with custom system prompt`);

    // Step 2: Process each chunk to extract triples with the custom system prompt
    const allTriples: Array<Triple & { confidence: number, metadata: any }> = [];
    
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      console.log(`Processing chunk ${i + 1}/${chunks.length} (${chunk.length} chars) with custom system prompt`);
      
      try {
        // Create messages with the custom system prompt and the chunk
        const messages = [
          new SystemMessage(customSystemPrompt),
          new HumanMessage(chunk)
        ];
        const response = await this.llm!.invoke(messages);
        
        // Convert response to triples
        const responseText = response.content as string;
        
        // Create a simple triple parser for the response (since we're not using LangChain's parser)
        const simpleTriples = this.parseTripleLines(responseText);
        
        // Convert to the expected format with confidence and metadata
        const structuredTriples = simpleTriples.map(triple => ({
          ...triple,
          confidence: 0.9, // Default confidence for custom prompt extraction
          metadata: {
            entityTypes: [], // Empty entity types as they're not provided
            source: chunk.substring(0, 100) + "...", // First 100 chars as context
            context: `${triple.subject} ${triple.predicate} ${triple.object}`
          }
        }));
        
        allTriples.push(...structuredTriples);
      } catch (error) {
        console.error(`Error processing chunk ${i + 1} with custom system prompt:`, error);
      }
    }

    // Step 3: Post-process to remove duplicates and normalize
    const processedTriples = this.postProcessTriples(allTriples);
    console.log(`Extracted ${processedTriples.length} unique triples after post-processing with custom system prompt`);

    return processedTriples;
  }
  
  /**
   * Helper method to parse triple lines from LLM output
   * @private
   */
  private parseTripleLines(text: string): Triple[] {
    const triples: Triple[] = [];
    const lines = text.split('\n');
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      
      // Try different regex patterns to extract triples
      const patterns = [
        // Standard format: ('subject', 'relation', 'object')
        /\('([^']+)',\s*'([^']+)',\s*'([^']+)'\)/,
        // Double quotes: ("subject", "relation", "object")
        /\("([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\)/,
        // No parentheses: "subject", "relation", "object"
        /"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"/,
        // Mixed quotes: ('subject', "relation", 'object')
        /\(['"]([^'"]+)['"],\s*['"]([^'"]+)['"],\s*['"]([^'"]+)['"]\)/,
        // Plain text: subject, relation, object
        /^([^,]+),\s*([^,]+),\s*(.+)$/
      ];
      
      let match = null;
      for (const pattern of patterns) {
        match = trimmed.match(pattern);
        if (match) break;
      }
      
      if (match) {
        triples.push({
          subject: match[1].trim(),
          predicate: match[2].trim(),
          object: match[3].trim()
        });
      }
    }
    
    return triples;
  }
}

/**
 * Process a document and extract triples with metadata
 * @param text Document text
 * @param useLangChain Whether to use LangChain's extraction (optional)
 * @param useGraphTransformer Whether to use LLMGraphTransformer (optional)
 * @param options Custom prompt options (optional)
 * @returns Extracted triples with metadata
 */
export async function processDocument(
  text: string, 
  useLangChain = false,
  useGraphTransformer = false,
  options?: PromptOptions
): Promise<Array<Triple & { confidence: number, metadata: any }>> {
  if (useLangChain) {
    if (useGraphTransformer) {
      // Pass graphTransformerPrompt if available
      return await processDocumentWithGraphTransformer(text, options?.graphTransformerPrompt);
    } else {
      // Initialize text processor with custom extraction prompt if available
      const processor = TextProcessor.getInstance();
      
      // If a custom extraction prompt is provided, use it for this invocation
      if (options?.extractionPrompt) {
        return await processor.processTextWithCustomPrompt(text, options.extractionPrompt);
      } else {
        return await processor.processText(text);
      }
    }
  }
  
  // Use default processor with potential custom system prompt
  const processor = TextProcessor.getInstance();
  
  // If a custom system prompt is provided, use it for this invocation
  if (options?.systemPrompt) {
    return await processor.processTextWithCustomSystemPrompt(text, options.systemPrompt);
  } else {
    return await processor.processText(text);
  }
}

/**
 * Process a document using LangChain's LLMGraphTransformer
 * @param text Document text
 * @param customGraphPrompt Optional custom prompt for the graph transformer
 * @returns Extracted triples with metadata
 */
async function processDocumentWithGraphTransformer(
  text: string,
  customGraphPrompt?: string
): Promise<Array<Triple & { confidence: number, metadata: any }>> {
  const processor = TextProcessor.getInstance();
  
  // Initialize LLM if not already done
  if (!processor.getLLM()) {
    await processor.initialize();
  }
  
  // Ensure we have an LLM
  const llm = processor.getLLM();
  if (!llm) {
    throw new Error("xAI API key is required for triple extraction. Please set XAI_API_KEY in your environment variables.");
  }
  
  // Use the existing LLM with LLMGraphTransformer
  const llmTransformerOptions: any = {
    llm,
    // Optional configurations
    allowedNodes: ["Person", "Organization", "Concept", "Location", "Event", "Product"],
    allowedRelationships: ["RELATED_TO", "PART_OF", "LOCATED_IN", "WORKS_AT", "CREATED", "BELONGS_TO", "HAS_PROPERTY"],
    nodeProperties: ["name", "type", "description"]
  };
  
  // Add custom prompt if provided
  if (customGraphPrompt) {
    llmTransformerOptions.customPrompt = customGraphPrompt;
  }
  
  const llmTransformer = new LLMGraphTransformer(llmTransformerOptions);
  
  // Create LangChain document from text
  const documents = [new Document({ pageContent: text })];
  
  try {
    // Extract graph documents
    const graphDocuments = await llmTransformer.convertToGraphDocuments(documents);
    
    // Convert graph nodes and relationships to triples
    const triples: Array<Triple & { confidence: number, metadata: any }> = [];
    
    if (graphDocuments.length > 0) {
      const graphDoc = graphDocuments[0];
      
      // Process each relationship as a triple
      for (const relationship of graphDoc.relationships) {
        // Use type assertion to handle potential mixed types
        const rel = relationship as unknown as {
          source: { id: string, type: string, properties?: Record<string, any> },
          target: { id: string, type: string, properties?: Record<string, any> },
          type: string
        };
          
        triples.push({
          subject: rel.source.id,
          predicate: rel.type.toLowerCase(),
          object: rel.target.id,
          confidence: 0.9, // Default high confidence for LLM-extracted relationships
          metadata: {
            entityTypes: [rel.source.type, rel.target.type],
            source: text.substring(0, 100) + "...", // First 100 chars as source context
            context: `${rel.source.id} ${rel.type} ${rel.target.id}`,
            sourceProperties: rel.source.properties || {},
            targetProperties: rel.target.properties || {}
          }
        });
      }
    }
    
    return triples;
  } catch (error) {
    console.error("Error processing with LLMGraphTransformer:", error);
    throw new Error(`Failed to process with LangChain: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Extract entity types from a text passage
 * @param text Text to analyze
 * @returns Map of entity names to their types
 */
export async function extractEntityTypes(text: string): Promise<Map<string, string[]>> {
  const processor = TextProcessor.getInstance();
  const triples = await processor.processText(text);
  
  const entityTypes = new Map<string, string[]>();
  
  for (const triple of triples) {
    if (triple.metadata && triple.metadata.entityTypes) {
      // Extract subject type
      if (triple.metadata.entityTypes[0]) {
        const subjectType = entityTypes.get(triple.subject) || [];
        if (!subjectType.includes(triple.metadata.entityTypes[0])) {
          subjectType.push(triple.metadata.entityTypes[0]);
        }
        entityTypes.set(triple.subject, subjectType);
      }
      
      // Extract object type
      if (triple.metadata.entityTypes[1]) {
        const objectType = entityTypes.get(triple.object) || [];
        if (!objectType.includes(triple.metadata.entityTypes[1])) {
          objectType.push(triple.metadata.entityTypes[1]);
        }
        entityTypes.set(triple.object, objectType);
      }
    }
  }
  
  return entityTypes;
}

/**
 * Split text into sentences and generate embeddings
 * @param text Text to process
 * @param documentId Optional document identifier
 * @returns Array of sentence embeddings
 */
export async function processSentenceEmbeddings(
  text: string, 
  documentId?: string
): Promise<SentenceEmbedding[]> {
  const processor = TextProcessor.getInstance();
  
  // Split text into sentences
  const sentences = await processor.splitIntoSentences(text);
  
  // Generate embeddings for the sentences
  return await processor.generateSentenceEmbeddings(sentences, documentId);
} 