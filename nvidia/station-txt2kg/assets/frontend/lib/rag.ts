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
 * Retrieval Augmented Generation (RAG) implementation using Qdrant and LangChain
 * This module provides a RetrievalQA chain using Qdrant as the vector store
 * Uses NVIDIA API for LLM inference
 * 
 * Architecture:
 * - Uses 'document-embeddings' collection for Pure RAG (stores full text chunks)
 * - Separate from 'entity-embeddings' collection used for knowledge graph entities
 * - Documents are retrieved via semantic similarity and fed to LLM for answer generation
 */

import { ChatOpenAI } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { QdrantVectorStore } from "@langchain/community/vectorstores/qdrant";
import { Embeddings } from "@langchain/core/embeddings";
import { EmbeddingsService } from './embeddings';

// Custom embeddings adapter to use our EmbeddingsService with LangChain
class CustomEmbeddings extends Embeddings {
  private embeddingsService: EmbeddingsService;

  constructor() {
    super({});
    this.embeddingsService = EmbeddingsService.getInstance();
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    await this.embeddingsService.initialize();
    return await this.embeddingsService.encode(documents);
  }

  async embedQuery(query: string): Promise<number[]> {
    await this.embeddingsService.initialize();
    const embeddings = await this.embeddingsService.encode([query]);
    return embeddings[0];
  }
}

export class RAGService {
  private static instance: RAGService;
  private vectorStore: QdrantVectorStore | null = null;
  private embeddingsService: CustomEmbeddings;
  private llm: ChatOpenAI | null = null;
  private initialized: boolean = false;
  private isInitializing: boolean = false;
  private collectionName: string = 'document-embeddings';

  private constructor() {
    this.embeddingsService = new CustomEmbeddings();
  }

  /**
   * Get the singleton instance of RAGService
   */
  public static getInstance(): RAGService {
    if (!RAGService.instance) {
      RAGService.instance = new RAGService();
    }
    return RAGService.instance;
  }

  /**
   * Initialize the RAG service
   */
  public async initialize(): Promise<void> {
    if (this.initialized || this.isInitializing) {
      return;
    }

    this.isInitializing = true;

    try {
      console.log('Initializing RAG service...');

      // Check for NVIDIA API key
      const nvidiaApiKey = process.env.NVIDIA_API_KEY;
      if (!nvidiaApiKey) {
        const error = new Error('NVIDIA_API_KEY is required for RAG service. Please set the NVIDIA_API_KEY environment variable.');
        console.error('❌ RAG Initialization Error:', error.message);
        throw error;
      }

      // Initialize NVIDIA LLM using ChatOpenAI with NVIDIA's base URL
      this.llm = new ChatOpenAI({
        modelName: "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature: 0.2,
        maxTokens: 1024,
        openAIApiKey: nvidiaApiKey,
        configuration: {
          baseURL: "https://integrate.api.nvidia.com/v1",
          timeout: 120000, // 120 second timeout
        },
        modelKwargs: {
          top_p: 0.95,
          frequency_penalty: 0,
          presence_penalty: 0
        }
      });

      console.log('✅ NVIDIA LLM initialized successfully');

      // Initialize Qdrant vector store
      const qdrantUrl = process.env.QDRANT_URL || 'http://localhost:6333';

      this.vectorStore = await QdrantVectorStore.fromExistingCollection(
        this.embeddingsService,
        {
          url: qdrantUrl,
          collectionName: this.collectionName,
          contentPayloadKey: 'text', // Map payload.text to pageContent
        }
      );

      console.log('✅ Qdrant vector store connected successfully');

      this.initialized = true;
      console.log('✅ RAG service initialized successfully');
    } catch (error) {
      console.error('❌ Error initializing RAG service:', error);
      this.isInitializing = false;
      throw error;
    } finally {
      this.isInitializing = false;
    }
  }

  /**
   * Store documents in Qdrant for retrieval
   * @param documents Array of text documents to store
   * @param metadata Optional metadata for the documents
   */
  public async storeDocuments(
    documents: string[],
    metadata?: Record<string, any>[]
  ): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!documents || documents.length === 0) {
      console.warn('No documents provided to store');
      return;
    }

    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }

    console.log(`Storing ${documents.length} documents in Qdrant`);

    // Create Document objects with metadata
    const docs = documents.map((text, i) => new Document({
      pageContent: text,
      metadata: {
        timestamp: new Date().toISOString(),
        ...(metadata && metadata[i] ? metadata[i] : {})
      }
    }));

    // Store in Qdrant using LangChain
    await this.vectorStore.addDocuments(docs);
    console.log(`✅ Successfully stored ${docs.length} document embeddings`);
  }

  /**
   * Perform question answering with document retrieval using proper RAG implementation
   * @param query User query
   * @param topK Number of most similar documents to retrieve
   * @returns Object containing the answer and document count
   */
  public async retrievalQA(query: string, topK: number = 5): Promise<{ answer: string; documentCount: number }> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.llm) {
      throw new Error('LLM not initialized');
    }

    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }

    console.log(`🔍 Performing RAG query with topK=${topK}`);

    // Use LangChain's similarity search to retrieve relevant documents
    const similarDocs = await this.vectorStore.similaritySearch(query, topK);

    if (!similarDocs || similarDocs.length === 0) {
      console.log('⚠️ No relevant documents found, falling back to LLM general knowledge');

      // Define prompt template for standalone LLM response
      const fallbackPromptTemplate = PromptTemplate.fromTemplate(`
You are a helpful assistant answering questions based on your general knowledge.
Since no specific information was found in the knowledge base, please provide the best answer you can.

Question: {query}

Answer:
`);

      // Create fallback chain
      const fallbackChain = RunnableSequence.from([
        {
          query: () => query,
        },
        fallbackPromptTemplate,
        this.llm,
        new StringOutputParser(),
      ]);

      // Execute fallback chain
      const answer = await fallbackChain.invoke({});
      return {
        answer: `[Note: No specific information was found in the knowledge base. This answer is based on general knowledge.]\n\n${answer}`,
        documentCount: 0
      };
    }

    console.log(`✅ Found ${similarDocs.length} relevant document chunks`);

    // Log first document structure for debugging
    if (similarDocs.length > 0) {
      console.log('📄 First document chunk structure:', {
        hasPageContent: !!similarDocs[0].pageContent,
        pageContentLength: similarDocs[0].pageContent?.length || 0,
        hasMetadata: !!similarDocs[0].metadata,
        metadataKeys: similarDocs[0].metadata ? Object.keys(similarDocs[0].metadata) : []
      });
    }

    // Extract text from retrieved documents
    // Support both pageContent (LangChain standard) and metadata.text (legacy format)
    const context = similarDocs
      .map((doc) => {
        // Try pageContent first (LangChain standard)
        if (doc.pageContent && doc.pageContent.trim().length > 0) {
          return doc.pageContent;
        }
        // Fall back to metadata.text (legacy Qdrant storage format)
        if (doc.metadata?.text && doc.metadata.text.trim().length > 0) {
          return doc.metadata.text;
        }
        return '';
      })
      .filter((text) => text.length > 0)
      .join('\n\n');

    console.log(`📝 Extracted context length: ${context.length} characters`);

    if (!context || context.trim().length === 0) {
      console.log('⚠️ Retrieved documents have no content, falling back to LLM');
      const fallbackPromptTemplate = PromptTemplate.fromTemplate(`
You are a helpful assistant answering questions based on your general knowledge.

Question: {query}

Answer:
`);

      const fallbackChain = RunnableSequence.from([
        {
          query: () => query,
        },
        fallbackPromptTemplate,
        this.llm,
        new StringOutputParser(),
      ]);

      const answer = await fallbackChain.invoke({});
      return {
        answer: `[Note: No specific information was found in the knowledge base. This answer is based on general knowledge.]\n\n${answer}`,
        documentCount: similarDocs.length
      };
    }

    // Define prompt template for RAG
    const promptTemplate = PromptTemplate.fromTemplate(`
Answer the question based only on the following context from the knowledge base.
If you cannot find the answer in the context, say "I cannot find this information in the knowledge base."

Context:
{context}

Question: {query}

Answer:
`);

    // Create retrieval chain using RunnableSequence
    const retrievalChain = RunnableSequence.from([
      {
        context: () => context,
        query: () => query,
      },
      promptTemplate,
      this.llm,
      new StringOutputParser(),
    ]);

    console.log('🤖 Generating answer with NVIDIA LLM...');

    // Execute chain
    try {
      const answer = await retrievalChain.invoke({});
      console.log('✅ RAG query completed successfully');
      console.log(`📝 Answer length: ${answer.length} characters`);
      console.log(`📄 Retrieved ${similarDocs.length} document chunks`);
      return {
        answer,
        documentCount: similarDocs.length
      };
    } catch (error) {
      console.error('❌ Error generating answer with NVIDIA LLM:', error);
      throw error;
    }
  }

  /**
   * Perform similar document retrieval without QA
   * @param query Query text
   * @param topK Number of documents to retrieve
   * @returns Array of retrieved documents with similarity scores
   */
  public async retrieveSimilarDocuments(
    query: string,
    topK: number = 5
  ): Promise<Array<{ text: string; score: number; metadata?: any }>> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }

    // Use LangChain's similarity search with scores
    const results = await this.vectorStore.similaritySearchWithScore(query, topK);

    return results.map(([doc, score]) => ({
      text: doc.pageContent,
      score: score,
      metadata: doc.metadata
    }));
  }
}

export default RAGService.getInstance(); 