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
import { getShouldStopEmbeddings, resetStopEmbeddings } from "@/app/api/stop-embeddings/route";

/**
 * Embeddings service for generating sentence embeddings using SentenceTransformer
 * Can use either local SentenceTransformer or remote Text Embeddings Inference API
 */
export class EmbeddingsService {
  private apiUrl: string;
  private modelId: string;
  private static instance: EmbeddingsService;
  private dimension: number = 384; // Dimension for all-MiniLM-L6-v2 (default model)
  private useNvidiaApi: boolean = false;
  private nvidiaApiKey: string = '';
  private nvidiaModel: string = 'nvidia/llama-3.2-nv-embedqa-1b-v2';
  private nvidiaInputType: string = 'query';

  private constructor() {
    this.apiUrl = process.env.EMBEDDINGS_API_URL || 'http://localhost:8000';
    this.modelId = process.env.EMBEDDINGS_MODEL_ID || 'all-MiniLM-L6-v2';
    
    // Always get NVIDIA API key from environment variables
    this.nvidiaApiKey = process.env.NVIDIA_API_KEY || '';
    
    // Try to get settings from localStorage if we're in the browser
    if (typeof window !== 'undefined') {
      const embeddingsProvider = localStorage.getItem('embeddings_provider') || '';
      this.useNvidiaApi = this.nvidiaApiKey !== '' && embeddingsProvider === 'nvidia';
      
      // Get NVIDIA model from localStorage if available
      const storedNvidiaModel = localStorage.getItem('nvidia_embeddings_model');
      if (storedNvidiaModel) {
        this.nvidiaModel = storedNvidiaModel;
      }
    } else {
      // Server-side code (API routes)
      const embeddingsProvider = process.env.EMBEDDINGS_PROVIDER || '';
      this.useNvidiaApi = this.nvidiaApiKey !== '' && embeddingsProvider === 'nvidia';
      
      // Get NVIDIA model if specified
      if (process.env.NVIDIA_EMBEDDINGS_MODEL) {
        this.nvidiaModel = process.env.NVIDIA_EMBEDDINGS_MODEL;
      }
    }
    
    // Override dimension if using NVIDIA model (llama-3.2-nv-embedqa-1b-v2 has 4096 dimensions)
    if (this.useNvidiaApi) {
      this.dimension = 4096;
    }
    
    console.log('EmbeddingsService initialized with useNvidiaApi:', this.useNvidiaApi, 'model:', this.useNvidiaApi ? this.nvidiaModel : this.modelId);
  }

  /**
   * Get the singleton instance of EmbeddingsService
   */
  public static getInstance(): EmbeddingsService {
    if (!EmbeddingsService.instance) {
      EmbeddingsService.instance = new EmbeddingsService();
    }
    return EmbeddingsService.instance;
  }
  
  /**
   * Reset the singleton instance to force reinitialization with new settings
   */
  public static reset(): void {
    if (EmbeddingsService.instance) {
      console.log('Resetting EmbeddingsService instance to pick up new settings');
      EmbeddingsService.instance = undefined as any;
    }
  }

  /**
   * Initialize the embeddings service
   */
  public async initialize(): Promise<void> {
    try {
      // If using NVIDIA API, check if the API key is valid
      if (this.useNvidiaApi) {
        if (!this.nvidiaApiKey) {
          throw new Error('NVIDIA API key is not set');
        }
        console.log(`Embeddings service initialized successfully using NVIDIA model: ${this.nvidiaModel}`);
        return;
      }
      
      // Check if the API is available
      const response = await fetch(`${this.apiUrl}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to connect to embeddings API: ${response.statusText}`);
      }

      console.log(`Embeddings service initialized successfully using model: ${this.modelId}`);
    } catch (error) {
      console.error('Error initializing embeddings service:', error);
      console.warn('Continuing without embeddings service. Embeddings will not be available.');
    }
  }

  /**
   * Get the dimension of the embeddings
   */
  public getDimension(): number {
    return this.dimension;
  }

  /**
   * Generate embeddings for a batch of texts
   * @param texts Array of texts to encode
   * @param batchSize Batch size for API requests
   * @returns Promise resolving to array of embeddings
   */
  public async encode(texts: string[], batchSize: number = 32): Promise<number[][]> {
    if (!texts || texts.length === 0) {
      return [];
    }

    // Process in batches to avoid overwhelming the API
    const results: number[][] = [];
    
    for (let i = 0; i < texts.length; i += batchSize) {
      // Check if embeddings generation should be stopped
      if (getShouldStopEmbeddings()) {
        console.log(`Embeddings generation stopped by user at batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(texts.length/batchSize)}`);
        resetStopEmbeddings(); // Reset the flag for next time
        throw new Error('Embeddings generation stopped by user');
      }
      
      const batch = texts.slice(i, i + batchSize);
      console.log(`Encoding batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(texts.length/batchSize)}`);
      
      try {
        let batchResults;
        if (this.useNvidiaApi) {
          batchResults = await this.encodeWithNvidia(batch);
        } else {
          batchResults = await this.encodeBatch(batch);
        }
        results.push(...batchResults);
      } catch (error) {
        console.error(`Error encoding batch ${i}-${i + batch.length}:`, error);
        // Fill with zeros for failed batches
        for (let j = 0; j < batch.length; j++) {
          results.push(new Array(this.dimension).fill(0));
        }
      }
    }
    
    return results;
  }

  /**
   * Generate embeddings for a single batch of texts using NVIDIA API
   * @param texts Array of texts to encode in a single batch
   * @returns Promise resolving to array of embeddings
   */
  private async encodeWithNvidia(texts: string[]): Promise<number[][]> {
    try {
      const response = await fetch('https://integrate.api.nvidia.com/v1/embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.nvidiaApiKey}`,
        },
        body: JSON.stringify({
          input: texts,
          model: this.nvidiaModel,
          input_type: this.nvidiaInputType,
          encoding_format: "float",
          truncate: "NONE"
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`NVIDIA API request failed with status ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      return data.data.map((item: any) => item.embedding);
    } catch (error) {
      console.error('Error calling NVIDIA embeddings API:', error);
      throw error;
    }
  }

  /**
   * Generate embeddings for a single batch of texts
   * @param texts Array of texts to encode in a single batch
   * @returns Promise resolving to array of embeddings
   */
  private async encodeBatch(texts: string[]): Promise<number[][]> {
    try {
      const response = await fetch(`${this.apiUrl}/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input: texts,
          model: this.modelId,
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return data.data.map((item: any) => item.embedding);
    } catch (error) {
      console.error('Error calling embeddings API:', error);
      throw error;
    }
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  public cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same length');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    if (normA === 0 || normB === 0) {
      return 0; // Handle zero vectors
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Find most similar texts based on cosine similarity
   * @param query Query text
   * @param candidates Array of candidate texts
   * @param topK Number of results to return
   * @returns Promise resolving to array of [index, similarity] pairs
   */
  public async findSimilarTexts(
    query: string,
    candidates: string[],
    topK: number = 5
  ): Promise<[number, number][]> {
    if (!candidates || candidates.length === 0) {
      return [];
    }

    // Generate embeddings
    const queryEmbedding = (await this.encode([query]))[0];
    const candidateEmbeddings = await this.encode(candidates);
    
    // Calculate similarities
    const similarities: [number, number][] = candidateEmbeddings.map(
      (embedding, index) => [index, this.cosineSimilarity(queryEmbedding, embedding)]
    );
    
    // Sort by similarity (descending) and return top k
    return similarities
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK);
  }
}

export default EmbeddingsService.getInstance(); 