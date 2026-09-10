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
import OpenAI from 'openai';
import { Agent } from 'http';
import { Agent as HttpsAgent } from 'https';

export interface LLMOptions {
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  stream?: boolean;
}

export interface OllamaOptions {
  baseUrl?: string;
  model?: string;
}

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface StreamCallbacks {
  onToken?: (token: string) => void;
  onComplete?: (fullResponse: string) => void;
  onError?: (error: any) => void;
}

export interface BatchOptions extends LLMOptions {
  concurrency?: number;
  batchSize?: number;
}

export interface BatchResult {
  results: string[];
  errors: Array<{ index: number; error: string; attempts?: number }>;
  successCount: number;
  totalCount: number;
  totalAttempts?: number;
  averageAttempts?: number;
}

/**
 * Service for interacting with LLMs through different providers
 */
export class LLMService {
  private static instance: LLMService;
  
  // NVIDIA client for accessing models hosted on NVIDIA API
  private nvidiaClient: OpenAI | null = null;
  
  // Ollama client for local models
  private ollamaClient: OpenAI | null = null;
  private ollamaBaseUrl: string = 'http://localhost:11434/v1';
  
  // vLLM client for local models with advanced features
  private vllmClient: OpenAI | null = null;
  private vllmBaseUrl: string = 'http://localhost:8001/v1';
  
  // xAI client would go here
  
  private constructor() {
    // Initialize NVIDIA client if API key is available
    const nvidiaApiKey = process.env.NVIDIA_API_KEY;
    if (nvidiaApiKey) {
      // Create HTTPS agent with connection pooling for NVIDIA API
      const httpsAgent = new HttpsAgent({
        keepAlive: true,
        maxSockets: 10,
        maxFreeSockets: 5,
        timeout: 60000,
        keepAliveMsecs: 30000
      });

      this.nvidiaClient = new OpenAI({
        apiKey: nvidiaApiKey,
        baseURL: 'https://integrate.api.nvidia.com/v1',
        httpAgent: httpsAgent
      });
    }

    // Initialize Ollama client with connection pooling - no API key needed for local Ollama
    const ollamaUrl = process.env.OLLAMA_BASE_URL || this.ollamaBaseUrl;
    
    // Create HTTP agent with connection pooling for Ollama
    const httpAgent = new Agent({
      keepAlive: true,
      maxSockets: 15, // Higher for local connections
      maxFreeSockets: 8,
      timeout: 1800000, // 30 minutes for large model inference
      keepAliveMsecs: 30000
    });

    this.ollamaClient = new OpenAI({
      apiKey: 'ollama', // Ollama doesn't require a real API key
      baseURL: ollamaUrl,
      httpAgent: httpAgent,
      timeout: 1800000 // 30 minutes timeout for large model inference
    });
    this.ollamaBaseUrl = ollamaUrl;

    // Initialize vLLM client with connection pooling - no API key needed for local vLLM
    const vllmUrl = process.env.VLLM_BASE_URL || this.vllmBaseUrl;
    
    // Create HTTP agent with connection pooling for vLLM
    const vllmHttpAgent = new Agent({
      keepAlive: true,
      maxSockets: 15, // Higher for local connections
      maxFreeSockets: 8,
      timeout: 120000, // Longer timeout for vLLM inference
      keepAliveMsecs: 30000
    });

    this.vllmClient = new OpenAI({
      apiKey: 'vllm', // vLLM doesn't require a real API key
      baseURL: vllmUrl,
      httpAgent: vllmHttpAgent
    });
    this.vllmBaseUrl = vllmUrl;
  }
  
  /**
   * Get the singleton instance of LLMService
   */
  public static getInstance(): LLMService {
    if (!LLMService.instance) {
      LLMService.instance = new LLMService();
    }
    return LLMService.instance;
  }
  
  /**
   * Check if a specific provider is configured
   */
  public isProviderConfigured(provider: 'nvidia' | 'ollama'): boolean {
    switch (provider) {
      case 'nvidia':
        return this.nvidiaClient !== null;
      case 'ollama':
        return this.ollamaClient !== null;
      default:
        return false;
    }
  }
  
  /**
   * Generate a completion using the NVIDIA API
   */
  public async generateNvidiaCompletion(
    model: string,
    messages: LLMMessage[],
    options: LLMOptions = {}
  ): Promise<string> {
    if (!this.nvidiaClient) {
      throw new Error('NVIDIA API client not configured. Set NVIDIA_API_KEY environment variable.');
    }
    
    const { temperature = 0.7, maxTokens = 1024, topP = 0.9, stream = false } = options;
    
    try {
      const completion = await this.nvidiaClient.chat.completions.create({
        model,
        messages,
        temperature,
        max_tokens: maxTokens,
        top_p: topP,
        stream,
      });
      
      if (!stream) {
        // Check if completion is a stream
        if ('choices' in completion) {
          return completion.choices[0]?.message?.content || '';
        } else {
          throw new Error('Unexpected response format');
        }
      } else {
        throw new Error('Streaming not supported in this method. Use generateNvidiaCompletionStream instead.');
      }
    } catch (error) {
      console.error('Error generating NVIDIA completion:', error);
      throw error;
    }
  }
  
  /**
   * Generate a streaming completion using the NVIDIA API
   */
  public async generateNvidiaCompletionStream(
    model: string,
    messages: LLMMessage[],
    callbacks: StreamCallbacks,
    options: LLMOptions = {}
  ): Promise<void> {
    if (!this.nvidiaClient) {
      throw new Error('NVIDIA API client not configured. Set NVIDIA_API_KEY environment variable.');
    }
    
    const { temperature = 0.7, maxTokens = 1024, topP = 0.9 } = options;
    
    try {
      const stream = await this.nvidiaClient.chat.completions.create({
        model,
        messages,
        temperature,
        max_tokens: maxTokens,
        top_p: topP,
        stream: true,
      });
      
      let fullResponse = '';
      
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || '';
        if (content) {
          fullResponse += content;
          callbacks.onToken?.(content);
        }
      }
      
      callbacks.onComplete?.(fullResponse);
    } catch (error) {
      console.error('Error generating NVIDIA streaming completion:', error);
      callbacks.onError?.(error);
      throw error;
    }
  }

  /**
   * Generate a completion using the Ollama API with retry logic
   */
  public async generateOllamaCompletion(
    model: string,
    messages: LLMMessage[],
    options: LLMOptions & { maxRetries?: number } = {}
  ): Promise<string> {
    if (!this.ollamaClient) {
      throw new Error('Ollama client not configured.');
    }
    
    const { temperature = 0.7, maxTokens = 1024, topP = 0.9, stream = false, maxRetries = 3 } = options;
    
    return this.retryWithBackoff(async () => {
      const completion = await this.ollamaClient!.chat.completions.create({
        model,
        messages,
        temperature,
        max_tokens: maxTokens,
        top_p: topP,
        stream,
      });
      
      if (!stream) {
        // Check if completion is a stream
        if ('choices' in completion) {
          return completion.choices[0]?.message?.content || '';
        } else {
          throw new Error('Unexpected response format');
        }
      } else {
        throw new Error('Streaming not supported in this method. Use generateOllamaCompletionStream instead.');
      }
    }, maxRetries);
  }

  /**
   * Generate a streaming completion using the Ollama API
   */
  public async generateOllamaCompletionStream(
    model: string,
    messages: LLMMessage[],
    callbacks: StreamCallbacks,
    options: LLMOptions = {}
  ): Promise<void> {
    if (!this.ollamaClient) {
      throw new Error('Ollama client not configured.');
    }
    
    const { temperature = 0.7, maxTokens = 1024, topP = 0.9 } = options;
    
    try {
      const stream = await this.ollamaClient.chat.completions.create({
        model,
        messages,
        temperature,
        max_tokens: maxTokens,
        top_p: topP,
        stream: true,
      });
      
      let fullResponse = '';
      
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || '';
        if (content) {
          fullResponse += content;
          callbacks.onToken?.(content);
        }
      }
      
      callbacks.onComplete?.(fullResponse);
    } catch (error) {
      console.error('Error generating Ollama streaming completion:', error);
      callbacks.onError?.(error);
      throw error;
    }
  }

  /**
   * Generate a completion using the vLLM API with retry logic
   */
  public async generateVllmCompletion(
    model: string,
    messages: LLMMessage[],
    options: LLMOptions & { maxRetries?: number } = {}
  ): Promise<string> {
    if (!this.vllmClient) {
      throw new Error('vLLM client not configured.');
    }
    
    const { temperature = 0.7, maxTokens = 1024, topP = 0.9, stream = false, maxRetries = 3 } = options;
    
    return this.retryWithBackoff(async () => {
      if (!stream) {
        const completion = await this.vllmClient!.chat.completions.create({
          model,
          messages,
          temperature,
          max_tokens: maxTokens,
          top_p: topP,
          stream: false,
        });
        
        if (completion && completion.choices && completion.choices.length > 0) {
          return completion.choices[0]?.message?.content || '';
        } else {
          throw new Error('Unexpected response format');
        }
      } else {
        throw new Error('Streaming not supported in this method. Use generateVllmCompletionStream instead.');
      }
    }, maxRetries);
  }

  /**
   * Generate a streaming completion using the vLLM API
   */
  public async generateVllmCompletionStream(
    model: string,
    messages: LLMMessage[],
    callbacks: StreamCallbacks,
    options: LLMOptions = {}
  ): Promise<void> {
    if (!this.vllmClient) {
      throw new Error('vLLM client not configured.');
    }
    
    const { temperature = 0.7, maxTokens = 1024, topP = 0.9 } = options;
    
    try {
      const stream = await this.vllmClient.chat.completions.create({
        model,
        messages,
        temperature,
        max_tokens: maxTokens,
        top_p: topP,
        stream: true,
      });
      
      let fullResponse = '';
      
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || '';
        if (content) {
          fullResponse += content;
          callbacks.onToken?.(content);
        }
      }
      
      callbacks.onComplete?.(fullResponse);
    } catch (error) {
      console.error('Error in vLLM streaming completion:', error);
      callbacks.onError?.(error instanceof Error ? error : new Error(String(error)));
    }
  }

  /**
   * Test Ollama connection and list available models
   */
  public async testOllamaConnection(): Promise<{ connected: boolean; models?: string[]; error?: string }> {
    if (!this.ollamaClient) {
      return { connected: false, error: 'Ollama client not configured' };
    }

    try {
      // Try to list models to test the connection
      const response = await fetch(`${this.ollamaBaseUrl.replace('/v1', '')}/api/tags`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      const models = data.models?.map((model: any) => model.name) || [];
      
      return { connected: true, models };
    } catch (error) {
      console.error('Error testing Ollama connection:', error);
      return { 
        connected: false, 
        error: error instanceof Error ? error.message : String(error) 
      };
    }
  }
  
  /**
   * Generic method to generate completions based on the selected model
   */
  public async generateCompletion(
    providerId: string,
    messages: LLMMessage[],
    options: LLMOptions = {}
  ): Promise<string> {
    // Extract provider and model information
    if (providerId.startsWith('nvidia-')) {
      // For NVIDIA models
      const modelMap: Record<string, string> = {
        'nvidia-llama': 'meta/llama-3.1-70b-instruct',
        'nvidia-mixtral': 'mistralai/mixtral-8x7b-instruct-v0.1',
        'nvidia-nemotron': 'nvdev/nvidia/llama-3.1-nemotron-70b-instruct',
      };
      
      const modelName = modelMap[providerId] || 'meta/llama-3.1-70b-instruct';
      return this.generateNvidiaCompletion(modelName, messages, options);
    } else if (providerId.startsWith('ollama-')) {
      // For Ollama models - extract model name from providerId
      const modelName = providerId.replace('ollama-', '');
      return this.generateOllamaCompletion(modelName, messages, options);
    } else if (providerId.startsWith('vllm-')) {
      // For vLLM models - extract model name from providerId
      const modelName = providerId.replace('vllm-', '');
      return this.generateVllmCompletion(modelName, messages, options);
    } else {
      throw new Error(`Unsupported provider: ${providerId}`);
    }
  }
  
  /**
   * Generic method to generate streaming completions based on the selected model
   */
  public async generateCompletionStream(
    providerId: string,
    messages: LLMMessage[],
    callbacks: StreamCallbacks,
    options: LLMOptions = {}
  ): Promise<void> {
    // Extract provider and model information
    if (providerId.startsWith('nvidia-')) {
      // For NVIDIA models
      const modelMap: Record<string, string> = {
        'nvidia-llama': 'meta/llama-3.1-70b-instruct',
        'nvidia-mixtral': 'mistralai/mixtral-8x7b-instruct-v0.1',
        'nvidia-nemotron': 'nvdev/nvidia/llama-3.1-nemotron-70b-instruct',
      };
      
      const modelName = modelMap[providerId] || 'meta/llama-3.1-70b-instruct';
      return this.generateNvidiaCompletionStream(modelName, messages, callbacks, options);
    } else if (providerId.startsWith('ollama-')) {
      // For Ollama models - extract model name from providerId
      const modelName = providerId.replace('ollama-', '');
      return this.generateOllamaCompletionStream(modelName, messages, callbacks, options);
    } else if (providerId.startsWith('vllm-')) {
      // For vLLM models - extract model name from providerId
      const modelName = providerId.replace('vllm-', '');
      return this.generateVllmCompletionStream(modelName, messages, callbacks, options);
    } else {
      throw new Error(`Unsupported provider: ${providerId}`);
    }
  }

  /**
   * Retry a function with exponential backoff
   */
  private async retryWithBackoff<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    baseDelay: number = 1000
  ): Promise<T> {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        if (attempt === maxRetries - 1) {
          // Last attempt failed, throw the error
          throw error;
        }
        
        // Calculate delay with exponential backoff and jitter
        const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
        console.warn(`Attempt ${attempt + 1} failed, retrying in ${delay.toFixed(0)}ms:`, 
          error instanceof Error ? error.message : String(error));
        
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    throw new Error('Max retries exceeded');
  }

  /**
   * Process multiple requests in parallel with controlled concurrency and retry logic
   */
  private async processBatch<T, R>(
    items: T[],
    processor: (item: T, index: number) => Promise<R>,
    concurrency: number = 5,
    maxRetries: number = 3
  ): Promise<Array<{ result?: R; error?: string; index: number; attempts?: number }>> {
    const results: Array<{ result?: R; error?: string; index: number; attempts?: number }> = [];
    
    // Process items in chunks to control concurrency
    for (let i = 0; i < items.length; i += concurrency) {
      const chunk = items.slice(i, i + concurrency);
      const chunkPromises = chunk.map(async (item, chunkIndex) => {
        const globalIndex = i + chunkIndex;
        let attempts = 0;
        
        try {
          const result = await this.retryWithBackoff(
            async () => {
              attempts++;
              return processor(item, globalIndex);
            },
            maxRetries
          );
          return { result, index: globalIndex, attempts };
        } catch (error) {
          return { 
            error: error instanceof Error ? error.message : String(error), 
            index: globalIndex,
            attempts
          };
        }
      });
      
      const chunkResults = await Promise.all(chunkPromises);
      results.push(...chunkResults);
    }
    
    return results;
  }

  /**
   * Generate batch completions using Ollama with controlled concurrency
   */
  public async generateOllamaBatchCompletion(
    model: string,
    messagesBatch: LLMMessage[][],
    options: BatchOptions = {}
  ): Promise<BatchResult> {
    const { concurrency = 5, ...llmOptions } = options;
    
    if (!this.ollamaClient) {
      throw new Error('Ollama client not configured.');
    }

    console.log(`Starting batch processing of ${messagesBatch.length} requests with concurrency ${concurrency}`);
    
    const batchResults = await this.processBatch(
      messagesBatch,
      async (messages, index) => {
        console.log(`Processing batch item ${index + 1}/${messagesBatch.length}`);
        return this.generateOllamaCompletion(model, messages, llmOptions);
      },
      concurrency,
      3 // maxRetries
    );

    const results: string[] = [];
    const errors: Array<{ index: number; error: string; attempts?: number }> = [];
    let successCount = 0;
    let totalAttempts = 0;

    batchResults.forEach(({ result, error, index, attempts }) => {
      totalAttempts += attempts || 1;
      if (error) {
        errors.push({ index, error, attempts });
        results[index] = ''; // Placeholder for failed requests
      } else {
        results[index] = result || '';
        successCount++;
      }
    });

    console.log(`Batch processing completed: ${successCount}/${messagesBatch.length} successful, total attempts: ${totalAttempts}`);

    return {
      results,
      errors,
      successCount,
      totalCount: messagesBatch.length,
      totalAttempts,
      averageAttempts: messagesBatch.length > 0 ? totalAttempts / messagesBatch.length : 0
    };
  }

  /**
   * Generate batch completions using NVIDIA API with controlled concurrency
   */
  public async generateNvidiaBatchCompletion(
    model: string,
    messagesBatch: LLMMessage[][],
    options: BatchOptions = {}
  ): Promise<BatchResult> {
    const { concurrency = 3, ...llmOptions } = options; // Lower concurrency for external API
    
    if (!this.nvidiaClient) {
      throw new Error('NVIDIA API client not configured.');
    }

    console.log(`Starting NVIDIA batch processing of ${messagesBatch.length} requests with concurrency ${concurrency}`);
    
    const batchResults = await this.processBatch(
      messagesBatch,
      async (messages, index) => {
        console.log(`Processing NVIDIA batch item ${index + 1}/${messagesBatch.length}`);
        return this.generateNvidiaCompletion(model, messages, llmOptions);
      },
      concurrency,
      3 // maxRetries
    );

    const results: string[] = [];
    const errors: Array<{ index: number; error: string; attempts?: number }> = [];
    let successCount = 0;
    let totalAttempts = 0;

    batchResults.forEach(({ result, error, index, attempts }) => {
      totalAttempts += attempts || 1;
      if (error) {
        errors.push({ index, error, attempts });
        results[index] = ''; // Placeholder for failed requests
      } else {
        results[index] = result || '';
        successCount++;
      }
    });

    console.log(`NVIDIA batch processing completed: ${successCount}/${messagesBatch.length} successful, total attempts: ${totalAttempts}`);

    return {
      results,
      errors,
      successCount,
      totalCount: messagesBatch.length,
      totalAttempts,
      averageAttempts: messagesBatch.length > 0 ? totalAttempts / messagesBatch.length : 0
    };
  }

  /**
   * Generic batch processing method that routes to appropriate provider
   */
  public async generateBatchCompletion(
    providerId: string,
    messagesBatch: LLMMessage[][],
    options: BatchOptions = {}
  ): Promise<BatchResult> {
    if (providerId.startsWith('nvidia-')) {
      const modelMap: Record<string, string> = {
        'nvidia-llama': 'meta/llama-3.1-70b-instruct',
        'nvidia-mixtral': 'mistralai/mixtral-8x7b-instruct-v0.1',
        'nvidia-nemotron': 'nvdev/nvidia/llama-3.1-nemotron-70b-instruct',
      };
      
      const modelName = modelMap[providerId] || 'meta/llama-3.1-70b-instruct';
      return this.generateNvidiaBatchCompletion(modelName, messagesBatch, options);
    } else if (providerId.startsWith('ollama-')) {
      const modelName = providerId.replace('ollama-', '');
      return this.generateOllamaBatchCompletion(modelName, messagesBatch, options);
    } else {
      throw new Error(`Unsupported provider for batch processing: ${providerId}`);
    }
  }
}

// Export a singleton instance for convenience
export const llmService = LLMService.getInstance(); 