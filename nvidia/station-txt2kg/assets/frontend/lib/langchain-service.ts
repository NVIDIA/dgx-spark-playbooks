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
import { ChatOpenAI } from "@langchain/openai";
import { SystemMessage } from "@langchain/core/messages";

/**
 * Service for interacting with LLMs through LangChain integrations
 */
export class LangChainService {
  private static instance: LangChainService;
  
  // Cache for created models to avoid recreating them
  private modelCache: Record<string, ChatOpenAI> = {};
  
  private constructor() {}
  
  /**
   * Get the singleton instance of LangChainService
   */
  public static getInstance(): LangChainService {
    if (!LangChainService.instance) {
      LangChainService.instance = new LangChainService();
    }
    return LangChainService.instance;
  }
  
  /**
   * Get or create a ChatOpenAI model instance for the NVIDIA Nemotron model
   */
  public async getNemotronModel(options?: {
    temperature?: number;
    maxTokens?: number;
  }): Promise<ChatOpenAI> {
    const modelId = "nvidia/llama-3.3-nemotron-super-49b-v1.5";
    const cacheKey = `nemotron-${options?.temperature || 0.7}-${options?.maxTokens || 8192}`;
    
    console.log(`Requesting Nemotron model (cacheKey: ${cacheKey})`);
    
    if (this.modelCache[cacheKey]) {
      console.log(`Using cached model for: ${cacheKey}`);
      return this.modelCache[cacheKey];
    }
    
    // Try to get API key from server endpoint if in browser
    let apiKey: string | undefined;
    
    if (typeof window !== 'undefined') {
      try {
        console.log("Fetching API key from server endpoint");
        const response = await fetch('/api/config');
        if (!response.ok) {
          throw new Error(`Failed to fetch API key: ${response.statusText}`);
        }
        const config = await response.json();
        apiKey = config.nemotronApiKey;
        console.log(`Retrieved API key from server: ${apiKey ? "Yes" : "No"}`);
      } catch (error) {
        console.error("Error fetching API key:", error);
      }
    } else {
      // Server-side, use environment variable directly
      apiKey = process.env.NVIDIA_API_KEY;
      console.log(`Retrieved API key from environment: ${apiKey ? "Yes" : "No"}`);
    }
    
    // If no API key is found, throw an error
    if (!apiKey) {
      console.error('No API key found for Nemotron model');
      throw new Error('No API key found for Nemotron model. Please add NVIDIA_API_KEY to your environment variables.');
    }
    
    console.log(`Creating new ChatOpenAI instance for: ${modelId}`);

    try {
      // Create a new ChatOpenAI instance
      const model = new ChatOpenAI({
        modelName: modelId,
        temperature: options?.temperature || 0.6,
        maxTokens: options?.maxTokens || 8192,
        openAIApiKey: apiKey,
        configuration: {
          baseURL: "https://integrate.api.nvidia.com/v1",
          timeout: 120000, // 120 second timeout for larger model
        },
        modelKwargs: {
          top_p: 0.95,
          frequency_penalty: 0,
          presence_penalty: 0
        }
      });
      
      console.log(`Created ChatOpenAI instance for model: ${modelId}`);
      
      // Test the model with a simple request before caching
      console.log("Testing model with a simple message...");
      try {
        const testResult = await model.invoke([new SystemMessage("Hello")]);
        console.log("Model test successful, response:", testResult.content.toString().substring(0, 50) + "...");
        
        // Cache the model
        this.modelCache[cacheKey] = model;
        
        return model;
      } catch (testError) {
        console.error("Model test failed with error:", testError);
        throw new Error(`Model test failed: ${testError instanceof Error ? testError.message : String(testError)}`);
      }
    } catch (error) {
      console.error("Error creating or testing Nemotron model:", error);
      throw new Error(`Failed to initialize Nemotron model: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  
  /**
   * Get or create a ChatOpenAI model instance for any NVIDIA model
   */
  public async getNvidiaModel(
    modelId: string, 
    options?: {
      temperature?: number;
      maxTokens?: number;
    }
  ): Promise<ChatOpenAI> {
    const cacheKey = `nvidia-${modelId}-${options?.temperature || 0.7}-${options?.maxTokens || 8192}`;
    
    if (this.modelCache[cacheKey]) {
      return this.modelCache[cacheKey];
    }
    
    // Try to get API key from server endpoint if in browser
    let apiKey: string | undefined;
    
    if (typeof window !== 'undefined') {
      try {
        console.log("Fetching API key from server endpoint");
        const response = await fetch('/api/config');
        if (!response.ok) {
          throw new Error(`Failed to fetch API key: ${response.statusText}`);
        }
        const config = await response.json();
        
        // Get the appropriate API key based on the model
        apiKey = config.nvidiaApiKey;
      } catch (error) {
        console.error("Error fetching API key:", error);
      }
    } else {
      // Server-side, use environment variables directly
      apiKey = process.env.NVIDIA_API_KEY;
    }
    
    // If no API key is found, throw an error
    if (!apiKey) {
      throw new Error(`No API key found for NVIDIA model: ${modelId}`);
    }
    
    // Create a new ChatOpenAI instance
    const model = new ChatOpenAI({
      modelName: modelId,
      temperature: options?.temperature || 0.7,
      maxTokens: options?.maxTokens || 8192,
      openAIApiKey: apiKey,
      configuration: {
        baseURL: "https://integrate.api.nvidia.com/v1",
        timeout: 60000, // 60 second timeout
      },
      // Use specific version of ChatCompletions API
      modelKwargs: {
        "response_format": { "type": "text" }
      }
    });
    
    console.log(`Created ChatOpenAI instance for model: ${modelId}`);
    
    // Test the model with a simple request before caching
    try {
      await model.invoke([new SystemMessage("Hello")]);
      console.log("Model test successful");
    } catch (error) {
      console.error("Model test failed:", error);
      throw new Error(`Failed to initialize model: ${error instanceof Error ? error.message : String(error)}`);
    }
    
    // Cache the model
    this.modelCache[cacheKey] = model;
    
    return model;
  }

  /**
   * Get or create a ChatOpenAI model instance for Ollama models
   */
  public async getOllamaModel(
    modelId: string, 
    options?: {
      temperature?: number;
      maxTokens?: number;
      baseURL?: string;
    }
  ): Promise<ChatOpenAI> {
    const baseURL = options?.baseURL || process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1';
    const cacheKey = `ollama-${modelId}-${options?.temperature || 0.7}-${options?.maxTokens || 8192}-${baseURL}`;
    
    if (this.modelCache[cacheKey]) {
      return this.modelCache[cacheKey];
    }
    
    console.log(`Creating new ChatOpenAI instance for Ollama model: ${modelId}`);

    try {
      // Create a new ChatOpenAI instance for Ollama
      const model = new ChatOpenAI({
        modelName: modelId,
        temperature: options?.temperature || 0.7,
        maxTokens: options?.maxTokens || 8192,
        openAIApiKey: 'ollama', // Ollama doesn't require a real API key
        configuration: {
          baseURL: baseURL,
          timeout: 1800000, // 30 minutes timeout for large model inference
          maxRetries: 0, // Disable retries to avoid additional delays
        },
        modelKwargs: {
          "response_format": { "type": "text" }
        }
      });
      
      console.log(`Created ChatOpenAI instance for Ollama model: ${modelId}`);
      
      // Test the model with a simple request before caching
      console.log("Testing Ollama model with a simple message...");
      try {
        const testResult = await model.invoke([new SystemMessage("Hello")]);
        console.log("Ollama model test successful, response:", testResult.content.toString().substring(0, 50) + "...");
        
        // Cache the model
        this.modelCache[cacheKey] = model;
        
        return model;
      } catch (testError) {
        console.error("Ollama model test failed with error:", testError);
        throw new Error(`Ollama model test failed: ${testError instanceof Error ? testError.message : String(testError)}`);
      }
    } catch (error) {
      console.error("Error creating or testing Ollama model:", error);
      throw new Error(`Failed to initialize Ollama model: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Get or create a ChatOpenAI model instance for vLLM models
   */
  public async getVllmModel(
    modelId: string, 
    options?: {
      temperature?: number;
      maxTokens?: number;
      baseURL?: string;
    }
  ): Promise<ChatOpenAI> {
    const baseURL = options?.baseURL || process.env.VLLM_BASE_URL || 'http://localhost:8001/v1';
    const cacheKey = `vllm-${modelId}-${options?.temperature || 0.7}-${options?.maxTokens || 8192}-${baseURL}`;
    
    if (this.modelCache[cacheKey]) {
      return this.modelCache[cacheKey];
    }
    
    console.log(`Creating new ChatOpenAI instance for vLLM model: ${modelId}`);

    try {
      // Create a new ChatOpenAI instance for vLLM
      const model = new ChatOpenAI({
        modelName: modelId,
        temperature: options?.temperature || 0.7,
        maxTokens: options?.maxTokens || 8192,
        openAIApiKey: 'vllm', // vLLM doesn't require a real API key
        configuration: {
          baseURL: baseURL,
          timeout: 120000, // 2 minute timeout for vLLM inference
        },
        modelKwargs: {
          "response_format": { "type": "text" }
        }
      });
      
      console.log(`Created ChatOpenAI instance for vLLM model: ${modelId}`);
      
      // Test the model with a simple request before caching
      console.log("Testing vLLM model with a simple message...");
      try {
        const testResult = await model.invoke([new SystemMessage("Hello")]);
        console.log("vLLM model test successful, response:", testResult.content.toString().substring(0, 50) + "...");
        
        // Cache the model
        this.modelCache[cacheKey] = model;
        
        return model;
      } catch (testError) {
        console.error("vLLM model test failed with error:", testError);
        throw new Error(`vLLM model test failed: ${testError instanceof Error ? testError.message : String(testError)}`);
      }
    } catch (error) {
      console.error("Error creating vLLM model:", error);
      throw new Error(`Failed to initialize vLLM model: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
}

// Export a singleton instance for convenience
export const langChainService = LangChainService.getInstance(); 