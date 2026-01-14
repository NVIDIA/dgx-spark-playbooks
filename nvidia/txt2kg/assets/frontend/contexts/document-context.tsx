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
"use client"

import type React from "react"

import { createContext, useContext, useState, useEffect } from "react"
import { type Triple, processTextWithChunking, processTextWithChunkingPyG, triplesToGraph } from "@/utils/text-processing"
import { useRouter } from "next/navigation"
import { toast } from "@/hooks/use-toast"
import { type PromptConfigurations } from "@/components/prompt-configuration"

export type Document = {
  id: string
  name: string
  status: "New" | "Processing" | "Processed" | "Error"
  uploadStatus: "Uploading" | "Uploaded"
  size: string
  file: File
  content?: string
  triples?: Triple[]
  graph?: {
    nodes: Array<{ id: string; label: string }>
    edges: Array<{ source: string; target: string; label: string }>
  }
  error?: string
  chunkCount?: number
  extractedDate?: Date
  processingMethod?: 'default' | 'langchain' | 'graphtransformer' | 'fallback'
  embeddings?: {
    count: number
    generated: Date
    status: "New" | "Processing" | "Processed" | "Error"
    error?: string
  }
}

export type LLMProvider = 'nvidia' | 'ollama';

export type ProcessingOptions = {
  useLangChain?: boolean;
  useGraphTransformer?: boolean;
  promptConfigs?: PromptConfigurations;
  llmProvider?: LLMProvider;
  ollamaModel?: string;
  ollamaBaseUrl?: string;
  chunkSize?: number;
  overlapSize?: number;
  chunkingMethod?: 'optimized' | 'pyg';
};

type DocumentContextType = {
  documents: Document[]
  addDocuments: (files: File[]) => void
  deleteDocuments: (documentIds: string[]) => void
  clearDocuments: () => void
  processDocuments: (selectedDocIds?: string[], options?: ProcessingOptions) => Promise<void>
  // Legacy method for backward compatibility
  processDocumentsLegacy: (useLangChain: boolean, selectedDocIds?: string[], useGraphTransformer?: boolean, promptConfigs?: PromptConfigurations) => Promise<void>
  isProcessing: boolean
  updateTriples: (documentId: string, triples: Triple[]) => void
  addTriple: (documentId: string, triple: Triple) => void
  editTriple: (documentId: string, index: number, triple: Triple) => void
  deleteTriple: (documentId: string, index: number) => void
  openGraphVisualization: (documentId?: string) => Promise<void>
  generateEmbeddings: (documentId: string) => Promise<void>
  isGeneratingEmbeddings: boolean
  viewTriples?: (documentId: string) => void
}

const DocumentContext = createContext<DocumentContextType | undefined>(undefined)

// Utility function to generate UUID with fallback
const generateUUID = (): string => {
  // Check if crypto.randomUUID is available
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    try {
      return crypto.randomUUID();
    } catch (error) {
      console.warn('crypto.randomUUID failed, using fallback:', error);
    }
  }
  
  // Fallback UUID generation
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
};

export function DocumentProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const [documents, setDocuments] = useState<Document[]>([])
  const [isInitialized, setIsInitialized] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isGeneratingEmbeddings, setIsGeneratingEmbeddings] = useState(false)
  const [apiKey, setApiKey] = useState<string | null>(null)

  // Load API key from localStorage on client-side only
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // API key loading removed - xAI integration has been removed
    }
  }, []);

  // Load from localStorage on client-side only
  useEffect(() => {
    if (!isInitialized) {
      try {
        const savedDocuments = localStorage.getItem('txt2kg_documents')
        if (savedDocuments) {
          const parsedDocuments = JSON.parse(savedDocuments)
          
          // Reconstruct documents with placeholder File objects
          const reconstructedDocs = parsedDocuments.map((doc: any) => {
            // Create a blob from the content if available
            let file: File;
            if (doc.content) {
              // Create a File object from the content string we previously saved
              const blob = new Blob([doc.content], { type: 'text/plain' });
              file = new File([blob], doc.name, { type: 'text/plain' });
            } else {
              // Create an empty placeholder if no content is available
              file = new File([], doc.name, { type: 'text/plain' });
            }
            
            return {
              ...doc,
              file
            };
          });
          
          console.log(`Restored ${reconstructedDocs.length} documents from localStorage`);
          setDocuments(reconstructedDocs);
        }
      } catch (error) {
        console.error('Error loading documents from localStorage:', error);
      }
      
      setIsInitialized(true);
    }
  }, [isInitialized]);

  // Save documents to localStorage whenever they change, but only after initialization
  useEffect(() => {
    if (isInitialized) {
      try {
        if (documents.length > 0) {
          // Serialize documents for localStorage storage
          // We need to ensure large documents don't exceed localStorage limits
          // Focus on saving processed data (triples & graph) rather than raw content for large files
          const documentsToSave = documents.map(doc => {
            // Don't save content for very large documents to avoid localStorage limits
            // But keep it for smaller ones to avoid reprocessing
            const shouldSaveContent = !doc.content || doc.content.length < 100000;
            
            return {
              ...doc,
              // Omit the actual File object as it can't be serialized
              file: {
                name: doc.file.name,
                size: doc.file.size,
                type: doc.file.type
              },
              // Only include content for smaller documents
              content: shouldSaveContent ? doc.content : undefined
            };
          });
          
          localStorage.setItem('txt2kg_documents', JSON.stringify(documentsToSave));
          console.log(`Saved ${documents.length} documents to localStorage`);
        } else {
          // Clear localStorage if documents array is empty
          localStorage.removeItem('txt2kg_documents');
          console.log('Cleared documents from localStorage');
        }
      } catch (error) {
        console.error('Error saving documents to localStorage:', error);
      }
    }
  }, [documents, isInitialized])

  const addDocuments = (files: File[]) => {
    const newDocuments = files.map((file) => ({
      id: generateUUID(),
      name: file.name,
      status: "New" as const,
      uploadStatus: "Uploaded" as const,
      size: (file.size / 1024).toFixed(2), // Convert to KB
      file,
    }))

    setDocuments((prev) => [...prev, ...newDocuments])
  }

  const deleteDocuments = (documentIds: string[]) => {
    setDocuments((prev) => prev.filter((doc) => !documentIds.includes(doc.id)))
  }

  const clearDocuments = () => {
    setDocuments([])
  }

  const updateDocumentStatus = (id: string, status: Document["status"], updates: Partial<Document> = {}) => {
    console.log(`Updating document ${id} status to: ${status}`);
    setDocuments((prev) => {
      const updated = prev.map((doc) => (doc.id === id ? { ...doc, status, ...updates } : doc));
      
      // Force UI refresh by adding timestamp to document state
      // This ensures React detects the change and re-renders components
      const timestamped = updated.map(doc => ({
        ...doc,
        _lastUpdated: Date.now() // Adding timestamp helps React detect changes
      }));
      
      return timestamped;
    });
    
    // Trigger a custom event for components that need to refresh
    if (typeof window !== 'undefined') {
      console.log('Dispatching document-status-changed event');
      window.dispatchEvent(new CustomEvent('document-status-changed', { 
        detail: { documentId: id, status }
      }));
    }
  }

  const updateTriples = (documentId: string, triples: Triple[]) => {
    // Helper function to normalize text
    const normalizeText = (text: string): string => {
      return text.replace(/['"()]/g, '').trim();
    };

    // Normalize triples before saving
    const normalizedTriples = triples.map(triple => ({
      subject: normalizeText(triple.subject),
      predicate: normalizeText(triple.predicate),
      object: normalizeText(triple.object)
    }));

    setDocuments((prev) =>
      prev.map((doc) => {
        if (doc.id === documentId) {
          const graph = triplesToGraph(normalizedTriples)
          return { ...doc, triples: normalizedTriples, graph }
        }
        return doc
      }),
    )
  }

  const addTriple = (documentId: string, triple: Triple) => {
    // Helper function to normalize text with null/undefined checks
    const normalizeText = (text: string | null | undefined): string => {
      if (!text || typeof text !== 'string') return '';
      return text.replace(/['"()]/g, '').trim();
    };

    // Normalize the new triple
    const normalizedTriple = {
      subject: normalizeText(triple.subject),
      predicate: normalizeText(triple.predicate),
      object: normalizeText(triple.object)
    };

    setDocuments((prev) =>
      prev.map((doc) => {
        if (doc.id === documentId && doc.triples) {
          const newTriples = [...doc.triples, normalizedTriple]
          const graph = triplesToGraph(newTriples)
          return { ...doc, triples: newTriples, graph }
        }
        return doc
      }),
    )
  }

  const editTriple = (documentId: string, index: number, triple: Triple) => {
    // Helper function to normalize text with null/undefined checks
    const normalizeText = (text: string | null | undefined): string => {
      if (!text || typeof text !== 'string') return '';
      return text.replace(/['"()]/g, '').trim();
    };

    // Normalize the edited triple
    const normalizedTriple = {
      subject: normalizeText(triple.subject),
      predicate: normalizeText(triple.predicate),
      object: normalizeText(triple.object)
    };

    setDocuments((prev) =>
      prev.map((doc) => {
        if (doc.id === documentId && doc.triples) {
          const newTriples = [...doc.triples]
          newTriples[index] = normalizedTriple
          const graph = triplesToGraph(newTriples)
          return { ...doc, triples: newTriples, graph }
        }
        return doc
      }),
    )
  }

  const deleteTriple = (documentId: string, index: number) => {
    setDocuments((prev) =>
      prev.map((doc) => {
        if (doc.id === documentId && doc.triples) {
          const newTriples = doc.triples.filter((_, i) => i !== index)
          const graph = triplesToGraph(newTriples)
          return { ...doc, triples: newTriples, graph }
        }
        return doc
      }),
    )
  }

  const readFileContent = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      // Check if it's a valid file with size
      if (file.size === 0) {
        // Handle zero-byte files
        console.warn(`File ${file.name} is empty (0 bytes)`);
        reject(new Error('File is empty (0 bytes)'));
        return;
      }
      
      // If the file isn't a real file (like from localStorage), handle that case
      if (!(file instanceof Blob) || (file.size === 0 && file.type === '')) {
        console.warn(`File ${file.name} appears to be a placeholder or invalid`);
        reject(new Error('Invalid file reference - likely a placeholder'));
        return;
      }
      
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        if (!content || content.trim() === '') {
          console.warn(`File ${file.name} content is empty or whitespace only`);
          reject(new Error('File content is empty'));
          return;
        }
        resolve(content);
      };
      reader.onerror = (e) => {
        console.error(`Error reading file ${file.name}:`, e);
        reject(e);
      };
      reader.readAsText(file);
    });
  }

  const extractTriplesFromChunk = async (chunk: string, systemPrompt?: string): Promise<Triple[]> => {
    console.log(`Extracting triples from chunk of length: ${chunk.length}`)
    
    // Create headers with API key if available
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    }
    
    // Add API key to headers if available
    if (apiKey) {
      headers["X-API-Key"] = apiKey
    }

    // Prepare request body with optional custom system prompt
    const requestBody: any = { text: chunk };
    if (systemPrompt) {
      requestBody.systemPrompt = systemPrompt;
    }

    // Add LLM provider information based on selected model
    const selectedModel = localStorage.getItem("selectedModel");
    if (selectedModel) {
      try {
        const model = JSON.parse(selectedModel);
        if (model.provider === "ollama") {
          requestBody.llmProvider = "ollama";
          requestBody.ollamaModel = model.model || "llama3.1:8b";
          console.log(`🦙 Using Ollama model: ${requestBody.ollamaModel}`);
        } else if (model.provider === "vllm") {
          requestBody.llmProvider = "vllm";
          requestBody.vllmModel = model.model;
          requestBody.vllmBaseUrl = model.baseURL || "http://localhost:8001/v1";
          console.log(`🚀 Using vLLM model: ${requestBody.vllmModel}`);
        } else if (model.id === "nvidia-nemotron" || model.id === "nvidia-nemotron-nano") {
          requestBody.llmProvider = "nvidia";
          requestBody.nvidiaModel = model.model; // Pass the actual model name
          console.log(`🖥️ Using NVIDIA model: ${model.model}`);
        }
      } catch (e) {
        // Ignore parsing errors, will use default
        console.log(`⚠️ Error parsing selected model, using default`);
      }
    } else {
      console.log(`⚠️ No selected model found, using default`);
    }

    const response = await fetch("/api/extract-triples", {
      method: "POST",
      headers,
      body: JSON.stringify(requestBody),
      // Rely on server-side timeout configuration instead of client-side AbortSignal
    })

    console.log("API response status:", response.status)

    const data = await response.json()

    if (!response.ok) {
      console.error("API error:", data)
      throw new Error(data.error || "Failed to extract triples")
    }

    console.log("API response data:", data)
    console.log("Triples count:", data.triples?.length || 0)

    return data.triples || []
  }

  // New processDocuments method with better options structure
  const processDocuments = async (
    selectedDocIds?: string[], 
    options?: ProcessingOptions
  ) => {
    console.log('🔍 processDocuments called with:', {
      selectedDocIds,
      selectedCount: selectedDocIds?.length || 0,
      options
    });
    
    const {
      useLangChain = false,
      useGraphTransformer = false,
      promptConfigs,
      llmProvider = 'ollama',
      ollamaModel = 'qwen3:1.7b',
      ollamaBaseUrl = 'http://localhost:11434/v1',
      chunkSize = 64000,
      overlapSize = 2000,
      chunkingMethod = 'optimized'
    } = options || {};

    return processDocumentsImpl(useLangChain, selectedDocIds, useGraphTransformer, promptConfigs, {
      llmProvider,
      ollamaModel,
      ollamaBaseUrl,
      chunkSize,
      overlapSize,
      chunkingMethod
    });
  };

  // Legacy method for backward compatibility
  const processDocumentsLegacy = async (
    useLangChain: boolean, 
    selectedDocIds?: string[], 
    useGraphTransformer?: boolean,
    promptConfigs?: PromptConfigurations
  ) => {
    return processDocumentsImpl(useLangChain, selectedDocIds, useGraphTransformer, promptConfigs);
  };

  const processDocumentsImpl = async (
    useLangChain: boolean, 
    selectedDocIds?: string[], 
    useGraphTransformer?: boolean,
    promptConfigs?: PromptConfigurations,
    llmOptions?: {
      llmProvider?: LLMProvider;
      ollamaModel?: string;
      ollamaBaseUrl?: string;
      chunkSize?: number;
      overlapSize?: number;
      chunkingMethod?: 'optimized' | 'pyg';
    }
  ) => {
    console.log('🔍 processDocumentsImpl called with:', {
      useLangChain,
      selectedDocIds,
      selectedCount: selectedDocIds?.length || 0,
      useGraphTransformer,
      totalDocuments: documents.length
    });
    
    // If selectedDocIds is explicitly provided, use it
    // If not provided, don't process anything (instead of processing all docs)
    const docIdsToProcess = selectedDocIds || [];
    
    console.log('🔍 Document IDs to process:', docIdsToProcess);
    
    // Get selected documents - filter by the provided selectedDocIds array
    const docsToProcess = documents.filter(
      (doc) => docIdsToProcess.includes(doc.id) && 
      (doc.status === "New" || doc.status === "Processed" || doc.status === "Error")
    );

    console.log('🔍 Documents to process:', docsToProcess.map(d => ({ id: d.id, name: d.name, status: d.status })));

    if (docsToProcess.length === 0) {
      console.log("❌ No documents to process - either none selected or none have valid status");
      return;
    }

    setIsProcessing(true);
    
    try {
      // Check which documents are already processed in ArangoDB
      console.log('🔍 Checking which documents are already processed in ArangoDB...');
      let alreadyProcessedDocs: Set<string> = new Set();
      
      try {
        const response = await fetch('/api/graph-db/check-document', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            documentNames: docsToProcess.map(d => d.name)
          })
        });
        
        if (response.ok) {
          const result = await response.json();
          if (result.processedDocuments) {
            Object.entries(result.processedDocuments).forEach(([docName, isProcessed]) => {
              if (isProcessed) {
                alreadyProcessedDocs.add(docName);
              }
            });
            console.log(`✅ Found ${alreadyProcessedDocs.size} documents already processed in ArangoDB:`, Array.from(alreadyProcessedDocs));
          }
        }
      } catch (checkError) {
        console.warn('⚠️ Could not check for already processed documents, continuing anyway:', checkError);
      }

      // Process each document sequentially
      for (const doc of docsToProcess) {
        // Skip if document is already processed in ArangoDB
        if (alreadyProcessedDocs.has(doc.name)) {
          console.log(`⏭️ Skipping document "${doc.name}" - already processed in ArangoDB`);
          updateDocumentStatus(doc.id, "Processed", {
            triples: doc.triples || [],
            graph: doc.graph,
            error: undefined
          });
          toast({
            title: "Document Skipped",
            description: `"${doc.name}" is already stored in ArangoDB`,
            duration: 3000,
          });
          continue;
        }
        // Update status to Processing before we begin
        updateDocumentStatus(doc.id, "Processing");
        
        try {
          // Read file content if not already available
          let content = doc.content;
          if (!content) {
            content = await readFileContent(doc.file);
          }

          console.log(`🚀 Processing document ${doc.name}, useLangChain: ${useLangChain}, isCSV: ${doc.name.toLowerCase().endsWith('.csv')}`);
          
          // Handle CSV files specially - always use row-as-document processing regardless of LangChain setting
          if (doc.name.toLowerCase().endsWith('.csv')) {
            console.log('📊 Processing CSV file with row-as-document approach:', doc.name);
            
            try {
              const triples = await parseCSVContent(content);
              console.log(`✅ CSV processing complete: ${triples.length} triples extracted`);
              
              // Send to process-document API
              const response = await fetch('/api/process-document', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                  text: content,
                  filename: doc.name,
                  triples: triples,
                  useLangChain: useLangChain, // Pass through the original setting
                  useGraphTransformer: useGraphTransformer,
                  systemPrompt: promptConfigs?.systemPrompt,
                  extractionPrompt: promptConfigs?.extractionPrompt,
                  graphTransformerPrompt: promptConfigs?.graphTransformerPrompt
                })
              });
              
              if (!response.ok) {
                throw new Error(`Document processing failed: ${response.statusText}`);
              }
              
              const result = await response.json();
              
              // Update the document with triples and graph
              updateDocumentStatus(doc.id, "Processed", {
                triples: triples,
                graph: triplesToGraph(triples),
                metadata: {
                  totalTriples: triples.length,
                  processingMethod: 'csv_row_as_document',
                  langchainUsed: useLangChain,
                  graphTransformerUsed: useGraphTransformer
                }
              });
              
              console.log(`✅ Document ${doc.name} processed successfully with ${triples.length} triples`);
            } catch (error) {
              console.error(`❌ Error processing CSV file ${doc.name}:`, error);
              updateDocumentStatus(doc.id, "Error", undefined, error instanceof Error ? error.message : 'Unknown error');
            }
            
            continue; // Skip the rest of the processing for CSV files
          }
          
          if (useLangChain) {
            // Use process-document endpoint with useLangChain flag
            console.log(`Processing document ${doc.name} with LangChain via process-document API...`);
            
            // Extract triples using the default method first (for fallback)
            let triples: Triple[] = [];
            try {
              // Convert JSON to text if it's a JSON file
              let processedContent = content;
              if (doc.name.toLowerCase().endsWith('.json')) {
                processedContent = convertJsonToText(content);
              }
              
              // Pass the custom system prompt if available
              const systemPrompt = promptConfigs?.systemPrompt;
              triples = await processTextWithChunking(
                processedContent, 
                (chunk) => extractTriplesFromChunk(chunk, systemPrompt)
              );
              
              // Call the process-document API endpoint with useLangChain flag
              // NOTE: This no longer automatically stores triples in Neo4j.
              // Storage in Neo4j is now handled manually through the UI's "Store in Graph DB" button.
              console.log(`Sending ${triples.length} triples to process-document API with useLangChain=true ${useGraphTransformer ? 'using GraphTransformer' : ''}`);
              
              // Include prompt configurations in the request body
              const requestBody: any = { 
                text: doc.name.toLowerCase().endsWith('.json') ? convertJsonToText(content) : content,
                filename: doc.name,
                triples: triples,
                useLangChain: true,
                useGraphTransformer: useGraphTransformer
              };
              
              // Add LLM provider options if available
              if (llmOptions) {
                if (llmOptions.llmProvider) {
                  requestBody.llmProvider = llmOptions.llmProvider;
                }
                if (llmOptions.ollamaModel) {
                  requestBody.ollamaModel = llmOptions.ollamaModel;
                }
                if (llmOptions.ollamaBaseUrl) {
                  requestBody.ollamaBaseUrl = llmOptions.ollamaBaseUrl;
                }
              }
              
              // Add prompt configs if available
              if (promptConfigs) {
                if (useGraphTransformer && promptConfigs.graphTransformerPrompt) {
                  requestBody.graphTransformerPrompt = promptConfigs.graphTransformerPrompt;
                } else if (promptConfigs.defaultExtractionPrompt) {
                  requestBody.extractionPrompt = promptConfigs.defaultExtractionPrompt;
                }
              }
              
              const response = await fetch('/api/process-document', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
              });

              if (!response.ok) {
                const errorText = await response.text();
                console.error(`Document processing API error: ${response.status} ${response.statusText}`, errorText);
                throw new Error(`Document processing failed: ${response.statusText} - ${errorText}`);
              }

              const result = await response.json();
              console.log(`Received response from process-document API with ${result.triples?.length || 0} triples`);
              
              // Update the document with triples and graph
              const resultTriples = result.triples || triples; // Fall back to original triples if none returned
              console.log(`Updating document status to "Processed" with ${resultTriples.length} triples`);
              updateDocumentStatus(doc.id, "Processed", {
                content,
                triples: resultTriples,
                graph: triplesToGraph(resultTriples),
                extractedDate: new Date(),
                processingMethod: useGraphTransformer ? 'graphtransformer' : 'langchain'
              });
            } catch (processingError) {
              console.error(`Error in LangChain processing for ${doc.name}:`, processingError);
              
              // If we have fallback triples, still mark as processed but include the error
              if (triples.length > 0) {
                console.log(`Using ${triples.length} fallback triples despite processing error`);
                updateDocumentStatus(doc.id, "Processed", {
                  content,
                  triples,
                  graph: triplesToGraph(triples),
                  extractedDate: new Date(),
                  error: processingError instanceof Error ? processingError.message : "Unknown error during LangChain processing",
                  processingMethod: 'fallback'
                });
              } else {
                // If no fallback triples, mark as error
                throw processingError;
              }
            }
          } else {
            // Use default processing (original implementation)
            console.log(`Processing document ${doc.name} using default processor...`);
            
            // Note: CSV files are handled above, so this only processes non-CSV files
            {
              // For non-CSV files, use the text chunking approach
              console.log(`Processing text document with chunking: ${doc.name}`);
              
              // Convert JSON to text if it's a JSON file
              let processedContent = content;
              if (doc.name.toLowerCase().endsWith('.json')) {
                processedContent = convertJsonToText(content);
                console.log(`Converted JSON file ${doc.name} to text format for processing`);
              }
              
              // Use custom system prompt if available
              const systemPrompt = promptConfigs?.systemPrompt;
              const chunkSize = llmOptions?.chunkSize || 512;
              const overlapSize = llmOptions?.overlapSize || 0;
              const chunkingMethod = llmOptions?.chunkingMethod || 'pyg';
              
              let triples: Triple[];
              if (chunkingMethod === 'pyg') {
                // Use PyTorch Geometric's exact chunking method with configurable chunk size and overlap
                const pygChunkSize = chunkSize || 512; // Use configured chunk size or default to 512
                const pygOverlapSize = overlapSize || 0; // Use configured overlap or default to 0 (original PyG behavior)
                triples = await processTextWithChunkingPyG(
                  processedContent, 
                  (chunk) => extractTriplesFromChunk(chunk, systemPrompt),
                  pygChunkSize,
                  pygOverlapSize
                );
              } else {
                // Use optimized chunking with overlap
                triples = await processTextWithChunking(
                  processedContent, 
                  (chunk) => extractTriplesFromChunk(chunk, systemPrompt),
                  chunkSize,
                  overlapSize
                );
              }
              
              // Send to process-document API - no longer automatically stores in Neo4j
              // Storage in Neo4j is now handled manually through the UI's "Store in Graph DB" button
              const requestBody: any = { 
                text: processedContent,
                filename: doc.name,
                triples: triples,
                useLangChain: false
              };
              
              // Add system prompt if available
              if (promptConfigs?.systemPrompt) {
                requestBody.systemPrompt = promptConfigs.systemPrompt;
              }
              
              const response = await fetch('/api/process-document', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
              });
              
              if (!response.ok) {
                throw new Error(`Document processing failed: ${response.statusText}`);
              }
              
              // Update the document with triples and graph
              updateDocumentStatus(doc.id, "Processed", {
                content,
                triples,
                graph: triplesToGraph(triples),
                chunkCount: Math.ceil(content.length / 512), // Approximate chunk count
                extractedDate: new Date()
              });
            }
          }
        } catch (error) {
          console.error(`Error processing document ${doc.name}:`, error);
          updateDocumentStatus(doc.id, "Error", {
            error: error instanceof Error ? error.message : "Unknown error"
          });
        }
      }
    } finally {
      // Add a small delay before turning off the processing state
      // This gives time for all UI updates to complete
      console.log("Processing complete, finalizing UI updates...");
      
      // Force a final UI refresh by dispatching an event immediately
      if (typeof window !== 'undefined') {
        console.log("Dispatching processing-complete event");
        window.dispatchEvent(new CustomEvent('processing-complete'));
      }
      
      // Reset the processing state
      setIsProcessing(false);
      console.log("Processing state reset, UI should be updated");
    }
  }

  // Helper function to process CSV content - each row as a document for LLM extraction
  const parseCSVContent = async (csvContent: string): Promise<Triple[]> => {
    console.log('🔍 parseCSVContent called with content length:', csvContent.length);
    console.log('Processing CSV content with row-as-document approach');
    
    // Split the CSV content into lines
    const lines = csvContent.split('\n').filter(line => line.trim().length > 0);
    
    if (lines.length < 2) {
      throw new Error("CSV file must contain a header row and at least one data row");
    }
    
    // Parse the header row
    const header = lines[0].split(',').map(h => h.trim().replace(/^"(.*)"$/, '$1'));
    console.log(`CSV headers: ${header.join(', ')}`);
    
    // Get data rows (skip header)
    const dataRows = lines.slice(1);
    console.log(`Processing ${dataRows.length} data rows as individual documents`);
    
    let allTriples: Triple[] = [];
    const BATCH_SIZE = 50; // Store every 50 rows
    let currentBatch: Triple[] = [];
    let storedTriples = 0;
    
    // Process each row as a separate document
    for (let rowIdx = 0; rowIdx < dataRows.length; rowIdx++) {
      const line = dataRows[rowIdx];
      
      try {
        // Parse CSV row into fields
        const fields: string[] = [];
        let fieldStart = 0;
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
          if (line[i] === '"') {
            inQuotes = !inQuotes;
          } else if (line[i] === ',' && !inQuotes) {
            fields.push(line.substring(fieldStart, i).trim().replace(/^"(.*)"$/, '$1'));
            fieldStart = i + 1;
          }
        }
        
        // Add the last field
        fields.push(line.substring(fieldStart).trim().replace(/^"(.*)"$/, '$1'));
        
        // Create document text from the row data
        let documentText = '';
        for (let i = 0; i < Math.min(header.length, fields.length); i++) {
          if (fields[i] && fields[i].trim()) {
            documentText += `${header[i]}: ${fields[i]}\n`;
          }
        }
        
        // Skip empty rows
        if (!documentText.trim()) {
          console.warn(`Skipping empty CSV row ${rowIdx + 1}`);
          continue;
        }
        
        console.log(`Processing row ${rowIdx + 1} as document: ${documentText.substring(0, 100)}...`);
        
        // Extract triples from this row's text using the existing extraction function
        try {
          console.log(`🔄 Calling extractTriplesFromChunk for row ${rowIdx + 1}`);
          // Note: promptConfigs is not available in this scope, so we'll pass undefined for now
          const rowTriples = await extractTriplesFromChunk(documentText, undefined);
          
          console.log(`📥 extractTriplesFromChunk returned:`, rowTriples);
          
          if (rowTriples && Array.isArray(rowTriples)) {
            console.log(`✅ Extracted ${rowTriples.length} triples from row ${rowIdx + 1}`);
            allTriples = allTriples.concat(rowTriples);
            currentBatch = currentBatch.concat(rowTriples);
            
            // Store batch every BATCH_SIZE rows or on last row
            if (currentBatch.length >= BATCH_SIZE || rowIdx === dataRows.length - 1) {
              try {
                console.log(`💾 Storing batch: ${currentBatch.length} triples (rows ${storedTriples + 1}-${rowIdx + 1})`);
                
                // Store batch to database via API
                const batchResponse = await fetch('/api/graph-db/triples', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ 
                    triples: currentBatch,
                    source: `CSV batch ${Math.floor(storedTriples / BATCH_SIZE) + 1}`
                  })
                });
                
                if (batchResponse.ok) {
                  storedTriples += currentBatch.length;
                  console.log(`✅ Batch stored successfully! Progress: ${storedTriples} total triples stored`);
                } else {
                  console.error(`❌ Failed to store batch: ${batchResponse.statusText}`);
                  // Continue processing even if storage fails
                }
                
                currentBatch = []; // Reset batch
              } catch (batchError) {
                console.error(`❌ Error storing batch at row ${rowIdx + 1}:`, batchError);
                // Continue processing even if one batch fails
              }
            }
          } else {
            console.warn(`⚠️ No valid triples returned for row ${rowIdx + 1}`);
          }
        } catch (error) {
          console.error(`❌ Error extracting triples from row ${rowIdx + 1}:`, error);
          continue;
        }
        
      } catch (parseError) {
        console.error(`Error parsing CSV row ${rowIdx + 1}:`, parseError);
        continue;
      }
    }
    
    console.log(`🏁 Successfully extracted ${allTriples.length} triples from ${dataRows.length} CSV rows`);
    console.log('Final triples array:', allTriples);
    return allTriples;
  }

  // Helper function to convert JSON content to readable text format
  const convertJsonToText = (jsonContent: string): string => {
    try {
      // Parse the JSON to validate it
      const jsonData = JSON.parse(jsonContent);
      
      // Convert JSON to a readable text format that preserves structure and relationships
      const formatJsonObject = (obj: any, indent: number = 0): string => {
        const spaces = '  '.repeat(indent);
        
        if (obj === null || obj === undefined) {
          return 'null';
        }
        
        if (typeof obj === 'string' || typeof obj === 'number' || typeof obj === 'boolean') {
          return String(obj);
        }
        
        if (Array.isArray(obj)) {
          if (obj.length === 0) return '[]';
          const items = obj.map((item, index) => 
            `${spaces}  Item ${index + 1}: ${formatJsonObject(item, indent + 1)}`
          ).join('\n');
          return `[\n${items}\n${spaces}]`;
        }
        
        if (typeof obj === 'object') {
          const entries = Object.entries(obj);
          if (entries.length === 0) return '{}';
          
          const props = entries.map(([key, value]) => 
            `${spaces}  ${key}: ${formatJsonObject(value, indent + 1)}`
          ).join('\n');
          return `{\n${props}\n${spaces}}`;
        }
        
        return String(obj);
      };
      
      // Create a descriptive text representation
      let textContent = `JSON Document Content:\n\n`;
      textContent += formatJsonObject(jsonData);
      
      return textContent;
    } catch (error) {
      console.warn('Failed to parse JSON, treating as plain text:', error);
      // If JSON parsing fails, return the original content as-is
      return jsonContent;
    }
  }

  const openGraphVisualization = async (documentId?: string) => {
    // Find the document to visualize
    const doc = documentId
      ? documents.find((d) => d.id === documentId && d.status === "Processed" && d.triples && d.triples.length > 0)
      : documents.find((d) => d.status === "Processed" && d.triples && d.triples.length > 0)

    if (!doc || !doc.triples) {
      console.warn("No suitable document found for graph visualization")
      return
    }

    try {
      // Create a timestamp to ensure we have unique localStorage keys that don't conflict
      const timestamp = Date.now();
      
      // Always store in localStorage as a backup with a timestamp suffix
      try {
        // Store with both the old keys (for backward compatibility) and new timestamped keys
        localStorage.setItem("graphTriples", JSON.stringify(doc.triples))
        localStorage.setItem("graphDocumentName", doc.name)
        
        // Also store with timestamp for uniqueness
        localStorage.setItem(`graphTriples_${timestamp}`, JSON.stringify(doc.triples))
        localStorage.setItem(`graphDocumentName_${timestamp}`, doc.name)
        
        console.log(`Stored ${doc.triples.length} triples in localStorage for document: ${doc.name}`)
      } catch (localStorageError) {
        console.error("LocalStorage error:", localStorageError);
        alert("Warning: Unable to save graph data to browser storage. The graph may not persist if you navigate away.");
        // Continue with API storage even if localStorage fails
      }

      // Try the API approach
      try {
        const response = await fetch("/api/graph-data", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            triples: doc.triples,
            documentName: doc.name,
            timestamp // Include timestamp for correlation
          }),
        })

        if (response.ok) {
          const { graphId } = await response.json()
          console.log(`Successfully stored graph data with ID: ${graphId}`)
          // Use Next.js router.replace to avoid building up history stack
          router.replace(`/graph?id=${graphId}&ts=${timestamp}`)
        } else {
          console.warn(`API storage failed (${response.status}): ${await response.text()}`)
          // If API fails, use localStorage fallback with timestamp parameter
          router.replace(`/graph?source=local&ts=${timestamp}`)
        }
      } catch (apiError) {
        console.error("Error with API storage:", apiError)
        // Navigate using localStorage fallback with timestamp
        router.replace(`/graph?source=local&ts=${timestamp}`)
      }
    } catch (error) {
      console.error("Error preparing graph data:", error)
      alert("Failed to prepare graph data. See console for details.")
    }
  }

  const generateEmbeddings = async (documentId: string) => {
    // Add more detailed diagnostics
    const doc = documents.find(d => d.id === documentId);
    
    if (!doc) {
      toast({
        title: "Document Not Found",
        description: `Could not find document with ID: ${documentId}`,
        variant: "destructive",
        duration: 3000,
      });
      return;
    }
    
    // If content already exists, use it right away
    if (doc.content && doc.content.trim() !== '') {
      await processEmbeddings(doc.id, doc.name, doc.content);
      return;
    }
    
    // Document exists but content is not loaded - log debug info
    console.log(`Attempting to load content for document: ${doc.name}`);
    console.log(`File info: size=${doc.file.size}, type=${doc.file.type}`);
    
    // Check if the document was loaded from localStorage and might have a corrupted file reference
    const isLikelyFromLocalStorage = doc.file.size === 0 || !(doc.file instanceof Blob);
    
    if (isLikelyFromLocalStorage) {
      toast({
        title: "File Reference Issue",
        description: "This document was restored from browser storage and cannot access its original file. Please re-upload the file or process it again first.",
        variant: "destructive",
        duration: 5000,
      });
      return;
    }
    
    try {
      // Document exists but content might not be loaded - try to load it
      const content = await readFileContent(doc.file);
      if (content && content.trim() !== '') {
        // Update the document with content first
        setDocuments(prevDocs => 
          prevDocs.map(d => {
            if (d.id === documentId) {
              return {
                ...d,
                content: content
              };
            }
            return d;
          })
        );
        
        // Continue with the loaded content
        await processEmbeddings(doc.id, doc.name, content);
      } else {
        toast({
          title: "Empty Document",
          description: "The document file appears to be empty",
          variant: "destructive",
          duration: 3000,
        });
      }
    } catch (error) {
      toast({
        title: "Content Loading Error",
        description: `Failed to load document content: ${error instanceof Error ? error.message : String(error)}`,
        variant: "destructive",
        duration: 5000,
      });
    }
  };

  // Helper function to handle the actual embeddings processing
  const processEmbeddings = async (documentId: string, documentName: string, content: string) => {
    setIsGeneratingEmbeddings(true);
    try {
      console.log(`Generating embeddings for document: ${documentName}`);
      
      // Update embeddings status to show it's processing, without changing main document status
      setDocuments(prevDocs => 
        prevDocs.map(d => {
          if (d.id === documentId) {
            return {
              ...d,
              embeddings: {
                count: d.embeddings?.count || 0,
                generated: d.embeddings?.generated || new Date(),
                status: "Processing" as const
              }
            };
          }
          return d;
        })
      );
      
      const response = await fetch('/api/embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          documentId: documentId,
          content: content,
          documentName: documentName
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to generate embeddings: ${await response.text()}`);
      }
      
      const result = await response.json();
      console.log('Embeddings generation result:', result);
      
      // Update embeddings status to show it's processed
      setDocuments(prevDocs => 
        prevDocs.map(d => {
          if (d.id === documentId) {
            return {
              ...d,
              embeddings: {
                count: result.embeddings,
                generated: new Date(),
                status: "Processed" as const
              }
            };
          }
          return d;
        })
      );
      
      // Show a toast notification
      toast({
        title: "Embeddings Generated",
        description: `Successfully generated ${result.embeddings} embeddings for "${documentName}"`,
        duration: 5000,
      });
      
    } catch (error) {
      console.error('Error generating embeddings:', error);
      
      // Update embeddings status to show there was an error
      setDocuments(prevDocs => 
        prevDocs.map(d => {
          if (d.id === documentId) {
            return {
              ...d,
              embeddings: {
                count: d.embeddings?.count || 0,
                generated: d.embeddings?.generated || new Date(),
                status: "Error" as const,
                error: error instanceof Error ? error.message : String(error)
              }
            };
          }
          return d;
        })
      );
      
      toast({
        title: "Embeddings Generation Failed",
        description: `Failed to generate embeddings: ${error instanceof Error ? error.message : String(error)}`,
        variant: "destructive",
        duration: 5000,
      });
    } finally {
      setIsGeneratingEmbeddings(false);
    }
  };

  return (
    <DocumentContext.Provider
      value={{
        documents,
        addDocuments,
        deleteDocuments,
        clearDocuments,
        processDocuments,
    processDocumentsLegacy,
        isProcessing,
        updateTriples,
        addTriple,
        editTriple,
        deleteTriple,
        openGraphVisualization,
        generateEmbeddings,
        isGeneratingEmbeddings
      }}
    >
      {children}
    </DocumentContext.Provider>
  )
}

export function useDocuments() {
  const context = useContext(DocumentContext)
  if (context === undefined) {
    throw new Error("useDocuments must be used within a DocumentProvider")
  }
  return context
}