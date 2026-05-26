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

import { useState, useEffect } from "react"
import { useDocuments } from "@/contexts/document-context"
import { Button } from "@/components/ui/button"
import { Sparkles, Loader2, CheckCircle, AlertCircle, FileText, Zap, Cpu, X, ChevronUp, ChevronDown } from "lucide-react"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import React from "react"
import { AdvancedOptions } from "@/components/advanced-options";
import { PromptConfiguration, PromptConfigurations } from "@/components/prompt-configuration";
import { useShiftSelect } from "@/hooks/use-shift-select"

interface EmbeddingsGeneratorProps {
  showTripleExtraction?: boolean;
}

type Document = {
  id: string;
  name: string;
  status: string;
  uploadStatus: string;
  size: string;
  triples?: any[];
  embeddings?: {
    count: number;
    generated: Date;
    status: "New" | "Processing" | "Processed" | "Error";
    error?: string;
  };
};

interface ContentProps {
  documents: Document[];
  selectedDocs: string[];
  handleSelectAll: () => void;
  handleItemClick: (item: Document, event?: React.MouseEvent) => void;
  isSelected: (itemId: string) => boolean;
  error: string | null;
  status: string;
}

interface EmbeddingsContentProps extends ContentProps {
  generateEmbeddings: () => void;
  isGenerating: boolean;
  useLangChain: boolean;
  setUseLangChain: (value: boolean) => void;
  useSentenceChunking: boolean;
  setUseSentenceChunking: (value: boolean) => void;
  embeddingsProvider: string;
  handleStopEmbeddings: () => void;
}

interface TriplesContentProps extends ContentProps {
  extractTriples: (promptConfigs?: PromptConfigurations) => void;
  isProcessing: boolean;
  useLangChain: boolean;
  setUseLangChain: (value: boolean) => void;
  useSentenceChunking: boolean;
  setUseSentenceChunking: (value: boolean) => void;
  useEntityExtraction: boolean;
  setUseEntityExtraction: (value: boolean) => void;
  error: string | null;
  status: string;
  handleStopProcessing: () => void;
}

export function EmbeddingsGenerator({ showTripleExtraction = false }: EmbeddingsGeneratorProps) {
  const { documents, processDocuments, generateEmbeddings: contextGenerateEmbeddings } = useDocuments()
  const [isGenerating, setIsGenerating] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [useLangChain, setUseLangChain] = useState(false)
  const [useSentenceChunking, setUseSentenceChunking] = useState(true)
  const [useEntityExtraction, setUseEntityExtraction] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState("")
  const [langChainMethod, setLangChainMethod] = React.useState<'default' | 'graphtransformer'>(
    'default'
  );
  const [embeddingsProvider, setEmbeddingsProvider] = useState<string>(
    typeof window !== 'undefined' ? localStorage.getItem("embeddings_provider") || "local" : "local"
  );

  // Use shift-select hook for document selection
  const {
    selectedItems: selectedDocs,
    setSelectedItems: setSelectedDocs,
    handleItemClick,
    handleSelectAll,
    isSelected
  } = useShiftSelect({
    items: documents,
    getItemId: (doc) => doc.id,
    canSelect: (doc) => doc.status === "New" || doc.status === "Processed" || doc.status === "Error",
    onSelectionChange: (selectedIds) => {
      // Optional: handle selection change if needed
    }
  })

  // Listen for embeddings settings changes
  useEffect(() => {
    const handleEmbeddingsSettingsChanged = () => {
      const updatedProvider = localStorage.getItem("embeddings_provider") || "local";
      setEmbeddingsProvider(updatedProvider);
      console.log("Embeddings generator detected embeddings settings change:", updatedProvider);
    };
    
    window.addEventListener('embeddings-settings-changed', handleEmbeddingsSettingsChanged);
    
    return () => {
      window.removeEventListener('embeddings-settings-changed', handleEmbeddingsSettingsChanged);
    };
  }, []);



  // Handle tab navigation
  const handleTabChange = (tab: string) => {
    const tabElement = document.querySelector(`[data-value="${tab}"]`)
    if (tabElement && 'click' in tabElement) {
      (tabElement as HTMLElement).click()
    }
  }

  // When LangChain is toggled off, disable dependent options
  useEffect(() => {
    if (!useLangChain) {
      setUseSentenceChunking(false)
      setUseEntityExtraction(false)
    }
    
    // Dispatch custom event to update embedding model info in Processing Summary
    const event = new CustomEvent('langChainToggled', {
      detail: { useLangChain }
    });
    window.dispatchEvent(event);
  }, [useLangChain])

  // Simulate embedding generation
  const generateEmbeddings = async () => {
    if (selectedDocs.length === 0) {
      setError("Please select at least one document")
      return
    }

    setError(null)
    setIsGenerating(true)
    setStatus("Preparing documents for embedding generation...")

    try {
      // Process each selected document
      for (let i = 0; i < selectedDocs.length; i++) {
        const docId = selectedDocs[i];
        const doc = documents.find(d => d.id === docId);
        
        if (!doc) {
          console.error(`Document with ID ${docId} not found`);
          continue;
        }
        
        setStatus(`Generating embeddings for document ${i+1} of ${selectedDocs.length}: ${doc.name}`);
        await contextGenerateEmbeddings(docId);
      }
      
      setStatus("Embedding generation complete!");
      setTimeout(() => {
        setIsGenerating(false);
        setStatus("");
      }, 1500);
    } catch (error) {
      console.error("Error generating embeddings:", error);
      setError("Failed to generate embeddings. Please try again.");
      setIsGenerating(false);
    }
  }

  // Extract triples from documents
  const extractTriples = async (options?: PromptConfigurations & { chunkSize?: number; overlapSize?: number; chunkingMethod?: 'optimized' | 'pyg' }) => {
    if (selectedDocs.length === 0) {
      setError("Please select at least one document")
      return
    }

    setError(null)
    setIsProcessing(true)
    setStatus("Preparing documents for triple extraction...")
    
    // Set up a listener for the processing-complete event
    const handleProcessingComplete = () => {
      console.log("Processing complete event received in embeddings-generator");
      setIsProcessing(false);
      setStatus("");
    };
    
    window.addEventListener('processing-complete', handleProcessingComplete);

    try {
      // Update the processing status display
      const docNames = selectedDocs.map(id => 
        documents.find(d => d.id === id)?.name || 'Unknown'
      ).join(', ');
      
      // Determine the processing method based on selected model and options
      let processingMethod = 'default extractor';
      try {
        const selectedModel = localStorage.getItem("selectedModel");
        if (selectedModel) {
          const model = JSON.parse(selectedModel);
          if (model.provider === "ollama") {
            processingMethod = `Ollama ${model.model || 'qwen3:1.7b'}`;
          } else if (model.id?.startsWith("nvidia-")) {
            processingMethod = 'NVIDIA Nemotron';
          }
        }
      } catch (e) {
        // Fallback to default if parsing fails
      }
      
      if (useLangChain) {
        processingMethod += langChainMethod === 'graphtransformer' ? ' with LLMGraphTransformer' : ' with LangChain';
      }
      
      setStatus(`Processing ${selectedDocs.length} document(s): ${docNames} using ${processingMethod}`);
      
      // Call processDocuments with the selected document IDs and processing options
      const useGraphTransformer = useLangChain && langChainMethod === 'graphtransformer';
      await processDocuments(selectedDocs, {
        useLangChain,
        useGraphTransformer,
        promptConfigs: options || undefined,
        chunkSize: options?.chunkSize,
        overlapSize: options?.overlapSize,
        chunkingMethod: options?.chunkingMethod
      });
      
      // Navigate to the edit tab after processing is complete
      setTimeout(() => {
        // Clean up the event listener
        window.removeEventListener('processing-complete', handleProcessingComplete);
        
        // Navigate to the edit tab
        handleTabChange("edit");
      }, 500);
    } catch (error) {
      console.error("Error processing documents:", error)
      setError("Failed to process documents. Please try again.")
      setIsProcessing(false)
      setStatus("")
      
      // Clean up the event listener
      window.removeEventListener('processing-complete', handleProcessingComplete);
    }
  }

  // Stop processing function
  const handleStopProcessing = async () => {
    try {
      const response = await fetch('/api/stop-processing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        setStatus("Processing stopped by user");
        setError(null);
        setIsProcessing(false);
        setIsGenerating(false);
      } else {
        setError("Failed to stop processing. Please try again.");
      }
    } catch (error) {
      console.error("Error stopping processing:", error);
      setError("Failed to stop processing. Please try again.");
    }
  }

  // Stop embeddings generation function
  const handleStopEmbeddings = async () => {
    try {
      const response = await fetch('/api/stop-embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        setStatus("Embeddings generation stopped by user");
        setError(null);
        setIsGenerating(false);
      } else {
        setError("Failed to stop embeddings generation. Please try again.");
      }
    } catch (error) {
      console.error("Error stopping embeddings generation:", error);
      setError("Failed to stop embeddings generation. Please try again.");
    }
  }

  return (
    <div className="space-y-4">
      {showTripleExtraction ? (
        <Tabs defaultValue="triples" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="triples">Triple Extraction</TabsTrigger>
            <TabsTrigger value="embeddings">Embeddings</TabsTrigger>
          </TabsList>
          
          <TabsContent value="embeddings" className="space-y-4 pt-4">
            <EmbeddingsContent
              documents={documents}
              selectedDocs={selectedDocs}
              handleSelectAll={handleSelectAll}
              handleItemClick={handleItemClick}
              isSelected={isSelected}
              generateEmbeddings={generateEmbeddings}
              isGenerating={isGenerating}
              useLangChain={useLangChain}
              setUseLangChain={setUseLangChain}
              useSentenceChunking={useSentenceChunking}
              setUseSentenceChunking={setUseSentenceChunking}
              error={error}
              status={status}
              embeddingsProvider={embeddingsProvider}
              handleStopEmbeddings={handleStopEmbeddings}
            />
          </TabsContent>
          
          <TabsContent value="triples" className="space-y-4 pt-4">
            <TriplesContent
              documents={documents}
              selectedDocs={selectedDocs}
              handleSelectAll={handleSelectAll}
              handleItemClick={handleItemClick}
              isSelected={isSelected}
              extractTriples={extractTriples}
              isProcessing={isProcessing}
              useLangChain={useLangChain}
              setUseLangChain={setUseLangChain}
              useSentenceChunking={useSentenceChunking}
              setUseSentenceChunking={setUseSentenceChunking}
              useEntityExtraction={useEntityExtraction}
              setUseEntityExtraction={setUseEntityExtraction}
              error={error}
              status={status}
              handleStopProcessing={handleStopProcessing}
            />
          </TabsContent>
        </Tabs>
      ) : (
        <EmbeddingsContent
          documents={documents}
          selectedDocs={selectedDocs}
          handleSelectAll={handleSelectAll}
          handleItemClick={handleItemClick}
          isSelected={isSelected}
          generateEmbeddings={generateEmbeddings}
          isGenerating={isGenerating}
          useLangChain={useLangChain}
          setUseLangChain={setUseLangChain}
          useSentenceChunking={useSentenceChunking}
          setUseSentenceChunking={setUseSentenceChunking}
          error={error}
          status={status}
          embeddingsProvider={embeddingsProvider}
          handleStopEmbeddings={handleStopEmbeddings}
        />
      )}
    </div>
  )
}

// Embeddings content component
function EmbeddingsContent({
  documents,
  selectedDocs,
  handleSelectAll,
  handleItemClick,
  isSelected,
  generateEmbeddings,
  isGenerating,
  useLangChain,
  setUseLangChain,
  useSentenceChunking,
  setUseSentenceChunking,
  error,
  status,
  embeddingsProvider,
  handleStopEmbeddings
}: EmbeddingsContentProps) {
  // Helper function to get embeddings status icon
  const getEmbeddingsStatusIcon = (doc: Document) => {
    // Use embeddings status if available, otherwise show 'New'
    const embeddingsStatus = doc.embeddings?.status || "New";
    
    switch (embeddingsStatus) {
      case "New":
        return <span className="h-2 w-2 rounded-full bg-cyan-400 mr-2"></span>;
      case "Processing":
        return <Loader2 className="h-3.5 w-3.5 text-yellow-500 mr-2 animate-spin" />;
      case "Processed":
        return <CheckCircle className="h-3.5 w-3.5 text-green-500 mr-2" />;
      case "Error":
        return <AlertCircle className="h-3.5 w-3.5 text-destructive mr-2" />;
      default:
        return <span className="h-2 w-2 rounded-full bg-gray-400 mr-2"></span>;
    }
  };
  
  // Helper function to get embeddings status text
  const getEmbeddingsStatusText = (doc: Document) => {
    if (doc.embeddings?.status === "Processed") {
      return `${doc.embeddings.count} vectors`;
    } else if (doc.embeddings?.status) {
      return doc.embeddings.status;
    } else {
      return "Ready";
    }
  };
  
  return (
    <>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 rounded-md bg-nvidia-green/15 flex items-center justify-center">
            <Sparkles className="h-3 w-3 text-nvidia-green" />
          </div>
          <h3 className="text-base font-semibold text-foreground">Generate Embeddings</h3>
        </div>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="text-xs text-muted-foreground cursor-help flex items-center hover:text-foreground transition-colors">
                <InfoIcon className="h-4 w-4 mr-1" />
                What are embeddings?
              </div>
            </TooltipTrigger>
            <TooltipContent className="max-w-[280px]">
              <p className="text-xs">
                Embeddings are vector representations of your documents that enable semantic search and similarity matching between documents.
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      
      <p className="text-sm text-muted-foreground">
        Generate vector embeddings for semantic search and document similarity
      </p>
      
      {/* Current embeddings provider indicator */}
      <div className="flex items-center text-xs text-muted-foreground mt-1 mb-3 bg-background/40 border border-border/40 rounded-md p-2">
        <Cpu className="h-3.5 w-3.5 mr-1.5" />
        <span>Using: <span className="font-medium text-primary">{embeddingsProvider === "nvidia" ? "NVIDIA API" : "Local Sentence Transformer"}</span></span>
      </div>

      <div className="space-y-3 pt-2">
        <h4 className="text-sm font-medium">Processing Options</h4>
        
        <div className="space-y-2">
          <div className="flex items-center space-x-2">
            <Switch 
              id="use-langchain-embeddings" 
              checked={useLangChain}
              onCheckedChange={setUseLangChain}
              disabled={isGenerating}
            />
            <Label htmlFor="use-langchain-embeddings" className="text-sm cursor-pointer">Use LangChain</Label>
          </div>
          
          {useLangChain && (
            <div className="mt-3">
              <AdvancedOptions title="LangChain Options" defaultOpen={false}>
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <Switch 
                      id="use-sentence-chunking-embeddings" 
                      checked={useSentenceChunking}
                      onCheckedChange={setUseSentenceChunking}
                      disabled={isGenerating}
                    />
                    <Label 
                      htmlFor="use-sentence-chunking-embeddings" 
                      className="text-sm cursor-pointer"
                    >
                      Use Sentence Chunking
                    </Label>
                  </div>
                  <p className="text-xs text-muted-foreground pl-7">
                    Split documents into sentences for more accurate embeddings
                  </p>
                </div>
              </AdvancedOptions>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-destructive/10 border border-destructive rounded-md p-3 flex items-start gap-2">
          <AlertCircle className="h-4 w-4 text-destructive mt-0.5 flex-shrink-0" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      <div className="border rounded-md overflow-hidden mt-4">
        <div className="bg-muted/30 p-3 flex items-center justify-between">
          <div className="flex items-center">
            <input
              type="checkbox"
              className="rounded border-border text-primary focus:ring-primary mr-3 h-4 w-4"
              checked={selectedDocs.length === documents.filter(doc => doc.uploadStatus === "Uploaded").length && 
                      documents.filter(doc => doc.uploadStatus === "Uploaded").length > 0}
              onChange={handleSelectAll}
              disabled={documents.length === 0 || isGenerating}
            />
            <span className="text-sm font-medium">
              {selectedDocs.length > 0 ? (
            <span className="text-nvidia-green text-xs">{selectedDocs.length} selected</span>
              ) : (
                <span className="text-xs">Select all</span>
              )}
            </span>
          </div>

          <div className="flex gap-3 pt-4">
            <Button
              size="default"
              disabled={isGenerating || selectedDocs.length === 0}
              onClick={generateEmbeddings}
              className="bg-nvidia-green hover:bg-nvidia-green/90 text-white font-medium px-6 py-2 gap-2"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Generating</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  <span>Generate Embeddings</span>
                </>
              )}
            </Button>
            
            {isGenerating && (
              <Button
                size="default"
                variant="destructive"
                onClick={handleStopEmbeddings}
                className="px-4 py-2 gap-2"
              >
                <X className="h-4 w-4" />
                <span>Stop</span>
              </Button>
            )}
          </div>
        </div>

        <div className="max-h-[250px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="w-12 py-2 px-3 text-left"></th>
                <th className="py-2 px-3 text-left">Document</th>
                <th className="py-2 px-3 text-left">Size</th>
                <th className="py-2 px-3 text-left">Triple Status</th>
                <th className="py-2 px-3 text-left">Embeddings Status</th>
              </tr>
            </thead>
            <tbody>
              {documents.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-6 text-center text-muted-foreground">
                    No documents available for embedding generation
                  </td>
                </tr>
              ) : (
                documents.map((doc) => (
                  <tr key={doc.id} className="border-t hover:bg-muted/20"
                      onClick={(e) => handleItemClick(doc, e)}>
                    <td className="py-2 px-3" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        className="rounded border-border text-primary focus:ring-primary h-4 w-4"
                        checked={isSelected(doc.id)}
                        onChange={(e) => handleItemClick(doc, e)}
                        disabled={isGenerating}
                      />
                    </td>
                    <td className="py-2 px-3 font-medium">{doc.name}</td>
                    <td className="py-2 px-3">{doc.size}</td>
                    <td className="py-2 px-3">
                      <div className="flex items-center">
                        {doc.status === "New" && (
                          <span className="h-2 w-2 rounded-full bg-cyan-400 mr-2"></span>
                        )}
                        {doc.status === "Processing" && (
                          <Loader2 className="h-3.5 w-3.5 text-yellow-500 mr-2 animate-spin" />
                        )}
                        {doc.status === "Processed" && (
                          <CheckCircle className="h-3.5 w-3.5 text-green-500 mr-2" />
                        )}
                        {doc.status === "Error" && (
                          <AlertCircle className="h-3.5 w-3.5 text-destructive mr-2" />
                        )}
                        <span>{doc.status}</span>
                      </div>
                    </td>
                    <td className="py-2 px-3">
                      <div className="flex items-center">
                        {getEmbeddingsStatusIcon(doc)}
                        <span>
                          {getEmbeddingsStatusText(doc)}
                          {doc.embeddings?.error && (
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span className="ml-1 cursor-help text-destructive">
                                    <AlertCircle className="h-3 w-3 inline" />
                                  </span>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p className="text-xs max-w-[200px]">{doc.embeddings.error}</p>
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          )}
                        </span>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {isGenerating && (
        <div className="border rounded-md p-4 bg-primary/5 mt-4">
          <div className="flex items-center gap-2 text-sm">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            <span>{status}</span>
          </div>
          <div className="mt-2 h-1 w-full bg-muted overflow-hidden rounded-full">
            <div className="h-full bg-primary rounded-full animate-progress"></div>
          </div>
        </div>
      )}
    </>
  )
}

// Add this function near the top of the file
function RadioButton({ id, name, value, checked, onChange, disabled = false, children }: {
  id: string;
  name: string;
  value: string;
  checked: boolean;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center space-x-2">
      <input
        type="radio"
        id={id}
        name={name}
        value={value}
        checked={checked}
        onChange={onChange}
        disabled={disabled}
        className="h-4 w-4 border-border text-primary focus:ring-primary"
      />
      <label htmlFor={id} className={`text-sm cursor-pointer ${disabled ? "opacity-50" : ""}`}>
        {children}
      </label>
    </div>
  );
}

// Triple extraction content component
function TriplesContent({
  documents,
  selectedDocs,
  handleSelectAll,
  handleItemClick,
  isSelected,
  extractTriples,
  isProcessing,
  useLangChain,
  setUseLangChain,
  useSentenceChunking,
  setUseSentenceChunking,
  useEntityExtraction,
  setUseEntityExtraction,
  error,
  status,
  handleStopProcessing
}: TriplesContentProps) {
  // Add sorting state
  const [sortField, setSortField] = useState<'name' | 'size' | 'status'>('name')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')

  // Sort documents based on current sort field and direction
  const sortedDocuments = React.useMemo(() => {
    return [...documents].sort((a, b) => {
      let aValue: string | number
      let bValue: string | number

      switch (sortField) {
        case 'name':
          aValue = a.name.toLowerCase()
          bValue = b.name.toLowerCase()
          break
        case 'size':
          aValue = parseFloat(a.size) || 0
          bValue = parseFloat(b.size) || 0
          break
        case 'status':
          aValue = a.status
          bValue = b.status
          break
        default:
          aValue = a.name.toLowerCase()
          bValue = b.name.toLowerCase()
      }

      if (sortDirection === 'asc') {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0
      }
    })
  }, [documents, sortField, sortDirection])

  // Handle column header click for sorting
  const handleSort = (field: 'name' | 'size' | 'status') => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('asc')
    }
  }
  const [langChainMethod, setLangChainMethod] = React.useState<'default' | 'graphtransformer'>(
    'default'
  );
  const [promptConfigs, setPromptConfigs] = useState<PromptConfigurations | null>(null);
  
  // Chunk configuration state
  const [chunkSize, setChunkSize] = useState<number>(512);
  const [overlapSize, setOverlapSize] = useState<number>(0);
  const [chunkingMethod, setChunkingMethod] = useState<'optimized' | 'pyg'>('pyg');

  // Handle radio button changes for LangChain method
  const handleLangChainMethodChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setLangChainMethod(e.target.value as 'default' | 'graphtransformer');
  };

  // Load prompt configurations from localStorage on component mount
  useEffect(() => {
    try {
      const savedConfigs = localStorage.getItem("promptConfigurations");
      if (savedConfigs) {
        setPromptConfigs(JSON.parse(savedConfigs));
      }
    } catch (err) {
      console.error("Error loading prompt configurations:", err);
    }
  }, []);

  // Handle prompt configuration changes
  const handlePromptConfigsChange = (configs: PromptConfigurations) => {
    setPromptConfigs(configs);
  };

  // Update actual flag used by API based on both useLangChain and langChainMethod
  React.useEffect(() => {
    // This effect is used to monitor langChainMethod changes
    // The actual implementation of different methods is handled in the API
  }, [langChainMethod]);

  // Handle extract triples button click
  const handleExtractTriples = () => {
    const options = {
      ...(promptConfigs || {}),
      chunkSize,
      overlapSize,
      chunkingMethod
    };
    extractTriples(options);
  };

  return (
    <>
      <div className="flex items-center gap-3 mb-4">
        <div className="w-6 h-6 rounded-md bg-nvidia-green/15 flex items-center justify-center">
          <Zap className="h-3 w-3 text-nvidia-green" />
        </div>
        <h3 className="text-base font-semibold text-foreground">Knowledge Graph Triple Extraction</h3>
      </div>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        Extract structured knowledge triples from documents for knowledge graph construction
      </p>

      <div className="space-y-4">
        <h4 className="text-sm font-semibold text-foreground">Processing Options</h4>
        
        <div className="space-y-2">
          {/* Hidden: Use LangChain toggle - LangChain is always used for triple extraction */}
          {/* <div className="flex items-center space-x-2">
            <Switch
              id="use-langchain-triples"
              checked={useLangChain}
              onCheckedChange={setUseLangChain}
              disabled={isProcessing}
            />
            <Label htmlFor="use-langchain-triples" className="text-sm cursor-pointer">Use LangChain</Label>
          </div> */}
          {/* <p className="text-xs text-muted-foreground pl-7">
            Leverages LangChain for knowledge extraction from documents
          </p> */}

          {false && useLangChain && (
            <div className="mt-3">
              <AdvancedOptions title="LangChain Options" defaultOpen={false}>
                <div className="space-y-3">
                  <div>
                    <h5 className="text-sm font-medium mb-2">LangChain Method</h5>
                    
                    <RadioButton
                      id="default-extractor"
                      name="langchain-method"
                      value="default"
                      checked={langChainMethod === 'default'}
                      onChange={handleLangChainMethodChange}
                      disabled={isProcessing}
                    >
                      Default Extractor
                    </RadioButton>
                    <p className="text-xs text-muted-foreground ml-6 mb-2">
                      Uses the standard LangChain extraction pipeline
                    </p>
                    
                    <RadioButton
                      id="graph-transformer"
                      name="langchain-method"
                      value="graphtransformer"
                      checked={langChainMethod === 'graphtransformer'}
                      onChange={handleLangChainMethodChange}
                      disabled={isProcessing}
                    >
                      LLMGraphTransformer
                    </RadioButton>
                    <p className="text-xs text-muted-foreground ml-6 mb-3">
                      Uses LangChain's specialized graph structure transformer
                    </p>
                  </div>
                  
                  <div>
                    <div className="flex items-center space-x-2">
                      <Switch 
                        id="use-sentence-chunking-triples" 
                        checked={useSentenceChunking}
                        onCheckedChange={setUseSentenceChunking}
                        disabled={isProcessing}
                      />
                      <Label 
                        htmlFor="use-sentence-chunking-triples" 
                        className="text-sm cursor-pointer"
                      >
                        Use Sentence Chunking
                      </Label>
                    </div>
                    <p className="text-xs text-muted-foreground pl-7 mb-3">
                      Split documents into sentences for more accurate triple extraction
                    </p>
                  </div>
                  
                  <div>
                    <div className="flex items-center space-x-2">
                      <Switch 
                        id="use-entity-extraction-triples" 
                        checked={useEntityExtraction}
                        onCheckedChange={setUseEntityExtraction}
                        disabled={isProcessing}
                      />
                      <Label 
                        htmlFor="use-entity-extraction-triples" 
                        className="text-sm cursor-pointer"
                      >
                        Entity Extraction
                      </Label>
                    </div>
                    <p className="text-xs text-muted-foreground pl-7">
                      Automatically detect and extract entities from documents
                    </p>
                  </div>
                </div>
              </AdvancedOptions>
            </div>
          )}
        </div>
      </div>
      
      {/* Chunk Configuration */}
      <div className="mt-4">
        <AdvancedOptions title="Chunk Configuration" defaultOpen={false}>
          <div className="space-y-4">
            {/* Chunking Method Selection */}
            <div>
              <Label className="text-sm font-medium mb-3 block">Chunking Method</Label>
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <input
                    type="radio"
                    id="chunking-optimized"
                    name="chunking-method"
                    value="optimized"
                    checked={chunkingMethod === 'optimized'}
                    onChange={(e) => setChunkingMethod(e.target.value as 'optimized' | 'pyg')}
                    disabled={isProcessing}
                    className="w-4 h-4 text-primary border-border focus:ring-primary"
                  />
                  <Label htmlFor="chunking-optimized" className="text-sm cursor-pointer">
                    Large chunks
                  </Label>
                </div>
                <p className="text-xs text-muted-foreground ml-6 mb-2">
                  Large chunks with overlap for modern LLMs like Gemma3:27b. Best for efficiency.
                </p>
                
                <div className="flex items-center space-x-2">
                  <input
                    type="radio"
                    id="chunking-pyg"
                    name="chunking-method"
                    value="pyg"
                    checked={chunkingMethod === 'pyg'}
                    onChange={(e) => setChunkingMethod(e.target.value as 'optimized' | 'pyg')}
                    disabled={isProcessing}
                    className="w-4 h-4 text-primary border-border focus:ring-primary"
                  />
                  <Label htmlFor="chunking-pyg" className="text-sm cursor-pointer">
                    Default (configurable size and overlap)
                  </Label>
                </div>
                <p className="text-xs text-muted-foreground ml-6 mb-3">
                  PyG's txt2kg.py chunking algorithm with configurable chunk size and overlap.
                </p>
              </div>
            </div>
            
            <div>
              <Label htmlFor="chunk-size" className="text-sm font-medium">
                Chunk Size (characters)
              </Label>
              <div className="mt-1">
                <input
                  id="chunk-size"
                  type="number"
                  min="1000"
                  max="128000"
                  step="1000"
                  value={chunkSize}
                  onChange={(e) => setChunkSize(Number(e.target.value))}
                  disabled={isProcessing}
                  className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                />
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Larger chunks provide more context but use more GPU memory and may lose detailed information.
              </p>
            </div>
            
            <div>
              <Label htmlFor="overlap-size" className="text-sm font-medium">
                Overlap Size (characters)
              </Label>
              <div className="mt-1">
                <input
                  id="overlap-size"
                  type="number"
                  min="0"
                  max="8000"
                  step="100"
                  value={overlapSize}
                  onChange={(e) => setOverlapSize(Number(e.target.value))}
                  disabled={isProcessing}
                  className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                />
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Overlap between chunks to preserve context across boundaries. Set to 0 for original PyG behavior.
              </p>
            </div>
            
            <div className="bg-muted/30 rounded-md p-3">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-4 h-4 rounded bg-primary/20 flex items-center justify-center">
                  <span className="text-xs text-primary">ℹ</span>
                </div>
                <span className="text-xs font-medium">Current Configuration</span>
              </div>
              <div className="text-xs text-muted-foreground space-y-1">
                {chunkingMethod === 'pyg' ? (
                  <>
                    <div>• Method: PyTorch Geometric (enhanced with overlap)</div>
                    <div>• Estimated chunks for 64KB document: ~{Math.ceil(64000 / Math.max(1, chunkSize - overlapSize))}</div>
                    <div>• Chunk size: {chunkSize.toLocaleString()} characters</div>
                    <div>• Overlap: {overlapSize} characters {overlapSize === 0 ? '(original PyG)' : '(enhanced)'}</div>
                    <div>• Best for: {overlapSize === 0 ? 'PyG compatibility' : 'Enhanced context preservation'}</div>
                  </>
                ) : (
                  <>
                    <div>• Method: Optimized for modern LLMs</div>
                    <div>• Estimated chunks for 64KB document: ~{Math.ceil(64000 / chunkSize)}</div>
                    <div>• GPU memory per chunk: ~{Math.round(chunkSize / 1000)}MB</div>
                    <div>• Overlap: {overlapSize} characters</div>
                    <div>• Processing efficiency: {chunkSize >= 32000 ? 'Optimal' : chunkSize >= 16000 ? 'Good' : 'Basic'}</div>
                  </>
                )}
              </div>
            </div>
          </div>
        </AdvancedOptions>
      </div>
      
      {/* Advanced Options with Prompt Configuration */}
      <div className="mt-4">
        <AdvancedOptions title="Prompt Configuration">
          <PromptConfiguration 
            onChange={handlePromptConfigsChange}
            initialConfigs={promptConfigs || undefined}
            langChainMethod={langChainMethod}
            useLangChain={useLangChain}
          />
        </AdvancedOptions>
      </div>
      
      {error && (
        <div className="bg-destructive/10 border border-destructive rounded-md p-3 flex items-start gap-2 mt-4">
          <AlertCircle className="h-4 w-4 text-destructive mt-0.5 flex-shrink-0" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      <div className="border rounded-md overflow-hidden mt-4">
        <div className="bg-muted/30 p-3 flex items-center justify-between">
          <div className="flex items-center">
            <input
              type="checkbox"
              className="rounded border-border text-primary focus:ring-primary mr-3 h-4 w-4"
              checked={selectedDocs.length === documents.filter(doc => (doc.status === "New" || doc.status === "Processed" || doc.status === "Error")).length && 
                      documents.filter(doc => (doc.status === "New" || doc.status === "Processed" || doc.status === "Error")).length > 0}
              onChange={handleSelectAll}
              disabled={documents.filter(doc => (doc.status === "New" || doc.status === "Processed" || doc.status === "Error")).length === 0 || isProcessing}
            />
            <span className="text-sm font-medium">
              {selectedDocs.length > 0 ? (
                <span className="text-primary">{selectedDocs.length} selected</span>
              ) : (
                "Select all"
              )}
            </span>
          </div>
          
          <div className="flex gap-3 pt-4">
            <Button
              size="default"
              onClick={handleExtractTriples}
              disabled={selectedDocs.length === 0 || isProcessing}
              className="bg-nvidia-green hover:bg-nvidia-green/90 text-white font-medium px-6 py-2 gap-2"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Zap className="h-4 w-4" />
                  <span>Extract Triples</span>
                </>
              )}
            </Button>
            
            {isProcessing && (
              <Button
                size="default"
                variant="destructive"
                onClick={handleStopProcessing}
                className="px-4 py-2 gap-2"
              >
                <X className="h-4 w-4" />
                <span>Stop</span>
              </Button>
            )}
          </div>
        </div>

        <div className="max-h-[200px] overflow-y-auto">
          {documents.length === 0 ? (
            <div className="p-6 text-center">
              <p className="text-muted-foreground">No documents available for processing</p>
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="w-12 py-2 px-3 text-left"></th>
                  <th 
                    className="py-2 px-3 text-left cursor-pointer hover:bg-muted/30 transition-colors"
                    onClick={() => handleSort('name')}
                  >
                    <div className="flex items-center gap-1">
                      <span>Document</span>
                      {sortField === 'name' && (
                        sortDirection === 'asc' ? 
                          <ChevronUp className="h-3 w-3" /> : 
                          <ChevronDown className="h-3 w-3" />
                      )}
                    </div>
                  </th>
                  <th 
                    className="py-2 px-3 text-left cursor-pointer hover:bg-muted/30 transition-colors"
                    onClick={() => handleSort('size')}
                  >
                    <div className="flex items-center gap-1">
                      <span>Size</span>
                      {sortField === 'size' && (
                        sortDirection === 'asc' ? 
                          <ChevronUp className="h-3 w-3" /> : 
                          <ChevronDown className="h-3 w-3" />
                      )}
                    </div>
                  </th>
                  <th 
                    className="py-2 px-3 text-left cursor-pointer hover:bg-muted/30 transition-colors"
                    onClick={() => handleSort('status')}
                  >
                    <div className="flex items-center gap-1">
                      <span>Status</span>
                      {sortField === 'status' && (
                        sortDirection === 'asc' ? 
                          <ChevronUp className="h-3 w-3" /> : 
                          <ChevronDown className="h-3 w-3" />
                      )}
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedDocuments.map((doc) => (
                  <tr 
                    key={doc.id} 
                    className={`border-b last:border-b-0 hover:bg-muted/20 ${(doc.status === "New" || doc.status === "Processed" || doc.status === "Error") && !isProcessing ? "cursor-pointer" : ""}`}
                    onClick={(e) => (doc.status === "New" || doc.status === "Processed" || doc.status === "Error") && !isProcessing && handleItemClick(doc, e)}
                  >
                    <td className="pl-3 py-3" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        className="rounded border-border text-primary focus:ring-primary h-4 w-4"
                        checked={isSelected(doc.id)}
                        onChange={(e) => {
                          e.stopPropagation();
                          handleItemClick(doc, e);
                        }}
                        disabled={(doc.status !== "New" && doc.status !== "Processed" && doc.status !== "Error") || isProcessing}
                      />
                    </td>
                    <td className="px-3 py-3 font-medium text-foreground">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-muted-foreground" />
                        <span>{doc.name}</span>
                      </div>
                    </td>
                    <td className="px-3 py-3">
                      {doc.status === "New" && (
                        <span className="inline-flex items-center">
                          <span className="h-2 w-2 rounded-full bg-cyan-400 mr-2"></span>
                          <span>{doc.status}</span>
                        </span>
                      )}
                      {doc.status === "Processing" && (
                        <span className="inline-flex items-center">
                          <Loader2 className="h-4 w-4 text-yellow-500 mr-2 animate-spin" />
                          <span>{doc.status}</span>
                        </span>
                      )}
                      {doc.status === "Processed" && (
                        <span className="inline-flex items-center">
                          <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                          <span>{doc.status}</span>
                        </span>
                      )}
                      {doc.status === "Error" && (
                        <span className="inline-flex items-center">
                          <AlertCircle className="h-4 w-4 text-destructive mr-2" />
                          <span>{doc.status}</span>
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-3">{doc.size} KB</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {isProcessing && status && (
        <div className="border rounded-md p-4 bg-primary/5 mt-4">
          <div className="flex items-center gap-2 text-sm">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            <span>{status}</span>
          </div>
          <div className="mt-2 h-1 w-full bg-muted overflow-hidden rounded-full">
            <div className="h-full bg-primary rounded-full animate-progress"></div>
          </div>
        </div>
      )}
    </>
  )
}

function InfoIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4" />
      <path d="M12 8h.01" />
    </svg>
  )
} 