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
import React from "react"
import { Database, FileText, Cpu, Sparkles, Zap } from "lucide-react"
import { ModelSelector } from "@/components/model-selector"
import { EmbeddingsGenerator } from "@/components/embeddings-generator"
import { useDocuments } from "@/contexts/document-context"
import { useState, useEffect } from "react"
import { OllamaIcon } from "@/components/ui/ollama-icon"

export function ConfigureTab() {
  // Use state from the parent component
  const [selectedModelInfo, setSelectedModelInfo] = useState({ 
    name: "Ollama Qwen3 1.7B", 
    icon: <OllamaIcon className="h-4 w-4 text-orange-500" />
  })
  const [embeddingModelInfo, setEmbeddingModelInfo] = useState("all-MiniLM-L6-v2")
  const [embeddingsProvider, setEmbeddingsProvider] = useState<string>("local")
  const [nvidiaEmbeddingsModel, setNvidiaEmbeddingsModel] = useState<string>("nvidia/llama-3.2-nv-embedqa-1b-v2")
  
  // Update model info when component mounts and when localStorage changes
  useEffect(() => {
    // Initial load from localStorage
    const updateModelInfo = () => {
      try {
        const savedModel = localStorage.getItem("selectedModel")
        if (savedModel) {
          const model = JSON.parse(savedModel)
          setSelectedModelInfo({
            name: model.name,
            icon: getModelIcon(model.id)
          })
        }
        
        // Load embedding settings
        const provider = localStorage.getItem("embeddings_provider") || "local"
        setEmbeddingsProvider(provider)
        
        // Load NVIDIA model if using NVIDIA
        if (provider === "nvidia") {
          const model = localStorage.getItem("nvidia_embeddings_model") || "nvidia/llama-3.2-nv-embedqa-1b-v2"
          setNvidiaEmbeddingsModel(model)
        }
      } catch (e) {
        console.error("Error loading model info:", e)
      }
    }
    
    // Update on load
    updateModelInfo()
    
    // Set up event listener for storage changes
    window.addEventListener('storage', updateModelInfo)
    
    // Custom event for when model selection changes
    const handleModelChange = (e: CustomEvent) => {
      if (e.detail?.model) {
        setSelectedModelInfo({
          name: e.detail.model.name,
          icon: getModelIcon(e.detail.model.id)
        })
      }
    }
    
    // Listen for LangChain toggle changes
    const handleLangChainToggle = (e: CustomEvent) => {
      if (e.detail?.useLangChain !== undefined) {
        // When LangChain is enabled, use GTE-large model, otherwise use default model
        if (embeddingsProvider === "local") {
          setEmbeddingModelInfo(e.detail.useLangChain ? "Alibaba-NLP/gte-modernbert-base" : "all-MiniLM-L6-v2")
        }
      }
    }
    
    // Listen for embeddings settings changes
    const handleEmbeddingsSettingsChanged = () => {
      const provider = localStorage.getItem("embeddings_provider") || "local"
      setEmbeddingsProvider(provider)
      
      if (provider === "nvidia") {
        const model = localStorage.getItem("nvidia_embeddings_model") || "nvidia/llama-3.2-nv-embedqa-1b-v2"
        setNvidiaEmbeddingsModel(model)
      } else {
        // Local provider - use sentence transformers
        const useLangChain = localStorage.getItem("useLangChain") === "true"
        setEmbeddingModelInfo(useLangChain ? "Alibaba-NLP/gte-modernbert-base" : "all-MiniLM-L6-v2")
      }
    }
    
    // Listen for custom model change events
    window.addEventListener('modelSelected', handleModelChange as EventListener)
    window.addEventListener('langChainToggled', handleLangChainToggle as EventListener)
    window.addEventListener('embeddings-settings-changed', handleEmbeddingsSettingsChanged as EventListener)
    
    return () => {
      window.removeEventListener('storage', updateModelInfo)
      window.removeEventListener('modelSelected', handleModelChange as EventListener)
      window.removeEventListener('langChainToggled', handleLangChainToggle as EventListener)
      window.removeEventListener('embeddings-settings-changed', handleEmbeddingsSettingsChanged as EventListener)
    }
  }, [embeddingsProvider])
  
  // Function to get the appropriate icon based on model ID
  const getModelIcon = (modelId: string) => {
    if (modelId?.startsWith("nvidia-")) {
      return <Cpu className="h-4 w-4 text-green-500" />
    } else if (modelId?.startsWith("ollama-")) {
      return <OllamaIcon className="h-4 w-4 text-orange-500" />
    }
    return <Cpu className="h-4 w-4 text-gray-500" />
  }
  
  // Calculate which model to display
  const displayEmbeddingModel = embeddingsProvider === "nvidia" ? nvidiaEmbeddingsModel : embeddingModelInfo
  const embeddingProviderIcon = embeddingsProvider === "nvidia" ? 
    <Cpu className="h-4 w-4 text-green-500" /> : 
    <Database className="h-4 w-4 text-indigo-500" />
  const embeddingTooltip = embeddingsProvider === "nvidia" ? 
    "Using NVIDIA API embedding model" : 
    "Using sentence-transformers service model"
  
  return (
    <div className="grid md:grid-cols-2 gap-8 lg:gap-12">
      {/* Left column: Model selection */}
      <div className="space-y-8">
        <div className="nvidia-build-card">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-8 h-8 rounded-lg bg-nvidia-green/15 flex items-center justify-center">
              <Cpu className="h-4 w-4 text-nvidia-green" />
            </div>
            <h2 className="text-lg font-semibold text-foreground">Current Configuration</h2>
          </div>
          <div className="space-y-4">
            <div className="bg-muted/10 border border-border/20 p-4 rounded-xl">
              <h3 className="text-sm font-semibold text-foreground mb-3">Selected Model</h3>
              <div className="flex items-center gap-3 text-sm">
                <div className="w-6 h-6 rounded-md bg-muted/30 flex items-center justify-center">
                  {selectedModelInfo.icon}
                </div>
                <span className="text-foreground font-medium">{selectedModelInfo.name}</span>
              </div>
            </div>
            
            <div className="bg-muted/10 border border-border/20 p-4 rounded-xl">
              <h3 className="text-sm font-semibold text-foreground mb-3">Embedding Model</h3>
              <div className="flex items-center gap-3 text-sm">
                <div className="w-6 h-6 rounded-md bg-muted/30 flex items-center justify-center">
                  {embeddingProviderIcon}
                </div>
                <span className="text-foreground font-medium truncate flex-1">{displayEmbeddingModel}</span>
              </div>
            </div>
            
            <ProcessingSummary />
          </div>
        </div>
        
        <div className="nvidia-build-card">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-8 h-8 rounded-lg bg-nvidia-green/15 flex items-center justify-center">
              <Sparkles className="h-4 w-4 text-nvidia-green" />
            </div>
            <h2 className="text-lg font-semibold text-foreground">Select Triple Extraction Model</h2>
          </div>
          
          <div className="space-y-4">
            <ModelSelector />
          </div>
        </div>
      </div>
      
      {/* Right column: Document Processing */}
      <div className="space-y-8">
        <div className="nvidia-build-card">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-8 h-8 rounded-lg bg-nvidia-green/15 flex items-center justify-center">
              <Zap className="h-4 w-4 text-nvidia-green" />
            </div>
            <h2 className="text-lg font-semibold text-foreground">Process Documents</h2>
          </div>
          <EmbeddingsGenerator showTripleExtraction={true} />
        </div>
      </div>
    </div>
  )
}

// Create a wrapper component that uses the document context
function ProcessingSummary() {
  const { documents } = useDocuments()
  
  // Count documents with "New" status that are ready for processing
  const docsReadyCount = documents.filter(doc => doc.status === "New").length
  
  return (
    <div className="bg-muted/10 border border-border/20 p-4 rounded-xl">
      <h3 className="text-sm font-semibold text-foreground mb-3">Documents Ready</h3>
      <div className="flex items-center gap-3 text-sm">
        <div className="w-6 h-6 rounded-md bg-muted/30 flex items-center justify-center">
          <FileText className="h-4 w-4 text-nvidia-green" />
        </div>
        <span className="text-foreground font-medium">
          {docsReadyCount === 1 
            ? '1 document ready for processing' 
            : `${docsReadyCount} documents ready for processing`}
        </span>
      </div>
    </div>
  )
} 