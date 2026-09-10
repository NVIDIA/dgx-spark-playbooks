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
import { ChevronDown, Cpu, Server, RefreshCw } from "lucide-react"
import { OllamaIcon } from "@/components/ui/ollama-icon"

interface LLMModel {
  id: string
  name: string
  model: string
  provider: string
  description?: string
}

// NVIDIA API models (always available if API key is set)
const NVIDIA_MODELS: LLMModel[] = [
  {
    id: "nvidia-nemotron-super",
    name: "Nemotron Super 49B",
    model: "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    provider: "nvidia",
    description: "NVIDIA API (requires key)"
  },
  {
    id: "nvidia-nemotron-nano",
    name: "Nemotron Nano 9B v2",
    model: "nvidia/nvidia-nemotron-nano-9b-v2",
    provider: "nvidia",
    description: "NVIDIA API - Fast & efficient"
  },
]

export function LLMSelectorCompact() {
  const [models, setModels] = useState<LLMModel[]>([])
  const [selectedModel, setSelectedModel] = useState<LLMModel | null>(null)
  const [isOpen, setIsOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  // Fetch available models from running backends
  const fetchAvailableModels = async () => {
    setIsLoading(true)
    const availableModels: LLMModel[] = []

    // Check vLLM first (port 8001)
    try {
      const vllmResponse = await fetch('/api/vllm/models', { 
        signal: AbortSignal.timeout(3000) 
      })
      if (vllmResponse.ok) {
        const data = await vllmResponse.json()
        if (data.models && Array.isArray(data.models)) {
          data.models.forEach((model: any) => {
            const modelId = model.id || model.name || model
            availableModels.push({
              id: `vllm-${modelId}`,
              name: modelId.split('/').pop() || modelId,
              model: modelId,
              provider: "vllm",
              description: "vLLM (GPU-accelerated)"
            })
          })
        }
      }
    } catch (e) {
      // vLLM not available
      console.log("vLLM not available")
    }

    // Check Ollama (port 11434)
    try {
      const ollamaResponse = await fetch('/api/ollama/tags', { 
        signal: AbortSignal.timeout(3000) 
      })
      if (ollamaResponse.ok) {
        const data = await ollamaResponse.json()
        if (data.models && Array.isArray(data.models)) {
          data.models.forEach((model: any) => {
            const modelName = model.name || model
            availableModels.push({
              id: `ollama-${modelName}`,
              name: modelName,
              model: modelName,
              provider: "ollama",
              description: "Local Ollama model"
            })
          })
        }
      }
    } catch (e) {
      // Ollama not available
      console.log("Ollama not available")
    }

    // Always add NVIDIA API models
    availableModels.push(...NVIDIA_MODELS)

    setModels(availableModels)
    
    // Set default selected model
    if (availableModels.length > 0) {
      // Try to restore saved selection
      try {
        const saved = localStorage.getItem("selectedModelForRAG")
        if (saved) {
          const savedModel: LLMModel = JSON.parse(saved)
          const found = availableModels.find(m => m.id === savedModel.id)
          if (found) {
            setSelectedModel(found)
            setIsLoading(false)
            return
          }
        }
      } catch (e) {
        // Ignore
      }
      
      // Default to first available local model (vLLM or Ollama), not NVIDIA API
      const localModel = availableModels.find(m => m.provider === "vllm" || m.provider === "ollama")
      setSelectedModel(localModel || availableModels[0])
    }
    
    setIsLoading(false)
  }

  // Fetch models on mount
  useEffect(() => {
    fetchAvailableModels()
  }, [])

  // Save selected model to localStorage and dispatch event
  const handleSelectModel = (model: LLMModel) => {
    setSelectedModel(model)
    setIsOpen(false)
    localStorage.setItem("selectedModelForRAG", JSON.stringify(model))
    
    // Dispatch event for other components
    window.dispatchEvent(new CustomEvent('ragModelSelected', {
      detail: { model }
    }))
  }

  const getModelIcon = (provider: string) => {
    if (provider === "ollama") {
      return <OllamaIcon className="h-3 w-3 text-orange-500" />
    }
    if (provider === "vllm") {
      return <Server className="h-3 w-3 text-purple-500" />
    }
    return <Cpu className="h-3 w-3 text-green-500" />
  }

  const getProviderLabel = (provider: string) => {
    switch (provider) {
      case "ollama": return "Ollama"
      case "vllm": return "vLLM"
      case "nvidia": return "NVIDIA API"
      default: return provider
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 text-sm border border-border/40 rounded-lg bg-background/50">
        <RefreshCw className="h-3 w-3 animate-spin text-muted-foreground" />
        <span className="text-muted-foreground">Loading models...</span>
      </div>
    )
  }

  if (!selectedModel) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 text-sm border border-border/40 rounded-lg bg-background/50 text-muted-foreground">
        No models available
      </div>
    )
  }

  // Group models by provider
  const groupedModels = models.reduce((acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = []
    }
    acc[model.provider].push(model)
    return acc
  }, {} as Record<string, LLMModel[]>)

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-label={`Select LLM model. Currently selected: ${selectedModel.name}`}
        className="flex items-center gap-2 px-3 py-1.5 text-sm border border-border/40 rounded-lg bg-background/50 hover:bg-muted/30 transition-colors"
      >
        {getModelIcon(selectedModel.provider)}
        <span className="font-medium">{selectedModel.name}</span>
        <ChevronDown className={`h-3 w-3 text-muted-foreground transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />
          
          {/* Dropdown */}
          <div 
            className="absolute top-full left-0 mt-2 w-72 border border-border/40 rounded-lg bg-popover shadow-lg z-50 overflow-hidden"
            role="listbox"
            aria-label="Available LLM models"
          >
            <div className="p-2 border-b border-border/40 bg-muted/30 flex items-center justify-between">
              <h4 className="text-xs font-semibold text-foreground">Select LLM for Answer Generation</h4>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  fetchAvailableModels()
                }}
                className="p-1 hover:bg-muted/50 rounded"
                title="Refresh models"
              >
                <RefreshCw className="h-3 w-3 text-muted-foreground" />
              </button>
            </div>
            <div className="max-h-80 overflow-y-auto">
              {Object.entries(groupedModels).map(([provider, providerModels]) => (
                <div key={provider}>
                  <div className="px-3 py-1.5 text-xs font-semibold text-muted-foreground bg-muted/20 border-b border-border/20">
                    {getProviderLabel(provider)}
                  </div>
                  {providerModels.map((model) => (
                    <button
                      key={model.id}
                      type="button"
                      role="option"
                      aria-selected={selectedModel.id === model.id}
                      onClick={() => handleSelectModel(model)}
                      className={`w-full flex items-start gap-2 p-3 hover:bg-muted/50 transition-colors text-left ${
                        selectedModel.id === model.id ? 'bg-nvidia-green/10' : ''
                      }`}
                    >
                      <div className="mt-0.5">
                        {getModelIcon(model.provider)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-foreground truncate">
                          {model.name}
                        </div>
                        {model.description && (
                          <div className="text-xs text-muted-foreground">
                            {model.description}
                          </div>
                        )}
                      </div>
                      {selectedModel.id === model.id && (
                        <div className="w-2 h-2 rounded-full bg-nvidia-green flex-shrink-0 mt-1.5" />
                      )}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
