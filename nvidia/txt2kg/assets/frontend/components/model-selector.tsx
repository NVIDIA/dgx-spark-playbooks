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

import { useState, useEffect, useRef } from "react"
import { ChevronDown, Cpu, Server, RefreshCw } from "lucide-react"
import { OllamaIcon } from "@/components/ui/ollama-icon"

interface Model {
  id: string
  name: string
  icon: React.ReactNode
  description: string
  model: string
  baseURL: string
  provider: string
  apiKeyName?: string
}

// NVIDIA API models (always available)
const NVIDIA_MODELS: Model[] = [
  {
    id: "nvidia-nemotron",
    name: "NVIDIA Llama 3.3 Nemotron Super 49B",
    icon: <Cpu className="h-4 w-4 text-green-500" />,
    description: "NVIDIA hosted Nemotron Super 49B v1.5 model",
    model: "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    apiKeyName: "NVIDIA_API_KEY",
    baseURL: "https://integrate.api.nvidia.com/v1",
    provider: "nvidia",
  },
  {
    id: "nvidia-nemotron-nano",
    name: "NVIDIA Nemotron Nano 9B v2",
    icon: <Cpu className="h-4 w-4 text-green-500" />,
    description: "NVIDIA hosted Nemotron Nano 9B v2 - Faster and more efficient",
    model: "nvidia/nvidia-nemotron-nano-9b-v2",
    apiKeyName: "NVIDIA_API_KEY",
    baseURL: "https://integrate.api.nvidia.com/v1",
    provider: "nvidia",
  },
]

// Helper to create model objects
const createOllamaModel = (modelName: string): Model => ({
  id: `ollama-${modelName}`,
  name: `Ollama ${modelName}`,
  icon: <OllamaIcon className="h-4 w-4 text-orange-500" />,
  description: `Local Ollama model`,
  model: modelName,
  baseURL: "http://localhost:11434/v1",
  provider: "ollama",
})

const createVllmModel = (modelName: string): Model => ({
  id: `vllm-${modelName}`,
  name: modelName.split('/').pop() || modelName,
  icon: <Server className="h-4 w-4 text-purple-500" />,
  description: "vLLM (GPU-accelerated)",
  model: modelName,
  baseURL: "http://localhost:8001/v1",
  provider: "vllm",
})

export function ModelSelector() {
  const [models, setModels] = useState<Model[]>([])
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [isOpen, setIsOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const buttonRef = useRef<HTMLButtonElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [mounted, setMounted] = useState(false)

  // Fetch available models from running backends
  const fetchAvailableModels = async () => {
    setIsLoading(true)
    const availableModels: Model[] = []

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
            availableModels.push(createVllmModel(modelId))
          })
        }
      }
    } catch (e) {
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
            availableModels.push(createOllamaModel(modelName))
          })
        }
      }
    } catch (e) {
      console.log("Ollama not available")
    }

    // Always add NVIDIA API models
    availableModels.push(...NVIDIA_MODELS)

    setModels(availableModels)
    
    // Set default selected model
    if (availableModels.length > 0) {
      // Try to restore saved selection
      try {
        const saved = localStorage.getItem("selectedModel")
        if (saved) {
          const savedModel = JSON.parse(saved)
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
      
      // Default to first available local model (vLLM or Ollama)
      const localModel = availableModels.find(m => m.provider === "vllm" || m.provider === "ollama")
      setSelectedModel(localModel || availableModels[0])
    }
    
    setIsLoading(false)
  }

  // Dispatch custom event when model changes
  const updateSelectedModel = (model: Model) => {
    setSelectedModel(model)
    localStorage.setItem("selectedModel", JSON.stringify(model))
    
    // Dispatch a custom event with the selected model data
    const event = new CustomEvent('modelSelected', {
      detail: { model }
    })
    window.dispatchEvent(event)
  }

  // Fetch models on mount
  useEffect(() => {
    fetchAvailableModels()
  }, [])

  // Set mounted state after component mounts (for SSR compatibility)
  useEffect(() => {
    setMounted(true)
  }, [])

  // Close on outside click and Escape
  useEffect(() => {
    const handleMouseDown = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsOpen(false)
    }
    document.addEventListener('mousedown', handleMouseDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handleMouseDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  // Listen for Ollama model updates
  useEffect(() => {
    const handleOllamaUpdate = () => {
      console.log("Ollama models updated, reloading...")
      fetchAvailableModels()
    }

    window.addEventListener('ollama-models-updated', handleOllamaUpdate)
    
    return () => {
      window.removeEventListener('ollama-models-updated', handleOllamaUpdate)
    }
  }, [])

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 bg-card border border-border rounded-lg px-4 py-2 text-sm">
        <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />
        <span className="text-muted-foreground">Loading models...</span>
      </div>
    )
  }

  if (!selectedModel) {
    return (
      <div className="flex items-center gap-2 bg-card border border-border rounded-lg px-4 py-2 text-sm text-muted-foreground">
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
  }, {} as Record<string, Model[]>)

  const getProviderLabel = (provider: string) => {
    switch (provider) {
      case "ollama": return "Ollama (Local)"
      case "vllm": return "vLLM (GPU-accelerated)"
      case "nvidia": return "NVIDIA API (Cloud)"
      default: return provider
    }
  }

  return (
    <div ref={containerRef} className="relative">
      <button
        ref={buttonRef}
        className="flex items-center gap-2 bg-card border border-border rounded-lg px-4 py-2 text-sm hover:bg-muted/30 transition-colors"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center gap-2">
          {selectedModel.icon}
          <span className="font-medium">{selectedModel.name}</span>
        </div>
        <ChevronDown className="h-4 w-4 text-muted-foreground ml-2" />
      </button>

      {isOpen && mounted && (
        <div 
          className="absolute bg-card border border-border rounded-md shadow-md overflow-hidden max-h-96 overflow-y-auto z-50"
          style={{
            width: "320px",
            bottom: "calc(100% + 4px)",
            left: 0,
          }}
        >
          <div className="px-3 py-2 border-b border-border/60 bg-muted/30 flex items-center justify-between">
            <span className="text-xs font-semibold text-foreground">Select Model</span>
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
          <div>
            {Object.entries(groupedModels).map(([provider, providerModels]) => (
              <div key={provider}>
                <div className="px-3 py-1.5 text-xs font-semibold text-muted-foreground bg-muted/20 border-b border-border/20">
                  {getProviderLabel(provider)}
                </div>
                <ul>
                  {providerModels.map((model) => (
                    <li key={model.id}>
                      <button
                        className={`w-full text-left px-3 py-2 hover:bg-muted/30 text-sm flex flex-col gap-1 ${model.id === selectedModel.id ? 'bg-primary/10' : ''}`}
                        onClick={() => {
                          updateSelectedModel(model)
                          setIsOpen(false)
                        }}
                      >
                        <span className="flex items-center gap-2">
                          {model.icon}
                          <span className={`font-medium ${model.id === selectedModel.id ? 'text-primary' : ''}`}>{model.name}</span>
                        </span>
                        <span className="text-xs text-muted-foreground pl-6">{model.description}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
