"use client"

import { useState, useEffect } from "react"
import { ChevronDown, Cpu } from "lucide-react"
import { OllamaIcon } from "@/components/ui/ollama-icon"

interface LLMModel {
  id: string
  name: string
  model: string
  provider: string
  description?: string
}

// Default models
const DEFAULT_MODELS: LLMModel[] = [
  {
    id: "ollama-llama3.1:8b",
    name: "Llama 3.1 8B",
    model: "llama3.1:8b",
    provider: "ollama",
    description: "Local Ollama model"
  },
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
  const [models, setModels] = useState<LLMModel[]>(DEFAULT_MODELS)
  const [selectedModel, setSelectedModel] = useState<LLMModel>(DEFAULT_MODELS[0])
  const [isOpen, setIsOpen] = useState(false)

  // Load Ollama models from settings
  useEffect(() => {
    try {
      const selectedOllamaModels = localStorage.getItem("selected_ollama_models")
      if (selectedOllamaModels) {
        const modelNames: string[] = JSON.parse(selectedOllamaModels)
        const ollamaModels: LLMModel[] = modelNames.map(name => ({
          id: `ollama-${name}`,
          name: name,
          model: name,
          provider: "ollama",
          description: "Local Ollama model"
        }))
        
        // Combine with default models, avoiding duplicates
        const defaultOllamaIds = DEFAULT_MODELS
          .filter(m => m.provider === "ollama")
          .map(m => m.model)
        const uniqueOllamaModels = ollamaModels.filter(
          m => !defaultOllamaIds.includes(m.model)
        )
        
        const allModels = [...DEFAULT_MODELS, ...uniqueOllamaModels]
        setModels(allModels)
      }
    } catch (error) {
      console.error("Error loading Ollama models:", error)
    }
  }, [])

  // Load selected model from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem("selectedModelForRAG")
      if (saved) {
        const savedModel: LLMModel = JSON.parse(saved)
        setSelectedModel(savedModel)
      }
    } catch (error) {
      console.error("Error loading selected model:", error)
    }
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
    return <Cpu className="h-3 w-3 text-green-500" />
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
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
          <div className="absolute top-full left-0 mt-2 w-64 border border-border/40 rounded-lg bg-popover shadow-lg z-50 overflow-hidden">
            <div className="p-2 border-b border-border/40 bg-muted/30">
              <h4 className="text-xs font-semibold text-foreground">Select LLM for Answer Generation</h4>
            </div>
            <div className="max-h-64 overflow-y-auto">
              {models.map((model) => (
                <button
                  key={model.id}
                  type="button"
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
          </div>
        </>
      )}
    </div>
  )
}

