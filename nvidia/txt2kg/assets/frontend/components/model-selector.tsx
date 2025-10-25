"use client"

import { useState, useEffect, useRef } from "react"
import { createPortal } from "react-dom"
import { ChevronDown, Sparkles, Cpu, Server } from "lucide-react"
import { OllamaIcon } from "@/components/ui/ollama-icon"

// Base models - NVIDIA NeMo as default (first in list)
const baseModels = [
  {
    id: "nvidia-nemotron",
    name: "NVIDIA Llama 3.3 Nemotron Super 49B",
    icon: <Cpu className="h-4 w-4 text-green-500" />,
    description: "NVIDIA hosted Nemotron Super 49B v1.5 model",
    model: "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    apiKeyName: "NVIDIA_API_KEY",
    baseURL: "https://integrate.api.nvidia.com/v1",
  },
  {
    id: "nvidia-nemotron-nano",
    name: "NVIDIA Nemotron Nano 9B v2",
    icon: <Cpu className="h-4 w-4 text-green-500" />,
    description: "NVIDIA hosted Nemotron Nano 9B v2 - Faster and more efficient",
    model: "nvidia/nvidia-nemotron-nano-9b-v2",
    apiKeyName: "NVIDIA_API_KEY",
    baseURL: "https://integrate.api.nvidia.com/v1",
  },
  // Preset Ollama model
  {
    id: "ollama-llama3.1:8b",
    name: "Ollama llama3.1:8b",
    icon: <OllamaIcon className="h-4 w-4 text-orange-500" />,
    description: "Local Ollama server with llama3.1:8b model",
    model: "llama3.1:8b",
    baseURL: "http://localhost:11434/v1",
    provider: "ollama",
  },
]

// vLLM models removed per user request

// Helper function to create Ollama model objects
const createOllamaModel = (modelName: string) => ({
  id: `ollama-${modelName}`,
  name: `Ollama ${modelName}`,
  icon: <OllamaIcon className="h-4 w-4 text-orange-500" />,
  description: `Local Ollama server with ${modelName} model`,
  model: modelName,
  baseURL: "http://localhost:11434/v1",
  provider: "ollama",
})

export function ModelSelector() {
  const [models, setModels] = useState(() => [...baseModels])
  const [selectedModel, setSelectedModel] = useState(() => {
    // Try to find a default Ollama model first
    const defaultOllama = models.find(m => m.provider === "ollama")
    return defaultOllama || models[0]
  })
  const [isOpen, setIsOpen] = useState(false)
  const buttonRef = useRef<HTMLButtonElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [mounted, setMounted] = useState(false)

  // Load configured Ollama models
  const loadOllamaModels = () => {
    try {
      const selectedOllamaModels = localStorage.getItem("selected_ollama_models")
      if (selectedOllamaModels) {
        const modelNames = JSON.parse(selectedOllamaModels)
        // Filter out models that are already in baseModels to avoid duplicates
        const baseModelNames = baseModels.filter(m => m.provider === "ollama").map(m => m.model)
        const filteredModelNames = modelNames.filter((name: string) => !baseModelNames.includes(name))
        const ollamaModels = filteredModelNames.map(createOllamaModel)
        const newModels = [...baseModels, ...ollamaModels]
        setModels(newModels)
        return newModels
      }
    } catch (error) {
      console.error("Error loading Ollama models:", error)
    }
    // Return base models if no Ollama models configured
    return [...baseModels]
  }

  // Dispatch custom event when model changes
  const updateSelectedModel = (model: any) => {
    setSelectedModel(model)
    
    // Dispatch a custom event with the selected model data
    const event = new CustomEvent('modelSelected', {
      detail: { model }
    })
    window.dispatchEvent(event)
  }

  useEffect(() => {
    // Save selected model to localStorage
    localStorage.setItem("selectedModel", JSON.stringify(selectedModel))
  }, [selectedModel])

  // Initialize models and selected model
  useEffect(() => {
    const loadedModels = loadOllamaModels()
    
    // Try to restore selected model from localStorage
    const savedModel = localStorage.getItem("selectedModel")
    if (savedModel) {
      try {
        const parsed = JSON.parse(savedModel)
        // Find matching model in our current models array
        const matchingModel = loadedModels.find(m => m.id === parsed.id)
        if (matchingModel) {
          updateSelectedModel(matchingModel)
        } else {
          // If saved model not found, use first available model
          updateSelectedModel(loadedModels[0])
        }
      } catch (e) {
        console.error("Error parsing saved model", e)
        updateSelectedModel(loadedModels[0])
      }
    } else {
      // If no model in localStorage, use first available model
      updateSelectedModel(loadedModels[0])
    }
  }, [])

  // Listen for Ollama model updates
  useEffect(() => {
    const handleOllamaUpdate = (event: CustomEvent) => {
      console.log("Ollama models updated, reloading...")
      const newModels = loadOllamaModels()
      
      // Check if current selected model still exists
      const currentModelStillExists = newModels.find(m => m.id === selectedModel.id)
      if (!currentModelStillExists) {
        // Select first available model if current one is no longer available
        updateSelectedModel(newModels[0])
      }
    }

    window.addEventListener('ollama-models-updated', handleOllamaUpdate as EventListener)
    
    return () => {
      window.removeEventListener('ollama-models-updated', handleOllamaUpdate as EventListener)
    }
  }, [selectedModel.id])

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
          className="absolute bg-card border border-border rounded-md shadow-md overflow-hidden max-h-80 overflow-y-auto z-50"
          style={{
            width: "288px",
            bottom: "calc(100% + 4px)",
            left: 0,
          }}
        >
          <ul className="divide-y divide-border/60">
            {models.map((model) => (
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
      )}
    </div>
  )
}

