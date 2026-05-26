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
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, Server, CheckCircle, XCircle, RefreshCw } from "lucide-react"
import { OllamaIcon } from "@/components/ui/ollama-icon"

interface OllamaConnectionProps {
  onConnectionChange?: (connected: boolean, models?: string[]) => void
}

export function OllamaConnection({ onConnectionChange }: OllamaConnectionProps) {
  const [baseUrl, setBaseUrl] = useState('http://localhost:11434')
  const [isConnecting, setIsConnecting] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState('llama3.2')
  const [error, setError] = useState<string | null>(null)
  const [lastChecked, setLastChecked] = useState<Date | null>(null)

  // Load settings from localStorage
  useEffect(() => {
    const savedUrl = localStorage.getItem('ollama_base_url')
    const savedModel = localStorage.getItem('ollama_model')
    
    if (savedUrl) setBaseUrl(savedUrl)
    if (savedModel) setSelectedModel(savedModel)
    
    // Auto-test connection on load
    testConnection()
  }, [])

  // Save settings to localStorage
  useEffect(() => {
    localStorage.setItem('ollama_base_url', baseUrl)
    localStorage.setItem('ollama_model', selectedModel)
  }, [baseUrl, selectedModel])

  const testConnection = async () => {
    setIsConnecting(true)
    setError(null)

    try {
      const response = await fetch('/api/ollama?action=test-connection')
      const result = await response.json()

      if (result.connected) {
        setIsConnected(true)
        setAvailableModels(result.models || [])
        setLastChecked(new Date())
        onConnectionChange?.(true, result.models)
      } else {
        setIsConnected(false)
        setError(result.error || 'Connection failed')
        onConnectionChange?.(false)
      }
    } catch (err) {
      setIsConnected(false)
      setError(err instanceof Error ? err.message : 'Connection test failed')
      onConnectionChange?.(false)
    } finally {
      setIsConnecting(false)
    }
  }

  const formatUrl = (url: string) => {
    // Ensure the URL ends with the correct path for Ollama API
    let formatted = url.trim()
    if (!formatted.startsWith('http://') && !formatted.startsWith('https://')) {
      formatted = 'http://' + formatted
    }
    return formatted
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <OllamaIcon className="h-5 w-5 text-orange-500" />
          Ollama Connection
        </CardTitle>
        <CardDescription>
          Connect to your local Ollama server for offline LLM processing
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="ollama-url">Ollama Server URL</Label>
          <div className="flex gap-2">
            <Input
              id="ollama-url"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="http://localhost:11434"
              className="flex-1"
            />
            <Button 
              onClick={testConnection} 
              disabled={isConnecting}
              variant="outline"
            >
              {isConnecting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              Test
            </Button>
          </div>
        </div>

        {/* Connection Status */}
        <div className="flex items-center gap-2">
          <Label>Status:</Label>
          {isConnected ? (
            <Badge variant="default" className="bg-green-500">
              <CheckCircle className="h-3 w-3 mr-1" />
              Connected
            </Badge>
          ) : (
            <Badge variant="destructive">
              <XCircle className="h-3 w-3 mr-1" />
              Disconnected
            </Badge>
          )}
          {lastChecked && (
            <span className="text-xs text-muted-foreground">
              Last checked: {lastChecked.toLocaleTimeString()}
            </span>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <XCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Available Models */}
        {isConnected && availableModels.length > 0 && (
          <div className="space-y-2">
            <Label>Available Models ({availableModels.length})</Label>
            <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto">
              {availableModels.map((model) => (
                <Badge
                  key={model}
                  variant={model === selectedModel ? "default" : "outline"}
                  className="cursor-pointer justify-start"
                  onClick={() => setSelectedModel(model)}
                >
                  {model}
                </Badge>
              ))}
            </div>
            <div className="text-xs text-muted-foreground">
              Selected model: <strong>{selectedModel}</strong>
            </div>
          </div>
        )}

        {/* Instructions */}
        {!isConnected && (
          <Alert>
            <OllamaIcon className="h-4 w-4" />
            <AlertDescription>
              Make sure Ollama is installed and running. Visit{" "}
              <a 
                href="https://ollama.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="underline"
              >
                ollama.com
              </a>{" "}
              for installation instructions.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  )
}
