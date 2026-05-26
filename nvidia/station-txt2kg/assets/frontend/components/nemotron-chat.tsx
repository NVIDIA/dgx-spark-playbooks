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
import { langChainService } from "@/lib/langchain-service"
import { HumanMessage, SystemMessage } from "@langchain/core/messages"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Spinner } from "@/components/ui/spinner"
import { ChatOpenAI } from "@langchain/openai"

export function NemotronChat() {
  const [input, setInput] = useState("")
  const [response, setResponse] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [modelReady, setModelReady] = useState(false)
  const [model, setModel] = useState<ChatOpenAI | null>(null)
  
  // Initialize the Nemotron model directly from environment variables
  useEffect(() => {
    const initializeModel = async () => {
      try {
        setError(null)
        // Initialize model using environment variable API key
        const nemotronModel = await langChainService.getNemotronModel({
          temperature: 0.7,
          maxTokens: 1024,
        })
        setModel(nemotronModel)
        setModelReady(true)
      } catch (modelError) {
        console.error("Error initializing model:", modelError)
        setError(`Error initializing model: ${modelError instanceof Error ? modelError.message : String(modelError)}`)
      }
    }
    
    // Initialize model on component mount
    initializeModel()
    
    // Listen for model selection changes
    const handleModelSelection = (event: any) => {
      if (event.detail?.model?.id === 'nvidia-nemotron') {
        // Re-initialize the model if the Nemotron model is selected
        initializeModel()
      }
    }
    
    // Add event listener for model selection changes
    window.addEventListener('modelSelected', handleModelSelection)
    
    // Cleanup event listener
    return () => {
      window.removeEventListener('modelSelected', handleModelSelection)
    }
  }, [])
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!input.trim() || !model) return
    
    setIsLoading(true)
    setError(null)
    
    try {
      console.log("Starting generation with input:", input.substring(0, 50) + "...")
      
      // Create messages
      const messages = [
        new SystemMessage("You are a helpful, concise assistant."),
        new HumanMessage(input)
      ]
      
      console.log("Invoking model with messages")
      
      // Generate response using the cached model
      const result = await model.invoke(messages)
      
      console.log("Generation completed successfully")
      
      // Update the response
      setResponse(result.content.toString())
      
    } catch (err) {
      console.error("Error generating response:", err)
      
      // More detailed error info
      if (err instanceof Error) {
        setError(`Error: ${err.message}\n${err.stack || ""}`)
      } else {
        setError(`Unknown error: ${String(err)}`)
      }
    } finally {
      setIsLoading(false)
    }
  }
  
  return (
    <div className="p-6 bg-card rounded-lg border border-border">
      <h2 className="text-xl font-bold mb-4">Chat with NVIDIA Nemotron</h2>
      
      {error && (
        <div className="bg-destructive/10 text-destructive rounded-md p-3 mb-4">
          {error}
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something..."
          className="min-h-[100px]"
          disabled={isLoading || !modelReady}
        />
        
        <Button 
          type="submit" 
          disabled={isLoading || !input.trim() || !modelReady}
          className="w-full"
        >
          {isLoading ? <Spinner className="mr-2" /> : null}
          {isLoading ? "Generating..." : "Submit"}
        </Button>
      </form>
      
      {response && (
        <div className="mt-6">
          <h3 className="font-medium mb-2">Response:</h3>
          <div className="p-4 bg-muted rounded-md whitespace-pre-wrap">
            {response}
          </div>
        </div>
      )}
    </div>
  )
} 