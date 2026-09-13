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

import { useEffect, useState } from "react"
import type { Triple } from "@/utils/text-processing"
import { GraphVisualization } from "@/components/graph-visualization"
import { NvidiaIcon } from "@/components/nvidia-icon"
import { ArrowLeft, AlertCircle, Network } from "lucide-react"
import { GraphDataForm } from "@/components/graph-data-form"
import { useRouter } from "next/navigation"

export default function GraphPage() {
  const router = useRouter()
  const [triples, setTriples] = useState<Triple[]>([])
  const [documentName, setDocumentName] = useState<string>("")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [dataSource, setDataSource] = useState<"neo4j" | "api" | "local" | "none">("none")

  useEffect(() => {
    const loadGraphData = async () => {
      try {
        setLoading(true)
        setError(null)

        // Check URL parameters
        const params = new URLSearchParams(window.location.search)
        const graphId = params.get("id")
        const source = params.get("source")
        
        console.log("Loading graph with params:", { graphId, source, url: window.location.href })

        // First try to load from Neo4j if available
        if (source !== "local") {
          try {
            console.log("Attempting to load graph data from Neo4j")
            const neo4jResponse = await fetch('/api/neo4j')
            
            if (neo4jResponse.ok) {
              const neo4jData = await neo4jResponse.json()
              if (neo4jData.triples && Array.isArray(neo4jData.triples) && neo4jData.triples.length > 0) {
                console.log("Successfully loaded graph data from Neo4j")
                setTriples(neo4jData.triples)
                setDocumentName(neo4jData.documentName || "Neo4j Graph")
                setDataSource("neo4j")
                setLoading(false)
                return
              } else {
                console.warn("Neo4j returned empty or invalid data, trying other methods")
              }
            } else {
              console.warn(`Neo4j returned status ${neo4jResponse.status}, trying other methods`)
            }
          } catch (neo4jError) {
            console.error("Error accessing Neo4j:", neo4jError)
            console.log("Continuing with other data sources")
          }
        }

        // If source=local is specified, use localStorage directly
        if (source === "local") {
          console.log("Using localStorage as specified in URL")
          return loadFromLocalStorage()
        }

        // If we have a graph ID, try to load from API
        if (graphId) {
          try {
            console.log("Loading graph data from API with ID:", graphId)
            console.log("Fetching from /api/graph-data?id=" + graphId)
            const response = await fetch(`/api/graph-data?id=${graphId}`)

            if (response.ok) {
              const data = await response.json()
              console.log("Successfully loaded graph data from API")
              setTriples(data.triples)
              setDocumentName(data.documentName)
              setDataSource("api")
              return
            } else {
              // Check if this is our special response indicating localStorage should be used
              let errorData;
              try {
                errorData = await response.json();
                console.log("Error response data:", errorData);
                
                if (errorData.useLocalStorage) {
                  console.log("Server indicated to use localStorage fallback")
                  loadFromLocalStorage();
                  return;
                }
              } catch (jsonError) {
                // Response wasn't JSON, try to get text
                const errorText = await response.text();
                console.error(`Failed to load graph data from API (${response.status}):`, errorText)
              }
              
              // Always try localStorage as a fallback
              console.log("Attempting to fall back to localStorage")
              try {
                loadFromLocalStorage()
                return
              } catch (localStorageError) {
                // If both API and localStorage fail, throw a more descriptive error
                throw new Error(`Graph data not found (ID: ${graphId}). The data may have been lost due to server restart or session expiration.`)
              }
            }
          } catch (apiError) {
            console.error("Error accessing API:", apiError)
            // Fall back to localStorage
            console.log("Falling back to localStorage due to API error")
            try {
              loadFromLocalStorage()
              return
            } catch (localStorageError) {
              throw new Error(`Could not load graph data: ${apiError}. Local storage fallback also failed.`)
            }
          }
        } else {
          // No graph ID, try localStorage
          console.log("No graph ID found, trying localStorage")
          loadFromLocalStorage()
          return
        }
      } catch (error) {
        console.error("Error loading graph data:", error)
        setError(error instanceof Error ? error.message : "Unknown error loading graph data")
        setTriples([])
      } finally {
        setLoading(false)
      }
    }

    const loadFromLocalStorage = () => {
      try {
        // Check if localStorage is available
        if (typeof window === 'undefined' || !window.localStorage) {
          throw new Error("LocalStorage is not available in this browser")
        }
        
        // Check URL parameters
        const params = new URLSearchParams(window.location.search)
        const timestamp = params.get("ts")
        
        // Try timestamped version first if timestamp is provided
        let storedTriples = null
        let storedDocName = null
        
        if (timestamp) {
          console.log(`Looking for timestamped data with ts=${timestamp}`)
          storedTriples = localStorage.getItem(`graphTriples_${timestamp}`)
          storedDocName = localStorage.getItem(`graphDocumentName_${timestamp}`)
        }
        
        // Fall back to non-timestamped version if timestamped version not found
        if (!storedTriples) {
          console.log("Timestamped data not found or no timestamp provided, falling back to default keys")
          storedTriples = localStorage.getItem("graphTriples")
          storedDocName = localStorage.getItem("graphDocumentName")
        }

        if (!storedTriples) {
          console.warn("No triples data found in localStorage")
          setTriples([])
          setError("No graph data found in localStorage. Please return to the application and create a new graph.")
          return
        }

        try {
          const parsedTriples = JSON.parse(storedTriples)
          
          if (!Array.isArray(parsedTriples)) {
            setTriples([])
            throw new Error("Invalid graph data format in localStorage")
          }
          
          console.log(`Successfully parsed triples from localStorage: ${parsedTriples.length} items`)
          setTriples(parsedTriples)
          setDataSource("local")

          if (storedDocName) {
            setDocumentName(storedDocName)
          } else {
            setDocumentName("Unnamed Document")
          }
        } catch (parseError) {
          console.error("Error parsing JSON from localStorage:", parseError)
          setTriples([])
          throw new Error("The stored graph data appears to be corrupted. Please return to the application and create a new graph.")
        }
      } catch (localStorageError) {
        console.error("Error loading from localStorage:", localStorageError)
        setTriples([])
        throw localStorageError instanceof Error 
          ? localStorageError 
          : new Error("Failed to load graph data from localStorage")
      }
    }

    loadGraphData().catch((err) => {
      console.error("Unhandled error in loadGraphData:", err)
      setError(err instanceof Error ? err.message : "Unknown error loading graph data")
      setLoading(false)
    })
  }, [])

  const handleBackClick = () => {
    router.push("/")
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center">
          <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center mb-4 animate-pulse">
            <Network className="h-8 w-8 text-primary" />
          </div>
          <p className="text-primary">Loading graph data...</p>
        </div>
      </div>
    )
  }

  if (error || triples.length === 0) {
    return (
      <div className="min-h-screen bg-background">
        <header className="border-b border-border/50 backdrop-blur-md bg-background/80 sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <NvidiaIcon className="h-8 w-8" />
              <div>
                <span className="text-xl font-bold gradient-text">txt2kg</span>
                <span className="ml-2 text-xs bg-primary/20 text-primary px-2 py-0.5 rounded-full">
                  Knowledge Graph Visualization
                </span>
              </div>
            </div>
            <button onClick={handleBackClick} className="btn-outline">
              <ArrowLeft className="h-4 w-4" />
              Back
            </button>
          </div>
        </header>

        <main className="container mx-auto px-4 py-8">
          <div className="glass-card rounded-xl p-6 mb-8 border border-destructive/30">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-full bg-destructive/20 flex items-center justify-center flex-shrink-0">
                <AlertCircle className="h-6 w-6 text-destructive" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-destructive mb-2">Error Loading Graph Data</h3>
                <p className="text-foreground mb-2">{error || "No graph data found"}</p>
                <p className="text-muted-foreground text-sm">
                  This could be due to browser storage limitations or a missing graph ID.
                </p>
              </div>
            </div>
          </div>

          <div className="glass-card rounded-xl p-6">
            <h2 className="text-xl font-bold mb-6">Alternative Methods</h2>

            <div className="space-y-8">
              <div>
                <h3 className="text-lg font-medium text-primary mb-3 flex items-center gap-2">
                  <ArrowLeft className="h-5 w-5" />
                  Method 1: Return to Main App
                </h3>
                <p className="text-foreground mb-4">Go back to the main application and try opening the graph again.</p>
                <button onClick={handleBackClick} className="btn-primary inline-flex">
                  Return to Main App
                </button>
              </div>

              <div className="border-t border-border pt-8">
                <h3 className="text-lg font-medium text-primary mb-3 flex items-center gap-2">
                  <Network className="h-5 w-5" />
                  Method 2: Manual Data Input
                </h3>
                <GraphDataForm />
              </div>
            </div>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border/50 backdrop-blur-md bg-background/80 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <NvidiaIcon className="h-8 w-8" />
            <div>
              <span className="text-xl font-bold gradient-text">txt2kg</span>
              <span className="ml-2 text-xs bg-primary/20 text-primary px-2 py-0.5 rounded-full">
                Knowledge Graph Visualization
              </span>
            </div>
          </div>
          <button onClick={handleBackClick} className="btn-outline">
            <ArrowLeft className="h-4 w-4" />
            Back to Application
          </button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Network className="h-6 w-6 text-primary" />
            <span>Knowledge Graph:</span>
            <span className="text-primary">{documentName}</span>
          </h1>

          <div className="text-xs bg-primary/10 text-primary px-2 py-1 rounded-full">
            {dataSource === "neo4j" ? "Data from Neo4j" : 
             dataSource === "api" ? "Data from API" : "Data from Local Storage"}
          </div>
        </div>

        <div className="glass-card rounded-xl overflow-hidden h-[calc(100vh-200px)]">
          <GraphVisualization triples={triples} fullscreen />
        </div>
      </main>
    </div>
  )
}

