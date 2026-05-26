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

/**
 * 3D Graph Visualization Page
 * 
 * This page provides a 3D visualization of knowledge graphs with multiple data source options:
 * 
 * Usage:
 * 1. Stored triples: /graph3d?source=stored - Uses triples from the graph database (ArangoDB/Neo4j)
 * 2. URL triples: /graph3d?triples=[...] - Uses triples passed directly in URL parameters
 * 3. localStorage: /graph3d?storageId=xyz - Uses triples from browser localStorage
 * 4. Sample data: /graph3d - Uses built-in sample data when no other source is available
 * 
 * Additional parameters:
 * - layout: force|hierarchical|radial - Sets the graph layout type
 * - highlightedNodes: JSON array of node names to highlight
 * 
 * Examples:
 * - /graph3d?source=stored&layout=force
 * - /graph3d?source=local&triples=[{"subject":"A","predicate":"relates_to","object":"B"}]
 */

import { useEffect, useState, useCallback } from "react"
import dynamic from "next/dynamic"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Separator } from "@/components/ui/separator"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Loader2, Cpu, Monitor, Settings, Brain, Layers, Zap, ChevronDown, ChevronRight } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

// Dynamically import the ForceGraphWrapper component with SSR disabled
const ForceGraphWrapper = dynamic(
  () => import("@/components/force-graph-wrapper").then(mod => mod.ForceGraphWrapper),
  { ssr: false }
)

// Dynamically import the WebGPU 3D Viewer component with SSR disabled
const WebGPU3DViewer = dynamic(
  () => import("@/components/webgpu-3d-viewer").then(mod => mod.WebGPU3DViewer),
  { ssr: false }
)

interface PerformanceMetrics {
  renderingTime: number
  clusteringTime?: number
  totalNodes: number
  totalLinks: number
  memoryUsage?: number
}

export default function Graph3DPage() {
  const [graphData, setGraphData] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [debugInfo, setDebugInfo] = useState<string>("")
  const [highlightedNodes, setHighlightedNodes] = useState<string[]>([])
  const [layoutType, setLayoutType] = useState<string>("3d")
  const [useEnhancedWebGPU, setUseEnhancedWebGPU] = useState<boolean>(false)
  const [enableClustering, setEnableClustering] = useState<boolean>(true)
  const [enableClusterColors, setEnableClusterColors] = useState<boolean>(false)
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null)
  const [showClusteringControls, setShowClusteringControls] = useState<boolean>(false)
  const [clusteringOptionsExpanded, setClusteringOptionsExpanded] = useState<boolean>(false)
  
  // Semantic clustering options
  const [clusteringMethod, setClusteringMethod] = useState<string>("hybrid")
  const [semanticAlgorithm, setSemanticAlgorithm] = useState<string>("hierarchical")
  const [numberOfClusters, setNumberOfClusters] = useState<number | null>(null)
  const [similarityThreshold, setSimilarityThreshold] = useState<number>(0.7)
  const [nameWeight, setNameWeight] = useState<number>(0.6)
  const [contentWeight, setContentWeight] = useState<number>(0.3)
  const [spatialWeight, setSpatialWeight] = useState<number>(0.1)
  
  const { toast } = useToast()

  // Handle clustering performance updates
  const handleClusteringUpdate = useCallback((metrics: PerformanceMetrics) => {
    setPerformanceMetrics(metrics)
    if (metrics.clusteringTime && !useEnhancedWebGPU) {
      toast({
        title: "Hybrid GPU/CPU Clustering Complete",
        description: `Server GPU processed ${metrics.totalNodes.toLocaleString()} nodes in ${metrics.clusteringTime.toFixed(0)}ms`,
      })
    }
  }, [toast, useEnhancedWebGPU])

  // Update performance metrics when graph data changes
  useEffect(() => {
    if (graphData) {
      const nodeCount = graphData.nodes?.length || 0
      const linkCount = graphData.links?.length || 0
      const tripleCount = graphData.triples?.length || 0
      
      if (nodeCount > 0 || tripleCount > 0) {
        setPerformanceMetrics({
          renderingTime: 0,
          totalNodes: nodeCount || Math.ceil(tripleCount * 0.6), // Estimate nodes from triples
          totalLinks: linkCount || Math.ceil(tripleCount * 0.8), // Estimate links from triples
        })
      }
    }
  }, [graphData])

  useEffect(() => {
    // Fetch graph data
    const fetchGraphData = async () => {
      try {
        setIsLoading(true)
        
        // Check URL parameters
        const params = new URLSearchParams(window.location.search)
        const graphId = params.get("id")
        const triplesParam = params.get("triples")
        const layoutParam = params.get("layout")
        const highlightedNodesParam = params.get("highlightedNodes")
        const storageId = params.get("storageId")
        const source = params.get("source")
        
        // Set layout type from URL parameter
        if (layoutParam) {
          setLayoutType(layoutParam)
          console.log("Layout type set from URL:", layoutParam)
        }
        
        // Set highlighted nodes from URL parameter
        if (highlightedNodesParam) {
          try {
            const parsedHighlightedNodes = JSON.parse(decodeURIComponent(highlightedNodesParam))
            if (Array.isArray(parsedHighlightedNodes)) {
              setHighlightedNodes(parsedHighlightedNodes)
              console.log("Highlighted nodes set from URL:", parsedHighlightedNodes)
            }
          } catch (parseError) {
            console.error("Failed to parse highlightedNodes from URL:", parseError)
          }
        }
        
        console.log("URL parameters:", { 
          graphId: graphId || "not provided", 
          hasTriples: !!triplesParam,
          hasStorageId: !!storageId,
          layout: layoutParam || "default",
          highlightedNodes: highlightedNodesParam ? "provided" : "not provided",
          source: source || "auto",
          allParams: Object.fromEntries(params.entries())
        });
        
        // Try to load from localStorage if storageId is provided
        if (storageId) {
          try {
            console.log("Found storageId in URL, attempting to retrieve data from localStorage:", storageId);
            const storedData = localStorage.getItem(storageId);
            
            if (!storedData) {
              console.error("No data found in localStorage for storageId:", storageId);
              setError("Could not find the graph data in your browser storage. It may have expired.");
              setIsLoading(false);
              return;
            }
            
            const triples = JSON.parse(storedData);
            console.log("Successfully retrieved triples from localStorage:", { 
              count: triples.length,
              sample: triples.slice(0, 2)
            });
            
            setGraphData({ triples });
            // setDebugInfo(`Using ${triples.length} triples from browser storage (ID: ${storageId})`);
            setIsLoading(false);
            
            // Clean up localStorage after retrieval to prevent buildup
            // Only do this for older IDs to prevent issues with multiple tabs/windows
            const currentTime = Date.now();
            const idTimestamp = parseInt(storageId.split('_')[1] || '0', 10);
            
            // If the ID is older than 5 minutes, clean it up
            if (currentTime - idTimestamp > 5 * 60 * 1000) {
              console.log("Cleaning up old localStorage entry:", storageId);
              localStorage.removeItem(storageId);
            }
            
            return;
          } catch (storageError) {
            console.error("Error retrieving data from localStorage:", storageError);
            setDebugInfo("Failed to retrieve triples from browser storage, falling back to API");
            // Continue to other methods if parsing fails
          }
        }
        
        // If we have triples passed directly in the URL param
        if (triplesParam) {
          try {
            console.log("Found triples data in URL parameter, attempting to parse")
            const triples = JSON.parse(decodeURIComponent(triplesParam))
            console.log("Successfully parsed triples from URL:", { 
              count: triples.length,
              sample: triples.slice(0, 2)
            });
            setGraphData({ triples })
            setDebugInfo("Using triples data from URL parameter")
            setIsLoading(false)
            return
          } catch (parseError) {
            console.error("Error parsing triples from URL:", parseError)
            setDebugInfo("Failed to parse triples from URL, falling back to API")
            // Continue to other methods if parsing fails
          }
        }
        
        // Determine data source based on URL parameters
        let endpoint: string;
        let useStoredTriples = false;
        
        if (graphId) {
          endpoint = `/api/graph-data?id=${graphId}`;
        } else if (source === 'stored' || (!triplesParam && !storageId)) {
          // Use stored triples if explicitly requested or if no other data source is available
          endpoint = '/api/graph-db/triples';
          useStoredTriples = true;
        } else {
          // Fall back to sample data
          endpoint = '/api/graph-data';
        }
        
        console.log(`Fetching graph data from API: ${endpoint}`);
        setDebugInfo(`Fetching from ${endpoint}`)
        
        const response = await fetch(endpoint)
        
        if (!response.ok) {
          console.error(`API responded with status ${response.status}: ${response.statusText}`)
          
          // If we were trying to fetch stored triples and it failed, fall back to sample data
          if (useStoredTriples) {
            console.log("Stored triples failed, falling back to sample graph data");
            setDebugInfo("No stored triples available, using sample data");
            const fallbackResponse = await fetch('/api/graph-data');
            if (fallbackResponse.ok) {
              const fallbackData = await fallbackResponse.json();
              setGraphData(fallbackData);
              setIsLoading(false);
              return;
            }
          }
          
          setDebugInfo(`API error: ${response.status} ${response.statusText}`)
          throw new Error(`Error fetching graph data: ${response.statusText}`)
        }
        
        const data = await response.json()
        console.log("API response received:", {
          dataExists: !!data,
          hasNodes: data && Array.isArray(data.nodes),
          hasLinks: data && Array.isArray(data.links),
          hasTriples: data && Array.isArray(data.triples),
          nodeCount: data && Array.isArray(data.nodes) ? data.nodes.length : 0,
          linkCount: data && Array.isArray(data.links) ? data.links.length : 0,
          tripleCount: data && Array.isArray(data.triples) ? data.triples.length : 0,
          dataType: typeof data,
          keys: data ? Object.keys(data) : [],
          rawData: JSON.stringify(data).substring(0, 200) + "..."
        });
        
        // Validate the data structure - can be either nodes/links or triples format
        if (!data) {
          setDebugInfo("API returned empty data")
          throw new Error('No data received from API');
        }
        
        // Handle stored triples response format
        if (useStoredTriples && data.triples && Array.isArray(data.triples)) {
          console.log("Processing stored triples from graph database");
          setGraphData({ triples: data.triples });
          setDebugInfo(`Using ${data.triples.length} stored triples from ${data.databaseType || 'graph database'}`);
          setIsLoading(false);
          return;
        }
        
        if ((!Array.isArray(data.nodes) || !Array.isArray(data.links)) && 
            !Array.isArray(data.triples)) {
          setDebugInfo(`Invalid data format: ${Object.keys(data).join(", ")}`)
          throw new Error('Invalid graph data structure: missing required data arrays');
        }
        
        if (Array.isArray(data.triples)) {
          setDebugInfo(`Using triples data (${data.triples.length} triples) from API`)
        } else if (Array.isArray(data.nodes) && Array.isArray(data.links)) {
          setDebugInfo(`Using nodes/links data (${data.nodes.length} nodes, ${data.links.length} links) from API`)
        }
        
        console.log("Setting graph data in state...");
        setGraphData(data)
        setIsLoading(false)
      } catch (err) {
        console.error('Failed to load graph data:', err)
        setError(`Failed to load graph data: ${err instanceof Error ? err.message : String(err)}`)
        setIsLoading(false)
      }
    }

    fetchGraphData()
  }, [])

  useEffect(() => {
    // Add overflow: hidden to the body element when the component mounts
    document.body.style.overflow = "hidden"

    // Clean up the effect when the component unmounts
    return () => {
      document.body.style.overflow = "auto"
    }
  }, [])

  // Display error or loading state
  if (error || isLoading) {
    return (
      <div className="h-screen w-screen overflow-hidden bg-black flex items-center justify-center">
        {isLoading && (
          <div className="text-center">
            <p className="mb-4 text-white">Loading graph data...</p>
            <div className="w-16 h-16 border-4 border-gray-700 border-t-green-500 rounded-full animate-spin mx-auto"></div>
            {debugInfo && (
              <p className="mt-4 text-xs text-gray-400">{debugInfo}</p>
            )}
          </div>
        )}
        
        {error && (
          <div className="bg-black/80 border border-red-900 text-red-400 px-6 py-4 rounded-lg max-w-lg">
            <p>{error}</p>
            {debugInfo && (
              <p className="mt-2 text-xs text-gray-500">{debugInfo}</p>
            )}
            <div className="mt-4 flex gap-2">
              <button 
                onClick={() => window.location.reload()} 
                className="bg-red-900/50 hover:bg-red-900 text-white py-1 px-3 rounded text-sm"
              >
                Retry
              </button>
              <button 
                onClick={() => window.location.href = '/'}
                className="bg-gray-800 hover:bg-gray-700 text-white py-1 px-3 rounded text-sm"
              >
                Return to Home
              </button>
            </div>
          </div>
        )}
      </div>
    )
  }

  // Only render the graph when data is ready
  return (
    <div className="h-screen w-screen overflow-hidden">
      {graphData && (
        <>
          {/* Controls Panel */}
          <div className="absolute top-20 left-2 z-50 flex flex-col gap-2 max-w-sm">
            {/* Main Controls Row */}
            <div className="flex items-center gap-4">
              {/* <div className="text-xs text-gray-500">
                {graphData.nodes && graphData.links ? (
                  `Rendering graph with ${graphData.nodes.length || 0} nodes and ${graphData.links.length || 0} links`
                ) : graphData.triples ? (
                  `Rendering graph from ${graphData.triples.length || 0} triples`
                ) : (
                  "Rendering graph data"
                )}
              </div> */}
              
              {/* WebGPU Mode Toggle */}
              {/* <button
                onClick={() => setUseEnhancedWebGPU(!useEnhancedWebGPU)}
                className="bg-gray-800/80 hover:bg-gray-700/80 px-3 py-1 rounded text-xs text-white border border-gray-600 transition-colors"
              >
                {useEnhancedWebGPU ? '🔧 Enhanced WebGPU' : '🎮 Standard 3D'}
              </button> */}
              
              {/* Clustering Controls Toggle */}
              <button
                onClick={() => setShowClusteringControls(!showClusteringControls)}
                className="bg-blue-800/80 hover:bg-blue-700/80 px-3 py-1 rounded text-xs text-white border border-blue-600 transition-colors flex items-center gap-1"
              >
                <Settings className="w-3 h-3" />
                Clustering
              </button>
            </div>

            {/* Debug Info */}
            {debugInfo && (
              <div className="bg-gray-800/80 px-2 py-1 rounded text-xs text-gray-300">{debugInfo}</div>
            )}

            {/* Enhanced Clustering Controls Panel */}
            {showClusteringControls && (
              <Card className="bg-black/95 border-gray-700 text-white max-w-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Brain className="w-4 h-4" />
                    Smart Clustering Controls
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Advanced semantic and spatial clustering options
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Enable Clustering Toggle */}
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <label className="text-sm font-medium">Enable Clustering</label>
                      <p className="text-xs text-gray-400">
                        GPU-accelerated graph clustering
                      </p>
                    </div>
                    <Switch
                      checked={enableClustering}
                      onCheckedChange={setEnableClustering}
                    />
                  </div>

                  {enableClustering && (
                    <>
                      <Separator className="bg-gray-600" />
                      
                      {/* Collapsible Clustering Method Options */}
                      <Collapsible open={clusteringOptionsExpanded} onOpenChange={setClusteringOptionsExpanded}>
                        <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded-lg hover:bg-gray-800/50 transition-colors">
                          <div className="flex items-center gap-2">
                            <Settings className="w-4 h-4" />
                            <span className="text-sm font-medium">Clustering Options</span>
                          </div>
                          {clusteringOptionsExpanded ? (
                            <ChevronDown className="w-4 h-4" />
                          ) : (
                            <ChevronRight className="w-4 h-4" />
                          )}
                        </CollapsibleTrigger>
                        
                        <CollapsibleContent className="space-y-4 pt-2">
                          {/* Clustering Method Selection */}
                          <div className="space-y-2">
                            <Label className="text-sm font-medium flex items-center gap-2">
                              <Layers className="w-3 h-3" />
                              Clustering Method
                            </Label>
                            <Select value={clusteringMethod} onValueChange={setClusteringMethod}>
                              <SelectTrigger className="bg-gray-800 border-gray-600 text-white">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent className="bg-gray-800 border-gray-600 text-white">
                                <SelectItem value="spatial">🌐 Spatial - Position-based</SelectItem>
                                <SelectItem value="semantic">🧠 Semantic - Name similarity</SelectItem>
                                <SelectItem value="hybrid">⚡ Hybrid - Smart combination</SelectItem>
                              </SelectContent>
                            </Select>
                            <p className="text-xs text-gray-400">
                              {clusteringMethod === "spatial" && "Groups nodes by 3D coordinates"}
                              {clusteringMethod === "semantic" && "Groups nodes by name/content similarity"}
                              {clusteringMethod === "hybrid" && "Combines semantic and spatial features"}
                            </p>
                          </div>

                          {/* Algorithm Selection */}
                          <div className="space-y-2">
                            <Label className="text-sm font-medium">Algorithm</Label>
                            <Select value={semanticAlgorithm} onValueChange={setSemanticAlgorithm}>
                              <SelectTrigger className="bg-gray-800 border-gray-600 text-white">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent className="bg-gray-800 border-gray-600 text-white">
                                <SelectItem value="hierarchical">🌳 Hierarchical</SelectItem>
                                <SelectItem value="kmeans">🎯 K-Means</SelectItem>
                                <SelectItem value="dbscan">🔍 DBSCAN</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>

                          {/* Number of Clusters (for K-means and Hierarchical) */}
                          {(semanticAlgorithm === "kmeans" || semanticAlgorithm === "hierarchical") && (
                            <div className="space-y-2">
                              <Label className="text-sm font-medium">Number of Clusters</Label>
                              <Input
                                type="number"
                                value={numberOfClusters || ""}
                                onChange={(e) => setNumberOfClusters(e.target.value ? parseInt(e.target.value) : null)}
                                placeholder="Auto"
                                className="bg-gray-800 border-gray-600 text-white"
                                min="2"
                                max="50"
                              />
                              <p className="text-xs text-gray-400">Leave empty for automatic selection</p>
                            </div>
                          )}

                          {/* Similarity Threshold (for DBSCAN) */}
                          {semanticAlgorithm === "dbscan" && (
                            <div className="space-y-2">
                              <Label className="text-sm font-medium">Similarity Threshold</Label>
                              <Slider
                                value={[similarityThreshold]}
                                onValueChange={(value) => setSimilarityThreshold(value[0])}
                                min={0.1}
                                max={1.0}
                                step={0.05}
                                className="w-full"
                              />
                              <p className="text-xs text-gray-400">
                                {similarityThreshold.toFixed(2)} - Higher values create fewer, tighter clusters
                              </p>
                            </div>
                          )}

                          {/* Hybrid Weights (for hybrid method) */}
                          {clusteringMethod === "hybrid" && (
                            <div className="space-y-3">
                              <Label className="text-sm font-medium">Feature Weights</Label>
                              
                              <div className="space-y-2">
                                <div className="flex justify-between text-xs">
                                  <span>Name Similarity</span>
                                  <span>{nameWeight.toFixed(1)}</span>
                                </div>
                                <Slider
                                  value={[nameWeight]}
                                  onValueChange={(value) => setNameWeight(value[0])}
                                  min={0}
                                  max={1}
                                  step={0.1}
                                  className="w-full"
                                />
                              </div>
                              
                              <div className="space-y-2">
                                <div className="flex justify-between text-xs">
                                  <span>Content Similarity</span>
                                  <span>{contentWeight.toFixed(1)}</span>
                                </div>
                                <Slider
                                  value={[contentWeight]}
                                  onValueChange={(value) => setContentWeight(value[0])}
                                  min={0}
                                  max={1}
                                  step={0.1}
                                  className="w-full"
                                />
                              </div>
                              
                              <div className="space-y-2">
                                <div className="flex justify-between text-xs">
                                  <span>Spatial Distance</span>
                                  <span>{spatialWeight.toFixed(1)}</span>
                                </div>
                                <Slider
                                  value={[spatialWeight]}
                                  onValueChange={(value) => setSpatialWeight(value[0])}
                                  min={0}
                                  max={1}
                                  step={0.1}
                                  className="w-full"
                                />
                              </div>
                              
                              <p className="text-xs text-gray-400">
                                Total: {(nameWeight + contentWeight + spatialWeight).toFixed(1)}
                              </p>
                            </div>
                          )}
                        </CollapsibleContent>
                      </Collapsible>

                      <Separator className="bg-gray-600" />

                      {/* Cluster Colors Toggle */}
                      <div className="flex items-center justify-between">
                        <div className="space-y-1">
                          <label className="text-sm font-medium">Cluster Colors</label>
                          <p className="text-xs text-gray-400">
                            Color nodes by cluster assignment
                          </p>
                        </div>
                        <Switch
                          checked={enableClusterColors}
                          onCheckedChange={setEnableClusterColors}
                        />
                      </div>
                    </>
                  )}

                  {/* Performance Metrics */}
                  {performanceMetrics && (
                    <>
                      <Separator className="bg-gray-600" />
                      <div className="space-y-2">
                        <label className="text-sm font-medium flex items-center gap-2">
                          <Zap className="w-3 h-3" />
                          Performance
                        </label>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <Badge variant="outline" className="justify-center">
                            {performanceMetrics.totalNodes.toLocaleString()} nodes
                          </Badge>
                          <Badge variant="outline" className="justify-center">
                            {performanceMetrics.totalLinks.toLocaleString()} links
                          </Badge>
                          {performanceMetrics.clusteringTime && (
                            <Badge variant="outline" className="justify-center">
                              {performanceMetrics.clusteringTime.toFixed(0)}ms cluster
                            </Badge>
                          )}
                          <Badge variant="outline" className="justify-center">
                            {performanceMetrics.renderingTime.toFixed(0)}ms render
                          </Badge>
                        </div>
                      </div>
                    </>
                  )}

                  {/* Clustering Status */}
                  {enableClustering && (
                    <Alert className="bg-black/50 border-gray-600">
                      <Monitor className="h-4 w-4" />
                      <AlertDescription className="text-xs">
                        {clusteringMethod === "spatial" && "Using spatial coordinate clustering"}
                        {clusteringMethod === "semantic" && "Using semantic name/content clustering"}
                        {clusteringMethod === "hybrid" && "Using hybrid semantic + spatial clustering"}
                        {" with "}
                        {semanticAlgorithm} algorithm
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
          {((graphData.nodes && graphData.links && graphData.nodes.length > 0) || 
            (graphData.triples && graphData.triples.length > 0)) ? (
            useEnhancedWebGPU ? (
              <WebGPU3DViewer
                graphData={graphData.nodes && graphData.links ? {
                  nodes: graphData.nodes,
                  links: graphData.links
                } : null}
                remoteServiceUrl="http://localhost:8083"
                enableClustering={enableClustering}
                onClusteringUpdate={handleClusteringUpdate}
                onError={(err) => {
                  console.error("Error from WebGPU3DViewer:", err);
                  setError(`Error in enhanced 3D renderer: ${err}`);
                  setDebugInfo(`Enhanced renderer error: ${err}`);
                }}
              />
            ) : (
              <ForceGraphWrapper 
                jsonData={graphData} 
                layoutType={layoutType}
                highlightedNodes={highlightedNodes}
                enableClustering={enableClustering}
                enableClusterColors={enableClusterColors}
                clusteringMode="hybrid" // Default to Hybrid GPU/CPU mode
                remoteServiceUrl="http://localhost:8083"
                onClusteringUpdate={handleClusteringUpdate}
                // Semantic clustering parameters
                clusteringMethod={clusteringMethod}
                semanticAlgorithm={semanticAlgorithm}
                numberOfClusters={numberOfClusters}
                similarityThreshold={similarityThreshold}
                nameWeight={nameWeight}
                contentWeight={contentWeight}
                spatialWeight={spatialWeight}
                onError={(err) => {
                  console.error("Error from ForceGraphWrapper:", err);
                  setError(`Error in graph renderer: ${err.message}`);
                  setDebugInfo(`Renderer error: ${err.message}`);
                }}
              />
            )
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="bg-black/80 border border-red-900 text-red-400 px-6 py-4 rounded-lg max-w-lg text-center">
                <p>Unable to render graph - invalid data structure</p>
                <p className="mt-2 text-xs text-gray-500">
                  The graph data must contain either nodes and links arrays or a triples array
                </p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
} 