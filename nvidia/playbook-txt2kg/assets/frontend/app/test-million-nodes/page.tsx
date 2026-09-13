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

import React, { useState, useEffect, useCallback, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { PyGraphistryViewer } from "@/components/pygraphistry-viewer"
import { ForceGraphWrapper } from "@/components/force-graph-wrapper"
import { useToast } from "@/hooks/use-toast"
import { 
  Play, 
  Square, 
  Zap, 
  Database, 
  Activity, 
  BarChart3, 
  Settings,
  AlertTriangle,
  CheckCircle,
  Clock,
  Eye,
  Download,
  Upload,
  Server
} from "lucide-react"

interface NodeObject {
  id: string
  name: string
  val?: number
  group?: string
  pagerank?: number
  betweenness?: number
  degree?: number
  x?: number
  y?: number
  z?: number
}

interface LinkObject {
  source: string
  target: string
  name: string
  weight?: number
}

interface GraphData {
  nodes: NodeObject[]
  links: LinkObject[]
}

interface GenerationStats {
  node_count: number
  edge_count: number
  generation_time: number
  density: number
  avg_degree: number
  pattern: string
  parameters: any
}

interface GenerationTask {
  task_id: string
  status: string
  progress: number
  message: string
  result?: {
    graph_data: GraphData
    stats: GenerationStats
  }
  error?: string
}

// Graph generation patterns
const GRAPH_PATTERNS = {
  RANDOM: 'random',
  SCALE_FREE: 'scale-free',
  SMALL_WORLD: 'small-world',
  CLUSTERED: 'clustered',
  HIERARCHICAL: 'hierarchical',
  GRID: 'grid'
} as const

type GraphPattern = typeof GRAPH_PATTERNS[keyof typeof GRAPH_PATTERNS]

export default function TestMillionNodesPage() {
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [generationProgress, setGenerationProgress] = useState(0)
  const [generationStats, setGenerationStats] = useState<GenerationStats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [currentTask, setCurrentTask] = useState<string | null>(null)
  
  // Generation parameters
  const [numNodes, setNumNodes] = useState(100000)
  const [graphPattern, setGraphPattern] = useState<GraphPattern>(GRAPH_PATTERNS.SCALE_FREE)
  const [avgDegree, setAvgDegree] = useState(5)
  const [numClusters, setNumClusters] = useState(100)
  const [smallWorldK, setSmallWorldK] = useState(6)
  const [smallWorldP, setSmallWorldP] = useState(0.1)
  const [seed, setSeed] = useState<number | null>(null)
  
  // Visualization mode
  const [visualizationMode, setVisualizationMode] = useState<'pygraphistry' | 'force-graph'>('pygraphistry')
  
  const { toast } = useToast()
  
  // Poll for task status
  // Removed polling useEffect since we're using direct API calls now
  
  const generateGraphData = useCallback(async () => {
    if (numNodes > 1000000) {
      toast({
        title: "Node Limit Exceeded",
        description: "Maximum 1 million nodes allowed for performance reasons.",
        variant: "destructive"
      })
      return
    }

    setIsGenerating(true)
    setError(null)
    setGenerationProgress(0)
    
    try {
      const requestBody = {
        num_nodes: numNodes,
        pattern: graphPattern,
        avg_degree: avgDegree,
        num_clusters: graphPattern === GRAPH_PATTERNS.CLUSTERED ? numClusters : undefined,
        small_world_k: graphPattern === GRAPH_PATTERNS.SMALL_WORLD ? smallWorldK : undefined,
        small_world_p: graphPattern === GRAPH_PATTERNS.SMALL_WORLD ? smallWorldP : undefined,
        seed: seed || undefined
      }
      
      toast({
        title: "Generation Started",
        description: `Starting generation of ${numNodes.toLocaleString()} nodes using ${graphPattern} pattern...`,
      })

      // First generate the graph data
      const generateResponse = await fetch('/api/pygraphistry/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      })
      
      if (!generateResponse.ok) {
        const errorData = await generateResponse.json()
        throw new Error(errorData.error || 'Failed to start graph generation')
      }

      const generateResult = await generateResponse.json()
      
      // Update progress
      setGenerationProgress(0.5)
      
      // If we have graph data directly, process it with the unified service
      if (generateResult.graph_data) {
        const visualizeRequest = {
          graph_data: generateResult.graph_data,
          processing_mode: "pygraphistry_cloud",
          layout_type: "force",
          gpu_acceleration: true,
          clustering: true
        }

        const visualizeResponse = await fetch('/api/pygraphistry/visualize', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(visualizeRequest)
        })

        if (!visualizeResponse.ok) {
          const errorData = await visualizeResponse.json()
          throw new Error(errorData.error || 'Failed to process graph visualization')
        }

        const result = await visualizeResponse.json()
        setGraphData(result.graph_data || generateResult.graph_data)
        setGenerationStats({
          node_count: generateResult.graph_data.nodes.length,
          edge_count: generateResult.graph_data.links.length,
          generation_time: 0,
          density: generateResult.graph_data.links.length / (generateResult.graph_data.nodes.length * (generateResult.graph_data.nodes.length - 1)),
          avg_degree: avgDegree,
          pattern: graphPattern,
          parameters: requestBody
        })
        setGenerationProgress(1.0)
        setIsGenerating(false)

        toast({
          title: "Graph Generated Successfully", 
          description: `Generated ${generateResult.graph_data.nodes.length.toLocaleString()} nodes and ${generateResult.graph_data.links.length.toLocaleString()} edges`,
        })
      } else {
        throw new Error('No graph data returned from generation service')
      }
      
    } catch (err: any) {
      console.error("Error generating graph:", err)
      setError(`Failed to generate graph: ${err.message}`)
      setIsGenerating(false)
      toast({
        title: "Generation Failed",
        description: err.message,
        variant: "destructive"
      })
    }
  }, [numNodes, graphPattern, avgDegree, numClusters, smallWorldK, smallWorldP, seed, toast])
  
  const clearGraph = useCallback(() => {
    setGraphData(null)
    setGenerationStats(null)
    setGenerationProgress(0)
    setError(null)
    setCurrentTask(null)
    if (isGenerating) {
      setIsGenerating(false)
    }
  }, [isGenerating])
  
  const exportGraphData = useCallback(() => {
    if (!graphData) return
    
    const dataStr = JSON.stringify(graphData, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    
    const link = document.createElement('a')
    link.href = url
    link.download = `graph_${numNodes}_nodes_${Date.now()}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }, [graphData, numNodes])
  
  const presetConfigs = [
    { name: "Small Test", nodes: 1000, pattern: GRAPH_PATTERNS.RANDOM, degree: 5 },
    { name: "Medium Test", nodes: 10000, pattern: GRAPH_PATTERNS.SCALE_FREE, degree: 3 },
    { name: "Large Test", nodes: 100000, pattern: GRAPH_PATTERNS.SCALE_FREE, degree: 3 },
    { name: "Huge Test", nodes: 500000, pattern: GRAPH_PATTERNS.CLUSTERED, degree: 4 },
    { name: "Million Nodes", nodes: 1000000, pattern: GRAPH_PATTERNS.CLUSTERED, degree: 2 },
  ]
  
  const memoryEstimate = useMemo(() => {
    // Rough estimate: each node ~100 bytes, each link ~150 bytes
    const nodeMemory = numNodes * 100 / 1024 / 1024 // MB
    const linkMemory = (numNodes * avgDegree / 2) * 150 / 1024 / 1024 // MB
    return nodeMemory + linkMemory
  }, [numNodes, avgDegree])
  
  return (
    <div className="container mx-auto p-4 max-w-7xl">
      <div className="flex flex-col gap-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Million Node Graph Test</h1>
            <p className="text-muted-foreground">
              Generate and visualize large-scale graphs with up to 1 million nodes using GPU acceleration
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-sm">
              <Server className="w-4 h-4 mr-1" />
              Backend Generation
            </Badge>
            <Badge variant="outline" className="text-sm">
              <Database className="w-4 h-4 mr-1" />
              PyGraphistry Ready
            </Badge>
            {memoryEstimate > 100 && (
              <Badge variant="destructive" className="text-sm">
                <AlertTriangle className="w-4 h-4 mr-1" />
                High Memory ({memoryEstimate.toFixed(0)}MB)
              </Badge>
            )}
          </div>
        </div>
        
        {/* Control Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Graph Generation Settings
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="parameters" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="parameters">Parameters</TabsTrigger>
                <TabsTrigger value="presets">Presets</TabsTrigger>
                <TabsTrigger value="advanced">Advanced</TabsTrigger>
              </TabsList>
              
              <TabsContent value="parameters" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="num-nodes">Number of Nodes</Label>
                    <Input
                      id="num-nodes"
                      type="number"
                      value={numNodes}
                      onChange={(e) => setNumNodes(Math.min(1000000, Math.max(1, parseInt(e.target.value) || 1)))}
                      min="1"
                      max="1000000"
                      step="1000"
                      disabled={isGenerating}
                    />
                    <p className="text-xs text-muted-foreground">Max: 1,000,000</p>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="graph-pattern">Graph Pattern</Label>
                    <Select 
                      value={graphPattern} 
                      onValueChange={(value: GraphPattern) => setGraphPattern(value)}
                      disabled={isGenerating}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value={GRAPH_PATTERNS.RANDOM}>Random (Erdős–Rényi)</SelectItem>
                        <SelectItem value={GRAPH_PATTERNS.SCALE_FREE}>Scale-Free (Barabási–Albert)</SelectItem>
                        <SelectItem value={GRAPH_PATTERNS.CLUSTERED}>Clustered Communities</SelectItem>
                        <SelectItem value={GRAPH_PATTERNS.SMALL_WORLD}>Small World (Watts-Strogatz)</SelectItem>
                        <SelectItem value={GRAPH_PATTERNS.HIERARCHICAL}>Hierarchical Tree</SelectItem>
                        <SelectItem value={GRAPH_PATTERNS.GRID}>Grid Network</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="avg-degree">Average Degree</Label>
                    <Input
                      id="avg-degree"
                      type="number"
                      value={avgDegree}
                      onChange={(e) => setAvgDegree(Math.min(50, Math.max(1, parseInt(e.target.value) || 1)))}
                      min="1"
                      max="50"
                      disabled={isGenerating}
                    />
                    <p className="text-xs text-muted-foreground">Connections per node</p>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="seed">Random Seed (Optional)</Label>
                    <Input
                      id="seed"
                      type="number"
                      value={seed || ''}
                      onChange={(e) => setSeed(e.target.value ? parseInt(e.target.value) : null)}
                      placeholder="Random"
                      disabled={isGenerating}
                    />
                    <p className="text-xs text-muted-foreground">For reproducible graphs</p>
                  </div>
                </div>
                
                {/* Pattern-specific parameters */}
                {graphPattern === GRAPH_PATTERNS.CLUSTERED && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t">
                    <div className="space-y-2">
                      <Label htmlFor="num-clusters">Number of Clusters</Label>
                      <Input
                        id="num-clusters"
                        type="number"
                        value={numClusters}
                        onChange={(e) => setNumClusters(Math.max(1, parseInt(e.target.value) || 1))}
                        min="1"
                        max="1000"
                        disabled={isGenerating}
                      />
                    </div>
                  </div>
                )}
                
                {graphPattern === GRAPH_PATTERNS.SMALL_WORLD && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t">
                    <div className="space-y-2">
                      <Label htmlFor="small-world-k">Initial Neighbors (k)</Label>
                      <Input
                        id="small-world-k"
                        type="number"
                        value={smallWorldK}
                        onChange={(e) => setSmallWorldK(Math.max(2, parseInt(e.target.value) || 2))}
                        min="2"
                        max="20"
                        disabled={isGenerating}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="small-world-p">Rewiring Probability (p)</Label>
                      <Input
                        id="small-world-p"
                        type="number"
                        step="0.01"
                        value={smallWorldP}
                        onChange={(e) => setSmallWorldP(Math.min(1, Math.max(0, parseFloat(e.target.value) || 0)))}
                        min="0"
                        max="1"
                        disabled={isGenerating}
                      />
                    </div>
                  </div>
                )}
                
                <div className="flex items-center justify-between pt-4">
                  <div className="flex items-center gap-4">
                    <Button
                      onClick={generateGraphData}
                      disabled={isGenerating}
                      className="flex items-center gap-2"
                    >
                      {isGenerating ? (
                        <>
                          <Clock className="w-4 h-4 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4" />
                          Generate Graph
                        </>
                      )}
                    </Button>
                    
                    <Button
                      onClick={clearGraph}
                      variant="outline"
                      disabled={!graphData && !isGenerating}
                    >
                      {isGenerating ? 'Cancel' : 'Clear'}
                    </Button>
                    
                    <Button
                      onClick={exportGraphData}
                      variant="outline"
                      disabled={!graphData}
                      className="flex items-center gap-2"
                    >
                      <Download className="w-4 h-4" />
                      Export
                    </Button>
                  </div>
                  
                  <div className="text-sm text-muted-foreground">
                    Estimated Memory: {memoryEstimate.toFixed(1)}MB
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="presets" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {presetConfigs.map((preset, index) => (
                    <Card key={index} className="cursor-pointer hover:shadow-md transition-shadow">
                      <CardContent className="p-4">
                        <div className="space-y-2">
                          <h3 className="font-semibold">{preset.name}</h3>
                          <div className="text-sm text-muted-foreground space-y-1">
                            <div>Nodes: {preset.nodes.toLocaleString()}</div>
                            <div>Pattern: {preset.pattern}</div>
                            <div>Degree: {preset.degree}</div>
                          </div>
                          <Button
                            size="sm"
                            onClick={() => {
                              setNumNodes(preset.nodes)
                              setGraphPattern(preset.pattern)
                              setAvgDegree(preset.degree)
                            }}
                            className="w-full"
                            disabled={isGenerating}
                          >
                            Load Preset
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </TabsContent>
              
              <TabsContent value="advanced" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Visualization Mode</Label>
                      <Select value={visualizationMode} onValueChange={(value: 'pygraphistry' | 'force-graph') => setVisualizationMode(value)}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="pygraphistry">PyGraphistry (GPU)</SelectItem>
                          <SelectItem value="force-graph">Force Graph (WebGL)</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        PyGraphistry recommended for large graphs (&gt;50k nodes)
                      </p>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="p-4 bg-muted rounded-lg">
                      <h4 className="font-medium mb-2">Backend Processing</h4>
                      <p className="text-sm text-muted-foreground">
                        Graphs are now generated on the backend using optimized NetworkX algorithms with GPU acceleration available through PyGraphistry.
                      </p>
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
        
        {/* Progress and Stats */}
        {(isGenerating || generationStats) && (
          <Card>
            <CardContent className="p-6">
              {isGenerating && (
                <div className="space-y-4">
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 animate-spin" />
                    <span>Generating graph on backend...</span>
                    {currentTask && (
                      <Badge variant="outline" className="text-xs">
                        Task: {currentTask}
                      </Badge>
                    )}
                  </div>
                  <Progress value={generationProgress} className="w-full" />
                  <p className="text-sm text-muted-foreground">
                    Progress: {generationProgress.toFixed(1)}%
                  </p>
                </div>
              )}
              
              {generationStats && (
                <div className="space-y-4">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span className="font-medium">Generation Complete</span>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-sm">
                    <div className="flex items-center gap-2">
                      <Activity className="w-4 h-4" />
                      <span>{generationStats.node_count.toLocaleString()} Nodes</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <BarChart3 className="w-4 h-4" />
                      <span>{generationStats.edge_count.toLocaleString()} Links</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Clock className="w-4 h-4" />
                      <span>{generationStats.generation_time.toFixed(2)}s</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Zap className="w-4 h-4" />
                      <span>{generationStats.pattern}</span>
                    </div>
                    <div>
                      <span>Density: {(generationStats.density * 100).toFixed(4)}%</span>
                    </div>
                    <div>
                      <span>Avg Degree: {generationStats.avg_degree.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}
        
        {/* Error Display */}
        {error && (
          <Card className="border-red-200 bg-red-50">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-red-700">
                <AlertTriangle className="w-4 h-4" />
                <span className="font-medium">Error:</span>
                <span>{error}</span>
              </div>
            </CardContent>
          </Card>
        )}
        
        {/* Visualization */}
        {graphData && !isGenerating && (
          <Card className="min-h-[600px]">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Eye className="w-5 h-5" />
                  Graph Visualization
                </CardTitle>
                <div className="flex items-center gap-2">
                  <Badge variant="outline">
                    {visualizationMode === 'pygraphistry' ? 'GPU Accelerated' : 'WebGL'}
                  </Badge>
                  <Badge variant="secondary">
                    {graphData.nodes.length.toLocaleString()} nodes
                  </Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div style={{ height: "600px", width: "100%" }}>
                {visualizationMode === 'pygraphistry' ? (
                  <PyGraphistryViewer
                    graphData={graphData}
                    onError={(err) => {
                      console.error("PyGraphistry error:", err)
                      setError(`Visualization error: ${err.message}`)
                    }}
                  />
                ) : (
                  <ForceGraphWrapper
                    jsonData={graphData}
                    fullscreen={false}
                    onError={(err) => {
                      console.error("Force graph error:", err)
                      setError(`Visualization error: ${err.message}`)
                    }}
                  />
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
} 