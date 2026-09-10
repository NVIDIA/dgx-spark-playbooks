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

import React, { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Loader2, Zap, Activity, Cpu, Cloud, ExternalLink, Settings } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'

interface GraphData {
  nodes: Array<{
    id: string
    name: string
    group?: string
    [key: string]: any
  }>
  links: Array<{
    source: string
    target: string
    name: string
    [key: string]: any
  }>
}

interface UnifiedGPUViewerProps {
  graphData: GraphData
  onError?: (error: Error) => void
}

interface ProcessingCapabilities {
  processing_modes: {
    pygraphistry_cloud: { available: boolean, description: string }
    local_gpu: { available: boolean, description: string } 
    local_cpu: { available: boolean, description: string }
  }
  has_rapids: boolean
  has_torch_geometric: boolean
  gpu_available: boolean
}

interface ProcessedData {
  processed_nodes: any[]
  processed_edges: any[]
  processing_mode: string
  embed_url?: string
  gpu_processed?: boolean
  layout_positions?: Record<string, [number, number]>
  clusters?: Record<string, number>
  centrality?: Record<string, Record<string, number>>
  stats: {
    node_count: number
    edge_count: number
    gpu_accelerated: boolean
    has_embed_url?: boolean
    layout_computed?: boolean
    clusters_computed?: boolean
    centrality_computed?: boolean
  }
  timestamp: string
}

export function UnifiedGPUViewer({ graphData, onError }: UnifiedGPUViewerProps) {
  const [isProcessing, setIsProcessing] = useState(false)
  const [processedData, setProcessedData] = useState<ProcessedData | null>(null)
  const [capabilities, setCapabilities] = useState<ProcessingCapabilities | null>(null)
  const [serviceHealth, setServiceHealth] = useState<'unknown' | 'healthy' | 'error'>('unknown')
  
  // Processing mode and options
  const [processingMode, setProcessingMode] = useState<'pygraphistry_cloud' | 'local_gpu' | 'local_cpu'>('pygraphistry_cloud')
  
  // PyGraphistry options
  const [layoutType, setLayoutType] = useState('force')
  const [gpuAcceleration, setGpuAcceleration] = useState(true)
  const [clustering, setClustering] = useState(false)
  
  // Local GPU options  
  const [layoutAlgorithm, setLayoutAlgorithm] = useState('force_atlas2')
  const [clusteringAlgorithm, setClusteringAlgorithm] = useState('leiden')
  const [computeCentrality, setComputeCentrality] = useState(true)
  
  const { toast } = useToast()
  const wsRef = useRef<WebSocket | null>(null)

  // Check service health and capabilities on mount
  useEffect(() => {
    checkServiceHealth()
    getCapabilities()
    setupWebSocket()
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  const checkServiceHealth = async () => {
    try {
      const response = await fetch('/api/unified-gpu/health')
      if (response.ok) {
        setServiceHealth('healthy')
      } else {
        setServiceHealth('error')
      }
    } catch (error) {
      console.error('Unified service health check failed:', error)
      setServiceHealth('error')
    }
  }

  const getCapabilities = async () => {
    try {
      const response = await fetch('/api/unified-gpu/capabilities')
      if (response.ok) {
        const caps = await response.json()
        setCapabilities(caps)
        
        // Set default mode based on capabilities
        if (caps.processing_modes.pygraphistry_cloud.available) {
          setProcessingMode('pygraphistry_cloud')
        } else if (caps.processing_modes.local_gpu.available) {
          setProcessingMode('local_gpu')
        } else {
          setProcessingMode('local_cpu')
        }
      }
    } catch (error) {
      console.error('Failed to get capabilities:', error)
    }
  }

  const setupWebSocket = () => {
    try {
      const ws = new WebSocket('ws://localhost:8080/ws')
      
      ws.onopen = () => {
        console.log('WebSocket connected to unified service')
      }
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          if (message.type === 'graph_processed') {
            setProcessedData(message.data)
            setIsProcessing(false)
            
            toast({
              title: "Processing Complete",
              description: `Processed ${message.data.stats.node_count} nodes with ${message.data.processing_mode}`,
            })
          }
        } catch (error) {
          console.error('WebSocket message error:', error)
        }
      }
      
      wsRef.current = ws
    } catch (error) {
      console.error('WebSocket setup failed:', error)
    }
  }

  const processGraph = async () => {
    if (!graphData?.nodes?.length || !graphData?.links?.length) {
      toast({
        title: "No Graph Data",
        description: "Please ensure graph data is loaded before processing.",
        variant: "destructive"
      })
      return
    }

    setIsProcessing(true)
    
    try {
      const requestData = {
        graph_data: {
          nodes: graphData.nodes,
          links: graphData.links
        },
        processing_mode: processingMode,
        
        // PyGraphistry options
        layout_type: layoutType,
        gpu_acceleration: gpuAcceleration,
        clustering: clustering,
        
        // Local GPU options
        layout_algorithm: layoutAlgorithm,
        clustering_algorithm: clusteringAlgorithm,
        compute_centrality: computeCentrality
      }

      const response = await fetch('/api/unified-gpu/visualize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      })

      if (!response.ok) {
        throw new Error(`Processing failed: ${response.statusText}`)
      }

      // Result will come via WebSocket or direct response
      const result = await response.json()
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        // Handle direct response if WebSocket not available
        setProcessedData(result)
        setIsProcessing(false)
        
        toast({
          title: "Processing Complete",
          description: `Processed ${result.stats.node_count} nodes with ${result.processing_mode}`,
        })
      }

    } catch (error) {
      console.error('Processing error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
      
      toast({
        title: "Processing Failed",
        description: errorMessage,
        variant: "destructive"
      })
      
      setIsProcessing(false)
      
      if (onError) {
        onError(error instanceof Error ? error : new Error(errorMessage))
      }
    }
  }

  const ServiceStatus = () => (
    <div className="flex items-center gap-2 text-sm">
      <div className={`w-2 h-2 rounded-full ${
        serviceHealth === 'healthy' ? 'bg-green-500' : 
        serviceHealth === 'error' ? 'bg-red-500' : 'bg-yellow-500'
      }`} />
      <span>
        Unified Service: {serviceHealth === 'healthy' ? 'Connected' : 
                         serviceHealth === 'error' ? 'Disconnected' : 'Checking...'}
      </span>
    </div>
  )

  const ProcessingModeSelector = () => (
    <div className="space-y-2">
      <label className="text-sm font-medium">Processing Mode</label>
      <Select value={processingMode} onValueChange={(value: any) => setProcessingMode(value)}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {capabilities?.processing_modes.pygraphistry_cloud.available && (
            <SelectItem value="pygraphistry_cloud">
              <div className="flex items-center gap-2">
                <Cloud className="w-4 h-4" />
                PyGraphistry Cloud
              </div>
            </SelectItem>
          )}
          {capabilities?.processing_modes.local_gpu.available && (
            <SelectItem value="local_gpu">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Local GPU (cuGraph)
              </div>
            </SelectItem>
          )}
          <SelectItem value="local_cpu">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4" />
              Local CPU
            </div>
          </SelectItem>
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        {capabilities?.processing_modes[processingMode]?.description}
      </p>
    </div>
  )

  const StatsDisplay = ({ stats }: { stats: ProcessedData['stats'] }) => (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
      <div className="flex items-center gap-2">
        <Activity className="w-4 h-4 text-blue-500" />
        <span>{stats.node_count.toLocaleString()} nodes</span>
      </div>
      <div className="flex items-center gap-2">
        <Activity className="w-4 h-4 text-green-500" />
        <span>{stats.edge_count.toLocaleString()} edges</span>
      </div>
      <div className="flex items-center gap-2">
        {stats.gpu_accelerated ? (
          <Zap className="w-4 h-4 text-yellow-500" />
        ) : (
          <Cpu className="w-4 h-4 text-gray-500" />
        )}
        <span>{stats.gpu_accelerated ? 'GPU' : 'CPU'} accelerated</span>
      </div>
    </div>
  )

  return (
    <div className="space-y-4">
      {/* Service Status */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              Unified GPU Visualization
            </span>
            <ServiceStatus />
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Processing Mode Selection */}
          <ProcessingModeSelector />

          {/* Processing Options */}
          <Tabs value={processingMode} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="pygraphistry_cloud" disabled={!capabilities?.processing_modes.pygraphistry_cloud.available}>
                Cloud
              </TabsTrigger>
              <TabsTrigger value="local_gpu" disabled={!capabilities?.processing_modes.local_gpu.available}>
                Local GPU
              </TabsTrigger>
              <TabsTrigger value="local_cpu">
                Local CPU
              </TabsTrigger>
            </TabsList>

            <TabsContent value="pygraphistry_cloud" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Layout Type</label>
                  <Select value={layoutType} onValueChange={setLayoutType}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="force">Force Directed</SelectItem>
                      <SelectItem value="circular">Circular</SelectItem>
                      <SelectItem value="hierarchical">Hierarchical</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">GPU Acceleration</label>
                    <Switch checked={gpuAcceleration} onCheckedChange={setGpuAcceleration} />
                  </div>
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">Auto-Clustering</label>
                    <Switch checked={clustering} onCheckedChange={setClustering} />
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="local_gpu" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Layout Algorithm</label>
                    <Select value={layoutAlgorithm} onValueChange={setLayoutAlgorithm}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="force_atlas2">Force Atlas 2</SelectItem>
                        <SelectItem value="spectral">Spectral Layout</SelectItem>
                        <SelectItem value="fruchterman_reingold">Fruchterman-Reingold</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Clustering Algorithm</label>
                    <Select value={clusteringAlgorithm} onValueChange={setClusteringAlgorithm}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="leiden">Leiden</SelectItem>
                        <SelectItem value="louvain">Louvain</SelectItem>
                        <SelectItem value="spectral">Spectral Clustering</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">Compute Centrality</label>
                    <Switch checked={computeCentrality} onCheckedChange={setComputeCentrality} />
                  </div>
                  {capabilities && (
                    <div className="p-2 bg-muted rounded text-xs">
                      RAPIDS cuGraph: {capabilities.has_rapids ? '✓ Available' : '✗ Not Available'}
                    </div>
                  )}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="local_cpu" className="space-y-4">
              <div className="p-4 bg-muted rounded-lg text-center">
                <p className="text-sm text-muted-foreground">
                  CPU fallback mode - basic processing without GPU acceleration
                </p>
              </div>
            </TabsContent>
          </Tabs>

          {/* Action Button */}
          <Button 
            onClick={processGraph}
            disabled={isProcessing || serviceHealth !== 'healthy'}
            className="w-full"
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Processing with {processingMode.replace('_', ' ')}...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Process Graph
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Results */}
      {processedData && (
        <Card className="flex-1">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle>Processing Results</CardTitle>
              <Badge variant="outline">
                {processedData.processing_mode.replace('_', ' ')}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Statistics */}
            <StatsDisplay stats={processedData.stats} />
            
            {/* Visualization */}
            <div className="w-full h-96 border rounded-lg overflow-hidden">
              {processedData.embed_url ? (
                // PyGraphistry Cloud embed
                <>
                  <div className="p-3 bg-blue-50 border-b flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-blue-700">
                        ☁️ PyGraphistry Cloud Visualization
                      </div>
                      <div className="text-xs text-blue-600">
                        Interactive GPU-accelerated visualization
                      </div>
                    </div>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => window.open(processedData.embed_url, '_blank')}
                    >
                      <ExternalLink className="w-4 h-4 mr-1" />
                      Open
                    </Button>
                  </div>
                  <iframe
                    src={processedData.embed_url}
                    className="w-full h-80"
                    title="PyGraphistry Visualization"
                    style={{ border: 'none' }}
                  />
                </>
              ) : (
                // Local processing result
                <div className="p-4 bg-green-50 border-b">
                  <div className="text-sm font-medium text-green-700">
                    🚀 Local {processedData.gpu_processed ? 'GPU' : 'CPU'} Processing Complete
                  </div>
                  <div className="text-xs text-green-600 mt-1">
                    {processedData.stats.layout_computed && '✓ Layout computed '}
                    {processedData.stats.clusters_computed && '✓ Clusters detected '}
                    {processedData.stats.centrality_computed && '✓ Centrality computed'}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
} 