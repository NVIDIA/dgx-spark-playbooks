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
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Loader2, Zap, Activity, BarChart3, Eye, ExternalLink, Info } from 'lucide-react'
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

interface PyGraphistryViewerProps {
  graphData: GraphData
  onError?: (error: Error) => void
}

interface VisualizationStats {
  node_count: number
  edge_count: number
  gpu_accelerated: boolean
  clustered: boolean
  layout_type: string
  avg_pagerank?: number
  max_pagerank?: number
  avg_betweenness?: number
  max_betweenness?: number
  density?: number
}

interface ProcessedGraphData {
  processed_nodes: any[]
  processed_edges: any[]
  embed_url?: string
  local_viz_data?: {
    nodes: any[]
    edges: any[]
    positions: Record<string, {x: number, y: number}>
    clusters: Record<string, number>
    layout_computed: boolean
    clusters_computed: boolean
  }
  stats: VisualizationStats & {
    has_embed_url?: boolean
    has_local_viz?: boolean
  }
  timestamp: string
}

export function PyGraphistryViewer({ graphData, onError }: PyGraphistryViewerProps) {
  const [isProcessing, setIsProcessing] = useState(false)
  const [processedData, setProcessedData] = useState<ProcessedGraphData | null>(null)
  const [gpuAcceleration, setGpuAcceleration] = useState(true)
  const [clustering, setClustering] = useState(false)
  const [layoutType, setLayoutType] = useState('force')
  const [serviceHealth, setServiceHealth] = useState<'unknown' | 'healthy' | 'error'>('unknown')
  const [isServiceInitialized, setIsServiceInitialized] = useState(false)
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const { toast } = useToast()

  // Check service health on mount
  useEffect(() => {
    checkServiceHealth()
  }, [])

  const checkServiceHealth = async () => {
    try {
      const response = await fetch('/api/pygraphistry/health')
      if (response.ok) {
        const health = await response.json()
        setServiceHealth('healthy')
        setIsServiceInitialized(health.pygraphistry_initialized)
      } else {
        setServiceHealth('error')
      }
    } catch (error) {
      console.error('Service health check failed:', error)
      setServiceHealth('error')
    }
  }

  const processWithPyGraphistry = async () => {
    if (!graphData?.nodes?.length || !graphData?.links?.length) {
      toast({
        title: "No Graph Data",
        description: "Please ensure graph data is loaded before processing with PyGraphistry.",
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
        layout_type: layoutType,
        gpu_acceleration: gpuAcceleration,
        clustering: clustering
      }

      const response = await fetch('/api/pygraphistry/visualize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      })

      if (!response.ok) {
        throw new Error(`PyGraphistry processing failed: ${response.statusText}`)
      }

      const result: ProcessedGraphData = await response.json()
      setProcessedData(result)
      
      toast({
        title: "GPU Processing Complete",
        description: `Processed ${result.stats.node_count} nodes and ${result.stats.edge_count} edges with ${result.stats.gpu_accelerated ? 'GPU' : 'CPU'} acceleration.`,
      })

    } catch (error) {
      console.error('PyGraphistry processing error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
      
      toast({
        title: "Processing Failed",
        description: errorMessage,
        variant: "destructive"
      })
      
      if (onError) {
        onError(error instanceof Error ? error : new Error(errorMessage))
      }
    } finally {
      setIsProcessing(false)
    }
  }

  const getGraphStats = async () => {
    if (!graphData?.nodes?.length || !graphData?.links?.length) return

    try {
      const response = await fetch('/api/pygraphistry/stats', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          nodes: graphData.nodes,
          links: graphData.links
        })
      })

      if (response.ok) {
        const stats = await response.json()
        toast({
          title: "Graph Statistics",
          description: `Density: ${(stats.density * 100).toFixed(2)}%, Avg PageRank: ${stats.avg_pagerank?.toFixed(4) || 'N/A'}`,
        })
      }
    } catch (error) {
      console.error('Failed to get graph stats:', error)
    }
  }

  const ServiceStatus = () => (
    <div className="flex items-center gap-2 text-sm">
      <div className={`w-2 h-2 rounded-full ${
        serviceHealth === 'healthy' ? 'bg-green-500' : 
        serviceHealth === 'error' ? 'bg-red-500' : 'bg-yellow-500'
      }`} />
      <span>
        PyGraphistry Service: {serviceHealth === 'healthy' ? 'Connected' : 
                               serviceHealth === 'error' ? 'Disconnected' : 'Checking...'}
      </span>
      {isServiceInitialized && (
        <Badge variant="outline" className="ml-2">
          <Zap className="w-3 h-3 mr-1" />
          GPU Ready
        </Badge>
      )}
    </div>
  )

  const StatsDisplay = ({ stats }: { stats: VisualizationStats }) => (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
      <div className="flex items-center gap-2">
        <Activity className="w-4 h-4" />
        <span>{stats.node_count} Nodes</span>
      </div>
      <div className="flex items-center gap-2">
        <BarChart3 className="w-4 h-4" />
        <span>{stats.edge_count} Edges</span>
      </div>
      <div className="flex items-center gap-2">
        <Zap className="w-4 h-4" />
        <span>{stats.gpu_accelerated ? 'GPU' : 'CPU'}</span>
      </div>
      <div className="flex items-center gap-2">
        <Eye className="w-4 h-4" />
        <span>{stats.layout_type}</span>
      </div>
      {stats.density !== undefined && (
        <div className="col-span-2">
          <span>Density: {(stats.density * 100).toFixed(2)}%</span>
        </div>
      )}
      {stats.avg_pagerank !== undefined && (
        <div className="col-span-2">
          <span>Avg PageRank: {stats.avg_pagerank.toFixed(4)}</span>
        </div>
      )}
    </div>
  )

  return (
    <div className="w-full h-full flex flex-col gap-4">
      {/* Control Panel */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              PyGraphistry GPU Visualization
            </CardTitle>
            <ServiceStatus />
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Configuration Controls */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center space-x-2">
              <Switch
                id="gpu-acceleration"
                checked={gpuAcceleration}
                onCheckedChange={setGpuAcceleration}
                disabled={!isServiceInitialized}
              />
              <label htmlFor="gpu-acceleration" className="text-sm font-medium">
                GPU Acceleration
              </label>
            </div>
            
            <div className="flex items-center space-x-2">
              <Switch
                id="clustering"
                checked={clustering}
                onCheckedChange={setClustering}
              />
              <label htmlFor="clustering" className="text-sm font-medium">
                Auto-Clustering
              </label>
            </div>
            
            <Select value={layoutType} onValueChange={setLayoutType}>
              <SelectTrigger>
                <SelectValue placeholder="Layout Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="force">Force-Directed</SelectItem>
                <SelectItem value="circular">Circular</SelectItem>
                <SelectItem value="hierarchical">Hierarchical</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <Button 
              onClick={processWithPyGraphistry}
              disabled={isProcessing || serviceHealth !== 'healthy'}
              className="flex items-center gap-2"
            >
              {isProcessing ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Zap className="w-4 h-4" />
              )}
              {isProcessing ? 'Processing...' : 'Process with GPU'}
            </Button>
            
            <Button 
              variant="outline"
              onClick={getGraphStats}
              disabled={serviceHealth !== 'healthy'}
              className="flex items-center gap-2"
            >
              <BarChart3 className="w-4 h-4" />
              Get Stats
            </Button>
            
            <Button 
              variant="outline"
              onClick={checkServiceHealth}
              className="flex items-center gap-2"
            >
              <Activity className="w-4 h-4" />
              Check Service
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Results Display */}
      {processedData && (
        <Card className="flex-1">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle>GPU-Accelerated Visualization</CardTitle>
              {processedData.embed_url && (
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => window.open(processedData.embed_url, '_blank')}
                  className="flex items-center gap-2"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in PyGraphistry
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Statistics */}
            <StatsDisplay stats={processedData.stats} />
            
            {            /* Embed Visualization */}
            {processedData.embed_url ? (
              <div className="w-full h-96 border rounded-lg overflow-hidden">
                <iframe
                  ref={iframeRef}
                  src={processedData.embed_url}
                  className="w-full h-full"
                  title="PyGraphistry Visualization"
                  allowFullScreen
                />
              </div>
            ) : processedData.local_viz_data ? (
              <div className="w-full h-96 border rounded-lg overflow-hidden">
                <div className="p-4 bg-blue-50 border-b">
                  <div className="flex items-center gap-2 text-blue-700">
                    <Info className="w-4 h-4" />
                    <span className="text-sm font-medium">Local Visualization Mode</span>
                  </div>
                  <div className="text-xs text-blue-600 mt-1">
                    PyGraphistry running in local mode. {processedData.local_viz_data.layout_computed ? 'Layout computed.' : 'No layout computed.'} 
                    {processedData.local_viz_data.clusters_computed ? ' Clusters detected.' : ''}
                  </div>
                </div>
                <div className="p-4 space-y-2">
                  <div className="text-sm">
                    <strong>Processed Data:</strong> {processedData.local_viz_data.nodes.length} nodes, {processedData.local_viz_data.edges.length} edges
                  </div>
                  {processedData.local_viz_data.layout_computed && (
                    <div className="text-sm text-green-600">
                      ✓ Layout positions computed
                    </div>
                  )}
                  {processedData.local_viz_data.clusters_computed && (
                    <div className="text-sm text-green-600">
                      ✓ Cluster analysis completed
                    </div>
                  )}
                  <div className="text-xs text-muted-foreground mt-2">
                    To enable interactive visualization, set up PyGraphistry cloud credentials (GRAPHISTRY_API_KEY or GRAPHISTRY_USERNAME/PASSWORD)
                  </div>
                </div>
              </div>
            ) : (
              <div className="w-full h-96 border rounded-lg flex items-center justify-center bg-muted">
                <div className="text-center space-y-2">
                  <div className="text-muted-foreground">
                    Processed data available - visualization embed not generated
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {processedData.processed_nodes.length} nodes, {processedData.processed_edges.length} edges processed
                  </div>
                  <div className="text-xs text-muted-foreground mt-2">
                    PyGraphistry may be running in local mode without cloud credentials
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Service Health Warning */}
      {serviceHealth === 'error' && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-destructive">
              <Activity className="w-4 h-4" />
              <span>PyGraphistry service is not available. Please ensure the service is running on port 8080.</span>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
} 