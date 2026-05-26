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
import { Loader2, Zap, Activity, Cpu, MonitorSpeaker } from 'lucide-react'
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

interface LocalGPUViewerProps {
  graphData: GraphData
  onError?: (error: Error) => void
}

interface GPUCapabilities {
  has_rapids: boolean
  has_torch_geometric: boolean
  gpu_available: boolean
  supported_layouts: string[]
  supported_clustering: string[]
}

interface ProcessedData {
  nodes: any[]
  edges: any[]
  gpu_processed: boolean
  layout_positions: Record<string, [number, number]>
  clusters: Record<string, number>
  centrality: Record<string, Record<string, number>>
  stats: {
    node_count: number
    edge_count: number
    gpu_accelerated: boolean
    layout_computed: boolean
    clusters_computed: boolean
    centrality_computed: boolean
  }
  timestamp: string
}

export function LocalGPUViewer({ graphData, onError }: LocalGPUViewerProps) {
  const [isProcessing, setIsProcessing] = useState(false)
  const [processedData, setProcessedData] = useState<ProcessedData | null>(null)
  const [capabilities, setCapabilities] = useState<GPUCapabilities | null>(null)
  const [serviceHealth, setServiceHealth] = useState<'unknown' | 'healthy' | 'error'>('unknown')
  
  // Processing options
  const [layoutAlgorithm, setLayoutAlgorithm] = useState('force_atlas2')
  const [clusteringAlgorithm, setClusteringAlgorithm] = useState('leiden')
  const [gpuAcceleration, setGpuAcceleration] = useState(true)
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
      const response = await fetch('/api/local-gpu/health')
      if (response.ok) {
        setServiceHealth('healthy')
      } else {
        setServiceHealth('error')
      }
    } catch (error) {
      console.error('Local GPU service health check failed:', error)
      setServiceHealth('error')
    }
  }

  const getCapabilities = async () => {
    try {
      const response = await fetch('/api/local-gpu/capabilities')
      if (response.ok) {
        const caps = await response.json()
        setCapabilities(caps)
        
        // Set defaults based on capabilities
        if (caps.supported_layouts?.length > 0) {
          setLayoutAlgorithm(caps.supported_layouts[0])
        }
        if (caps.supported_clustering?.length > 0) {
          setClusteringAlgorithm(caps.supported_clustering[0])
        }
      }
    } catch (error) {
      console.error('Failed to get GPU capabilities:', error)
    }
  }

  const setupWebSocket = () => {
    try {
      const ws = new WebSocket('ws://localhost:8081/ws')
      
      ws.onopen = () => {
        console.log('WebSocket connected to local GPU service')
      }
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          if (message.type === 'graph_processed') {
            setProcessedData(message.data)
            setIsProcessing(false)
            
            toast({
              title: "GPU Processing Complete",
              description: `Processed ${message.data.stats.node_count} nodes with ${message.data.gpu_processed ? 'GPU' : 'CPU'} acceleration.`,
            })
          }
        } catch (error) {
          console.error('WebSocket message error:', error)
        }
      }
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }
      
      wsRef.current = ws
    } catch (error) {
      console.error('WebSocket setup failed:', error)
    }
  }

  const processWithLocalGPU = async () => {
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
        layout_algorithm: layoutAlgorithm,
        clustering_algorithm: clusteringAlgorithm,
        gpu_acceleration: gpuAcceleration,
        compute_centrality: computeCentrality
      }

      const response = await fetch('/api/local-gpu/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      })

      if (!response.ok) {
        throw new Error(`Local GPU processing failed: ${response.statusText}`)
      }

      // The result will come via WebSocket, so we just wait
      toast({
        title: "Processing Started",
        description: "Graph processing started on local GPU. Results will appear shortly.",
      })

    } catch (error) {
      console.error('Local GPU processing error:', error)
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
        Local GPU Service: {serviceHealth === 'healthy' ? 'Connected' : 
                           serviceHealth === 'error' ? 'Disconnected' : 'Checking...'}
      </span>
      {capabilities?.gpu_available && (
        <Badge variant="outline" className="ml-2">
          <Zap className="w-3 h-3 mr-1" />
          cuGraph Ready
        </Badge>
      )}
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
        <span>{stats.gpu_accelerated ? 'GPU' : 'CPU'} processed</span>
      </div>
      {stats.layout_computed && (
        <div className="flex items-center gap-2">
          <MonitorSpeaker className="w-4 h-4 text-purple-500" />
          <span>Layout computed</span>
        </div>
      )}
      {stats.clusters_computed && (
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-orange-500" />
          <span>Clusters detected</span>
        </div>
      )}
      {stats.centrality_computed && (
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-pink-500" />
          <span>Centrality computed</span>
        </div>
      )}
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
              Local GPU Visualization
            </span>
            <ServiceStatus />
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* GPU Capabilities */}
          {capabilities && (
            <div className="p-3 bg-muted rounded-lg">
              <h4 className="font-medium mb-2">GPU Capabilities</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${capabilities.has_rapids ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span>RAPIDS cuGraph: {capabilities.has_rapids ? 'Available' : 'Not Available'}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${capabilities.has_torch_geometric ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span>PyTorch Geometric: {capabilities.has_torch_geometric ? 'Available' : 'Not Available'}</span>
                </div>
              </div>
            </div>
          )}

          {/* Processing Controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Layout Algorithm</label>
                <Select value={layoutAlgorithm} onValueChange={setLayoutAlgorithm}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {capabilities?.supported_layouts?.map(layout => (
                      <SelectItem key={layout} value={layout}>
                        {layout.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </SelectItem>
                    ))}
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
                    {capabilities?.supported_clustering?.map(clustering => (
                      <SelectItem key={clustering} value={clustering}>
                        {clustering.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">GPU Acceleration</label>
                <Switch 
                  checked={gpuAcceleration} 
                  onCheckedChange={setGpuAcceleration}
                  disabled={!capabilities?.gpu_available}
                />
              </div>

              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Compute Centrality</label>
                <Switch 
                  checked={computeCentrality} 
                  onCheckedChange={setComputeCentrality}
                />
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <Button 
              onClick={processWithLocalGPU}
              disabled={isProcessing || serviceHealth !== 'healthy'}
              className="flex-1"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Processing on {gpuAcceleration ? 'GPU' : 'CPU'}...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  Process with Local GPU
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {processedData && (
        <Card className="flex-1">
          <CardHeader className="pb-3">
            <CardTitle>Local GPU Processing Results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Statistics */}
            <StatsDisplay stats={processedData.stats} />
            
            {/* Visualization */}
            <div className="w-full h-96 border rounded-lg overflow-hidden">
              <div className="p-4 bg-blue-50 border-b">
                <div className="text-sm font-medium text-blue-700">
                  🚀 Local GPU Visualization (cuGraph Powered)
                </div>
                <div className="text-xs text-blue-600 mt-1">
                  Processing completed locally with {processedData.gpu_processed ? 'GPU' : 'CPU'} acceleration
                </div>
              </div>
              <div className="p-4">
                <iframe
                  src="http://localhost:8081"
                  className="w-full h-80"
                  title="Local GPU Visualization"
                  style={{ border: 'none' }}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
} 