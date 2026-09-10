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

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { useToast } from '@/hooks/use-toast'
import { 
  Zap, 
  Activity, 
  Cloud, 
  Cpu, 
  ExternalLink, 
  RefreshCw, 
  Settings,
  Download,
  Play,
  Pause
} from 'lucide-react'

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

interface RemoteGPUViewerProps {
  graphData: GraphData
  onError?: (error: Error) => void
  remoteServiceUrl?: string
}

interface RemoteRenderingSession {
  session_id: string
  embed_url: string
  gpu_processed: boolean
  stats: {
    node_count: number
    edge_count: number
    gpu_accelerated: boolean
    processing_time: number
    layout_computed: boolean
    clusters_computed: boolean
    centrality_computed: boolean
  }
  render_config: {
    quality: string
    interactive: boolean
    layout_algorithm: string
    clustering_algorithm: string
  }
  timestamp: string
}

export function RemoteGPUViewer({ graphData, onError, remoteServiceUrl = 'http://localhost:8082' }: RemoteGPUViewerProps) {
  const [isProcessing, setIsProcessing] = useState(false)
  const [session, setSession] = useState<RemoteRenderingSession | null>(null)
  const [serviceHealth, setServiceHealth] = useState<'unknown' | 'healthy' | 'error'>('unknown')
  
  // Rendering configuration
  const [layoutAlgorithm, setLayoutAlgorithm] = useState('force_atlas2')
  const [clusteringAlgorithm, setClusteringAlgorithm] = useState('leiden')
  const [renderQuality, setRenderQuality] = useState('high')
  const [computeCentrality, setComputeCentrality] = useState(true)
  const [interactiveMode, setInteractiveMode] = useState(true)
  
  // WebSocket connection for real-time updates
  const wsRef = useRef<WebSocket | null>(null)
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const { toast } = useToast()

  // Check service health on mount
  useEffect(() => {
    checkServiceHealth()
  }, [])

  // Setup WebSocket when session is created
  useEffect(() => {
    if (session?.session_id) {
      setupWebSocket(session.session_id)
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [session?.session_id])

  const checkServiceHealth = async () => {
    try {
      const response = await fetch(`${remoteServiceUrl}/api/health`)
      if (response.ok) {
        const health = await response.json()
        setServiceHealth('healthy')
        console.log('Remote GPU service health:', health)
      } else {
        setServiceHealth('error')
      }
    } catch (error) {
      console.error('Remote GPU service health check failed:', error)
      setServiceHealth('error')
    }
  }

  const setupWebSocket = (sessionId: string) => {
    try {
      const wsUrl = `${remoteServiceUrl.replace('http', 'ws')}/ws/${sessionId}`
      const ws = new WebSocket(wsUrl)
      
      ws.onopen = () => {
        console.log('WebSocket connected to remote rendering service')
      }
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          handleWebSocketMessage(message)
        } catch (error) {
          console.error('WebSocket message error:', error)
        }
      }
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }
      
      ws.onclose = () => {
        console.log('WebSocket connection closed')
      }
      
      wsRef.current = ws
    } catch (error) {
      console.error('WebSocket setup failed:', error)
    }
  }

  const handleWebSocketMessage = (message: any) => {
    if (message.type === 'parameter_update') {
      // Handle real-time parameter updates
      console.log('Parameter update received:', message.data)
      
      toast({
        title: "Visualization Updated",
        description: "Real-time parameter changes applied.",
      })
    }
  }

  // Detect when to recommend remote rendering based on your library capabilities
  const shouldUseRemoteRendering = useCallback((nodeCount: number, linkCount: number) => {
    // Enhanced detection based on your frontend capabilities
    const estimatedMemoryUsage = (nodeCount * 64) + (linkCount * 32); // bytes per element
    const maxWebGLNodes = typeof window !== 'undefined' && window.WebGL2RenderingContext ? 50000 : 10000;
    const maxWebGPUNodes = typeof window !== 'undefined' && 'gpu' in navigator ? 100000 : 25000;
    
    // Check available rendering capabilities
    const hasWebGPU = typeof window !== 'undefined' && 'gpu' in navigator;
    const hasWebGL2 = typeof window !== 'undefined' && window.WebGL2RenderingContext;
    
    // Memory considerations (Three.js geometry limits)
    const estimatedMemoryMB = estimatedMemoryUsage / (1024 * 1024);
    const maxClientMemory = hasWebGPU ? 512 : hasWebGL2 ? 256 : 128; // MB
    
    return {
      recommended: nodeCount > (hasWebGPU ? maxWebGPUNodes : maxWebGLNodes) || estimatedMemoryMB > maxClientMemory,
      reason: nodeCount > maxWebGLNodes 
        ? `${nodeCount.toLocaleString()} nodes exceed client rendering capacity`
        : estimatedMemoryMB > maxClientMemory 
        ? `Estimated ${estimatedMemoryMB.toFixed(1)}MB exceeds browser memory limits`
        : 'Client rendering suitable',
      capabilities: {
        webgpu: hasWebGPU,
        webgl2: hasWebGL2,
        maxNodes: hasWebGPU ? maxWebGPUNodes : maxWebGLNodes,
        estimatedMemory: estimatedMemoryMB
      }
    };
  }, []);

  // Enhanced library-aware configuration
  const optimizeConfigForLibraries = useCallback((nodeCount: number, config: any) => {
    // Optimize based on your 3d-force-graph and Three.js usage patterns
    const optimized = { ...config };
    
    // Three.js renderer optimizations
    if (nodeCount > 25000) {
      optimized.render_quality = 'high'; // Use instanced rendering
      optimized.enable_lod = true; // Level-of-detail for distant nodes
      optimized.max_texture_size = 2048; // Optimize for GPU memory
    } else if (nodeCount > 10000) {
      optimized.render_quality = 'medium';
      optimized.enable_lod = false;
      optimized.max_texture_size = 1024;
    }
    
    // D3.js force simulation optimizations 
    if (nodeCount > 50000) {
      optimized.physics_iterations = 100; // Reduced for large graphs
      optimized.alpha_decay = 0.05; // Faster convergence
      optimized.velocity_decay = 0.6; // More damping
    }
    
    // WebGL-specific optimizations
    optimized.webgl_features = {
      instance_rendering: nodeCount > 10000,
      texture_atlasing: nodeCount > 5000,
      frustum_culling: nodeCount > 15000,
      occlusion_culling: nodeCount > 25000
    };
    
    return optimized;
  }, []);

  const processGraphWithLibraryOptimization = useCallback(async () => {
    if (!graphData || serviceHealth !== 'healthy') return;
    
    const nodeCount = graphData.nodes?.length || 0;
    const linkCount = graphData.links?.length || 0;
    
    // Check if remote rendering is recommended
    const renderingAnalysis = shouldUseRemoteRendering(nodeCount, linkCount);
    
    if (!renderingAnalysis.recommended && nodeCount < 10000) {
      // Use local Three.js + D3.js rendering
      toast({
        title: "Local Rendering Recommended",
        description: `Local rendering optimal for ${nodeCount.toLocaleString()} nodes`,
      });
      return;
    }
    
    setIsProcessing(true);
    
    toast({
      title: "Remote GPU Processing",
      description: renderingAnalysis.reason,
    });
    
    try {
      // Optimize configuration for your library stack
      const optimizedConfig = optimizeConfigForLibraries(nodeCount, {
        layout_algorithm: layoutAlgorithm,
        clustering_algorithm: clusteringAlgorithm,
        render_quality: renderQuality,
        show_labels: nodeCount < 5000, // Labels only for smaller graphs
        animation_duration: Math.max(1000, Math.min(5000, nodeCount * 2)),
        background_color: '#000000', // Match your UI theme
        
        // Frontend library compatibility settings
        d3_version: "7.9.0", // Match your package.json
        threejs_version: "0.176.0",
        force_graph_version: "1.77.0",
        
        // Performance tuning based on your existing patterns
        webgl_optimization: true,
        gpu_memory_management: true,
        progressive_loading: nodeCount > 25000
      });
      
      // Process with remote GPU service
      const response = await fetch(`${remoteServiceUrl}/api/render`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          graph_data: graphData,
          config: optimizedConfig
        })
      });
      
      if (!response.ok) {
        throw new Error(`Rendering failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      setSession(result);
      
      toast({
        title: "Remote Processing Complete",
        description: `Graph with ${nodeCount.toLocaleString()} nodes processed successfully`,
      });
      
    } catch (error) {
      console.error('Remote rendering failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      toast({
        title: "Remote Rendering Failed",
        description: `Falling back to local rendering: ${errorMessage}`,
        variant: "destructive"
      });
      
      if (onError) {
        onError(error instanceof Error ? error : new Error(errorMessage));
      }
    } finally {
      setIsProcessing(false);
    }
  }, [graphData, serviceHealth, shouldUseRemoteRendering, optimizeConfigForLibraries, layoutAlgorithm, clusteringAlgorithm, renderQuality, remoteServiceUrl, toast, onError]);

  const processWithRemoteGPU = async () => {
    if (!graphData?.nodes?.length || !graphData?.links?.length) {
      toast({
        title: "No Graph Data",
        description: "Please ensure graph data is loaded before processing.",
        variant: "destructive"
      })
      return
    }

    // Check if graph is too large for browser-based rendering
    const nodeCount = graphData.nodes.length
    const shouldUseRemote = nodeCount > 1000 || renderQuality === 'ultra'

    if (!shouldUseRemote) {
      toast({
        title: "Consider Local Processing",
        description: `Graph has ${nodeCount} nodes. Remote rendering is optimized for larger graphs.`,
      })
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
        compute_centrality: computeCentrality,
        render_quality: renderQuality,
        interactive_mode: interactiveMode
      }

      const response = await fetch(`${remoteServiceUrl}/api/render`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      })

      if (!response.ok) {
        throw new Error(`Remote rendering failed: ${response.statusText}`)
      }

      const result = await response.json()
      setSession(result)
      setIsProcessing(false)

      toast({
        title: "Remote Rendering Complete",
        description: `Processed ${result.stats.node_count} nodes with ${result.gpu_processed ? 'GPU' : 'CPU'} acceleration in ${result.stats.processing_time.toFixed(2)}s.`,
      })

    } catch (error) {
      console.error('Remote GPU processing error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
      
      toast({
        title: "Remote Processing Failed",
        description: errorMessage,
        variant: "destructive"
      })
      
      setIsProcessing(false)
      
      if (onError) {
        onError(error instanceof Error ? error : new Error(errorMessage))
      }
    }
  }

  const updateRenderingParameters = async (updates: Partial<any>) => {
    if (!session?.session_id || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.warn('Cannot update parameters: no active session or WebSocket connection')
      return
    }

    try {
      wsRef.current.send(JSON.stringify({
        type: "update_params",
        ...updates
      }))
    } catch (error) {
      console.error('Failed to send parameter update:', error)
    }
  }

  const openInNewTab = () => {
    if (session?.embed_url) {
      const fullUrl = `${remoteServiceUrl}${session.embed_url}`
      window.open(fullUrl, '_blank')
    }
  }

  const refreshSession = async () => {
    if (!session?.session_id) return

    try {
      const response = await fetch(`${remoteServiceUrl}/api/session/${session.session_id}`)
      if (response.ok) {
        const sessionStatus = await response.json()
        console.log('Session status:', sessionStatus)
        
        toast({
          title: "Session Refreshed",
          description: `Session ${sessionStatus.session_id.substring(0, 8)}... is active`,
        })
      }
    } catch (error) {
      console.error('Failed to refresh session:', error)
    }
  }

  const ServiceStatus = () => (
    <div className="flex items-center gap-2 text-sm">
      <div className={`w-2 h-2 rounded-full ${
        serviceHealth === 'healthy' ? 'bg-green-500' : 
        serviceHealth === 'error' ? 'bg-red-500' : 'bg-yellow-500'
      }`} />
      <span>
        Remote GPU Service: {serviceHealth === 'healthy' ? 'Connected' : 
                           serviceHealth === 'error' ? 'Disconnected' : 'Checking...'}
      </span>
      {session && (
        <Badge variant="outline" className="ml-2">
          <Zap className="w-3 h-3 mr-1" />
          Session Active
        </Badge>
      )}
    </div>
  )

  const StatsDisplay = ({ stats }: { stats: RemoteRenderingSession['stats'] }) => (
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
      <div className="flex items-center gap-2">
        <Activity className="w-4 h-4 text-purple-500" />
        <span>{stats.processing_time.toFixed(2)}s processing</span>
      </div>
      {stats.layout_computed && (
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-orange-500" />
          <span>Layout computed</span>
        </div>
      )}
      {stats.clusters_computed && (
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-pink-500" />
          <span>Clusters detected</span>
        </div>
      )}
    </div>
  )

  const AutoRemoteCheck = () => {
    const nodeCount = graphData?.nodes?.length || 0
    const shouldAutoUseRemote = nodeCount > 10000
    
    if (shouldAutoUseRemote) {
      return (
        <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-center gap-2 text-blue-700 text-sm">
            <Cloud className="w-4 h-4" />
            <span className="font-medium">Large Graph Detected</span>
          </div>
          <div className="text-xs text-blue-600 mt-1">
            Graph has {nodeCount.toLocaleString()} nodes. Remote GPU rendering is recommended for optimal performance.
          </div>
        </div>
      )
    }
    return null
  }

  return (
    <div className="space-y-4">
      {/* Service Status */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Cloud className="w-5 h-5" />
              Remote GPU Rendering
            </span>
            <ServiceStatus />
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Auto remote check */}
          <AutoRemoteCheck />

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
                    <SelectItem value="force_atlas2">Force Atlas 2 (GPU)</SelectItem>
                    <SelectItem value="spectral">Spectral Layout (GPU)</SelectItem>
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
                    <SelectItem value="leiden">Leiden (GPU)</SelectItem>
                    <SelectItem value="louvain">Louvain (GPU)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Render Quality</label>
                <Select value={renderQuality} onValueChange={setRenderQuality}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="low">Low (Fast)</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                    <SelectItem value="ultra">Ultra (Million+ nodes)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Compute Centrality</label>
                  <Switch 
                    checked={computeCentrality} 
                    onCheckedChange={setComputeCentrality}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Interactive Mode</label>
                  <Switch 
                    checked={interactiveMode} 
                    onCheckedChange={setInteractiveMode}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <Button 
              onClick={processWithRemoteGPU} 
              disabled={isProcessing || serviceHealth !== 'healthy'}
              className="flex-1"
              variant="outline"
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  Basic Remote GPU
                </>
              )}
            </Button>
            
            <Button 
              onClick={processGraphWithLibraryOptimization} 
              disabled={isProcessing || serviceHealth !== 'healthy'}
              className="flex-1"
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Optimizing...
                </>
              ) : (
                <>
                  <Cpu className="w-4 h-4 mr-2" />
                  Library-Optimized GPU
                </>
              )}
            </Button>
            
            {session && (
              <Button variant="outline" onClick={refreshSession}>
                <RefreshCw className="w-4 h-4" />
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Visualization Results */}
      {session && (
        <Card className="flex-1">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle>Remote GPU Visualization</CardTitle>
              <div className="flex items-center gap-2">
                <Badge variant="outline">
                  Session: {session.session_id.substring(0, 8)}...
                </Badge>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={openInNewTab}
                >
                  <ExternalLink className="w-4 h-4 mr-1" />
                  Open
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Statistics */}
            <StatsDisplay stats={session.stats} />
            
            {/* Configuration Display */}
            <div className="p-3 bg-muted rounded-lg">
              <h4 className="font-medium mb-2">Render Configuration</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>Quality: {session.render_config.quality}</div>
                <div>Interactive: {session.render_config.interactive ? 'Yes' : 'No'}</div>
                <div>Layout: {session.render_config.layout_algorithm}</div>
                <div>Clustering: {session.render_config.clustering_algorithm}</div>
              </div>
            </div>
            
            {/* Iframe Visualization */}
            <div className="w-full h-96 border rounded-lg overflow-hidden">
              <div className="p-3 bg-green-50 border-b flex items-center justify-between">
                <div>
                  <div className="text-sm font-medium text-green-700">
                    🚀 Remote GPU Visualization
                  </div>
                  <div className="text-xs text-green-600">
                    Interactive {session.gpu_processed ? 'GPU' : 'CPU'}-accelerated visualization served remotely
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => updateRenderingParameters({ layout_algorithm: 'spectral' })}
                    disabled={!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN}
                  >
                    <Settings className="w-4 h-4" />
                  </Button>
                </div>
              </div>
              <iframe
                ref={iframeRef}
                src={`${remoteServiceUrl}${session.embed_url}`}
                className="w-full h-80"
                title="Remote GPU Visualization"
                style={{ border: 'none' }}
                allow="fullscreen"
              />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
} 