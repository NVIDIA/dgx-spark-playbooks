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

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Loader2, Play, Square, RotateCcw, Monitor, Wifi } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'
import { RemoteWebGPUClusteringClient, type ClusteringResult } from '@/utils/remote-webgpu-clustering'

interface WebRTCGraphViewerProps {
  graphData: {
    nodes: any[]
    links: any[]
  } | null
  remoteServiceUrl?: string
  autoRefresh?: boolean
  refreshInterval?: number
  onError?: (error: string) => void
}

interface StreamingStats {
  sessionId: string | null
  isStreaming: boolean
  lastFrameTime: Date | null
  frameCount: number
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error'
  processingTime: number | null
}

export function WebRTCGraphViewer({ 
  graphData, 
  remoteServiceUrl = 'http://localhost:8083',
  autoRefresh = true,
  refreshInterval = 1000,
  onError 
}: WebRTCGraphViewerProps) {
  const [client, setClient] = useState<RemoteWebGPUClusteringClient | null>(null)
  const [isInitializing, setIsInitializing] = useState(true)
  const [serviceAvailable, setServiceAvailable] = useState(false)
  const [capabilities, setCapabilities] = useState<any>(null)
  const [streamingStats, setStreamingStats] = useState<StreamingStats>({
    sessionId: null,
    isStreaming: false,
    lastFrameTime: null,
    frameCount: 0,
    connectionStatus: 'disconnected',
    processingTime: null
  })

  const imgRef = useRef<HTMLImageElement>(null)
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const { toast } = useToast()

  // Initialize remote client
  useEffect(() => {
    const initializeClient = async () => {
      try {
        const remoteClient = new RemoteWebGPUClusteringClient(remoteServiceUrl, false) // Disable proxy mode for WebSocket
        const available = await remoteClient.checkAvailability()
        
        if (available) {
          const caps = remoteClient.getCapabilities()
          setCapabilities(caps)
          setServiceAvailable(true)
          setClient(remoteClient)
          
          // Set up event listeners
          remoteClient.on('connected', () => {
            setStreamingStats(prev => ({ ...prev, connectionStatus: 'connected' }))
            toast({
              title: "Connected",
              description: "Connected to remote GPU service",
            })
          })
          
          remoteClient.on('disconnected', () => {
            setStreamingStats(prev => ({ ...prev, connectionStatus: 'disconnected' }))
          })
          
          remoteClient.on('error', (error: any) => {
            setStreamingStats(prev => ({ ...prev, connectionStatus: 'error' }))
            onError?.(`WebSocket error: ${error}`)
          })
          
          remoteClient.on('clusteringComplete', (result: ClusteringResult) => {
            setStreamingStats(prev => ({ 
              ...prev, 
              processingTime: result.processingTime 
            }))
          })
          
          // Connect WebSocket
          remoteClient.connectWebSocket()
          setStreamingStats(prev => ({ ...prev, connectionStatus: 'connecting' }))
          
        } else {
          setServiceAvailable(false)
          onError?.('Remote WebGPU service not available')
        }
      } catch (error) {
        console.error('Failed to initialize WebRTC client:', error)
        setServiceAvailable(false)
        onError?.(`Failed to connect: ${error}`)
      } finally {
        setIsInitializing(false)
      }
    }

    initializeClient()

    return () => {
      if (client) {
        client.dispose()
      }
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current)
      }
    }
  }, [remoteServiceUrl])

  // Start streaming
  const startStreaming = useCallback(async () => {
    if (!client || !graphData || !serviceAvailable) {
      toast({
        title: "Cannot Start Streaming",
        description: "Service not available or no graph data",
        variant: "destructive"
      })
      return
    }

    try {
      setStreamingStats(prev => ({ ...prev, isStreaming: true }))
      
      const sessionId = await client.startWebRTCStreaming(graphData.nodes, graphData.links)
      
      if (sessionId) {
        setStreamingStats(prev => ({ 
          ...prev, 
          sessionId,
          frameCount: 0,
          lastFrameTime: new Date()
        }))
        
        // Start frame refreshing
        if (autoRefresh) {
          startFrameRefresh(sessionId)
        } else {
          // Load initial frame
          loadFrame(sessionId)
        }
        
        toast({
          title: "Streaming Started",
          description: `WebRTC session ${sessionId.substring(0, 8)}... created`,
        })
      } else {
        throw new Error('Failed to create streaming session')
      }
    } catch (error) {
      console.error('Failed to start streaming:', error)
      setStreamingStats(prev => ({ ...prev, isStreaming: false }))
      onError?.(`Failed to start streaming: ${error}`)
    }
  }, [client, graphData, serviceAvailable, autoRefresh])

  // Stop streaming
  const stopStreaming = useCallback(async () => {
    if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current)
      refreshIntervalRef.current = null
    }

    if (client && streamingStats.sessionId) {
      try {
        await client.cleanupWebRTCSession(streamingStats.sessionId)
      } catch (error) {
        console.warn('Failed to cleanup session:', error)
      }
    }

    setStreamingStats(prev => ({
      ...prev,
      sessionId: null,
      isStreaming: false,
      lastFrameTime: null,
      frameCount: 0
    }))

    toast({
      title: "Streaming Stopped",
      description: "WebRTC session ended",
    })
  }, [client, streamingStats.sessionId])

  // Start frame refresh interval
  const startFrameRefresh = useCallback((sessionId: string) => {
    if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current)
    }

    refreshIntervalRef.current = setInterval(() => {
      loadFrame(sessionId)
    }, refreshInterval)
  }, [refreshInterval])

  // Load a single frame
  const loadFrame = useCallback((sessionId: string) => {
    if (!client || !imgRef.current) return

    const frameUrl = client.getStreamFrameUrl(sessionId)
    const img = imgRef.current

    // Add timestamp to prevent caching
    const urlWithTimestamp = `${frameUrl}?t=${Date.now()}`
    
    img.onload = () => {
      setStreamingStats(prev => ({
        ...prev,
        lastFrameTime: new Date(),
        frameCount: prev.frameCount + 1
      }))
    }

    img.onerror = () => {
      console.warn('Failed to load frame')
    }

    img.src = urlWithTimestamp
  }, [client])

  // Refresh current frame
  const refreshFrame = useCallback(() => {
    if (streamingStats.sessionId) {
      loadFrame(streamingStats.sessionId)
    }
  }, [streamingStats.sessionId, loadFrame])

  if (isInitializing) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Loader2 className="h-5 w-5 animate-spin" />
            Initializing WebRTC Viewer
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p>Connecting to remote GPU service...</p>
        </CardContent>
      </Card>
    )
  }

  if (!serviceAvailable) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-red-600">
            <Wifi className="h-5 w-5" />
            Service Unavailable
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertDescription>
              Remote WebGPU service is not available at {remoteServiceUrl}.
              Please ensure the service is running and accessible.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      {/* Service Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Monitor className="h-5 w-5" />
            WebRTC GPU Streaming
          </CardTitle>
          <CardDescription>
            Stream GPU-rendered visualizations from remote server
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-1">
              <p className="text-sm font-medium">Connection</p>
              <Badge variant={
                streamingStats.connectionStatus === 'connected' ? 'default' :
                streamingStats.connectionStatus === 'connecting' ? 'secondary' :
                streamingStats.connectionStatus === 'error' ? 'destructive' : 'outline'
              }>
                {streamingStats.connectionStatus}
              </Badge>
            </div>
            
            <div className="space-y-1">
              <p className="text-sm font-medium">GPU Available</p>
              <Badge variant={capabilities?.gpuAcceleration?.rapidsAvailable ? 'default' : 'secondary'}>
                {capabilities?.gpuAcceleration?.rapidsAvailable ? 'Yes' : 'CPU Only'}
              </Badge>
            </div>
            
            <div className="space-y-1">
              <p className="text-sm font-medium">Streaming</p>
              <Badge variant={streamingStats.isStreaming ? 'default' : 'outline'}>
                {streamingStats.isStreaming ? 'Active' : 'Inactive'}
              </Badge>
            </div>
            
            <div className="space-y-1">
              <p className="text-sm font-medium">Frame Count</p>
              <Badge variant="outline">
                {streamingStats.frameCount}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Controls */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-2">
            {!streamingStats.isStreaming ? (
              <Button
                onClick={startStreaming}
                disabled={!graphData?.nodes?.length || streamingStats.connectionStatus !== 'connected'}
                className="flex items-center gap-2"
              >
                <Play className="h-4 w-4" />
                Start Streaming
              </Button>
            ) : (
              <Button
                onClick={stopStreaming}
                variant="destructive"
                className="flex items-center gap-2"
              >
                <Square className="h-4 w-4" />
                Stop Streaming
              </Button>
            )}
            
            {streamingStats.isStreaming && !autoRefresh && (
              <Button
                onClick={refreshFrame}
                variant="outline"
                className="flex items-center gap-2"
              >
                <RotateCcw className="h-4 w-4" />
                Refresh Frame
              </Button>
            )}
          </div>
          
          {streamingStats.lastFrameTime && (
            <p className="text-sm text-gray-600 mt-2">
              Last frame: {streamingStats.lastFrameTime.toLocaleTimeString()}
              {streamingStats.processingTime && (
                <span className="ml-2">
                  (processed in {streamingStats.processingTime.toFixed(2)}s)
                </span>
              )}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Streamed Visualization */}
      {streamingStats.sessionId && (
        <Card>
          <CardHeader>
            <CardTitle>GPU-Rendered Visualization</CardTitle>
            <CardDescription>
              Session: {streamingStats.sessionId.substring(0, 8)}...
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative bg-gray-900 rounded-lg overflow-hidden">
              <img
                ref={imgRef}
                alt="GPU-rendered graph visualization"
                className="w-full h-auto max-h-[600px] object-contain"
                style={{ minHeight: '400px' }}
              />
              
              {streamingStats.isStreaming && autoRefresh && (
                <div className="absolute top-2 right-2">
                  <Badge variant="default" className="bg-green-600">
                    <div className="w-2 h-2 bg-green-300 rounded-full animate-pulse mr-2" />
                    Live
                  </Badge>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Capabilities Info */}
      {capabilities && (
        <Card>
          <CardHeader>
            <CardTitle>Service Capabilities</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <h4 className="font-medium mb-2">Available Modes:</h4>
                <ul className="space-y-1">
                  {Object.entries(capabilities.modes).map(([mode, info]: [string, any]) => (
                    <li key={mode} className="flex items-center gap-2">
                      <Badge variant={info.available ? 'default' : 'secondary'} className="text-xs">
                        {mode}
                      </Badge>
                      <span className="text-gray-600">{info.description}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div>
                <h4 className="font-medium mb-2">GPU Acceleration:</h4>
                <ul className="space-y-1">
                  <li className="flex items-center gap-2">
                    <Badge variant={capabilities?.gpuAcceleration?.rapidsAvailable ? 'default' : 'outline'} className="text-xs">
                      RAPIDS
                    </Badge>
                    <span className="text-gray-600">cuGraph/cuDF</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Badge variant={capabilities?.gpuAcceleration?.opencvAvailable ? 'default' : 'outline'} className="text-xs">
                      OpenCV
                    </Badge>
                    <span className="text-gray-600">Image processing</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Badge variant={capabilities?.gpuAcceleration?.plottingAvailable ? 'default' : 'outline'} className="text-xs">
                      Plotting
                    </Badge>
                    <span className="text-gray-600">Visualization</span>
                  </li>
                </ul>
              </div>
            </div>
            
            <div className="mt-4 pt-4 border-t">
              <p className="text-sm text-gray-600">
                Cluster dimensions: {capabilities?.clusterDimensions?.join(' × ') || 'N/A'} 
                ({capabilities?.maxClusterCount?.toLocaleString() || 'N/A'} total clusters)
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
