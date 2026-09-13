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

// @ts-nocheck
import React, { useEffect, useState, useCallback } from "react"
import dynamic from "next/dynamic"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, Play, Square, RotateCcw, Zap, Cpu, Monitor, Maximize, Minimize } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

// Dynamically import components with SSR disabled
const WebGPU3DViewer = dynamic(
  () => import("@/components/webgpu-3d-viewer").then(mod => mod.WebGPU3DViewer),
  { ssr: false }
)

const ForceGraphWrapper = dynamic(
  () => import("@/components/force-graph-wrapper").then(mod => mod.ForceGraphWrapper),
  { ssr: false }
)

interface TestConfiguration {
  nodeCount: number
  linkDensity: number
  use3D: boolean
  enableClustering: boolean
  graphType: 'random' | 'scale-free' | 'small-world' | 'hierarchical' | 'clustered'
}

interface PerformanceMetrics {
  generationTime: number
  renderingTime: number
  clusteringTime?: number
  totalNodes: number
  totalLinks: number
  memoryUsage?: number
}

export default function TestWebGPUClusteringPage() {
  const [testConfig, setTestConfig] = useState<TestConfiguration>({
    nodeCount: 2000, // Smaller for better clustering
    linkDensity: 0.08, // Higher density for clustering (8.0%)
    use3D: true,
    enableClustering: true,
    graphType: 'clustered' // Optimized for clustering
  })

  const [graphData, setGraphData] = useState<any>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isRendering, setIsRendering] = useState(false)
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [debugInfo, setDebugInfo] = useState<string>("")
  const [isFullscreen, setIsFullscreen] = useState(false)

  const { toast } = useToast()

  // Generate test graph data
  const generateTestGraph = useCallback(async () => {
    setIsGenerating(true)
    setError(null)
    setDebugInfo("Generating large test graph...")
    
    const startTime = performance.now()
    
    try {
      const nodes = []
      const links = []
      
      // Generate nodes based on graph type
      for (let i = 0; i < testConfig.nodeCount; i++) {
        let x, y, z
        
        switch (testConfig.graphType) {
          case 'hierarchical':
            // Hierarchical layout with levels
            const level = Math.floor(Math.log2(i + 1))
            const angleStep = (2 * Math.PI) / Math.pow(2, level)
            const angle = (i - Math.pow(2, level) + 1) * angleStep
            const radius = level * 50
            x = Math.cos(angle) * radius + (Math.random() - 0.5) * 20
            y = Math.sin(angle) * radius + (Math.random() - 0.5) * 20
            z = level * 30 + (Math.random() - 0.5) * 10
            break
            
          case 'small-world':
            // Small world with tight clusters - optimized for clustering algorithms
            const clusterSize = 200 // Smaller, tighter clusters
            const clusterId = Math.floor(i / clusterSize)
            const clustersPerRow = Math.ceil(Math.sqrt(Math.ceil(testConfig.nodeCount / clusterSize)))
            const clusterX = (clusterId % clustersPerRow) * 300
            const clusterY = Math.floor(clusterId / clustersPerRow) * 300
            // Tight clustering with some spread
            x = clusterX + (Math.random() - 0.5) * 80
            y = clusterY + (Math.random() - 0.5) * 80
            z = (Math.random() - 0.5) * 40
            break
            
          case 'clustered':
            // Highly clustered graph optimized for clustering algorithms
            const totalClusters = 8 // Fixed number of clusters
            const nodesPerCluster = Math.ceil(testConfig.nodeCount / totalClusters)
            const clusterIdx = Math.floor(i / nodesPerCluster)
            
            // Position clusters in a grid
            const clustersInRow = Math.ceil(Math.sqrt(totalClusters))
            const clusterXPos = (clusterIdx % clustersInRow) * 400
            const clusterYPos = Math.floor(clusterIdx / clustersInRow) * 400
            
            // Very tight clustering with minimal spread
            x = clusterXPos + (Math.random() - 0.5) * 60
            y = clusterYPos + (Math.random() - 0.5) * 60
            z = (Math.random() - 0.5) * 30
            break
            
          case 'scale-free':
            // Scale-free network with power-law distribution
            const r = Math.pow(Math.random(), 0.5) * 300
            const theta = Math.random() * 2 * Math.PI
            const phi = Math.acos(2 * Math.random() - 1)
            x = r * Math.sin(phi) * Math.cos(theta)
            y = r * Math.sin(phi) * Math.sin(theta)
            z = r * Math.cos(phi)
            break
            
          default: // random
            x = (Math.random() - 0.5) * 1000
            y = (Math.random() - 0.5) * 1000
            z = (Math.random() - 0.5) * 1000
        }
        
        nodes.push({
          id: i.toString(),
          name: `Node ${i}`,
          x,
          y,
          z,
          val: Math.random() * 10 + 1,
          color: `hsl(${(i * 137.508) % 360}, 70%, 60%)`, // Golden angle for color distribution
          group: Math.floor(i / 1000).toString()
        })
      }
      
      // Generate links based on graph type and density
      const targetLinkCount = Math.floor(testConfig.nodeCount * testConfig.linkDensity)
      const linkSet = new Set<string>() // Prevent duplicate links
      
      for (let i = 0; i < targetLinkCount && linkSet.size < targetLinkCount; i++) {
        let sourceId, targetId
        
        switch (testConfig.graphType) {
          case 'hierarchical':
            // Connect nodes in hierarchical structure
            sourceId = Math.floor(Math.random() * testConfig.nodeCount)
            // Prefer connections to nearby hierarchy levels
            const level = Math.floor(Math.log2(sourceId + 1))
            const levelStart = Math.pow(2, level) - 1
            const levelEnd = Math.min(Math.pow(2, level + 1) - 1, testConfig.nodeCount - 1)
            targetId = levelStart + Math.floor(Math.random() * (levelEnd - levelStart + 1))
            break
            
          case 'small-world':
            // Connect within clusters and few random long-distance connections
            sourceId = Math.floor(Math.random() * testConfig.nodeCount)
            if (Math.random() < 0.9) { // 90% local connections for strong clustering
              // Local connection within cluster
              const clusterSize = 200
              const clusterId = Math.floor(sourceId / clusterSize)
              const clusterStart = clusterId * clusterSize
              const clusterEnd = Math.min(clusterStart + clusterSize, testConfig.nodeCount)
              targetId = clusterStart + Math.floor(Math.random() * (clusterEnd - clusterStart))
            } else {
              // Random long-distance connection (bridges between clusters)
              targetId = Math.floor(Math.random() * testConfig.nodeCount)
            }
            break
            
          case 'clustered':
            // Highly clustered connections - 95% within cluster, 5% between clusters
            sourceId = Math.floor(Math.random() * testConfig.nodeCount)
            if (Math.random() < 0.95) {
              // Connect within the same cluster
              const totalClusters = 8
              const nodesPerCluster = Math.ceil(testConfig.nodeCount / totalClusters)
              const sourceCluster = Math.floor(sourceId / nodesPerCluster)
              const clusterStart = sourceCluster * nodesPerCluster
              const clusterEnd = Math.min(clusterStart + nodesPerCluster, testConfig.nodeCount)
              targetId = clusterStart + Math.floor(Math.random() * (clusterEnd - clusterStart))
            } else {
              // Bridge connection to a different cluster
              targetId = Math.floor(Math.random() * testConfig.nodeCount)
            }
            break
            
          case 'scale-free':
            // Preferential attachment for scale-free network
            sourceId = Math.floor(Math.random() * testConfig.nodeCount)
            // Use power-law distribution for target selection
            targetId = Math.floor(Math.pow(Math.random(), 2) * testConfig.nodeCount)
            break
            
          default: // random
            sourceId = Math.floor(Math.random() * testConfig.nodeCount)
            targetId = Math.floor(Math.random() * testConfig.nodeCount)
        }
        
        // Ensure no self-loops and no duplicates
        if (sourceId !== targetId) {
          const linkKey = sourceId < targetId ? `${sourceId}-${targetId}` : `${targetId}-${sourceId}`
          if (!linkSet.has(linkKey)) {
            linkSet.add(linkKey)
            links.push({
              source: sourceId.toString(),
              target: targetId.toString(),
              name: `edge_${sourceId}_${targetId}`,
              strength: Math.random() * 0.5 + 0.5
            })
          }
        }
      }
      
      const generationTime = performance.now() - startTime
      
      const newGraphData = { nodes, links }
      setGraphData(newGraphData)
      
      const metrics: PerformanceMetrics = {
        generationTime,
        renderingTime: 0,
        totalNodes: nodes.length,
        totalLinks: links.length
      }
      
      setPerformanceMetrics(metrics)
      setDebugInfo(`Generated ${nodes.length} nodes and ${links.length} links in ${generationTime.toFixed(2)}ms`)
      
      toast({
        title: "Test Graph Generated",
        description: `Created ${nodes.length.toLocaleString()} nodes and ${links.length.toLocaleString()} links`,
      })
      
    } catch (error) {
      console.error('Graph generation failed:', error)
      setError(`Graph generation failed: ${error}`)
    } finally {
      setIsGenerating(false)
    }
  }, [testConfig, toast])

  // Update configuration handlers
  const updateNodeCount = useCallback((value: number[]) => {
    setTestConfig((prev: TestConfiguration) => ({ ...prev, nodeCount: value[0] }))
  }, [])

  const updateLinkDensity = useCallback((value: number[]) => {
    setTestConfig((prev: TestConfiguration) => ({ ...prev, linkDensity: value[0] }))
  }, [])

  const updateGraphType = useCallback((type: TestConfiguration['graphType']) => {
    setTestConfig((prev: TestConfiguration) => ({ ...prev, graphType: type }))
  }, [])

  // Calculate estimated memory usage
  const estimatedMemoryMB = Math.round((testConfig.nodeCount * 200 + testConfig.nodeCount * testConfig.linkDensity * 100) / 1024)

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Configuration Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            
            {/* Node Count Slider */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium">Node Count</label>
                <Badge variant="outline">{testConfig.nodeCount.toLocaleString()}</Badge>
              </div>
              <Slider
                value={[testConfig.nodeCount]}
                onValueChange={updateNodeCount}
                min={1000}
                max={500000}
                step={1000}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>1K</span>
                <span>500K</span>
              </div>
            </div>

            {/* Link Density Slider */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium">Link Density</label>
                <Badge variant="outline">{(testConfig.linkDensity * 100).toFixed(3)}%</Badge>
              </div>
              <Slider
                value={[testConfig.linkDensity]}
                onValueChange={updateLinkDensity}
                min={0.00001}
                max={0.2}
                step={0.00001}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>0.001%</span>
                <span>20%</span>
              </div>
            </div>

            {/* Graph Type Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Graph Type</label>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                {[
                  { id: 'clustered', name: 'Clustered' },
                  { id: 'small-world', name: 'Small-World' },
                  { id: 'scale-free', name: 'Scale-Free' },
                  { id: 'hierarchical', name: 'Hierarchical' },
                  { id: 'random', name: 'Random' }
                ].map((type) => (
                  <Button
                    key={type.id}
                    variant={testConfig.graphType === type.id ? "default" : "outline"}
                    size="sm"
                    onClick={() => updateGraphType(type.id as TestConfiguration['graphType'])}
                    className="p-2"
                  >
                    <span className="font-medium">{type.name}</span>
                  </Button>
                ))}
              </div>
            </div>

            {/* Options */}
            <div className="flex items-center justify-between">
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Switch
                    id="use3d"
                    checked={testConfig.use3D}
                    onCheckedChange={(checked: boolean) => setTestConfig((prev: TestConfiguration) => ({ ...prev, use3D: checked }))}
                  />
                  <label htmlFor="use3d" className="text-sm font-medium">3D Visualization</label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch
                    id="clustering"
                    checked={testConfig.enableClustering}
                    onCheckedChange={(checked: boolean) => setTestConfig((prev: TestConfiguration) => ({ ...prev, enableClustering: checked }))}
                  />
                  <label htmlFor="clustering" className="text-sm font-medium">GPU Clustering</label>
                </div>
              </div>

              <div className="text-right">
                <Badge variant={estimatedMemoryMB > 1000 ? "destructive" : "secondary"}>
                  ~{estimatedMemoryMB}MB
                </Badge>
              </div>
            </div>

            {/* Generate Button */}
            <Button
              onClick={generateTestGraph}
              disabled={isGenerating}
              size="lg"
              className="w-full"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating Test Graph...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Generate Graph
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Performance Metrics */}
        {performanceMetrics && (
          <Card>
            <CardContent className="pt-6">
              <div className="flex justify-center gap-4">
                <Badge variant="outline">
                  {performanceMetrics.totalNodes.toLocaleString()} nodes
                </Badge>
                <Badge variant="outline">
                  {performanceMetrics.totalLinks.toLocaleString()} links
                </Badge>
                <Badge variant="outline">
                  {performanceMetrics.generationTime.toFixed(0)}ms
                </Badge>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Visualization */}
        {graphData && (
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  {testConfig.use3D ? <Monitor className="h-5 w-5" /> : <Cpu className="h-5 w-5" />}
                  {testConfig.use3D ? '3D' : '2D'}
                </CardTitle>
                {testConfig.use3D && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsFullscreen(!isFullscreen)}
                    className="flex items-center gap-2"
                  >
                    {isFullscreen ? (
                      <>
                        <Minimize className="h-4 w-4" />
                        Exit Fullscreen
                      </>
                    ) : (
                      <>
                        <Maximize className="h-4 w-4" />
                        Fullscreen
                      </>
                    )}
                  </Button>
                )}
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className={`relative ${isFullscreen ? 'fixed inset-0 z-50 bg-black' : 'h-[600px]'}`}>
                {testConfig.use3D ? (
                  <WebGPU3DViewer
                    graphData={graphData}
                    remoteServiceUrl="http://localhost:8083"
                    onError={(err: string) => {
                      console.error("WebGPU 3D Viewer Error:", err)
                      setError(`WebGPU 3D Viewer: ${err}`)
                    }}
                  />
                ) : (
                  <ForceGraphWrapper
                    jsonData={graphData}
                    layoutType="force"
                    highlightedNodes={[]}
                    onError={(err: Error) => {
                      console.error("Force Graph Error:", err)
                      setError(`Force Graph: ${err.message}`)
                    }}
                  />
                )}
                {isFullscreen && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsFullscreen(false)}
                    className="absolute top-4 right-4 z-10 bg-black/80 text-white border-white/20 hover:bg-black/60"
                  >
                    <Minimize className="h-4 w-4 mr-2" />
                    Exit Fullscreen
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
