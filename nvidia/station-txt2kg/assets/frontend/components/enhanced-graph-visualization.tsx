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

import React, { useMemo, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Cpu, Eye } from 'lucide-react'
import { ForceGraphWrapper } from './force-graph-wrapper'
import type { Triple } from '@/utils/text-processing'

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

interface EnhancedGraphVisualizationProps {
  graphData?: GraphData
  jsonData?: any // For backward compatibility with existing ForceGraphWrapper
  triples?: Triple[]
  fullscreen?: boolean
  layoutType?: string
  highlightedNodes?: string[]
  onError?: (error: Error) => void
}

export function EnhancedGraphVisualization({
  graphData,
  jsonData,
  triples,
  fullscreen = false,
  layoutType,
  highlightedNodes,
  onError
}: EnhancedGraphVisualizationProps) {
  const [gpuPreferred, setGpuPreferred] = useState(false)
  
  // Convert triples to graph data format if needed
  const processedGraphData = React.useMemo(() => {
    if (graphData) {
      return graphData
    }
    
    if (triples && triples.length > 0) {
      const nodes = new Map<string, any>()
      const links: any[] = []
      
      triples.forEach((triple, index) => {
        // Triple interface has simple string properties
        const subjectId = triple.subject
        const subjectName = triple.subject
        const objectId = triple.object
        const objectName = triple.object
        const predicateName = triple.predicate
        
        // Add nodes
        if (!nodes.has(subjectId)) {
          nodes.set(subjectId, {
            id: subjectId,
            name: subjectName,
            group: 'entity'
          })
        }
        
        if (!nodes.has(objectId)) {
          nodes.set(objectId, {
            id: objectId,
            name: objectName,
            group: 'entity'
          })
        }
        
        // Add link
        links.push({
          source: subjectId,
          target: objectId,
          name: predicateName
        })
      })
      
      return {
        nodes: Array.from(nodes.values()),
        links: links
      }
    }
    
    return null
  }, [graphData, triples])

  const handleError = useCallback((error: Error) => {
    console.error('Graph visualization error:', error)
    if (onError) {
      onError(error)
    }
  }, [onError])

  const nodeCount = processedGraphData?.nodes?.length || jsonData?.nodes?.length || 0
  const linkCount = processedGraphData?.links?.length || jsonData?.links?.length || 0

  return (
    <div className="w-full h-full">
      <Card className="h-full">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Eye className="w-5 h-5" />
              Knowledge Graph Visualization
            </CardTitle>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span>{nodeCount} nodes</span>
                <span>•</span>
                <span>{linkCount} edges</span>
              </div>
              <div className="flex items-center space-x-2">
                <Badge variant="secondary" className="flex items-center gap-1">
                  <Cpu className="w-3 h-3" />
                  WebGPU Accelerated
                </Badge>
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-6 h-[calc(100%-80px)]">
          <div className="h-full rounded-lg border overflow-hidden">
            <ForceGraphWrapper
              jsonData={jsonData || {
                nodes: processedGraphData?.nodes || [],
                links: processedGraphData?.links || []
              }}
              fullscreen={fullscreen}
              layoutType={layoutType}
              highlightedNodes={highlightedNodes}
              onError={handleError}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 