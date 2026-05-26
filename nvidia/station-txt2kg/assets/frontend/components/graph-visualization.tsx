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

import { useState, useEffect, useRef, useCallback } from "react"
import { FallbackGraph } from "./fallback-graph"
import { CuboidIcon as Cube, LayoutGrid } from "lucide-react"
import type { Triple } from "@/utils/text-processing"

interface GraphVisualizationProps {
  triples: Triple[]
  fullscreen?: boolean
  highlightedNodes?: string[]
  layoutType?: string
  initialMode?: '2d' | '3d'
}

export function GraphVisualization({ 
  triples, 
  fullscreen = false,
  highlightedNodes = [],
  layoutType = "force",
  initialMode = '2d'
}: GraphVisualizationProps) {
  // Default to 2D view unless explicitly set to 3D
  const [use3D, setUse3D] = useState(initialMode === '3d')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const loadTimerRef = useRef<NodeJS.Timeout | null>(null)
  
  // Handle 3D view errors that come from the iframe
  const handleIframeError = useCallback((event: MessageEvent) => {
    if (event.data && event.data.type === '3d-graph-error') {
      setError(event.data.message || 'Error loading 3D graph');
      setIsLoading(false);
    }
  }, []);
  
  // Handle 3D view in an iframe to completely isolate it from the main DOM
  useEffect(() => {
    if (use3D) {
      setIsLoading(true);
      setError(null);
      
      // Set a safety timeout in case the iframe never loads
      loadTimerRef.current = setTimeout(() => {
        setIsLoading(false);
      }, 10000); // 10 second timeout
      
      if (iframeRef.current) {
        // Create an event listener to know when the iframe is loaded
        const handleLoad = () => {
          if (loadTimerRef.current) {
            clearTimeout(loadTimerRef.current);
            loadTimerRef.current = null;
          }
          
          setTimeout(() => {
            setIsLoading(false);
          }, 2000);
        };
        
        // Add the event listener
        iframeRef.current.addEventListener('load', handleLoad);
        
        // Add message listener for error communication
        window.addEventListener('message', handleIframeError);
        
        try {
          // Get graph ID from URL if available
          const params = new URLSearchParams(window.location.search);
          const graphId = params.get("id");
          
          // Add highlighted nodes and layout type to the iframe parameters
          const highlightedNodesParam = highlightedNodes.length > 0 
            ? `&highlightedNodes=${encodeURIComponent(JSON.stringify(highlightedNodes))}` 
            : '';
          
          const timestamp = Date.now();
          const baseParams = `&fullscreen=${fullscreen}&layout=${layoutType}${highlightedNodesParam}&t=${timestamp}`;
          
          let iframeSrc = '';
          
          if (graphId) {
            // If we have a graph ID, we can just pass that
            iframeSrc = `/graph3d?id=${graphId}${baseParams}`;
          } else {
            // For large triples data, try to use stored database triples first
            const MAX_URL_TRIPLES = 100; // Maximum number of triples to include in URL
            
            if (triples.length > MAX_URL_TRIPLES) {
              console.log(`Large dataset detected (${triples.length} triples), attempting to use stored database triples`);
              
              // Try to store in database first, then use stored source
              fetch('/api/graph-db/triples', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                  triples: triples,
                  documentName: 'Graph Visualization Data'
                })
              }).then(response => {
                if (response.ok) {
                  console.log('Successfully stored triples in database, using stored source');
                  // Update iframe to use stored source
                  if (iframeRef.current) {
                    iframeRef.current.src = `/graph3d?source=stored${baseParams}`;
                  }
                } else {
                  console.warn('Failed to store in database, using localStorage fallback');
                  // Fallback to localStorage
                  const storageId = `graph_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
                  try {
                    localStorage.setItem(storageId, JSON.stringify(triples));
                    console.log(`Stored ${triples.length} triples in localStorage with ID: ${storageId}`);
                    if (iframeRef.current) {
                      iframeRef.current.src = `/graph3d?storageId=${storageId}${baseParams}`;
                    }
                  } catch (storageError) {
                    console.error("Both database and localStorage failed:", storageError);
                    console.warn(`Using limited triples (${MAX_URL_TRIPLES} of ${triples.length}) to avoid header size issues`);
                    const limitedTriples = triples.slice(0, MAX_URL_TRIPLES);
                    if (iframeRef.current) {
                      iframeRef.current.src = `/graph3d?triples=${encodeURIComponent(JSON.stringify(limitedTriples))}${baseParams}`;
                    }
                  }
                }
              }).catch(error => {
                console.error('Error storing triples in database:', error);
                // Fallback to localStorage
                const storageId = `graph_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
                try {
                  localStorage.setItem(storageId, JSON.stringify(triples));
                  console.log(`Stored ${triples.length} triples in localStorage with ID: ${storageId}`);
                  if (iframeRef.current) {
                    iframeRef.current.src = `/graph3d?storageId=${storageId}${baseParams}`;
                  }
                } catch (storageError) {
                  console.error("Both database and localStorage failed:", storageError);
                  console.warn(`Using limited triples (${MAX_URL_TRIPLES} of ${triples.length}) to avoid header size issues`);
                  const limitedTriples = triples.slice(0, MAX_URL_TRIPLES);
                  if (iframeRef.current) {
                    iframeRef.current.src = `/graph3d?triples=${encodeURIComponent(JSON.stringify(limitedTriples))}${baseParams}`;
                  }
                }
              });
              
              // Set initial iframe src to stored source (will be updated by the fetch above)
              iframeSrc = `/graph3d?source=stored${baseParams}`;
            } else {
              // For small data sets, just use the URL parameter approach
              iframeSrc = `/graph3d?triples=${encodeURIComponent(JSON.stringify(triples))}${baseParams}`;
            }
          }
          
          // Set the iframe source
          iframeRef.current.src = iframeSrc;
        } catch (err) {
          console.error("Error setting iframe source:", err);
          setError("Failed to prepare graph data for visualization");
          setIsLoading(false);
        }
        
        // Clean up
        return () => {
          if (loadTimerRef.current) {
            clearTimeout(loadTimerRef.current);
          }
          if (iframeRef.current) {
            iframeRef.current.removeEventListener('load', handleLoad);
          }
          window.removeEventListener('message', handleIframeError);
        };
      }
    }
  }, [use3D, triples, fullscreen, handleIframeError, highlightedNodes, layoutType]);
  
  // Handle switching to 2D view
  const switchTo2D = () => {
    setUse3D(false);
    setError(null);
  };
  
  // Handle switching to 3D view
  const switchTo3D = () => {
    setUse3D(true);
    setError(null);
  };
  
  return (
    <div className={`relative ${fullscreen ? "h-full" : "h-[500px]"}`}>
      {use3D ? (
        <div className="relative h-full w-full">
          <iframe
            ref={iframeRef}
            className="w-full h-full border-0"
            title="3D Graph Visualization"
            sandbox="allow-scripts allow-same-origin"
          />
          
          <button
            onClick={switchTo2D}
            className="absolute bottom-2 right-2 px-3 py-1.5 bg-black/70 hover:bg-black/90 text-white text-xs rounded-full flex items-center gap-1.5 z-20"
            disabled={isLoading}
          >
            <LayoutGrid className="h-3.5 w-3.5" />
            <span>2D View</span>
          </button>
          
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/70 z-10">
              <div className="flex flex-col items-center gap-3">
                <div className="animate-spin w-12 h-12 rounded-full border-t-2 border-l-2 border-primary border-r-transparent border-b-transparent"></div>
                <div className="text-primary font-medium">Loading 3D graph visualization...</div>
                <div className="text-xs text-gray-400">This may take a moment</div>
              </div>
            </div>
          )}
          
          {error && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-10">
              <div className="text-red-500 p-6 bg-black/90 rounded-lg max-w-md text-center">
                <p className="font-bold mb-3">Error loading 3D visualization</p>
                <p className="text-sm mb-4">{error}</p>
                <p className="text-xs mb-4 text-gray-400">Your browser may not support WebGL or 3D rendering.</p>
                <button
                  onClick={switchTo2D}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded"
                >
                  Switch to 2D View
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="relative h-full w-full">
          <FallbackGraph 
            triples={triples} 
            fullscreen={fullscreen}
            highlightedNodes={highlightedNodes}
          />
          
          <button
            onClick={switchTo3D}
            className="absolute bottom-2 right-2 px-3 py-1.5 bg-black/70 hover:bg-black/90 text-white text-xs rounded-full flex items-center gap-1.5"
          >
            <Cube className="h-3.5 w-3.5" />
            <span>3D View</span>
          </button>
        </div>
      )}
    </div>
  )
}

