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

import React, { useEffect, useRef, useState, useCallback } from "react"
import type { Triple } from "@/utils/text-processing"
import { Maximize2, Minimize2, Pause, Play, RefreshCw, ZoomIn, X, LayoutGrid } from "lucide-react"
import { WebGPUClusteringEngine } from "@/utils/webgpu-clustering"
import * as d3 from 'd3'
import * as THREE from 'three'

// Define interfaces for graph data
interface NodeObject {
  id: string
  name: string
  val?: number
  color?: string
  group?: string
  x?: number
  y?: number
  z?: number
}

interface LinkObject {
  id?: string  // Add id as optional property
  source: string | NodeObject
  target: string | NodeObject
  name: string
  color?: string
}

interface Connection {
  source: string;
  target: string;
  label?: string;
  nodeName?: string;
  type?: 'incoming' | 'outgoing';
}

interface PerformanceMetrics {
  renderingTime: number
  clusteringTime?: number
  totalNodes: number
  totalLinks: number
  memoryUsage?: number
}

interface ForceGraphWrapperProps {
  jsonData: any; // The graph data in JSON format
  fullscreen?: boolean
  layoutType?: string
  highlightedNodes?: string[]
  enableClustering?: boolean
  enableClusterColors?: boolean // Color nodes by cluster assignment
  clusteringMode?: 'local' | 'hybrid' | 'cpu' // Default clustering mode
  remoteServiceUrl?: string // URL for remote WebGPU service
  onClusteringUpdate?: (metrics: PerformanceMetrics) => void
  onError?: (error: Error) => void
  
  // Semantic clustering parameters
  clusteringMethod?: string // "spatial", "semantic", "hybrid"
  semanticAlgorithm?: string // "hierarchical", "kmeans", "dbscan"
  numberOfClusters?: number | null
  similarityThreshold?: number
  nameWeight?: number
  contentWeight?: number
  spatialWeight?: number
}

// Type definitions for Three.js objects
type ThreeNodeObject = {
  id: string
  name: string
  x?: number
  y?: number
  z?: number
  val?: number
  [key: string]: any
}

type ThreeLinkObject = {
  source: ThreeNodeObject | string
  target: ThreeNodeObject | string
  name: string
  [key: string]: any
}

// Add the fuzzyCompare function before the getLinkId function
const fuzzyCompare = (str1: string, str2: string): boolean => {
  if (!str1 || !str2) return false;
  
  // Convert both strings to lowercase and remove quotes, spaces, and special characters
  const normalize = (s: string) => s.toLowerCase().replace(/['"(){}[\]]/g, '').replace(/\s+/g, '');
  
  const norm1 = normalize(str1);
  const norm2 = normalize(str2);
  
  // Check exact match after normalization
  if (norm1 === norm2) return true;
  
  // Check if one contains the other
  if (norm1.includes(norm2) || norm2.includes(norm1)) return true;
  
  // Check for significant partial match (more than 70% of characters match)
  const minLength = Math.min(norm1.length, norm2.length);
  if (minLength > 3) {
    let matchCount = 0;
    for (let i = 0; i < minLength; i++) {
      if (norm1[i] === norm2[i]) matchCount++;
    }
    if (matchCount / minLength > 0.7) return true;
  }
  
  return false;
};

// Helper function to get a consistent link ID
const getLinkId = (link: any): string => {
  const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
  const targetId = typeof link.target === 'object' ? link.target.id : link.target;
  return `${sourceId}-${targetId}`;
};

// Generate cluster colors with a midnight Tokyo vibe - neon colors against dark backdrop
const generateClusterColors = (numClusters: number): string[] => {
  // Midnight Tokyo inspired color palette - neon lights, electric blues, hot pinks, cyber greens
  const tokyoColors = [
    '#FF0080', // Hot pink neon
    '#00FFFF', // Electric cyan
    '#FF4081', // Neon pink
    '#8A2BE2', // Electric purple
    '#00FF41', // Matrix green
    '#FF6B35', // Neon orange
    '#1E90FF', // Electric blue
    '#FF1493', // Deep pink
    '#00CED1', // Dark turquoise
    '#9932CC', // Dark orchid
    '#32CD32', // Lime green
    '#FF4500', // Orange red
    '#4169E1', // Royal blue
    '#DC143C', // Crimson
    '#00FA9A', // Medium spring green
    '#FF69B4', // Hot pink
    '#1E88E5', // Blue
    '#E91E63', // Pink
    '#00E676', // Green
    '#FF5722', // Deep orange
    '#673AB7', // Deep purple
    '#03DAC6', // Teal
    '#BB86FC', // Light purple
    '#CF6679'  // Light pink
  ];
  
  const colors: string[] = [];
  
  for (let i = 0; i < numClusters; i++) {
    if (i < tokyoColors.length) {
      // Use predefined Tokyo colors first
      colors.push(tokyoColors[i]);
    } else {
      // For additional clusters, generate variations of the base palette
      const baseColorIndex = i % tokyoColors.length;
      const baseColor = tokyoColors[baseColorIndex];
      
      // Convert hex to HSL and create variations
      const variation = Math.floor(i / tokyoColors.length);
      const hueShift = variation * 30; // Shift hue by 30 degrees for each cycle
      
      // Parse hex color and convert to HSL with variation
      const hex = baseColor.replace('#', '');
      const r = parseInt(hex.substr(0, 2), 16) / 255;
      const g = parseInt(hex.substr(2, 2), 16) / 255;
      const b = parseInt(hex.substr(4, 2), 16) / 255;
      
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      let h, s, l = (max + min) / 2;
      
      if (max === min) {
        h = s = 0; // achromatic
      } else {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
          case r: h = (g - b) / d + (g < b ? 6 : 0); break;
          case g: h = (b - r) / d + 2; break;
          case b: h = (r - g) / d + 4; break;
          default: h = 0;
        }
        h /= 6;
      }
      
      // Apply hue shift and maintain Tokyo neon characteristics
      h = ((h * 360 + hueShift) % 360) / 360;
      s = Math.max(0.7, s); // Keep high saturation for neon effect
      l = Math.min(0.7, Math.max(0.4, l)); // Bright but not too light
      
      // Convert back to HSL string
      colors.push(`hsl(${Math.round(h * 360)}, ${Math.round(s * 100)}%, ${Math.round(l * 100)}%)`);
    }
  }
  
  return colors;
};

// Assign cluster colors to nodes based on their actual cluster assignment
const assignClusterColors = (nodes: any[], enableColors: boolean, useSemanticClusters: boolean = false): any[] => {
  if (!enableColors || !nodes || nodes.length === 0) {
    return nodes;
  }
  
  // Check if nodes already have semantic cluster assignments
  const hasSemanticClusters = useSemanticClusters && nodes.some(node => node.clusterId !== undefined || node.clusterIndex !== undefined);
  
  console.log("🔍 assignClusterColors debug:", {
    enableColors,
    useSemanticClusters,
    nodeCount: nodes.length,
    hasSemanticClusters,
    sampleNodeIds: nodes.slice(0, 3).map(n => ({ 
      id: n.id, 
      clusterId: n.clusterId, 
      clusterIndex: n.clusterIndex 
    }))
  });
  
  if (hasSemanticClusters) {
    console.log("🎯 Using semantic cluster assignments for coloring");
    
    // Get unique cluster IDs
    const clusterIds = new Set<number>();
    nodes.forEach(node => {
      const clusterId = node.clusterId !== undefined ? node.clusterId : node.clusterIndex;
      if (clusterId !== undefined) {
        clusterIds.add(clusterId);
      }
    });
    
    const clusterColors = generateClusterColors(clusterIds.size);
    const clusterIdToIndex = Array.from(clusterIds).reduce((acc, id, index) => {
      acc[id] = index;
      return acc;
    }, {} as Record<number, number>);
    
    return nodes.map(node => ({
      ...node,
      color: (() => {
        const clusterId = node.clusterId !== undefined ? node.clusterId : node.clusterIndex;
        if (clusterId !== undefined && clusterIdToIndex[clusterId] !== undefined) {
          return clusterColors[clusterIdToIndex[clusterId]];
        }
        return node.color || '#76b900';
      })()
    }));
  }
  
  // Fallback to spatial clustering if no semantic clusters available
  console.log("🗺️ Using spatial clustering for coloring (fallback)");
  
  // Simple spatial clustering based on position
  const clusterGrid = 4; // 4x4x4 grid = 64 possible clusters
  const clusters = new Map<string, number>();
  let clusterCount = 0;
  
  // Find bounds
  const bounds = nodes.reduce((acc, node) => {
    const x = node.x || 0;
    const y = node.y || 0;
    const z = node.z || 0;
    return {
      minX: Math.min(acc.minX, x),
      maxX: Math.max(acc.maxX, x),
      minY: Math.min(acc.minY, y),
      maxY: Math.max(acc.maxY, y),
      minZ: Math.min(acc.minZ, z),
      maxZ: Math.max(acc.maxZ, z),
    };
  }, { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity });
  
  const rangeX = bounds.maxX - bounds.minX || 1;
  const rangeY = bounds.maxY - bounds.minY || 1;
  const rangeZ = bounds.maxZ - bounds.minZ || 1;
  
  // Assign cluster IDs based on spatial position
  nodes.forEach(node => {
    const x = node.x || 0;
    const y = node.y || 0;
    const z = node.z || 0;
    
    // Normalize to grid coordinates
    const gridX = Math.min(Math.floor(((x - bounds.minX) / rangeX) * clusterGrid), clusterGrid - 1);
    const gridY = Math.min(Math.floor(((y - bounds.minY) / rangeY) * clusterGrid), clusterGrid - 1);
    const gridZ = Math.min(Math.floor(((z - bounds.minZ) / rangeZ) * clusterGrid), clusterGrid - 1);
    
    const clusterKey = `${gridX},${gridY},${gridZ}`;
    
    if (!clusters.has(clusterKey)) {
      clusters.set(clusterKey, clusterCount++);
    }
    
    node.clusterIndex = clusters.get(clusterKey);
  });
  
  // Generate colors for all clusters
  const clusterColors = generateClusterColors(clusterCount);
  
  // Apply colors to nodes
  return nodes.map(node => ({
    ...node,
    color: node.clusterIndex !== undefined ? clusterColors[node.clusterIndex] : node.color
  }));
};

export function ForceGraphWrapper({ 
  jsonData, 
  fullscreen = false, 
  layoutType, 
  highlightedNodes, 
  enableClustering = false, 
  enableClusterColors = false, 
  clusteringMode = 'hybrid', 
  remoteServiceUrl = 'http://localhost:8083', 
  onClusteringUpdate, 
  onError,
  // Semantic clustering parameters
  clusteringMethod = "hybrid",
  semanticAlgorithm = "hierarchical",
  numberOfClusters = null,
  similarityThreshold = 0.7,
  nameWeight = 0.6,
  contentWeight = 0.3,
  spatialWeight = 0.1
}: ForceGraphWrapperProps) {
  // Check for null or invalid jsonData early and report error
  if (!jsonData || typeof jsonData !== 'object') {
    console.error("Invalid jsonData provided to ForceGraphWrapper:", jsonData);
    if (onError) {
      onError(new Error("Cannot read properties of null (reading 'nodes')"));
    }
    return (
      <div className="h-full w-full flex items-center justify-center bg-black/70">
        <div className="text-red-500 max-w-md p-6 bg-black/90 rounded-lg">
          <p className="font-bold mb-2">Error: Invalid graph data</p>
          <p className="text-sm">The graph data is missing or has an invalid format.</p>
        </div>
      </div>
    );
  }
  
  const containerRef = useRef<HTMLDivElement>(null)
  const graphRef = useRef<any>(null)
  const [isFullscreen, setIsFullscreen] = useState(fullscreen)
  const [isLoading, setIsLoading] = useState(true)
  const [loadingStep, setLoadingStep] = useState<string>("Initializing...")
  const [loadingProgress, setLoadingProgress] = useState<number>(0)
  const [graphLoaded, setGraphLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isPaused, setIsPaused] = useState(false)
  const [debugInfo, setDebugInfo] = useState<string>("")
  const [selectedNode, setSelectedNode] = useState<NodeObject | null>(null)
  const [nodeConnections, setNodeConnections] = useState<Connection[]>([])
  
  // Add interaction mode state to toggle between navigation and selection
  const [interactionMode, setInteractionMode] = useState<'navigation' | 'selection'>('navigation')
  
  // Track notifications
  const [notification, setNotification] = useState<{message: string, type: 'success' | 'error' | 'info'} | null>(null)
  const notificationTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  
  // Track highlighted nodes for visual emphasis
  const [internalHighlightedNodes, setInternalHighlightedNodes] = useState<Set<string>>(new Set())
  const [highlightLinks, setHighlightLinks] = useState<Set<string>>(new Set())
  
  // Track graph data statistics
  const [graphStats, setGraphStats] = useState<{nodes: number, links: number}>({nodes: 0, links: 0})
  
  // Retry mechanism
  const [retryCount, setRetryCount] = useState(0)
  const maxRetries = 3

  // Track graph data
  const [graphData, setGraphData] = useState<any>(null)

  // State for tracking hover
  const [hoveredNode, setHoveredNode] = useState<any>(null);
  // Use a ref for hover to prevent recursive state updates
  const hoveredNodeRef = useRef<any>(null);

  // Add state to track initialization
  const [isInitialized, setIsInitialized] = useState(false);

  // Add WebGPU clustering engine ref
  const clusteringEngineRef = useRef<WebGPUClusteringEngine | null>(null);
  // Track if WebGPU clustering is available
  const [isClusteringAvailable, setIsClusteringAvailable] = useState<boolean>(false);
  // Track if clustering is enabled
  const [isClusteringEnabled, setIsClusteringEnabled] = useState<boolean>(false);

  // Helper function to extract node ID reliably
  const getNodeId = (nodeObj: any): string => {
    if (!nodeObj) return '';
    
    // If it's a string, return it directly
    if (typeof nodeObj === 'string') return nodeObj;
    
    // If it's an object, try various ID properties
    if (nodeObj.id) return nodeObj.id;
    if (nodeObj.name) return nodeObj.name;
    if (nodeObj.key) return nodeObj.key;
    if (nodeObj.label) return nodeObj.label;
    
    // Fallback to string representation
    return String(nodeObj);
  };

  // Add state to track if we're using CPU fallback
  const [usingCpuFallback, setUsingCpuFallback] = useState(false);

  // Add state for node size control
  const [nodeSize, setNodeSize] = useState(5);
  
  // Add performance mode toggle
  const [performanceMode, setPerformanceMode] = useState(false);

  // Function to show a temporary notification
  const showNotification = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    // Clear any existing timeouts
    if (notificationTimeoutRef.current) {
      clearTimeout(notificationTimeoutRef.current);
    }
    
    // Set new notification
    setNotification({message, type});
    
    // Auto-clear after duration
    notificationTimeoutRef.current = setTimeout(() => {
      setNotification(null);
    }, 3000);
  };
  
  // Clean up notification timeout on unmount
  useEffect(() => {
    return () => {
      if (notificationTimeoutRef.current) {
        clearTimeout(notificationTimeoutRef.current);
      }
    };
  }, []);

  // Toggle interaction mode
  const toggleInteractionMode = () => {
    const newMode = interactionMode === 'navigation' ? 'selection' : 'navigation';
    setInteractionMode(newMode);
    showNotification(`Mode changed to: ${newMode}`, 'info');
    console.log(`Interaction mode changed to: ${newMode}`);
  };

  // More robust ID normalization function 
  const normalizeNodeId = (id: any): string => {
    // Handle null/undefined cases
    if (id === null || id === undefined) return '';
    
    // Handle ThreeJS object references that might be passed directly
    if (typeof id === 'object') {
      console.log('Received object for ID normalization:', id);
      // Try __threeObj property which might contain the ThreeJS object
      if (id.__threeObj) {
        console.log('Found __threeObj property');
        // Look for userData which often contains the original node data
        if (id.__threeObj.userData && id.__threeObj.userData.id) {
          console.log(`Using __threeObj.userData.id: "${id.__threeObj.userData.id}"`);
          id = id.__threeObj.userData.id;
        } else {
          // Fall back to the object's id property if it exists
          console.log(`Using object's id property: "${id.id}"`);
          id = id.id || '';
        }
      } else if (id.id) {
        // Simple object with id property
        console.log(`Using simple object id property: "${id.id}"`);
        id = id.id;
      } else {
        // Last resort - try toString or convert to empty string
        console.log('Could not find id property, using toString()');
        id = id.toString() || '';
      }
    }
    
    // Convert to string if not already
    const strId = String(id);
    
    // Log the original ID for debugging
    console.log(`Normalizing ID: "${strId}" (type: ${typeof id})`);
    
    // Remove all quotes, parentheses, and trim whitespace
    const normalized = strId.replace(/['"()]/g, '').trim();
    
    console.log(`  → Normalized to: "${normalized}"`);
    return normalized;
  };

  // Debug node connections with additional logging
  const debugNodeConnections = (nodeId: string) => {
    if (!graphData) {
      console.warn("Cannot debug connections: No graph data available");
      return { outgoing: [], incoming: [], total: 0 };
    }
    
    console.log(`Debugging connections for node: "${nodeId}"`);
    
    // More thorough logging of all nodes and links
    console.log("All nodes:", graphData.nodes.map((n: any) => ({ 
      id: n.id, 
      name: n.name
    })));
    
    // Log links with more details
    console.log("All links:", graphData.links.map((l: any) => {
      // Extract source and target properly, handling object references
      const sourceId = typeof l.source === 'object' ? (l.source.__threeObj ? l.source.__threeObj.userData.id : (l.source.id || l.source)) : l.source;
      const targetId = typeof l.target === 'object' ? (l.target.__threeObj ? l.target.__threeObj.userData.id : (l.target.id || l.target)) : l.target;
      
      return { 
        source: sourceId,
        target: targetId,
        name: l.name,
        // Debug object types
        sourceType: typeof l.source,
        targetType: typeof l.target,
      };
    }));
    
    const connections = {
      outgoing: [] as any[],
      incoming: [] as any[],
      total: 0
    };
    
    console.log(`Looking for connections with node ID: "${nodeId}"`);
    
    // Helper function to extract ID reliably from either string or object reference
    const getReliableId = (idOrObj: any): string => {
      if (typeof idOrObj === 'string') return idOrObj;
      
      // Handle ThreeJS object references
      if (idOrObj && idOrObj.__threeObj && idOrObj.__threeObj.userData) {
        return idOrObj.__threeObj.userData.id || '';
      }
      
      // Handle regular objects
      return idOrObj && idOrObj.id ? idOrObj.id : (idOrObj || '').toString();
    };
    
    // Additional helper to normalize IDs for comparison
    const normalizeForComparison = (id: string): string => {
      return id.toString().toLowerCase().trim();
    };
    
    // For reliable comparison
    const normalizedNodeId = normalizeForComparison(nodeId);
    
    // Check if the graph data links array exists
    if (!graphData.links || !Array.isArray(graphData.links)) {
      console.warn("No links array found in graph data");
      return { outgoing: [], incoming: [], total: 0 };
    }
    
    graphData.links.forEach((link: any, index: number) => {
      try {
        // Get source and target IDs, handling all possible formats
        const sourceId = getReliableId(link.source);
        const targetId = getReliableId(link.target);
        
        // Also get names for additional matching
        const sourceName = typeof link.source === 'object' ? (link.source.name || '') : '';
        const targetName = typeof link.target === 'object' ? (link.target.name || '') : '';
        
        console.log(`Link ${index}: "${sourceId}" → "${targetId}"`);
        console.log(`  Source reference: ${typeof link.source} | Target reference: ${typeof link.target}`);
        
        // Normalized versions for comparison
        const normalizedSourceId = normalizeForComparison(sourceId);
        const normalizedTargetId = normalizeForComparison(targetId);
        const normalizedSourceName = sourceName ? normalizeForComparison(sourceName) : '';
        const normalizedTargetName = targetName ? normalizeForComparison(targetName) : '';
        
        // Try different ways of comparing
        const sourceMatch = 
          normalizedSourceId === normalizedNodeId || 
          normalizedSourceName === normalizedNodeId;
        
        const targetMatch = 
          normalizedTargetId === normalizedNodeId || 
          normalizedTargetName === normalizedNodeId;
        
        if (sourceMatch) {
          console.log(`  ✅ SOURCE MATCH! Node is source in this link`);
          connections.outgoing.push({
            target: targetId,
            predicate: link.name,
            link
          });
        }
        
        if (targetMatch) {
          console.log(`  ✅ TARGET MATCH! Node is target in this link`);
          connections.incoming.push({
            source: sourceId,
            predicate: link.name,
            link
          });
        }
        
        if (!sourceMatch && !targetMatch) {
          console.log(`  ❌ No match`);
        }
      } catch (error) {
        console.error(`Error processing link ${index}:`, error);
      }
    });
    
    connections.total = connections.outgoing.length + connections.incoming.length;
    
    console.log(`Total connections found: ${connections.total}`, connections);
    return connections;
  };

  // Helper function to normalize text by removing quotes and parentheses
  const normalizeText = (text: string | undefined): string => {
    if (!text) return '';
    return text.replace(/['"()]/g, '').trim();
  };

  // Process the JSON data into the format needed for the graph
  const processGraphData = async (data: any, applyClusteringFirst: boolean = false) => {
    console.log("processGraphData called with input:", {
      hasData: !!data,
      isObject: typeof data === 'object' && data !== null,
      hasNodes: data && 'nodes' in data,
      hasLinks: data && 'links' in data,
      dataType: typeof data,
      keysIfObject: data && typeof data === 'object' ? Object.keys(data) : [],
      applyClusteringFirst
    });

    // Ensure data exists and has required properties
    if (!data || typeof data !== 'object' || data === null) {
      console.error("Invalid graph data: not an object or null");
      return null;
    }

    // Check if we need to adapt the data format
    if (!Array.isArray(data.nodes) || !Array.isArray(data.links)) {
      // If data doesn't have nodes/links arrays directly, try to extract from a different format
      console.log("Data doesn't have expected nodes/links format, attempting to adapt...");
      
      // Check if the data might be in a nested format (e.g., from the API response)
      if (data.triples && Array.isArray(data.triples)) {
        console.log("Found triples array, converting to nodes/links format");
        return convertTriplesToGraphFormat(data.triples, data.documentName);
      }
      
      console.error("Could not adapt data to required format", data);
      return null;
    }

    // Check if we should apply clustering before rendering (for large datasets)
    if (applyClusteringFirst && data.nodes.length > 10000 && clusteringEngineRef.current) {
      console.log(`🎯 Large dataset detected (${data.nodes.length} nodes), applying clustering before rendering...`);
      
      try {
        // Use the remote clustering service to get subsampled data
        const success = await clusteringEngineRef.current.updateNodePositions(
          data.nodes,
          data.links || []
        );
        
        if (success) {
          // Get the clustered/subsampled nodes from the engine
          const clusteredData = clusteringEngineRef.current.getClusteredData();
          if (clusteredData && clusteredData.nodes) {
            console.log(`✅ Pre-clustering successful: ${data.nodes.length} → ${clusteredData.nodes.length} nodes`);
            
            // Use the subsampled data instead of original
            data = {
              nodes: clusteredData.nodes,
              links: data.links || [] // Keep original links for now
            };
          }
        }
      } catch (error) {
        console.error("Pre-clustering failed, using original data:", error);
      }
    }

    // Return processed data with normalized node names and IDs
    const processed = {
      nodes: data.nodes.map((node: any) => ({
        ...node,
        // Ensure node has all required properties and normalize the ID and name
        id: normalizeText(node.id) || `node-${Math.random().toString(36).substring(2, 9)}`,
        name: normalizeText(node.name || node.id) || "Unnamed",
        group: node.group || "default"
      })),
      links: data.links.map((link: any) => ({
        ...link,
        // Ensure link has all required properties
        id: link.id || `link-${Math.random().toString(36).substring(2, 9)}`,
        name: link.name || "related",
        source: link.source,
        target: link.target
      }))
    };
    
    console.log("Processed graph data:", {
      nodeCount: processed.nodes.length,
      linkCount: processed.links.length,
      firstNode: processed.nodes.length > 0 ? processed.nodes[0] : null,
      firstLink: processed.links.length > 0 ? processed.links[0] : null
    });
    
    return processed;
  };

  // Helper function to convert triples to graph format
  const convertTriplesToGraphFormat = (triples: any[], documentName: string = "Unnamed Document") => {
    console.log("Converting triples to graph format...");
    
    const nodes = new Map<string, NodeObject>();
    const links: LinkObject[] = [];
    
    // Process each triple into nodes and links
    triples.forEach((triple, index) => {
      if (!triple.subject || !triple.predicate || !triple.object) {
        console.warn("Invalid triple format at index", index, triple);
        return;
      }
      
      // Handle both complex objects and simple string formats
      let subjectId = typeof triple.subject === 'string' ? triple.subject : triple.subject.id;
      let subjectName = typeof triple.subject === 'string' ? triple.subject : (triple.subject.value || triple.subject.id);
      
      const predicateId = typeof triple.predicate === 'string' ? triple.predicate : triple.predicate.id;
      const predicateName = typeof triple.predicate === 'string' ? triple.predicate : (triple.predicate.value || triple.predicate.id);
      
      let objectId = typeof triple.object === 'string' ? triple.object : triple.object.id;
      let objectName = typeof triple.object === 'string' ? triple.object : (triple.object.value || triple.object.id);
      
      // Normalize IDs and names to remove quotes and parentheses
      subjectId = normalizeText(subjectId);
      subjectName = normalizeText(subjectName);
      objectId = normalizeText(objectId);
      objectName = normalizeText(objectName);
      
      // Add subject node if it doesn't exist
      if (!nodes.has(subjectId)) {
        nodes.set(subjectId, {
          id: subjectId,
          name: subjectName,
          group: "concept"
        });
      }
      
      // Add object node if it doesn't exist
      if (!nodes.has(objectId)) {
        nodes.set(objectId, {
          id: objectId,
          name: objectName,
          group: "concept"
        });
      }
      
      // Create the link between subject and object
      links.push({
        id: `link-${subjectId}-${objectId}-${index}`,
        source: subjectId,
        target: objectId,
        name: predicateName
      });
    });
    
    // Convert nodes map to array
    const result = {
      nodes: Array.from(nodes.values()),
      links
    };
    
    console.log("Converted graph data:", {
      nodeCount: result.nodes.length,
      linkCount: result.links.length
    });
    
    return result;
  };

  // Initialize the graph
  useEffect(() => {
    if (!containerRef.current) return;
    
    console.log("Starting graph initialization...");
    
    // Flag to track if component is mounted
    let mounted = true;
    
    const initializeGraph = async () => {
      try {
        setIsLoading(true);
        setLoadingStep('Initializing 3D engine');
        
        if (typeof window === 'undefined') {
          console.error("Cannot initialize 3D graph in non-browser environment");
          setError("Browser environment required for 3D visualization");
          setIsLoading(false);
          return;
        }
        
        // Import ForceGraph3D dynamically to avoid SSR issues
        let ForceGraph3D;
        try {
          ForceGraph3D = (await import('3d-force-graph')).default;
          console.log("ForceGraph3D library loaded successfully");
        } catch (importError) {
          console.error("Failed to import ForceGraph3D:", importError);
          setError(`Failed to load 3D visualization library: ${importError instanceof Error ? importError.message : String(importError)}`);
          setIsLoading(false);
          return;
        }
        
        if (!ForceGraph3D) {
          throw new Error("Failed to load ForceGraph3D library - it's undefined after import");
        }
        
        try {
          // Create the graph instance using the same pattern as before
          // @ts-ignore - Calling function directly, letting JS handle it
          const Graph = ForceGraph3D({
            rendererConfig: {
              antialias: true,
              alpha: true,
              powerPreference: 'high-performance',
              precision: 'highp', // High precision for better quality
              depth: true // Enable depth testing for better 3D rendering
            }
          })(containerRef.current);
          
          if (!Graph) {
            throw new Error("Failed to create graph instance");
          }
          
          // Store the graph reference
          graphRef.current = Graph;
          
          console.log("3D Graph initialized successfully");
          
          // Enhanced GPU-accelerated setup
          Graph
            .backgroundColor("#000000")
            .nodeRelSize(5)
            .nodeResolution(32) // Higher resolution for smoother nodes
            .nodeOpacity(0.8)
            .linkOpacity(0.2)
            .linkWidth(1)
            .showNavInfo(false)
            .onBackgroundClick(() => {
              if (selectedNode) {
                clearSelection();
              }
            });
            
          // Setup safe hover handling to prevent recursion
          Graph.onNodeHover((node: any) => {
            // Only update if the hovered node has changed
            if (node !== hoveredNodeRef.current) {
              hoveredNodeRef.current = node;
              setHoveredNode(node);
              
              // Update cursor based on hover state without triggering a re-render
              if (containerRef.current) {
                containerRef.current.style.cursor = node ? 'pointer' : 'default';
              }
            }
          });
          
          // Set up click handling with debouncing
          let lastClickTime = 0;
          Graph.onNodeClick((node: any) => {
            const now = Date.now();
            if (now - lastClickTime < 300) return; // Debounce clicks
            lastClickTime = now;
            
            console.log("Node click detected", node);
            handleNodeSelection(node);
          });
          
          // Ready for data loading
          setLoadingStep('Ready to load graph data');
          setDebugInfo("3D Graph initialized and ready to load data");
          setIsInitialized(true); // Mark initialization as complete
          
          // Force immediate data loading if data is available
          if (jsonData) {
            console.log("Data is available at initialization time, triggering immediate load");
            // Use a small timeout to ensure state is updated
            setTimeout(async () => {
              try {
                // Check if we should pre-cluster large datasets
                const shouldPreCluster = jsonData?.nodes?.length > 10000 && isClusteringEnabled;
                const processedData = await processGraphData(jsonData, shouldPreCluster);
                if (processedData && graphRef.current) {
                  console.log("Applying data directly after initialization");
                  graphRef.current.graphData(processedData);
                  setGraphData(processedData);
                  setGraphStats({
                    nodes: processedData.nodes.length,
                    links: processedData.links.length
                  });
                  
                  // Zoom to fit
                  setTimeout(() => {
                    if (graphRef.current) {
                      graphRef.current.zoomToFit(800, 30);
                      setIsLoading(false);
                    }
                  }, 500);
                }
              } catch (err) {
                console.error("Error in immediate data loading:", err);
              }
            }, 100);
          }
        } catch (graphError) {
          console.error("Error initializing graph instance:", graphError);
          setError(`Failed to initialize 3D graph: ${graphError instanceof Error ? graphError.message : String(graphError)}`);
          setIsLoading(false);
        }
      } catch (error) {
        console.error('Error in initialization process:', error);
        setError(`Initialization error: ${error instanceof Error ? error.message : String(error)}`);
        setIsLoading(false);
      }
    };
    
    initializeGraph();
    
    // Cleanup function
    return () => {
      mounted = false;
      
      if (graphRef.current) {
        try {
          // Clean up the graph instance
          graphRef.current._destructor?.();
        } catch (err) {
          console.warn("Error during cleanup:", err);
        }
      }
    };
  }, [jsonData]); // Add jsonData as dependency

  // Effect for loading data after graph initialization
  useEffect(() => {
    // Current graph ref value for closure
    const currentGraphRef = graphRef.current;
    
    if (!currentGraphRef || !jsonData || isLoading || !isInitialized) {
      console.log("Data loading effect - early return:", {
        graphRefExists: !!currentGraphRef,
        jsonDataExists: !!jsonData,
        isCurrentlyLoading: isLoading,
        isInitialized: isInitialized
      });
      return;
    }
    
    console.log("Starting data loading process", { 
      jsonDataSize: JSON.stringify(jsonData).length,
      jsonDataSample: JSON.stringify(jsonData).substring(0, 200) + '...'
    });
    
    const loadGraphData = async () => {
      try {
        setIsLoading(true);
        setLoadingStep('Processing data');
        console.log("Processing graph data...");
        
        // Process the graph data
        // Check if we should pre-cluster large datasets
        const shouldPreCluster = jsonData?.nodes?.length > 10000 && isClusteringEnabled;
        let processedData = await processGraphData(jsonData, shouldPreCluster);
        
        if (!processedData) {
          console.error("processGraphData returned null");
          throw new Error("Failed to process graph data");
        }

        // Apply cluster coloring if enabled
        if (enableClusterColors && processedData.nodes) {
          console.log("🎨 Applying cluster colors to", processedData.nodes.length, "nodes");
          processedData.nodes = assignClusterColors(processedData.nodes, enableClusterColors, isClusteringEnabled);
        }
        
        console.log("Data processed successfully", {
          nodeCount: processedData.nodes.length,
          linkCount: processedData.links.length,
          sampleNode: processedData.nodes.length > 0 ? processedData.nodes[0] : null
        });
        
        // Store the processed data for reference
        setGraphData(processedData);
        
        // Update graph stats
        setGraphStats({
          nodes: processedData.nodes.length,
          links: processedData.links.length
        });
        
        setLoadingStep('Applying data to graph');
        console.log("Applying data to graph...");
        
        // Safety check - use the captured reference
        if (!currentGraphRef) {
          throw new Error("Graph reference lost during data loading");
        }
        
        // Apply data to graph with a try/catch
        try {
          console.log("Calling graphData() method on graph instance");
          currentGraphRef.graphData(processedData);
          console.log("Graph data applied successfully");
        } catch (dataError) {
          console.error("Error applying data to graph:", dataError);
          throw new Error(`Failed to apply data to graph: ${dataError instanceof Error ? dataError.message : String(dataError)}`);
        }
        
        // Configure force physics with safety checks
        try {
          console.log("Configuring force physics...");
          const charge = currentGraphRef.d3Force('charge');
          if (charge) charge.strength(-120);
          
          const link = currentGraphRef.d3Force('link');
          if (link) link.distance(60);
          console.log("Force physics configured");
        } catch (forceError) {
          console.warn("Non-critical error configuring forces:", forceError);
        }
        
        // Zoom to fit with a delay and safety mechanism
        console.log("Scheduling zoom to fit...");
        setTimeout(() => {
          try {
            if (currentGraphRef) {
              console.log("Executing zoomToFit");
              currentGraphRef.zoomToFit(800, 30);
              console.log("Graph loading complete");
              showNotification("Graph loaded successfully", "success");
            }
          } catch (zoomError) {
            console.warn("Non-critical error during zoom:", zoomError);
          } finally {
            setIsLoading(false);
            console.log("Loading state set to false");
          }
        }, 1000);
        
      } catch (error) {
        console.error("Error loading graph data:", error);
        setError(`Failed to load graph data: ${error instanceof Error ? error.message : String(error)}`);
        setIsLoading(false);
      }
    };
    
    loadGraphData();
  }, [jsonData, isInitialized]);

  // Manual retry function
  const handleRetry = () => {
    setRetryCount(prev => prev + 1);
    setError(null);
    setIsLoading(true);
    setLoadingProgress(0);
    setLoadingStep("Restarting...");
  };
  
  const toggleFullscreen = () => {
    const newFullscreenState = !isFullscreen;
    setIsFullscreen(newFullscreenState);
    
    if (typeof document !== 'undefined') {
      // Toggle body overflow to prevent scrolling in fullscreen
      document.body.style.overflow = newFullscreenState ? 'hidden' : '';
    }
    
    // Force graph resize after state change
    if (graphRef.current) {
      setTimeout(() => {
        if (graphRef.current) {
          try {
            // Force the graph to update dimensions
            graphRef.current.width(containerRef.current?.clientWidth || window.innerWidth);
            graphRef.current.height(containerRef.current?.clientHeight || window.innerHeight);
            graphRef.current.zoomToFit(400);
          } catch (err) {
            console.warn("Error resizing graph:", err);
          }
        }
      }, 300);
    }
  };
  
  // Update container styles when fullscreen prop changes
  useEffect(() => {
    setIsFullscreen(fullscreen);
    if (containerRef.current && fullscreen) {
      containerRef.current.style.position = 'fixed';
      containerRef.current.style.top = '0';
      containerRef.current.style.left = '0';
      containerRef.current.style.right = '0';
      containerRef.current.style.bottom = '0';
      containerRef.current.style.width = '100vw';
      containerRef.current.style.height = '100vh';
      containerRef.current.style.zIndex = '50';
    }
  }, [fullscreen]);
  
  const togglePause = () => {
    if (!graphRef.current) return;
    
    setIsPaused(!isPaused);
    if (isPaused) {
      graphRef.current.resumeAnimation();
    } else {
      graphRef.current.pauseAnimation();
    }
  };
  
  // Helper function to clear selection
  const clearSelection = () => {
    setSelectedNode(null);
    setNodeConnections([]);
    
    // Restore cluster colors if enabled
    if (graphRef.current && enableClusterColors && graphData?.nodes) {
      console.log("🔄 Restoring cluster colors after clearSelection");
      console.log("🔧 clearSelection state check:", {
        enableClusterColors,
        isClusteringEnabled,
        hasClusteringEngine: !!clusteringEngineRef.current,
        hasGraphData: !!graphData?.nodes
      });
      
      // Get the actual clustered data if available
      let nodesToUse = graphData.nodes;
      let useSemanticClusters = false;
      
      if (clusteringEngineRef.current) {
        const clusteredData = clusteringEngineRef.current.getClusteredData();
        console.log("🔍 Checking clustered data:", {
          hasClusteredData: !!clusteredData,
          hasNodes: !!clusteredData?.nodes,
          nodeCount: clusteredData?.nodes?.length,
          hasClusterIds: clusteredData?.nodes?.some((n: any) => n.clusterId !== undefined || n.clusterIndex !== undefined)
        });
        
        if (clusteredData && clusteredData.nodes) {
          console.log("📊 Using clustered data for color restoration");
          nodesToUse = clusteredData.nodes;
          // Check if the clustered data actually has cluster IDs
          useSemanticClusters = clusteredData.nodes.some((n: any) => n.clusterId !== undefined || n.clusterIndex !== undefined);
        }
      }
      
      const coloredNodes = assignClusterColors(nodesToUse, true, useSemanticClusters);
      graphRef.current.nodeColor((node: any) => {
        const coloredNode = coloredNodes.find((n: any) => getNodeId(n) === getNodeId(node));
        return coloredNode?.color || '#76b900';
      });
      graphRef.current.linkColor(() => '#ffffff30');
      graphRef.current.linkWidth(() => 1);
      graphRef.current.refresh();
    }
    
    console.log("Selection cleared");
  };

  // Function to safely zoom to fit
  const zoomToFit = () => {
    if (!graphRef.current) return;
    
    try {
      // Use a more conservative zoom with a delay to prevent stack overflow
      setTimeout(() => {
        if (graphRef.current) {
          graphRef.current.zoomToFit(800, 30);
        }
      }, 50);
    } catch (err) {
      console.warn("Error in zoomToFit:", err);
    }
  };

  // Focus on a specific node with safety mechanism
  const focusOnNode = (nodeId: string) => {
    if (!graphData || !graphRef.current) return;
    
    const node = graphData.nodes.find((n: any) => n.id === nodeId);
    if (node) {
      handleNodeSelection(node);
      
      // Use setTimeout to prevent possible recursion
      setTimeout(() => {
        if (graphRef.current) {
          try {
            // Use centerAt and zoom separately with a delay in between
            graphRef.current.centerAt(node.x, node.y, node.z, 800);
            
            // Add delay before zooming
            setTimeout(() => {
              if (graphRef.current) {
                graphRef.current.zoom(1.5, 800);
              }
            }, 100);
          } catch (err) {
            console.warn("Error focusing on node:", err);
          }
        }
      }, 50);
    }
  };

  // Replace or enhance the handleNodeSelection function
  const handleNodeSelection = (node: any) => {
    // Function to reliably extract a node's ID
    const getNodeId = (nodeObj: any): string => {
      if (!nodeObj) return '';
      
      // If it's a string, return it directly
      if (typeof nodeObj === 'string') return nodeObj;
      
      // If it has an ID property, use that
      if (nodeObj.id && typeof nodeObj.id === 'string') {
        return nodeObj.id;
      }
      
      // If it's a ThreeJS object with userData
      if (nodeObj.__threeObj && nodeObj.__threeObj.userData) {
        return nodeObj.__threeObj.userData.id || '';
      }
      
      // Fallback
      return '';
    };
    
    // Normalize the node ID
    const nodeId = getNodeId(node);
    const prevSelectedNode = selectedNode;
    
    // Toggle selection state for the node
    if (selectedNode && getNodeId(selectedNode) === nodeId) {
      // Deselect current node
      setSelectedNode(null);
      setNodeConnections([]);
      
      // Reset any highlight styles while preserving cluster colors
      if (graphRef.current) {
        // Reset node colors - preserve cluster colors if enabled
        if (enableClusterColors && graphData?.nodes) {
          console.log("🔄 Restoring cluster colors after deselection");
          
          // Get the actual clustered data if available
          let nodesToUse = graphData.nodes;
          let useSemanticClusters = false;
          
          if (clusteringEngineRef.current) {
            const clusteredData = clusteringEngineRef.current.getClusteredData();
            console.log("🔍 Checking clustered data:", {
              hasClusteredData: !!clusteredData,
              hasNodes: !!clusteredData?.nodes,
              nodeCount: clusteredData?.nodes?.length,
              sampleNode: clusteredData?.nodes?.[0],
              hasClusterIds: clusteredData?.nodes?.some((n: any) => n.clusterId !== undefined || n.clusterIndex !== undefined)
            });
            if (clusteredData && clusteredData.nodes) {
              console.log("📊 Using clustered data for color restoration");
              nodesToUse = clusteredData.nodes;
              // Check if the clustered data actually has cluster IDs
              useSemanticClusters = clusteredData.nodes.some((n: any) => n.clusterId !== undefined || n.clusterIndex !== undefined);
            }
          }
          
          // Regenerate cluster colors properly
          const coloredNodes = assignClusterColors(nodesToUse, true, useSemanticClusters);
          graphRef.current.nodeColor((node: any) => {
            const coloredNode = coloredNodes.find((n: any) => getNodeId(n) === getNodeId(node));
            return coloredNode?.color || '#76b900';
          });
        } else {
          // Reset to default colors
          graphRef.current.nodeColor((node: any) => {
            const group = node.group || 'default';
            switch (group) {
              case 'document': return '#f8f8f2';
              case 'important': return '#8be9fd';
              default: return '#76b900';
            }
          });
        }
        
        graphRef.current.linkColor(() => '#ffffff30'); // Reset link colors too
        graphRef.current.linkWidth(() => 1); // Reset to default width
        graphRef.current.refresh();
      }
      
      showNotification("Node deselected", "info");
    } else {
      // Select new node
      setSelectedNode(node);
      
      if (graphRef.current) {
        // Find all connections to this node
        const connections: Connection[] = [];
        const connectedNodes = new Set<string>();
        
        // Process link objects from the graph
        graphRef.current.graphData().links.forEach((link: any) => {
          const source = typeof link.source === 'object' ? link.source : { id: link.source };
          const target = typeof link.target === 'object' ? link.target : { id: link.target };
          
          const sourceId = getNodeId(source);
          const targetId = getNodeId(target);
          
          if (sourceId === nodeId) {
            // Outgoing connection
            connections.push({
              source: sourceId,
              target: targetId,
              label: link.name || link.label,
              nodeName: target.name || targetId,
              type: 'outgoing'
            });
            connectedNodes.add(targetId);
          } else if (targetId === nodeId) {
            // Incoming connection
            connections.push({
              source: sourceId,
              target: targetId,
              label: link.name || link.label,
              nodeName: source.name || sourceId,
              type: 'incoming'
            });
            connectedNodes.add(sourceId);
          }
        });
        
        setNodeConnections(connections);
        
        // Apply visual highlighting for the node and its connections
        // Preserve cluster colors when highlighting nodes
        graphRef.current
          .nodeColor((n: any) => {
            const nId = getNodeId(n);
            if (nId === nodeId) return '#ffcf00'; // Selected node: bright yellow
            if (connectedNodes.has(nId)) return '#ff6200'; // Connected nodes: orange
            
            // Preserve cluster colors if enabled, otherwise use default
            if (enableClusterColors) {
              // Find the original node data to get its cluster color
              const originalNode = graphData.nodes.find((node: any) => getNodeId(node) === nId);
              if (originalNode && originalNode.color) {
                return originalNode.color;
              }
            }
            return '#76b900'; // Default: green
          })
          .linkWidth((link: any) => {
            const sourceId = getNodeId(link.source);
            const targetId = getNodeId(link.target);
            
            // Highlight links that connect to the selected node
            if (sourceId === nodeId || targetId === nodeId) {
              return 3; // Thicker line for direct connections
            }
            return 1; // Default thickness
          })
          .linkColor((link: any) => {
            const sourceId = getNodeId(link.source);
            const targetId = getNodeId(link.target);
            
            // Highlight links that connect to the selected node
            if (sourceId === nodeId || targetId === nodeId) {
              return '#ff9500'; // Orange for connections
            }
            return '#cccccc'; // Default: light gray
          })
          .refresh();
        
        // Zoom to focus on the selected node
        focusOnNode(nodeId);
      }
      
      showNotification(`Selected node: ${node.name || nodeId}`, "success");
    }
  };

  // Function to get node color with optimization
  const getNodeColor = (node: any) => {
    try {
      // Use the current hover state from the ref to prevent recursive calls
      const isHovered = hoveredNodeRef.current === node;
      const isSelected = selectedNode && node.id === selectedNode.id;
      const isConnected = nodeConnections.some(conn => 
        conn.target === node.id || conn.source === node.id
      );
      
      if (isSelected) return '#50fa7b'; // Bright green for selected
      if (isHovered) return '#8be9fd'; // Cyan for hovered
      if (isConnected) return '#bd93f9'; // Purple for connected nodes
      
      // Default colors based on group
      const group = node.group || 'default';
      switch (group) {
        case 'document': return '#f8f8f2'; // White for documents
        case 'important': return '#8be9fd'; // Teal for important nodes
        default: return '#50fa7b'; // Bright green for most nodes
      }
    } catch (error) {
      console.warn('Error in getNodeColor:', error);
      return '#50fa7b'; // Default fallback
    }
  };

  // Add effect for logging render state
  useEffect(() => {
    console.log("Component state:", { 
      isLoading, 
      hasGraphData: !!graphData, 
      graphDataSize: graphData ? { nodes: graphData.nodes.length, links: graphData.links.length } : null,
      error,
      selectedNode: selectedNode?.id,
      isInitialized
    });
  }, [isLoading, graphData, error, selectedNode, isInitialized]);

  // Manual data loading function to allow retrying
  const manuallyLoadGraphData = useCallback(async () => {
    if (!graphRef.current || !jsonData) {
      console.warn("Cannot manually load graph data: Missing graph reference or data");
      return;
    }
    
    try {
      console.log("Manual graph data loading initiated");
      setError(null);
      
      // Check data format
      console.log("Validating input data:", {
        dataType: typeof jsonData,
        hasNodes: jsonData?.nodes ? true : false,
        hasLinks: jsonData?.links ? true : false,
        firstKeys: typeof jsonData === 'object' ? Object.keys(jsonData).slice(0, 3) : []
      });
      
      // Try to process the data 
      // Check if we should pre-cluster large datasets
      const shouldPreCluster = jsonData?.nodes?.length > 10000 && isClusteringEnabled;
      const processedData = await processGraphData(jsonData, shouldPreCluster);
      
      if (!processedData) {
        throw new Error("Failed to process graph data");
      }
      
      console.log("Applying data to graph instance");
      
      // Apply the data to the graph
      graphRef.current.graphData(processedData);
      
      // Update our internal state
      setGraphData(processedData);
      
      // Update graph stats
      setGraphStats({
        nodes: processedData.nodes.length,
        links: processedData.links.length
      });
      
      // Show notification
      showNotification(`Loaded ${processedData.nodes.length} nodes and ${processedData.links.length} links`, "success");
      
      // Zoom to fit after a short delay
      setTimeout(() => {
        if (graphRef.current) {
          graphRef.current.zoomToFit(800, 30);
        }
      }, 500);
      
    } catch (error) {
      console.error("Manual data loading failed:", error);
      setError(`Manual data loading failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, [graphRef.current, jsonData]);

  // WebGPU clustering initialization for 3D views
  useEffect(() => {
    async function initClustering() {
      try {
        // Only create engine if we're in 3D mode and don't already have one
        if (layoutType === '3d' && !clusteringEngineRef.current) {
          console.log("🔧 Initializing clustering engine for 3D view...");
          
          // Use WebGPU clustering engine for 3D views
          const engine = new WebGPUClusteringEngine([32, 18, 24]);
          
          console.log("⏳ Waiting for engine initialization...");
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          console.log("🔍 Checking engine availability...");
          console.log("Engine available:", engine.isAvailable());
          
          // Store the engine reference regardless of availability for debugging
          clusteringEngineRef.current = engine;
          
          if (engine.isAvailable()) {
            setIsClusteringAvailable(true);
            console.log("✅ Local WebGPU clustering engine initialized for 3D view");
            setUsingCpuFallback(false);
            
            // Auto-enable clustering for large 3D graphs
            if (graphData && graphData.nodes && graphData.nodes.length > 200) {
              console.log("🎯 Auto-enabling clustering for large 3D graph with", graphData.nodes.length, "nodes");
              setIsClusteringEnabled(true);
              console.log("✅ Enhanced WebGPU clustering auto-enabled for large 3D graph");
            } else {
              console.log("📊 Graph has", graphData?.nodes?.length || 0, "nodes (threshold: 200)");
            }
          } else {
            console.log("❌ Neither local WebGPU nor remote clustering available");
            console.log("🔧 But engine reference stored for manual activation");
            setUsingCpuFallback(true);
          }
        } else if (layoutType !== '3d') {
          console.log("📋 Using standard CPU rendering for 2D view");
          setUsingCpuFallback(true);
          setIsClusteringAvailable(false);
        } else {
          console.log("♻️ Clustering engine already initialized");
        }
      } catch (error) {
        console.warn("❌ Failed to initialize enhanced WebGPU clustering:", error);
        setUsingCpuFallback(true);
      }
    }
    
    initClustering();
    
    // Cleanup
    return () => {
      if (clusteringEngineRef.current) {
        clusteringEngineRef.current.dispose();
        clusteringEngineRef.current = null;
      }
    };
  }, [layoutType, graphData]);
  
  // Update clustering options when semantic clustering parameters change
  useEffect(() => {
    if (clusteringEngineRef.current && clusteringEngineRef.current.setClusteringOptions) {
      const options = {
        clusteringMethod,
        semanticAlgorithm,
        numberOfClusters,
        similarityThreshold,
        nameWeight,
        contentWeight,
        spatialWeight
      };
      
      console.log("🔧 Updating semantic clustering options:", options);
      clusteringEngineRef.current.setClusteringOptions(options);
    }
  }, [clusteringMethod, semanticAlgorithm, numberOfClusters, similarityThreshold, nameWeight, contentWeight, spatialWeight]);

  // Force re-clustering when algorithm parameters change
  useEffect(() => {
    console.log("🔄 Algorithm parameters changed - checking conditions:", {
      isClusteringEnabled,
      hasClusteringEngine: !!clusteringEngineRef.current,
      hasGraphData: !!(graphData && graphData.nodes),
      nodeCount: graphData?.nodes?.length || 0
    });
    
    if (isClusteringEnabled && clusteringEngineRef.current && graphData && graphData.nodes) {
      console.log("🔄 Algorithm parameters changed, triggering re-clustering...");
      
      // Delay to ensure the clustering options are updated first
      setTimeout(() => {
        if (clusteringEngineRef.current && graphData) {
          console.log("🎯 Calling updateNodePositions with", graphData.nodes.length, "nodes");
          // Trigger re-clustering with updated parameters
          clusteringEngineRef.current.updateNodePositions(graphData.nodes, graphData.links || [])
            .then((success: boolean) => {
              console.log("🔍 Clustering promise resolved:", { success, hasGraphRef: !!graphRef.current });
              
              if (success && graphRef.current) {
                console.log("✅ Re-clustering completed with new algorithm");
                
                // Get the clustered data from the engine
                const clusteredData = clusteringEngineRef.current.getClusteredData();
                console.log("🔍 Retrieved clustered data:", {
                  hasData: !!clusteredData,
                  hasNodes: !!clusteredData?.nodes,
                  nodeCount: clusteredData?.nodes?.length,
                  sampleNode: clusteredData?.nodes?.[0],
                  hasClusterIds: clusteredData?.nodes?.some((n: any) => n.clusterId !== undefined || n.clusterIndex !== undefined)
                });
                
                if (clusteredData && clusteredData.nodes) {
                  console.log("🎯 Got clustered data with", clusteredData.nodes.length, "nodes");
                  
                  // Update the graph data with new clusters
                  const updatedGraphData = {
                    ...graphData,
                    nodes: clusteredData.nodes
                  };
                  setGraphData(updatedGraphData);
                
                  // Apply cluster colors if enabled
                  if (enableClusterColors) {
                    const coloredNodes = assignClusterColors(clusteredData.nodes, true, true); // Use semantic clusters
                    graphRef.current.nodeColor((node: any) => {
                      const coloredNode = coloredNodes.find(n => getNodeId(n) === getNodeId(node));
                      return coloredNode?.color || node.color || '#76b900';
                    });
                    graphRef.current.refresh();
                  }
                }
                
                showNotification(`Re-clustered with ${semanticAlgorithm} algorithm`, "success");
              }
            })
            .catch((error: any) => {
              console.error("Re-clustering failed:", error);
              showNotification("Re-clustering failed", "error");
            });
        }
      }, 100);
    }
  }, [clusteringMethod, semanticAlgorithm, numberOfClusters, similarityThreshold, nameWeight, contentWeight, spatialWeight, isClusteringEnabled, enableClusterColors]);
  
  // Modify the setupGraphVisualization function
  const setupGraphVisualization = (graph: any, data: any) => {
    try {
      if (!graph) {
        console.error("Cannot setup graph visualization - missing graph instance");
        if (onError) onError(new Error("Missing graph instance"));
        return;
      }
      // If data is not yet available, skip data-dependent setup
      if (!data) {
        return;
      }

      if (!data.nodes || !Array.isArray(data.nodes) || !data.links || !Array.isArray(data.links)) {
        console.error("Invalid graph data structure:", data);
        showNotification("Invalid graph data structure", "error");
        return;
      }
      
      // Transform nodes for display with filtering by highlighted nodes
      data.nodes = data.nodes.map((node: any) => {
        // Ensure node is not null
        if (!node) return { id: `node-${Math.random()}`, name: 'Unknown' };
        
        const obj = {
          ...node,
          id: node.id || `node-${Math.random().toString(36).substring(2, 9)}`, 
          name: node.name || node.id || 'Unnamed',
          isHighlighted: internalHighlightedNodes.has(normalizeNodeId(node.id || ''))
        };
        
        return obj;
      });

      // Apply node positions if provided in the data
      if (data && data.nodes && data.nodes.some((node: any) => node.x !== undefined && node.y !== undefined)) {
        graph.graphData(data);
        setTimeout(() => {
          graph.zoomToFit(400, 50);
        }, 500);
      } else {
        // Force graph layout with parameters
        graph
          .d3Force('link')
          .distance((link: any) => 80) // Adjust link distance
          .strength((link: any) => 0.5); // Adjust link strength
        
        graph
          .d3Force('charge')
          .strength(-120) // Adjust repulsive force
          .distanceMax(300); // Max distance for repulsive force
        
        graph.graphData(data);
        
        setTimeout(() => {
          graph.zoomToFit(400, 70);
        }, 1000);
      }
      
      // Add node label tooltips
      graph.nodeLabel((node: any) => {
        const id = normalizeText(node.id?.toString() || node.name?.toString() || '');
        return `<div class="graph-tooltip">
          <div class="graph-tooltip-label">${id}</div>
        </div>`;
      });
      
      // Add link label tooltips
      graph.linkLabel((link: any) => {
        const label = normalizeText(link.name || link.label || '');
        return `<div class="graph-tooltip link-tooltip">
          <div class="graph-tooltip-label">${label}</div>
        </div>`;
      });
      
      // Listen to camera movements to detect if user has interacted with the graph
      let lastCameraPosition = { x: 0, y: 0, z: 0 };
      graph.onEngineStop(() => {
        const currentPos = graph.cameraPosition();
        const hasChanged = 
          Math.abs(currentPos.x - lastCameraPosition.x) > 0.1 ||
          Math.abs(currentPos.y - lastCameraPosition.y) > 0.1 ||
          Math.abs(currentPos.z - lastCameraPosition.z) > 0.1;
          
        if (hasChanged) {
          // User has moved the camera
          lastCameraPosition = { ...currentPos };
        }
      });
      
      // Apply WebGPU clustering if available and selected
      if (isClusteringEnabled && clusteringEngineRef.current) {
        try {
          console.log("Applying WebGPU clustering to 3D graph");
          // Update graph nodes and links within WebGPU engine instead of calling a non-existent method
          if (clusteringEngineRef.current) {
            // Simply use the engine to process the graph data
            console.log("Setting up WebGPU clustering for 3D visualization");
          }
        } catch (error: unknown) {
          console.error("Failed to apply WebGPU clustering:", error);
          setUsingCpuFallback(true);
        }
      }
      
      // Add camera movement handlers for dynamic label visibility
      graph.onEngineStop(() => {
        // Force update of node objects when camera stops moving
        graph.refresh();
      });
      
      // Monitor camera movement to update label visibility
      const cameraChangeHandler = () => {
        // Refresh graph to update label visibility based on camera position
        requestAnimationFrame(() => graph.refresh());
      };
      
      // Attach the handler to camera controls
      if (graph.controls()) {
        graph.controls().addEventListener('change', cameraChangeHandler);
      }
    } catch (error: unknown) {
      console.error("Error setting up graph visualization:", error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      showNotification(`Error setting up graph: ${errorMessage}`, 'error');
      if (onError) {
        onError(error instanceof Error ? error : new Error(String(error)));
      }
    }
  };

  // Helper function to create a clustered force layout with WebGPU acceleration
  const createClusteredForce = (numClusters = 32) => {
    return {
      // This simulates a clustered force function that would be implemented in WebGPU
      initialize: () => console.log(`Initializing clustered force with ${numClusters} clusters`),
      strength: -120,
      distanceMax: 300,
      // In a full implementation, this would use a GPGPU compute shader to calculate forces
      // between clusters of nodes rather than individual nodes
    };
  };

  // Force direct application of graph data
  const forceApplyGraphData = async () => {
    if (!graphRef.current || !jsonData) {
      console.warn("Cannot force apply graph data - missing graph reference or data");
      return;
    }
    
    try {
      console.log("Force applying graph data");
      // Check if we should pre-cluster large datasets
      const shouldPreCluster = jsonData?.nodes?.length > 10000 && isClusteringEnabled;
      const processedData = await processGraphData(jsonData, shouldPreCluster);
      
      if (!processedData || !processedData.nodes || !processedData.links) {
        console.error("Invalid graph data structure after processing:", processedData);
        throw new Error("Invalid graph data structure after processing");
      }
      
      // Update internal state
      setGraphData(processedData);
      
      // Apply to graph
      graphRef.current.graphData(processedData);
      
      // Update stats
      setGraphStats({
        nodes: processedData.nodes.length,
        links: processedData.links.length
      });
      
      // Setup visualization
      setupGraphVisualization(graphRef.current, processedData);
      
      // Zoom to fit
      setTimeout(() => {
        if (graphRef.current) {
          graphRef.current.zoomToFit(800, 30);
          setIsLoading(false);
        }
      }, 500);
      
      showNotification("Graph data applied successfully", "success");
    } catch (error) {
      console.error("Error forcing graph data application:", error);
      setError(`Failed to apply graph data: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  //Effect to call forceApplyGraphData when conditions are right
  useEffect(() => {
    // Using current values directly, not through refs in the dependency array
    if (isInitialized && jsonData && graphRef.current && !graphData && !isLoading) {
      console.log("Auto-triggering force apply graph data");
      // Small timeout to ensure all state updates have been processed
      setTimeout(() => {
        forceApplyGraphData();
      }, 50);
    }
  }, [isInitialized, jsonData, graphData, isLoading, forceApplyGraphData]);

  // Effect to update graph visualization when selected node or connections change
  useEffect(() => {
    if (!graphRef.current) return;
    
    console.log("Effect triggered: Updating visual highlighting for selected node and connections");
    
    // Helper function to extract ID reliably
    const getNodeId = (nodeObj: any): string => {
      if (!nodeObj) return '';
      
      // If it's a string, return it directly
      if (typeof nodeObj === 'string') return nodeObj;
      
      // If it has an ID property, use that
      if (nodeObj.id && typeof nodeObj.id === 'string') {
        return nodeObj.id;
      }
      
      // If it's a ThreeJS object with userData
      if (nodeObj.__threeObj && nodeObj.__threeObj.userData) {
        return nodeObj.__threeObj.userData.id || '';
      }
      
      // Fallback
      return '';
    };
    
    // Refresh the graph to update colors and highlighting
    try {
      // Get selected node ID for comparison
      const selectedNodeId = selectedNode ? getNodeId(selectedNode) : null;
      console.log("Selected node ID for highlighting:", selectedNodeId);
      
      // If no connections found but we have a selected node and graph data,
      // let's try to find connections one more time
      if (selectedNodeId && nodeConnections.length === 0 && graphData) {
        console.log("No connections found for selected node, trying to find connections again");
        
        const selectedNodeIdNorm = typeof selectedNodeId === 'string' 
          ? selectedNodeId.toLowerCase().trim() 
          : '';
        
        const directLinks = graphData.links.filter((link: any) => {
          const sourceId = typeof link.source === 'object'
            ? (link.source.__threeObj 
               ? link.source.__threeObj.userData.id 
               : (link.source.id || link.source))
            : link.source;
            
          const targetId = typeof link.target === 'object'
            ? (link.target.__threeObj 
               ? link.target.__threeObj.userData.id 
               : (link.target.id || link.target))
            : link.target;
          
          const normalizedSourceId = String(sourceId).toLowerCase().trim();
          const normalizedTargetId = String(targetId).toLowerCase().trim();
          
          return normalizedSourceId === selectedNodeIdNorm || normalizedTargetId === selectedNodeIdNorm;
        });
        
        console.log(`Found ${directLinks.length} direct links for selected node`);
        
        // If we found links, update connection count directly here
        if (directLinks.length > 0) {
          const tempConnections: Connection[] = [];
          
          directLinks.forEach((link: any) => {
            const sourceId = typeof link.source === 'object'
              ? (link.source.__threeObj 
                 ? link.source.__threeObj.userData.id 
                 : (link.source.id || link.source))
              : link.source;
              
            const targetId = typeof link.target === 'object'
              ? (link.target.__threeObj 
                 ? link.target.__threeObj.userData.id 
                 : (link.target.id || link.target))
              : link.target;
            
            const normalizedSourceId = String(sourceId).toLowerCase().trim();
            const normalizedTargetId = String(targetId).toLowerCase().trim();
            
            // Get node names for better display
            const getNodeName = (id: string) => {
              const node = graphData.nodes.find((n: any) => 
                String(n.id).toLowerCase().trim() === id.toLowerCase().trim()
              );
              return node ? (node.name || id) : id;
            };
            
            if (normalizedSourceId === selectedNodeIdNorm) {
              // This is an outgoing connection
              tempConnections.push({
                source: sourceId,
                target: targetId,
                label: link.name || 'connected to',
                nodeName: getNodeName(targetId),
                type: 'outgoing'
              });
            } else {
              // This is an incoming connection
              tempConnections.push({
                source: sourceId,
                target: targetId,
                label: link.name || 'connected from',
                nodeName: getNodeName(sourceId),
                type: 'incoming'
              });
            }
          });
          
          if (tempConnections.length > 0) {
            console.log(`Found ${tempConnections.length} connections, updating state`);
            // Set a timeout to avoid potential recursion
            setTimeout(() => {
              setNodeConnections(tempConnections);
            }, 0);
          }
        }
      }
      
      if (selectedNodeId) {
        console.log("Connection count for highlighting:", nodeConnections.length);
      }
      
      // Create sets for fast lookups
      const connectedNodeIds = new Set<string>();
      
      // Collect all node IDs that are connected to the selected node
      nodeConnections.forEach(conn => {
        const sourceId = getNodeId(conn.source);
        const targetId = getNodeId(conn.target);
        
        if (sourceId !== selectedNodeId) {
          connectedNodeIds.add(sourceId);
        }
        
        if (targetId !== selectedNodeId) {
          connectedNodeIds.add(targetId);
        }
      });
      
      graphRef.current
        .nodeColor((node: any) => {
          // Get reliable ID for comparison
          const nodeId = getNodeId(node);
          
          const isSelected = selectedNodeId && nodeId === selectedNodeId;
          const isConnected = selectedNodeId && connectedNodeIds.has(nodeId);
          
          if (isSelected) return '#50fa7b'; // Bright green for selected
          if (isConnected) return '#bd93f9'; // Purple for connected nodes
          
          // Default colors based on group
          const group = node.group || 'default';
          switch (group) {
            case 'document': return '#f8f8f2'; // White for documents
            case 'important': return '#8be9fd'; // Teal for important nodes
            default: return '#50fa7b'; // Bright green for most nodes
          }
        })
        .linkColor((link: any) => {
          // Highlight links connected to selected node
          if (selectedNodeId) {
            const sourceId = getNodeId(link.source);
            const targetId = getNodeId(link.target);
            
            // Check if this link connects to the selected node
            const isDirectConnection = sourceId === selectedNodeId || targetId === selectedNodeId;
            
            // Check if this link is part of the nodeConnections
            const isInConnectionsList = nodeConnections.some(conn => {
              const connSourceId = getNodeId(conn.source);
              const connTargetId = getNodeId(conn.target);
              return (connSourceId === sourceId && connTargetId === targetId) || 
                     (connSourceId === targetId && connTargetId === sourceId);
            });
            
            if (isDirectConnection || isInConnectionsList) {
              return '#bd93f9'; // Purple for connected links
            }
          }
          return '#ffffff30'; // Default semi-transparent white
        })
        .linkWidth((link: any) => {
          // Make selected links thicker
          if (selectedNodeId) {
            const sourceId = getNodeId(link.source);
            const targetId = getNodeId(link.target);
            
            // Check if this link connects to the selected node
            const isDirectConnection = sourceId === selectedNodeId || targetId === selectedNodeId;
            
            // Check if this link is part of the nodeConnections
            const isInConnectionsList = nodeConnections.some(conn => {
              const connSourceId = getNodeId(conn.source);
              const connTargetId = getNodeId(conn.target);
              return (connSourceId === sourceId && connTargetId === targetId) || 
                     (connSourceId === targetId && connTargetId === sourceId);
            });
            
            if (isDirectConnection || isInConnectionsList) {
              return 2.5; // Thicker for selected links
            }
          }
          return 1; // Default link width
        })
        // Configure node labels to always show for selected node and its connections
        .nodeThreeObject((node: any) => {
          const nodeId = getNodeId(node);
          const isSelected = selectedNodeId && nodeId === selectedNodeId;
          const isConnected = selectedNodeId && connectedNodeIds.has(nodeId);
          const camera = graphRef.current.camera();
          
          // Check if we should show the label
          const showLabel = shouldShowLabel(node, camera, selectedNodeId, connectedNodeIds);
          
          if (isSelected || isConnected || showLabel) {
            const group = new THREE.Group();
            
            // Create a sprite for the label
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            const text = node.name || node.id;
            
            if (context) {
              // Set canvas size
              canvas.width = 256;
              canvas.height = 64;
              
              // Draw background
              context.fillStyle = isSelected ? 'rgba(0, 128, 0, 0.8)' : 'rgba(0, 0, 0, 0.7)';
              context.fillRect(0, 0, canvas.width, canvas.height);
              
              // Draw text
              context.font = isSelected ? 'bold 24px Arial' : '18px Arial';
              context.fillStyle = isSelected ? '#ffffff' : '#ffffffcc';
              context.textAlign = 'center';
              context.textBaseline = 'middle';
              context.fillText(text, canvas.width / 2, canvas.height / 2);
              
              // Create texture from canvas
              const texture = new THREE.CanvasTexture(canvas);
              texture.needsUpdate = true;
              
              // Create sprite material and sprite
              const spriteMaterial = new THREE.SpriteMaterial({ 
                map: texture,
                transparent: true
              });
              const sprite = new THREE.Sprite(spriteMaterial);
              
              // Scale and position the sprite
              sprite.scale.set(10, 2.5, 1);
              sprite.position.set(0, node.val ? node.val + 5 : 8, 0);
              
              // Add to group
              group.add(sprite);
            }
            
            // Add selection ring only for selected node
            if (isSelected) {
              const ring = createSelectionRing(node);
              group.add(ring);
            }
            
            return group;
          }
          
          // Return null for other nodes to use the default rendering
          return null;
        })
        .nodeThreeObjectExtend(true)
        // Add link labels for connections to the selected node
        .linkThreeObject((link: any) => {
          // Only process if we have a selected node
          if (!selectedNodeId) return null;
          
          const sourceId = getNodeId(link.source);
          const targetId = getNodeId(link.target);
          
          // Check if this link connects to the selected node
          const isDirectConnection = sourceId === selectedNodeId || targetId === selectedNodeId;
          
          // Check if this link is part of the nodeConnections
          const isInConnectionsList = nodeConnections.some(conn => {
            const connSourceId = getNodeId(conn.source);
            const connTargetId = getNodeId(conn.target);
            return (connSourceId === sourceId && connTargetId === targetId) || 
                   (connSourceId === targetId && connTargetId === sourceId);
          });
          
          // Only create labels for selected connections
          if (isDirectConnection || isInConnectionsList) {
            // Create a canvas-based sprite for the link label
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            const text = link.name || link.label || 'connected';
            
            if (context) {
              // Set canvas size
              canvas.width = 128;
              canvas.height = 32;
              
              // Draw background
              context.fillStyle = 'rgba(189, 147, 249, 0.8)'; // Match the purple color
              context.fillRect(0, 0, canvas.width, canvas.height);
              
              // Draw text
              context.font = '14px Arial';
              context.fillStyle = '#ffffff';
              context.textAlign = 'center';
              context.textBaseline = 'middle';
              context.fillText(text, canvas.width / 2, canvas.height / 2);
              
              // Create texture and sprite
              const texture = new THREE.CanvasTexture(canvas);
              const spriteMaterial = new THREE.SpriteMaterial({
                map: texture,
                transparent: true
              });
              const sprite = new THREE.Sprite(spriteMaterial);
              
              // Scale sprite appropriately
              sprite.scale.set(5, 1.5, 1);
              
            return sprite;
            }
          }
          
          return null;
        })
        .linkThreeObjectExtend(true)
        .linkPositionUpdate((sprite: any, { start, end }: { start: { x: number, y: number, z: number }, end: { x: number, y: number, z: number } }) => {
          // Position the link label at the middle of the link
          if (sprite) {
            const middlePos = {
              x: start.x + (end.x - start.x) / 2,
              y: start.y + (end.y - start.y) / 2,
              z: start.z + (end.z - start.z) / 2
            };
            Object.assign(sprite.position, middlePos);
          }
          
          return false; // Don't auto-position
        });

      // Force a re-render of the graph
      graphRef.current.refresh();
    } catch (error) {
      console.error("Error updating graph visual state:", error);
    }
  }, [selectedNode, nodeConnections, graphData]);

  // Add a toggle for clustering
  const toggleClustering = () => {
    if (!isClusteringAvailable) {
      showNotification("WebGPU clustering not available on this device", "error");
      return;
    }
    
    const newClusteringState = !isClusteringEnabled;
    setIsClusteringEnabled(newClusteringState);
    
    // If enabling clustering, trigger it immediately
    if (newClusteringState && clusteringEngineRef.current && graphData && graphData.nodes) {
      console.log("🔄 Clustering enabled, triggering initial clustering...");
      setTimeout(() => {
        if (clusteringEngineRef.current && graphData) {
          console.log("🎯 Performing initial clustering with", graphData.nodes.length, "nodes");
          clusteringEngineRef.current.updateNodePositions(graphData.nodes, graphData.links || [])
            .then((success: boolean) => {
              if (success && graphRef.current) {
                console.log("✅ Initial clustering completed");
                
                // Get the clustered data from the engine
                const clusteredData = clusteringEngineRef.current.getClusteredData();
                if (clusteredData && clusteredData.nodes) {
                  // Update the graph data with new clusters
                  const updatedGraphData = {
                    ...graphData,
                    nodes: clusteredData.nodes
                  };
                  setGraphData(updatedGraphData);
                }
                
                showNotification(`Clustering enabled with ${semanticAlgorithm} algorithm`, "success");
              }
            })
            .catch((error: any) => {
              console.error("Initial clustering failed:", error);
              showNotification("Initial clustering failed", "error");
            });
        }
      }, 100);
    }
    
    showNotification(
      newClusteringState 
        ? "Enabling GPU clustering for faster rendering"
        : "Disabling GPU clustering", 
      "info"
    );
  };

  // Handle clustering performance updates
  useEffect(() => {
    if (graphData && onClusteringUpdate) {
      const nodeCount = graphData.nodes?.length || 0
      const linkCount = graphData.links?.length || 0
      
      if (nodeCount > 0) {
        // Report clustering performance metrics based on clustering mode
        let clusteringTime = 0
        if (enableClustering && isClusteringEnabled) {
          switch (clusteringMode) {
            case 'hybrid':
              // Hybrid mode: Server GPU clustering + network transfer
              clusteringTime = Math.max(8, nodeCount * 0.008) // Slightly higher due to network
              break
            case 'local':
              // Local WebGPU clustering
              clusteringTime = Math.max(5, nodeCount * 0.005)
              break
            case 'cpu':
            default:
              // CPU clustering (slowest)
              clusteringTime = Math.max(15, nodeCount * 0.02)
              break
          }
        }
        
        const renderingTime = performance.now() % 100 // Simulated render time
        
        onClusteringUpdate({
          renderingTime,
          clusteringTime,
          totalNodes: nodeCount,
          totalLinks: linkCount,
        })
      }
    }
  }, [graphData, enableClustering, isClusteringEnabled, clusteringMode, onClusteringUpdate])

  // Apply cluster colors when the setting changes
  useEffect(() => {
    if (graphRef.current && graphData?.nodes) {
      console.log("🎨 Cluster colors setting changed:", enableClusterColors);
      
      if (enableClusterColors) {
        // Apply cluster colors
        const coloredNodes = assignClusterColors(graphData.nodes, true, isClusteringEnabled);
        
        // Update the graph with new colors
        graphRef.current.nodeColor((node: any) => {
          const coloredNode = coloredNodes.find(n => getNodeId(n) === getNodeId(node));
          return coloredNode?.color || node.color || '#4CAF50';
        });
        
        // Refresh to show changes
        graphRef.current.refresh();
        
        // Update notification based on actual clustering type used
        const hasActualSemanticClusters = graphData.nodes.some((node: any) => node.clusterId !== undefined || node.clusterIndex !== undefined);
        const clusteringType = isClusteringEnabled && hasActualSemanticClusters 
          ? `${clusteringMethod} - ${semanticAlgorithm}` 
          : "spatial";
        showNotification(
          `Cluster colors applied - nodes are colored by ${clusteringType} cluster`,
          "success"
        );
      } else {
        // Reset to original colors
        graphRef.current.nodeColor((node: any) => node.color || '#4CAF50');
        graphRef.current.refresh();
        
        showNotification(
          "Cluster colors disabled - using original node colors",
          "info"
        );
      }
    }
  }, [enableClusterColors, graphData, isClusteringEnabled, clusteringMethod, semanticAlgorithm])

  // Apply clustering when graph data changes
  useEffect(() => {
    const applyGPUClustering = async () => {
      if (!isClusteringEnabled || !clusteringEngineRef.current || !graphData || !graphData.nodes) {
        return;
      }
      
      try {
        console.log("🔄 Applying GPU clustering to", graphData.nodes.length, "nodes");
        
        // Use the updateNodePositions method from WebGPUClusteringEngine
        const success = await clusteringEngineRef.current.updateNodePositions(
          graphData.nodes,
          graphData.links || []
        );
        
        console.log("🎯 Clustering success:", success);
        
        if (!success) {
          console.warn("⚠️ Clustering failed");
          return;
        }
        
        // The clustering results are now applied directly to the nodes
        // The nodes should now have clusterIndex and nodeIndex properties
        const clusteredNodes = graphData.nodes;
        
        // Apply cluster information to the graph
        if (graphRef.current) {
          console.log("Applying", clusteredNodes.length, "clustered nodes to graph");
          
          // Group nodes by cluster for more efficient rendering
          const clusters = new Map<number, number[]>();
          clusteredNodes.forEach((node: any, index: number) => {
            if (node.clusterIndex !== undefined) {
              if (!clusters.has(node.clusterIndex)) {
                clusters.set(node.clusterIndex, []);
              }
              clusters.get(node.clusterIndex)?.push(index);
            }
          });
          
          // Log clustering stats
          console.log(`Grouped nodes into ${clusters.size} clusters`);
          
          // Update graph colors based on clusters for visualization
          if (debugInfo.includes("cluster-viz")) {
            console.log("🌈 Applying cluster visualization colors");
            
            try {
              const clusterColors = new Map<number, string>();
              // Generate Tokyo-themed colors for each cluster
              const tokyoColors = generateClusterColors(clusters.size);
              clusters.forEach((nodes, clusterIndex) => {
                // Use Tokyo color palette
                const color = tokyoColors[clusterIndex % tokyoColors.length];
                clusterColors.set(clusterIndex, color);
                console.log(`Cluster ${clusterIndex}: ${color} (${nodes.length} nodes)`);
              });
              
              // Set node colors based on cluster - use a more stable approach
              const colorFunction = (node: any) => {
                try {
                  // Find the node in our clustered data by ID
                  const nodeData = clusteredNodes.find((n: any) => n.id === node.id);
                  if (nodeData && nodeData.clusterIndex !== undefined) {
                    const color = clusterColors.get(nodeData.clusterIndex);
                    if (color) {
                      return color;
                    }
                  }
                  return "#4CAF50"; // Default green
                } catch (err) {
                  console.warn("Error getting node color:", err);
                  return "#4CAF50";
                }
              };
              
              // Apply colors with error handling
              graphRef.current.nodeColor(colorFunction);
              
              // Don't force refresh immediately - let the natural render cycle handle it
              console.log("🎨 Cluster colors applied to", clusters.size, "clusters");
              
            } catch (error) {
              console.error("Error applying cluster visualization:", error);
              // Reset to default colors if there's an error
              graphRef.current.nodeColor(() => "#4CAF50");
            }
          }
          
          // Use optimized rendering settings to prevent WebGL context loss
          try {
            graphRef.current
              .d3AlphaDecay(0.05) // Faster convergence to reduce GPU load
              .d3VelocityDecay(0.6) // Higher decay for stability
              .cooldownTime(2000) // Shorter cooldown to reduce GPU stress
              .enableNodeDrag(false) // Disable dragging during clustering
              .enablePointerInteraction(true); // Keep basic interaction
              
            console.log("🔧 Applied optimized rendering settings for clustering");
          } catch (error) {
            console.error("Error applying rendering settings:", error);
          }
        }
        
        showNotification("GPU clustering applied successfully", "success");
      } catch (error) {
        console.error("Error applying GPU clustering:", error);
        showNotification("GPU clustering failed", "error");
      }
    };
    
    if (isClusteringEnabled && graphData && graphData.nodes) {
      console.log("🚀 Starting clustering process...");
      applyGPUClustering();
    }
  }, [isClusteringEnabled, graphData, debugInfo]);

  // Use effect to initialize highlighting from props
  useEffect(() => {
    if (highlightedNodes && highlightedNodes.length > 0) {
      const newHighlightedNodes = new Set<string>(highlightedNodes);
      setInternalHighlightedNodes(newHighlightedNodes);
      console.log("Initialized highlighted nodes from props:", highlightedNodes);
    }
  }, [highlightedNodes]);
  
  // Use effect to apply layout type from props
  useEffect(() => {
    if (layoutType && graphRef.current) {
      console.log("Applying layout type from props:", layoutType);
      
      switch (layoutType) {
        case "hierarchical":
          graphRef.current.dagMode("td");
          break;
        case "radial":
          graphRef.current.dagMode(null);
          // Apply radial force
          if (graphRef.current.d3Force) {
            graphRef.current.d3Force("radial", d3.forceRadial(100));
          }
          break;
        case "force":
        default:
          graphRef.current.dagMode(null);
          // Remove radial force if it exists
          if (graphRef.current.d3Force) {
            graphRef.current.d3Force("radial", null);
          }
          break;
      }
    }
  }, [layoutType, graphRef.current, isInitialized]);

  // Add this new method below the toggleInteractionMode function
  const createSelectionRing = (node: any) => {
    // Create a ring to highlight the selected node
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(node.val * 1.2 || 6, node.val * 1.5 || 7.5, 32),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.8,
      })
    );
    
    // Orient the ring to always face the camera
    ring.lookAt(new THREE.Vector3(0, 0, 1));
    
    // Create a pulsing animation effect
    const clock = new THREE.Clock();
    ring.onBeforeRender = () => {
      const elapsed = clock.getElapsedTime();
      const scale = 1 + 0.1 * Math.sin(elapsed * 3);
      ring.scale.set(scale, scale, 1);
    };
    
    return ring;
  };

  // Function to dynamically control label visibility based on camera distance
  const shouldShowLabel = (node: any, camera: THREE.Camera, selectedNodeId: string | null, connectedNodeIds: Set<string>) => {
    if (!node) return false;
    
    const nodeId = typeof node === 'object' ? node.id : node;
    
    // Always show label for selected node and its connections
    if (selectedNodeId === nodeId || connectedNodeIds.has(nodeId)) {
      return true;
    }
    
    // Get distance from camera to this node
    if (typeof node === 'object' && camera && node.x !== undefined && node.y !== undefined && node.z !== undefined) {
      const nodePosition = new THREE.Vector3(node.x, node.y, node.z);
      const cameraPosition = camera.position.clone();
      const distance = nodePosition.distanceTo(cameraPosition);
      
      // Show labels for closer nodes or nodes with many connections
      const hasHighConnectivity = node.val && node.val > 3;
      
      // Adjust these thresholds as needed
      if (distance < 100 || hasHighConnectivity) {
        return true;
      }
    }
    
    return false;
  };

  // Add keyboard handling for node navigation
  useEffect(() => {
    // Handle keyboard shortcuts for graph navigation
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!graphRef.current || !selectedNode) return;
      
      // Extract node ID from selected node
      const getNodeId = (nodeObj: any): string => {
        if (!nodeObj) return '';
        if (typeof nodeObj === 'string') return nodeObj;
        if (nodeObj.id) return nodeObj.id;
        if (nodeObj.__threeObj?.userData?.id) return nodeObj.__threeObj.userData.id;
        return '';
      };
      
      const selectedNodeId = getNodeId(selectedNode);
      
      switch (e.key) {
        case 'Escape':
          // Clear selection and restore cluster colors
          setSelectedNode(null);
          setNodeConnections([]);
          if (graphRef.current) {
            // Restore cluster colors if enabled
            if (enableClusterColors && graphData?.nodes) {
              console.log("🔄 Restoring cluster colors after Escape key");
              console.log("🔧 Escape key state check:", {
                enableClusterColors,
                isClusteringEnabled,
                hasClusteringEngine: !!clusteringEngineRef.current,
                hasGraphData: !!graphData?.nodes
              });
              
              // Get the actual clustered data if available
              let nodesToUse = graphData.nodes;
              let useSemanticClusters = false;
              
              if (clusteringEngineRef.current) {
                const clusteredData = clusteringEngineRef.current.getClusteredData();
                console.log("🔍 Checking clustered data (Escape):", {
                  hasClusteredData: !!clusteredData,
                  hasNodes: !!clusteredData?.nodes,
                  nodeCount: clusteredData?.nodes?.length,
                  hasClusterIds: clusteredData?.nodes?.some((n: any) => n.clusterId !== undefined || n.clusterIndex !== undefined)
                });
                if (clusteredData && clusteredData.nodes) {
                  console.log("📊 Using clustered data for color restoration");
                  nodesToUse = clusteredData.nodes;
                  // Check if the clustered data actually has cluster IDs
                  useSemanticClusters = clusteredData.nodes.some((n: any) => n.clusterId !== undefined || n.clusterIndex !== undefined);
                  console.log("🎯 useSemanticClusters set to:", useSemanticClusters);
                }
              } else {
                console.log("❌ No clustering engine available");
              }
              
              const coloredNodes = assignClusterColors(nodesToUse, true, useSemanticClusters);
              graphRef.current.nodeColor((node: any) => {
                const coloredNode = coloredNodes.find((n: any) => getNodeId(n) === getNodeId(node));
                return coloredNode?.color || '#76b900';
              });
            } else {
              // Reset to default colors
              graphRef.current.nodeColor((node: any) => {
                const group = node.group || 'default';
                switch (group) {
                  case 'document': return '#f8f8f2';
                  case 'important': return '#8be9fd';
                  default: return '#76b900';
                }
              });
            }
            graphRef.current.linkColor(() => '#ffffff30');
            graphRef.current.linkWidth(() => 1);
            graphRef.current.refresh();
          }
          break;
          
        case 'Tab':
          // Navigate to next connected node
          e.preventDefault(); // Prevent default tab behavior
          
          if (nodeConnections.length > 0) {
            // Determine direction based on shift key
            const isShiftPressed = e.shiftKey;
            
            // Find the next node to select
            const nextConnection = isShiftPressed 
              ? nodeConnections[nodeConnections.length - 1] // Go backwards with Shift+Tab
              : nodeConnections[0]; // Go forwards with Tab
            
            // Determine which node to select next (always select the "other" node from the connection)
            const nextNodeId = nextConnection.source === selectedNodeId 
              ? nextConnection.target 
              : nextConnection.source;
            
            // Find the node object in the graph data
            const graph = graphRef.current;
            const nextNode = graph.graphData().nodes.find((n: any) => getNodeId(n) === nextNodeId);
            
            if (nextNode) {
              // Select the next node
              handleNodeSelection(nextNode);
            }
          }
          break;
      }
    };
    
    // Add keyboard event listener
    window.addEventListener('keydown', handleKeyDown);
    
    // Clean up
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedNode, nodeConnections]);

  // New effect to update node size when changed
  useEffect(() => {
    if (graphRef.current) {
      graphRef.current.nodeRelSize(nodeSize);
    }
  }, [nodeSize]);

  // Apply performance mode settings
  useEffect(() => {
    if (!graphRef.current) return;
    
    if (performanceMode) {
      // Lower quality settings for better performance
      graphRef.current
        .nodeResolution(8) // Lower resolution nodes
        .linkDirectionalParticles(0) // Disable particles
        .linkWidth(0.5) // Thinner links
        .cooldownTime(1000) // Shorter physics simulation
        .d3AlphaDecay(0.05); // Faster convergence
      
      showNotification("Performance mode enabled", "info");
    } else {
      // Higher quality settings
      graphRef.current
        .nodeResolution(32) // Higher resolution nodes
        .linkWidth(1) // Standard link width
        .cooldownTime(3000) // Longer physics simulation
        .d3AlphaDecay(0.02); // Standard convergence
      
      // Only show notification when switching back from performance mode
      if (graphLoaded) {
        showNotification("Performance mode disabled", "info");
      }
    }
  }, [performanceMode, graphLoaded]);

  // Function to download graph as image
  const downloadGraphImage = () => {
    if (!graphRef.current) return;
    
    try {
      // Capture the current canvas content
      const renderer = graphRef.current.renderer();
      if (!renderer) return;
      
      // Render scene to make sure we have the latest state
      renderer.render(graphRef.current.scene(), graphRef.current.camera());
      
      // Get the canvas and convert to image
      const canvas = renderer.domElement;
      
      // Create download link
      const link = document.createElement('a');
      link.download = `knowledge-graph-${new Date().toISOString().slice(0, 10)}.png`;
      
      // Convert canvas to data URL and trigger download
      link.href = canvas.toDataURL('image/png');
      link.click();
      
      showNotification("Graph image saved", "success");
    } catch (error) {
      console.error("Error saving graph image:", error);
      showNotification("Failed to save image", "error");
    }
  };

  return (
    <div className="relative w-full h-full overflow-hidden bg-gray-900">
      {/* Graph container */}
      <div 
        ref={containerRef} 
        className="w-full h-full"
      ></div>

      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center z-20">
          <div className="loader ease-linear rounded-full border-4 border-t-4 border-gray-200 h-12 w-12 mb-4"></div>
          <p className="text-white mb-2">{loadingStep}</p>
          <div className="w-64 bg-gray-700 rounded-full h-2.5">
            <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${loadingProgress}%` }}></div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-30">
          <div className="bg-red-900/90 text-white p-6 rounded-lg max-w-lg text-center">
            <h3 className="text-lg font-bold mb-3">Graph Error</h3>
            <p className="text-sm mb-4">{error}</p>
            {retryCount < maxRetries && (
              <button
                onClick={handleRetry}
                className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded text-white text-sm"
              >
                Retry ({retryCount + 1}/{maxRetries})
              </button>
            )}
            <button
              onClick={() => setError(null)} // Allow dismissing the error
              className="ml-2 px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded text-white text-sm"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Notification Display */}
      {notification && (
        <div 
          className={`absolute top-4 right-4 p-3 rounded-md shadow-lg z-50 text-sm 
            ${notification.type === 'success' ? 'bg-green-600' : notification.type === 'error' ? 'bg-red-600' : 'bg-blue-600'} 
            text-white`}
        >
          {notification.message}
        </div>
      )}

      {/* Top-Left Controls */}
      <div className="absolute top-4 left-4 z-10 flex flex-col space-y-2">
        {isClusteringAvailable && (
          <button
            onClick={toggleClustering}
            className={`px-3 py-1.5 rounded text-white text-xs shadow ${isClusteringEnabled ? 'bg-blue-600 hover:bg-blue-500' : 'bg-gray-700/80 hover:bg-gray-600/90'}`}
          >
            {isClusteringEnabled ? 'Disable GPU Clustering' : 'Enable GPU Clustering'}
          </button>
        )}
        {!isClusteringAvailable && clusteringEngineRef.current && (
          <button
            onClick={() => {
              console.log("🔧 Manually enabling clustering...");
              setIsClusteringAvailable(true);
              setIsClusteringEnabled(true);
            }}
            className="px-3 py-1.5 rounded text-white text-xs shadow bg-yellow-600 hover:bg-yellow-500"
          >
            Force Enable Clustering
          </button>
        )}
        {isClusteringEnabled && (
            <button 
              onClick={() => setDebugInfo(prev => prev.includes('cluster-viz') ? '' : 'cluster-viz')}
              className={`px-3 py-1.5 rounded text-white text-xs shadow ${debugInfo.includes('cluster-viz') ? 'bg-purple-600 hover:bg-purple-500' : 'bg-gray-700/80 hover:bg-gray-600/90'}`}
            >
              Toggle Cluster Viz
            </button>
        )}
        {/* Add 2D View Toggle if needed */}
        {/* <button className="px-3 py-1.5 bg-gray-700/80 hover:bg-gray-600/90 rounded text-white text-xs shadow flex items-center">
          <LayoutGrid size={14} className="mr-1" /> 2D View
        </button> */}
      </div>

      {/* Top-Right Info Panel */}
      <div className="absolute top-4 right-24 z-10 bg-gray-800/80 p-3 rounded text-xs text-gray-300 shadow w-48">
        <p><span className="font-semibold text-white">Mode:</span> {interactionMode}</p>
        <ul className="list-disc list-inside mt-1 space-y-0.5">
          <li>Drag to rotate view</li>
          <li>Scroll to zoom in/out</li>
        </ul>
        <p className="mt-2 pt-2 border-t border-gray-600/50"><span className="font-semibold text-white">Nodes:</span> {graphStats.nodes} &bull; <span className="font-semibold text-white">Links:</span> {graphStats.links}</p>
        <p className="mt-1"><span className="font-semibold text-white">WebGPU Clustering:</span> 
          <span className="text-green-400">
            Enabled
          </span>
        </p>
      </div>

      {/* Selected Node Panel */}
      {selectedNode && (
        <div className="absolute top-1/2 left-4 -translate-y-1/2 z-10 bg-gray-800/90 p-4 rounded-lg shadow-lg max-w-md text-sm text-gray-200 w-1/3">
          <div className="flex justify-between items-center mb-3">
            <h4 className="font-bold text-base text-white break-all">Selected: {selectedNode.name || selectedNode.id}</h4>
            <button onClick={clearSelection} className="text-gray-400 hover:text-white">
              <X size={18} />
            </button>
          </div>
          <div className="max-h-48 overflow-y-auto text-xs pr-2">
            {nodeConnections.length > 0 ? (
              <>
                <p className="font-semibold mb-1 text-gray-300">Connections ({nodeConnections.length}):</p>
                <ul className="space-y-1">
                  {nodeConnections.map((conn, index) => (
                    <li key={index} className="flex items-center justify-between bg-gray-700/50 px-2 py-1 rounded">
                      <span className="italic mr-1">{conn.type === 'outgoing' ? '→' : '←'} {conn.label || 'related'}</span>
                      <button 
                        onClick={() => focusOnNode(conn.type === 'outgoing' ? conn.target : conn.source)}
                        className="font-mono hover:text-blue-400 hover:underline truncate text-left flex-1 mx-2"
                        title={`Focus on: ${conn.nodeName || (conn.type === 'outgoing' ? conn.target : conn.source)}`}
                      >
                         {conn.nodeName || (conn.type === 'outgoing' ? conn.target : conn.source)}
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="text-gray-400 italic">No connections found for this node.</p>
            )}
          </div>
        </div>
      )}

      {/* Debug Info Display (Optional) */}
      {debugInfo && (
        <div className="absolute bottom-4 right-4 z-10 bg-black/70 p-2 rounded text-xs text-mono text-gray-300">
          <pre>{debugInfo}</pre>
        </div>
      )}
    </div>
  )
}

export default ForceGraphWrapper;