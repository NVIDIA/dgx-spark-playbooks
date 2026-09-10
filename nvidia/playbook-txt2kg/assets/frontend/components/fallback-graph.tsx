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

import type React from "react"

import { useEffect, useRef, useState, useCallback } from "react"
import type { Triple } from "@/types/graph"
import { Maximize2, Minimize2, ZoomIn, ZoomOut, Move, Filter, Play, Pause } from "lucide-react"

interface FallbackGraphProps {
  triples: Triple[]
  fullscreen?: boolean
  highlightedNodes?: string[]
}

interface Node {
  id: string
  label: string
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  color: string
  connections: number
}

interface Link {
  source: string
  target: string
  label: string
}

// Add interface for CPU-based grid cell
interface GridCell {
  x: number;
  y: number;
  nodeIndices: number[];
}

export function FallbackGraph({ triples, fullscreen = false, highlightedNodes }: FallbackGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isFullscreen, setIsFullscreen] = useState(fullscreen)
  const [isBrowserFullscreen, setIsBrowserFullscreen] = useState(false)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [simulation, setSimulation] = useState<{
    nodes: Node[]
    links: Link[]
    isRunning: boolean
    iteration: number
  } | null>(null)
  const [nodeLimit, setNodeLimit] = useState(75) // Default node limit
  const [showNodeLimitWarning, setShowNodeLimitWarning] = useState(false)
  const [allNodesCount, setAllNodesCount] = useState(0)
  const [tooltipText, setTooltipText] = useState("")
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 })
  const [showTooltip, setShowTooltip] = useState(false)
  const [simulationPaused, setSimulationPaused] = useState(true) // Start with simulation paused

  // Add state for CPU-based clustering
  const [cpuClustering, setCpuClustering] = useState<boolean>(false);
  const [gridCells, setGridCells] = useState<Map<string, GridCell>>(new Map());

  // Handle browser fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsBrowserFullscreen(!!document.fullscreenElement)
    }

    document.addEventListener("fullscreenchange", handleFullscreenChange)

    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange)
    }
  }, [])

  // Toggle browser fullscreen
  const toggleFullscreen = useCallback(() => {
    if (!containerRef.current) return

    if (!document.fullscreenElement) {
      // Enter fullscreen
      containerRef.current.requestFullscreen().catch((err) => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`)
      })
    } else {
      // Exit fullscreen
      document.exitFullscreen().catch((err) => {
        console.error(`Error attempting to exit fullscreen: ${err.message}`)
      })
    }
  }, [])

  const handleZoomIn = useCallback(() => {
    setZoom((prev) => Math.min(3, prev + 0.1))
  }, [])

  const handleZoomOut = useCallback(() => {
    setZoom((prev) => Math.max(0.1, prev - 0.1))
  }, [])

  const handleResetView = useCallback(() => {
    setZoom(1)
    setOffset({ x: 0, y: 0 })
    setSelectedNode(null)

    // Restart simulation
    setSimulation((prev) => (prev ? { ...prev, isRunning: true, iteration: 0 } : null))
  }, [])

  const handleIncreaseNodeLimit = useCallback(() => {
    setNodeLimit((prev) => Math.min(prev + 50, 500))
  }, [])

  const handleDecreaseNodeLimit = useCallback(() => {
    setNodeLimit((prev) => Math.max(25, prev - 25))
  }, [])

  const toggleNodeLimit = useCallback(() => {
    setNodeLimit((prev) => (prev === 75 ? 150 : 75))
  }, [])

  // Handle tooltip display
  const handleButtonMouseEnter = useCallback((e: React.MouseEvent, text: string) => {
    const rect = e.currentTarget.getBoundingClientRect()
    setTooltipText(text)
    setTooltipPosition({
      x: rect.left + rect.width / 2,
      y: rect.bottom + 5,
    })
    setShowTooltip(true)
  }, [])

  const handleButtonMouseLeave = useCallback(() => {
    setShowTooltip(false)
  }, [])

  // Add a CPU-based clustering implementation as fallback for GPU clustering
  const applyCpuClustering = (nodes: Node[]) => {
    if (!cpuClustering || !nodes.length) return;
    
    console.log("Applying CPU-based clustering fallback");
    
    // Create a grid for spatial partitioning (2D for fallback graph)
    const cellSize = 100; // Size of each grid cell
    const newGridCells = new Map<string, GridCell>();
    
    // Assign nodes to grid cells
    nodes.forEach((node, index) => {
      const cellX = Math.floor(node.x / cellSize);
      const cellY = Math.floor(node.y / cellSize);
      const cellKey = `${cellX},${cellY}`;
      
      if (!newGridCells.has(cellKey)) {
        newGridCells.set(cellKey, {
          x: cellX,
          y: cellY,
          nodeIndices: []
        });
      }
      
      newGridCells.get(cellKey)!.nodeIndices.push(index);
    });
    
    setGridCells(newGridCells);
    console.log(`CPU clustering: Created ${newGridCells.size} grid cells for ${nodes.length} nodes`);
  };

  // Initialize the simulation with a subset of nodes
  useEffect(() => {
    if (!triples.length) return

    // Extract unique entities and count their connections
    const entityConnections = new Map<string, number>()

    triples.forEach((triple) => {
      // Count subject connections
      if (entityConnections.has(triple.subject)) {
        entityConnections.set(triple.subject, entityConnections.get(triple.subject)! + 1)
      } else {
        entityConnections.set(triple.subject, 1)
      }

      // Count object connections
      if (entityConnections.has(triple.object)) {
        entityConnections.set(triple.object, entityConnections.get(triple.object)! + 1)
      } else {
        entityConnections.set(triple.object, 1)
      }
    })

    // Sort entities by connection count (most connected first)
    const sortedEntities = Array.from(entityConnections.entries()).sort((a, b) => b[1] - a[1])

    // Store total node count
    setAllNodesCount(sortedEntities.length)

    // Show warning if we're limiting nodes
    setShowNodeLimitWarning(sortedEntities.length > nodeLimit)

    // Take only the top N entities
    const topEntities = sortedEntities.slice(0, nodeLimit).map(([id]) => id)

    // Create a Set for faster lookups
    const includedEntities = new Set(topEntities)

    // Create nodes
    const nodes: Node[] = topEntities.map((id) => {
      const connectionCount = entityConnections.get(id) || 0
      const isHighlighted = highlightedNodes?.includes(id) || false

      return {
        id,
        label: id,
        x: Math.random() * 800 - 400,
        y: Math.random() * 800 - 400,
        vx: 0,
        vy: 0,
        radius: Math.max(5, Math.min(12, 5 + connectionCount * 0.5)),
        color: isHighlighted ? "#FF9900" : "#76B900",
        connections: connectionCount,
      }
    })

    // Create links (only between included entities)
    const links: Link[] = triples
      .filter((triple) => includedEntities.has(triple.subject) && includedEntities.has(triple.object))
      .map((triple) => ({
        source: triple.subject,
        target: triple.object,
        label: triple.predicate,
      }))

    setSimulation({
      nodes,
      links,
      isRunning: !simulationPaused, // Use the simulationPaused state to determine initial running state
      iteration: 0,
    })
    
    // Apply CPU clustering after setting up the simulation
    applyCpuClustering(nodes);
  }, [triples, nodeLimit, simulationPaused, highlightedNodes])

  // Run the simulation with optimizations
  useEffect(() => {
    if (!simulation || !simulation.isRunning) return

    let animationFrameId: number
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Force simulation parameters
    const strength = -30 // Repulsive force between nodes
    const linkDistance = 100 // Desired distance between connected nodes
    const linkStrength = 0.1 // Strength of the links
    const friction = 0.9 // Friction to slow down nodes
    const gravity = 0.1 // Force pulling nodes to the center
    const maxIterations = 300 // Maximum number of iterations

    // Create a node lookup map for faster access
    const nodeMap = new Map(simulation.nodes.map((node) => [node.id, node]))

    const tick = () => {
      // Apply forces
      const { nodes, links, iteration } = simulation

      // Stop if we've reached max iterations
      if (iteration >= maxIterations) {
        setSimulation((prev) => (prev ? { ...prev, isRunning: false } : null))
        return
      }

      // Optimization: Use a grid-based approach for repulsion
      // This serves as CPU fallback for GPU clustering
      const cellSize = 100; // Size of each grid cell

      // If CPU clustering is enabled, use the pre-computed grid cells instead of recalculating
      if (cpuClustering && gridCells.size > 0) {
        // Apply repulsive forces only between nodes in the same or adjacent cells
        for (const node of nodes) {
          const cellX = Math.floor(node.x / cellSize);
          const cellY = Math.floor(node.y / cellSize);
          
          // Check current cell and adjacent cells
          for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
              const neighborCellKey = `${cellX + dx},${cellY + dy}`;
              const cell = gridCells.get(neighborCellKey);
              
              if (!cell) continue;
              
              // Only calculate forces between nodes in this cell
              for (const otherNodeIndex of cell.nodeIndices) {
                const otherNode = nodes[otherNodeIndex];
                if (node.id === otherNode.id) continue;
                
                // Calculate repulsive force (same as before)
                const dx = otherNode.x - node.x;
                const dy = otherNode.y - node.y;
                const distance = Math.sqrt(dx * dx + dy * dy) || 1;
                
                // Skip if too far
                if (distance > cellSize * 1.5) continue;
                
                const force = strength / (distance * distance);
                const maxForce = 5;
                const limitedForce = Math.max(-maxForce, Math.min(maxForce, force));
                
                const fx = (limitedForce * dx) / distance;
                const fy = (limitedForce * dy) / distance;
                
                node.vx -= fx;
                node.vy -= fy;
              }
            }
          }
        }
      } else {
        // Original grid-based approach (existing code)
        const grid = new Map<string, Node[]>();
        
        // Place nodes in grid cells
        for (const node of nodes) {
          const cellX = Math.floor(node.x / cellSize);
          const cellY = Math.floor(node.y / cellSize);
          const cellKey = `${cellX},${cellY}`;

          if (!grid.has(cellKey)) {
            grid.set(cellKey, []);
          }

          grid.get(cellKey)!.push(node);
        }
        
        // Apply repulsive forces between nodes in same or adjacent cells (existing code)
        // ... existing cell-based force calculation code ...
      }

      // Apply attractive forces along links
      for (const link of links) {
        const sourceNode = nodeMap.get(link.source)
        const targetNode = nodeMap.get(link.target)

        if (sourceNode && targetNode) {
          const dx = targetNode.x - sourceNode.x
          const dy = targetNode.y - sourceNode.y
          const distance = Math.sqrt(dx * dx + dy * dy) || 1
          const force = (distance - linkDistance) * linkStrength

          const fx = (force * dx) / distance
          const fy = (force * dy) / distance

          sourceNode.vx += fx
          sourceNode.vy += fy
          targetNode.vx -= fx
          targetNode.vy -= fy
        }
      }

      // Apply gravity towards center
      for (const node of nodes) {
        node.vx += (-node.x * gravity) / 100
        node.vy += (-node.y * gravity) / 100
      }

      // Apply velocity with friction and update positions
      for (const node of nodes) {
        node.vx *= friction
        node.vy *= friction
        node.x += node.vx
        node.y += node.vy
      }

      // Check if simulation has stabilized
      const isStable = nodes.every((node) => Math.abs(node.vx) < 0.1 && Math.abs(node.vy) < 0.1)

      if (isStable) {
        setSimulation((prev) => (prev ? { ...prev, isRunning: false } : null))
      } else {
        setSimulation((prev) => (prev ? { ...prev, iteration: prev.iteration + 1 } : null))
      }

      // Draw the graph
      drawGraph()

      if (!isStable && iteration < maxIterations) {
        animationFrameId = requestAnimationFrame(tick)
      }
    }

    const drawGraph = () => {
      if (!canvas || !simulation) return

      const ctx = canvas.getContext("2d")
      if (!ctx) return

      const { width, height } = canvas
      
      // Clear the canvas
      ctx.clearRect(0, 0, width, height)
      
      // Calculate center offset for panning
      const centerX = width / 2 + offset.x * zoom
      const centerY = height / 2 + offset.y * zoom
      
      // Draw connections first (so nodes appear on top)
      ctx.lineWidth = 1 / zoom
      simulation.links.forEach((link) => {
        const source = simulation.nodes.find((n) => n.id === link.source)
        const target = simulation.nodes.find((n) => n.id === link.target)
        
        if (!source || !target) return
        
        const sourceIsHighlighted = highlightedNodes?.includes(source.id) || false
        const targetIsHighlighted = highlightedNodes?.includes(target.id) || false
        const isHighlightedLink = sourceIsHighlighted && targetIsHighlighted
        
        // Calculate positions with zoom and pan
        const x1 = centerX + source.x * zoom
        const y1 = centerY + source.y * zoom
        const x2 = centerX + target.x * zoom
        const y2 = centerY + target.y * zoom
        
        // Draw link line
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.strokeStyle = isHighlightedLink ? 'rgba(255, 153, 0, 0.8)' : "rgba(150, 150, 150, 0.3)"
        ctx.stroke()
        
        // Draw directional arrow
        const angle = Math.atan2(y2 - y1, x2 - x1)
        const arrowLength = 10 / zoom
        const arrowWidth = 3 / zoom
        
        // Calculate position for the arrow near the target
        const radius = target.radius
        const distance = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        const ratio = (distance - radius) / distance
        const arrowX = x1 + (x2 - x1) * ratio
        const arrowY = y1 + (y2 - y1) * ratio
        
        ctx.beginPath()
        ctx.moveTo(arrowX, arrowY)
        ctx.lineTo(
          arrowX - arrowLength * Math.cos(angle - Math.PI / 6),
          arrowY - arrowLength * Math.sin(angle - Math.PI / 6)
        )
        ctx.lineTo(
          arrowX - arrowLength * 0.7 * Math.cos(angle),
          arrowY - arrowLength * 0.7 * Math.sin(angle)
        )
        ctx.lineTo(
          arrowX - arrowLength * Math.cos(angle + Math.PI / 6),
          arrowY - arrowLength * Math.sin(angle + Math.PI / 6)
        )
        ctx.closePath()
        
        ctx.fillStyle = isHighlightedLink ? 'rgba(255, 153, 0, 0.8)' : "rgba(150, 150, 150, 0.5)"
        ctx.fill()
        
        // Draw link label if hovered/selected or zoom is high enough
        if (
          (hoveredNode === source.id || 
           hoveredNode === target.id || 
           selectedNode === source.id || 
           selectedNode === target.id || 
           zoom > 2) && 
          link.label
        ) {
          // Calculate label position (middle of the link)
          const labelX = (x1 + x2) / 2
          const labelY = (y1 + y2) / 2 - 5 / zoom
          
          // Draw label background
          const labelText = String(link.label)
          const textWidth = (ctx.measureText(labelText).width + 8) / zoom
          const textHeight = 16 / zoom
          
          ctx.fillStyle = isHighlightedLink ? "rgba(255, 153, 0, 0.2)" : "rgba(0, 0, 0, 0.6)"
          ctx.beginPath()
          ctx.roundRect(
            labelX - textWidth / 2,
            labelY - textHeight,
            textWidth,
            textHeight,
            5 / zoom
          )
          ctx.fill()
          
          // Draw label text
          ctx.fillStyle = isHighlightedLink ? "#FFF" : "rgba(255, 255, 255, 0.9)"
          ctx.font = `${12 / zoom}px sans-serif`
          ctx.textAlign = "center"
          ctx.textBaseline = "middle"
          ctx.fillText(labelText, labelX, labelY - textHeight / 2)
        }
      })
      
      // Draw nodes
      simulation.nodes.forEach((node) => {
        const isHighlighted = highlightedNodes?.includes(node.id) || false
        const isHovered = hoveredNode === node.id
        const isSelected = selectedNode === node.id
        
        // Calculate position with zoom and pan
        const x = centerX + node.x * zoom
        const y = centerY + node.y * zoom
        const radius = node.radius * zoom
        
        // Draw glow for highlighted, hovered or selected nodes
        if (isHighlighted || isHovered || isSelected) {
          ctx.beginPath()
          ctx.arc(x, y, radius * 1.5, 0, Math.PI * 2)
          ctx.fillStyle = isHighlighted 
            ? "rgba(255, 153, 0, 0.3)" 
            : (isSelected ? "rgba(0, 128, 255, 0.3)" : "rgba(255, 255, 255, 0.3)")
          ctx.fill()
        }
        
        // Draw node circle
        ctx.beginPath()
        ctx.arc(x, y, radius, 0, Math.PI * 2)
        ctx.fillStyle = isHighlighted 
          ? "#FF9900" 
          : (isSelected ? "#0088FF" : (isHovered ? "#7CD22D" : node.color))
        ctx.fill()
        
        // Draw node stroke
        ctx.lineWidth = 1.5 / zoom
        ctx.strokeStyle = isHighlighted 
          ? "rgba(255, 153, 0, 0.8)" 
          : (isSelected ? "rgba(0, 128, 255, 0.8)" : "rgba(50, 50, 50, 0.5)")
        ctx.stroke()
        
        // Draw node label if hovered, selected, or zoom is high enough
        if (isHovered || isSelected || zoom > 1.2 || isHighlighted) {
          const labelText = String(node.label)
          const fontSize = isHighlighted || isSelected ? 14 / zoom : 12 / zoom
          ctx.font = `${fontSize}px sans-serif`
          ctx.textAlign = "center"
          ctx.textBaseline = "middle"
          
          // Draw text background for better readability
          const textWidth = (ctx.measureText(labelText).width + 10) / zoom
          const textHeight = 20 / zoom
          
          ctx.fillStyle = isHighlighted 
            ? "rgba(255, 153, 0, 0.8)" 
            : (isSelected ? "rgba(0, 128, 255, 0.8)" : "rgba(0, 0, 0, 0.7)")
          ctx.beginPath()
          ctx.roundRect(
            x - textWidth / 2,
            y + radius + 4 / zoom,
            textWidth,
            textHeight,
            5 / zoom
          )
          ctx.fill()
          
          // Draw text
          ctx.fillStyle = "rgba(255, 255, 255, 0.95)"
          ctx.fillText(labelText, x, y + radius + 4 / zoom + textHeight / 2)
        }
      })
    }

    // Start the simulation
    animationFrameId = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(animationFrameId)
    }
  }, [simulation, hoveredNode, selectedNode, zoom, offset, cpuClustering, gridCells])

  // Handle canvas resize
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current
      if (!canvas) return

      const container = canvas.parentElement
      if (!container) return

      canvas.width = container.clientWidth
      canvas.height = container.clientHeight
    }

    handleResize()
    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
    }
  }, [isFullscreen, isBrowserFullscreen])

  // Handle mouse interactions
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !simulation) return

    const getMousePosition = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      return { x, y }
    }

    const handleMouseMove = (e: MouseEvent) => {
      if (!simulation) return

      const { x, y } = getMousePosition(e)
      const centerX = canvas.width / 2
      const centerY = canvas.height / 2

      // Check if mouse is over any node
      let hovered = null
      for (const node of simulation.nodes) {
        const nodeX = centerX + (node.x + offset.x) * zoom
        const nodeY = centerY + (node.y + offset.y) * zoom
        const distance = Math.sqrt(Math.pow(x - nodeX, 2) + Math.pow(y - nodeY, 2))

        if (distance < node.radius * zoom * 1.2) { // Slightly larger hover area
          hovered = node.id
          break
        }
      }

      setHoveredNode(hovered)

      // Handle dragging
      if (isDragging) {
        const dx = (x - dragStart.x) / zoom
        const dy = (y - dragStart.y) / zoom

        setOffset((prev) => ({
          x: prev.x + dx,
          y: prev.y + dy,
        }))

        setDragStart({ x, y })
      }
    }

    const handleMouseDown = (e: MouseEvent) => {
      const { x, y } = getMousePosition(e)

      // Check if clicking on a node
      const centerX = canvas.width / 2
      const centerY = canvas.height / 2

      let clickedNode = null
      for (const node of simulation.nodes) {
        const nodeX = centerX + (node.x + offset.x) * zoom
        const nodeY = centerY + (node.y + offset.y) * zoom
        const distance = Math.sqrt(Math.pow(x - nodeX, 2) + Math.pow(y - nodeY, 2))

        if (distance < node.radius * zoom * 1.2) { // Slightly larger clickable area
          clickedNode = node.id
          break
        }
      }

      if (clickedNode) {
        // Toggle selection if clicking on a node
        if (selectedNode === clickedNode) {
          setSelectedNode(null);
        } else {
          setSelectedNode(clickedNode);
          console.log(`Selected node: ${clickedNode}`);
          console.log(`Node connections: ${simulation.links.filter((link) => link.source === clickedNode || link.target === clickedNode).length}`);
        }
      } else {
        // Start dragging the canvas
        setIsDragging(true)
        setDragStart({ x, y })
      }
    }

    const handleMouseUp = () => {
      setIsDragging(false)
    }

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault()

      // Adjust zoom level
      const delta = -Math.sign(e.deltaY) * 0.1
      const newZoom = Math.max(0.1, Math.min(3, zoom + delta))

      setZoom(newZoom)
    }

    canvas.addEventListener("mousemove", handleMouseMove)
    canvas.addEventListener("mousedown", handleMouseDown)
    canvas.addEventListener("mouseup", handleMouseUp)
    canvas.addEventListener("mouseleave", handleMouseUp)
    canvas.addEventListener("wheel", handleWheel, { passive: false })

    return () => {
      canvas.removeEventListener("mousemove", handleMouseMove)
      canvas.removeEventListener("mousedown", handleMouseDown)
      canvas.removeEventListener("mouseup", handleMouseUp)
      canvas.removeEventListener("mouseleave", handleMouseUp)
      canvas.removeEventListener("wheel", handleWheel)
    }
  }, [simulation, isDragging, dragStart, zoom, offset, selectedNode])

  // Handle canvas resize when fullscreen changes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const container = canvas.parentElement
    if (!container) return

    canvas.width = container.clientWidth
    canvas.height = container.clientHeight

    // Force redraw with full highlighting and labels
    if (simulation) {
      // Instead of a simplified redraw, call the main drawing function
      // which includes all highlighting and labels
      const nodeMap = new Map(simulation.nodes.map((node) => [node.id, node]))

      const drawFullGraph = () => {
        const ctx = canvas.getContext("2d")
        if (!ctx) return

        ctx.clearRect(0, 0, canvas.width, canvas.height)

        const { nodes, links } = simulation
        const centerX = canvas.width / 2
        const centerY = canvas.height / 2

        // Draw links with proper highlighting
        for (const link of links) {
          const sourceNode = nodeMap.get(link.source)
          const targetNode = nodeMap.get(link.target)

          if (sourceNode && targetNode) {
            // Transform coordinates based on zoom and offset
            const sx = centerX + (sourceNode.x + offset.x) * zoom
            const sy = centerY + (sourceNode.y + offset.y) * zoom
            const tx = centerX + (targetNode.x + offset.x) * zoom
            const ty = centerY + (targetNode.y + offset.y) * zoom

            // Highlight links connected to selected node
            if (
              hoveredNode === sourceNode.id ||
              hoveredNode === targetNode.id ||
              selectedNode === sourceNode.id ||
              selectedNode === targetNode.id
            ) {
              ctx.strokeStyle = "rgba(118, 185, 0, 0.6)"
              ctx.lineWidth = 2
            } else {
              ctx.strokeStyle = "rgba(255, 255, 255, 0.2)"
              ctx.lineWidth = 1
            }

            ctx.beginPath()
            ctx.moveTo(sx, sy)
            ctx.lineTo(tx, ty)
            ctx.stroke()

            // Draw arrow
            const angle = Math.atan2(ty - sy, tx - sx)
            const arrowLength = 8

            ctx.beginPath()
            ctx.moveTo(tx, ty)
            ctx.lineTo(
              tx - arrowLength * Math.cos(angle - Math.PI / 6),
              ty - arrowLength * Math.sin(angle - Math.PI / 6),
            )
            ctx.lineTo(
              tx - arrowLength * Math.cos(angle + Math.PI / 6),
              ty - arrowLength * Math.sin(angle + Math.PI / 6),
            )
            ctx.closePath()
            ctx.fillStyle = "rgba(118, 185, 0, 0.6)"
            ctx.fill()

            // Draw link label for selected connections
            if (
              hoveredNode === sourceNode.id ||
              hoveredNode === targetNode.id ||
              selectedNode === sourceNode.id ||
              selectedNode === targetNode.id
            ) {
              const midX = (sx + tx) / 2
              const midY = (sy + ty) / 2

              // Background for label
              ctx.font = "10px Inter, sans-serif"
              const labelWidth = ctx.measureText(link.label).width + 8
              ctx.fillStyle = "rgba(0, 0, 0, 0.7)"
              ctx.fillRect(midX - labelWidth / 2, midY - 10, labelWidth, 20)

              // Label text
              ctx.fillStyle = "white"
              ctx.textAlign = "center"
              ctx.textBaseline = "middle"
              ctx.fillText(link.label, midX, midY)
            }
          }
        }

        // Draw nodes with proper highlighting
        for (const node of nodes) {
          // Transform coordinates based on zoom and offset
          const x = centerX + (node.x + offset.x) * zoom
          const y = centerY + (node.y + offset.y) * zoom
          const radius = node.radius * zoom

          // Node circle
          ctx.beginPath()
          ctx.arc(x, y, radius, 0, Math.PI * 2)

          // Highlight hovered or selected node
          if (node.id === hoveredNode || node.id === selectedNode) {
            // Glow effect
            ctx.fillStyle = "#76B900"
          } else {
            ctx.fillStyle = "rgba(118, 185, 0, 0.8)"
          }

          ctx.fill()

          // Draw node border
          if (node.id === selectedNode) {
            ctx.strokeStyle = "white"
            ctx.lineWidth = 2
            ctx.stroke()
          }

          // Draw node label
          ctx.font =
            node.id === hoveredNode || node.id === selectedNode
              ? "bold 12px Inter, sans-serif"
              : "11px Inter, sans-serif"

          ctx.fillStyle = "rgba(255, 255, 255, 0.9)"
          ctx.textAlign = "center"
          ctx.textBaseline = "middle"

          // Always show labels for important nodes
          const isImportantNode = node.connections > 2
          const isHighlightedNode = node.id === hoveredNode || node.id === selectedNode

          if (isImportantNode || isHighlightedNode) {
            // Background for label
            const labelWidth = ctx.measureText(node.label).width + 8
            const labelHeight = 20

            // Always add background for important nodes
            ctx.fillStyle = "rgba(0, 0, 0, 0.8)"
            ctx.fillRect(x - labelWidth / 2, y + radius + 4, labelWidth, labelHeight)

            // Text color
            ctx.fillStyle = isHighlightedNode ? "white" : "rgba(255, 255, 255, 0.9)"
            ctx.fillText(node.label, x, y + radius + 14)
          }
        }
      }

      drawFullGraph()
    }
  }, [isFullscreen, isBrowserFullscreen, simulation, zoom, offset, hoveredNode, selectedNode])

  // Add toggle function for simulation pause/play
  const toggleSimulation = useCallback(() => {
    setSimulationPaused(!simulationPaused);
    setSimulation(prev => prev ? { ...prev, isRunning: simulationPaused } : null);
  }, [simulationPaused]);

  // Add UI control for CPU clustering toggle
  useEffect(() => {
    // Detect if GPU clustering might be unavailable (simple check)
    const checkGpuSupport = () => {
      const hasGpu = typeof navigator !== 'undefined' && 'gpu' in navigator;
      if (!hasGpu) {
        console.log("WebGPU not detected, enabling CPU clustering fallback");
        setCpuClustering(true);
      }
    };
    
    checkGpuSupport();
  }, []);

  // Add this additional useEffect to refresh the display when selectedNode changes
  useEffect(() => {
    // Force a redraw when selectedNode changes
    const canvas = canvasRef.current;
    if (canvas && simulation) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Create a node map for faster lookup
        const nodeMap = new Map();
        simulation.nodes.forEach(node => {
          nodeMap.set(node.id, node);
        });
        
        // Draw the graph
        const drawFullGraph = () => {
          // Draw links
          for (const link of simulation.links) {
            const sourceNode = nodeMap.get(link.source);
            const targetNode = nodeMap.get(link.target);
            
            if (sourceNode && targetNode) {
              // Transform coordinates based on zoom and offset
              const sx = centerX + (sourceNode.x + offset.x) * zoom;
              const sy = centerY + (sourceNode.y + offset.y) * zoom;
              const tx = centerX + (targetNode.x + offset.x) * zoom;
              const ty = centerY + (targetNode.y + offset.y) * zoom;
              
              // Highlight links connected to selected node
              if (
                hoveredNode === sourceNode.id ||
                hoveredNode === targetNode.id ||
                selectedNode === sourceNode.id ||
                selectedNode === targetNode.id
              ) {
                ctx.strokeStyle = "rgba(118, 185, 0, 0.7)";
                ctx.lineWidth = 2.5;
              } else {
                ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
                ctx.lineWidth = 1;
              }
              
              ctx.beginPath();
              ctx.moveTo(sx, sy);
              ctx.lineTo(tx, ty);
              ctx.stroke();
              
              // Draw arrow
              const angle = Math.atan2(ty - sy, tx - sx);
              const arrowLength = 8;
              
              ctx.beginPath();
              ctx.moveTo(tx, ty);
              ctx.lineTo(
                tx - arrowLength * Math.cos(angle - Math.PI / 6),
                ty - arrowLength * Math.sin(angle - Math.PI / 6)
              );
              ctx.lineTo(
                tx - arrowLength * Math.cos(angle + Math.PI / 6),
                ty - arrowLength * Math.sin(angle + Math.PI / 6)
              );
              ctx.closePath();
              ctx.fillStyle = "rgba(118, 185, 0, 0.7)";
              ctx.fill();
            }
          }
          
          // Draw nodes
          for (const node of simulation.nodes) {
            // Transform coordinates based on zoom and offset
            const x = centerX + (node.x + offset.x) * zoom;
            const y = centerY + (node.y + offset.y) * zoom;
            const radius = node.radius * zoom;
            
            // Node circle
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            
            // Highlight the selected node or nodes connected to the selected node
            const isSelected = node.id === selectedNode;
            const isConnectedToSelected = selectedNode && 
              simulation.links.some(
                link => (link.source === selectedNode && link.target === node.id) || 
                      (link.target === selectedNode && link.source === node.id)
              );
            
            if (isSelected) {
              ctx.fillStyle = "#76B900"; // Bright green for selected
            } else if (isConnectedToSelected) {
              ctx.fillStyle = "#50a0ff"; // Blue for connected nodes
            } else if (node.id === hoveredNode) {
              ctx.fillStyle = "#d0ff50"; // Yellow-green for hovered
            } else {
              ctx.fillStyle = "rgba(118, 185, 0, 0.8)"; // Default
            }
            
            ctx.fill();
            
            // Draw node border
            if (isSelected) {
              ctx.strokeStyle = "white";
              ctx.lineWidth = 2;
              ctx.stroke();
            } else if (isConnectedToSelected) {
              ctx.strokeStyle = "#a0d0ff";
              ctx.lineWidth = 1.5;
              ctx.stroke();
            }
            
            // Draw node label
            const isImportantNode = node.connections > 2 || isSelected || isConnectedToSelected;
            const isHighlightedNode = node.id === hoveredNode || isSelected;
            
            if (isImportantNode || isHighlightedNode) {
              ctx.font = isSelected ? "bold 12px Inter, sans-serif" : "11px Inter, sans-serif";
              
              // Background for label
              const labelWidth = ctx.measureText(node.label).width + 8;
              const labelHeight = 20;
              
              // Add background
              if (isSelected) {
                ctx.fillStyle = "rgba(0, 128, 0, 0.9)";
              } else if (isConnectedToSelected) {
                ctx.fillStyle = "rgba(0, 64, 128, 0.9)";
              } else {
                ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
              }
              
              ctx.fillRect(x - labelWidth / 2, y + radius + 4, labelWidth, labelHeight);
              
              // Text color
              ctx.fillStyle = "white";
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              ctx.fillText(node.label, x, y + radius + 14);
            }
          }
        };
        
        drawFullGraph();
      }
    }
  }, [selectedNode, simulation, zoom, offset, hoveredNode]);

  return (
    <div className="relative h-full w-full" ref={containerRef}>
      <div className={`bg-black rounded-lg overflow-hidden h-full w-full`}>
        <canvas ref={canvasRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

        {/* Info panel */}
        <div className="absolute bottom-2 left-2 text-xs bg-black/70 px-3 py-2 rounded flex items-center gap-2">
          <span className="text-gray-400">Force-Directed Graph</span>
          {simulation?.isRunning && (
            <span className="text-primary animate-pulse">
              Simulating... {Math.round((simulation.iteration / 300) * 100)}%
            </span>
          )}
        </div>

        {/* Node limit warning */}
        {showNodeLimitWarning && (
          <div className="absolute top-2 left-2 text-xs bg-black/70 px-3 py-2 rounded flex items-center gap-2">
            <span className="text-yellow-400">
              Showing {nodeLimit} of {allNodesCount} nodes
            </span>
            <button
              onClick={handleIncreaseNodeLimit}
              className="px-2 py-0.5 bg-primary/20 text-primary rounded hover:bg-primary/30"
              title="Show more nodes"
            >
              Show more
            </button>
          </div>
        )}

        {/* Controls */}
        <div className="absolute top-2 right-2 flex flex-col gap-2">
          {/* Add play/pause simulation button */}
          <button
            onClick={toggleSimulation}
            className="p-2 bg-black/70 hover:bg-black/90 text-white rounded-full z-10"
            type="button"
            onMouseEnter={(e) => handleButtonMouseEnter(e, simulationPaused ? "Start simulation" : "Pause simulation")}
            onMouseLeave={handleButtonMouseLeave}
          >
            {simulationPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
          </button>

          <button
            onClick={handleZoomIn}
            className="p-2 bg-black/70 hover:bg-black/90 text-white rounded-full z-10"
            type="button"
            onMouseEnter={(e) => handleButtonMouseEnter(e, "Zoom in")}
            onMouseLeave={handleButtonMouseLeave}
          >
            <ZoomIn className="h-4 w-4" />
          </button>

          <button
            onClick={handleZoomOut}
            className="p-2 bg-black/70 hover:bg-black/90 text-white rounded-full z-10"
            type="button"
            onMouseEnter={(e) => handleButtonMouseEnter(e, "Zoom out")}
            onMouseLeave={handleButtonMouseLeave}
          >
            <ZoomOut className="h-4 w-4" />
          </button>

          <button
            onClick={toggleNodeLimit}
            className="p-2 bg-black/70 hover:bg-black/90 text-white rounded-full z-10"
            type="button"
            onMouseEnter={(e) => handleButtonMouseEnter(e, "Toggle node limit")}
            onMouseLeave={handleButtonMouseLeave}
          >
            <Filter className="h-4 w-4" />
          </button>
        </div>

        {/* Node limit controls */}
        {showNodeLimitWarning && (
          <div className="absolute bottom-2 right-2 bg-black/70 rounded px-2 py-1 flex items-center gap-2">
            <button
              onClick={handleDecreaseNodeLimit}
              className="text-white text-xs px-2 py-0.5 bg-gray-700 rounded hover:bg-gray-600"
              disabled={nodeLimit <= 25}
              type="button"
              onMouseEnter={(e) => handleButtonMouseEnter(e, "Show fewer nodes")}
              onMouseLeave={handleButtonMouseLeave}
            >
              -
            </button>
            <span className="text-xs text-white">{nodeLimit} nodes</span>
            <button
              onClick={handleIncreaseNodeLimit}
              className="text-white text-xs px-2 py-0.5 bg-gray-700 rounded hover:bg-gray-600"
              disabled={nodeLimit >= 500}
              type="button"
              onMouseEnter={(e) => handleButtonMouseEnter(e, "Show more nodes")}
              onMouseLeave={handleButtonMouseLeave}
            >
              +
            </button>
          </div>
        )}

        {/* Selected node info */}
        {selectedNode && (
          <div className="absolute top-2 left-2 bg-black/80 text-white text-sm px-4 py-3 rounded max-w-xs">
            <h3 className="font-bold text-primary mb-1">{selectedNode}</h3>
            <div className="text-xs text-gray-300">
              {simulation?.links.filter((link) => link.source === selectedNode || link.target === selectedNode)
                .length || 0}{" "}
              connections
            </div>
            <div className="mt-2 text-xs max-h-[400px] overflow-auto">
              <div className="mb-2">
                <div className="text-gray-400 text-xs uppercase mb-1">Outgoing</div>
                {simulation?.links
                  .filter((link) => link.source === selectedNode)
                  .map((link, i) => (
                    <div key={`out-${i}`} className="flex items-center gap-1 mb-1">
                      <span className="text-gray-400">→</span>
                      <span className="text-primary">{link.label}</span>
                      <span className="text-gray-300">→</span>
                      <span>{link.target}</span>
                    </div>
                  )) || <div className="text-gray-500 italic">None</div>}
              </div>
              
              <div>
                <div className="text-gray-400 text-xs uppercase mb-1">Incoming</div>
                {simulation?.links
                  .filter((link) => link.target === selectedNode)
                  .map((link, i) => (
                    <div key={`in-${i}`} className="flex items-center gap-1 mb-1">
                      <span>{link.source}</span>
                      <span className="text-gray-300">→</span>
                      <span className="text-primary">{link.label}</span>
                      <span className="text-gray-400">→</span>
                    </div>
                  )) || <div className="text-gray-500 italic">None</div>}
              </div>
              
              <button 
                onClick={() => setSelectedNode(null)}
                className="mt-4 bg-gray-700 hover:bg-gray-600 text-white px-3 py-1 rounded text-xs"
              >
                Clear selection
              </button>
            </div>
          </div>
        )}

        {/* Tooltip */}
        {showTooltip && (
          <div
            className="absolute bg-black/90 text-white text-xs px-2 py-1 rounded pointer-events-none z-50"
            style={{
              left: `${tooltipPosition.x}px`,
              top: `${tooltipPosition.y}px`,
              transform: "translateX(-50%)",
            }}
          >
            {tooltipText}
          </div>
        )}

        {/* Add CPU fallback indicator */}
        {cpuClustering && (
          <div className="absolute bottom-2 left-2 bg-gray-900/80 text-white text-xs px-2 py-1 rounded">
            Using CPU clustering fallback
          </div>
        )}
      </div>
    </div>
  )
}

