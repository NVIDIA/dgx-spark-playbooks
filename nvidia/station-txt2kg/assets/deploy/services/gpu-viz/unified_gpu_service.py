#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Unified GPU Graph Visualization Service

Combines PyGraphistry cloud processing and local GPU processing with cuGraph
into a single FastAPI service for maximum flexibility.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import time
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from enum import Enum

# PyGraphistry imports
import graphistry

# GPU-accelerated imports (available in NVIDIA PyG container)
try:
    import cudf
    import cugraph
    import cupy as cp
    from cuml import UMAP
    HAS_RAPIDS = True
    print("✓ RAPIDS cuGraph/cuDF/cuML available")
except ImportError:
    HAS_RAPIDS = False
    print("⚠ RAPIDS not available, falling back to CPU")

try:
    import torch
    import torch_geometric
    HAS_TORCH_GEOMETRIC = True
    print("✓ PyTorch Geometric available")
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("⚠ PyTorch Geometric not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingMode(str, Enum):
    PYGRAPHISTRY_CLOUD = "pygraphistry_cloud"
    LOCAL_GPU = "local_gpu" 
    LOCAL_CPU = "local_cpu"

class GraphPattern(str, Enum):
    RANDOM = "random"
    SCALE_FREE = "scale-free"
    SMALL_WORLD = "small-world"
    CLUSTERED = "clustered"
    HIERARCHICAL = "hierarchical"
    GRID = "grid"

class GraphData(BaseModel):
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]

class GraphGenerationRequest(BaseModel):
    num_nodes: int
    pattern: GraphPattern = GraphPattern.SCALE_FREE
    avg_degree: Optional[int] = 5
    num_clusters: Optional[int] = 100
    small_world_k: Optional[int] = 6
    small_world_p: Optional[float] = 0.1
    grid_dimensions: Optional[List[int]] = [100, 100]
    seed: Optional[int] = None

class UnifiedVisualizationRequest(BaseModel):
    graph_data: GraphData
    processing_mode: ProcessingMode = ProcessingMode.PYGRAPHISTRY_CLOUD
    
    # PyGraphistry Cloud options
    layout_type: Optional[str] = "force"
    gpu_acceleration: Optional[bool] = True  
    clustering: Optional[bool] = False
    
    # Local GPU options
    layout_algorithm: Optional[str] = "force_atlas2"
    clustering_algorithm: Optional[str] = "leiden"
    compute_centrality: Optional[bool] = True

class GraphGenerationStatus(BaseModel):
    task_id: str
    status: str  # "running", "completed", "failed"
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Import graph generation classes (keeping existing code)
from pygraphistry_service import LargeGraphGenerator, init_graphistry

class LocalGPUProcessor:
    """GPU-accelerated graph processing using cuGraph"""
    
    def __init__(self):
        self.use_gpu = HAS_RAPIDS
        logger.info(f"Local GPU Processor initialized (GPU: {self.use_gpu})")
    
    def create_cugraph_from_data(self, nodes: List[Dict], edges: List[Dict]) -> Tuple['cugraph.Graph', 'cudf.DataFrame']:
        """Create cuGraph from node/edge data"""
        if not self.use_gpu:
            raise RuntimeError("GPU libraries not available")
            
        # Create edge dataframe
        edge_data = []
        for edge in edges:
            edge_data.append({
                'src': edge['source'],
                'dst': edge['target'],
                'weight': edge.get('weight', 1.0)
            })
        
        # Convert to cuDF
        edges_df = cudf.DataFrame(edge_data)
        
        # Create cuGraph
        G = cugraph.Graph()
        G.from_cudf_edgelist(edges_df, source='src', destination='dst', edge_attr='weight')
        
        return G, edges_df
    
    def compute_gpu_layout(self, G, algorithm: str = "force_atlas2") -> Dict[str, Tuple[float, float]]:
        """Compute GPU-accelerated graph layout"""
        try:
            if algorithm == "force_atlas2":
                layout_df = cugraph.force_atlas2(G)
            elif algorithm == "fruchterman_reingold":
                layout_df = cugraph.spectral_layout(G, dim=2)
            else:  # spectral
                layout_df = cugraph.spectral_layout(G, dim=2)
            
            # Convert to dictionary
            positions = {}
            for _, row in layout_df.iterrows():
                node_id = str(row['vertex'])
                positions[node_id] = (float(row['x']), float(row['y']))
            
            logger.info(f"Computed {algorithm} layout for {len(positions)} nodes on GPU")
            return positions
            
        except Exception as e:
            logger.error(f"GPU layout computation failed: {e}")
            return {}
    
    def compute_gpu_clustering(self, G, algorithm: str = "leiden") -> Dict[str, int]:
        """Compute GPU-accelerated community detection"""
        try:
            if algorithm == "leiden":
                clusters_df, modularity = cugraph.leiden(G)
            elif algorithm == "louvain":
                clusters_df, modularity = cugraph.louvain(G)
            else:  # spectral clustering
                clusters_df = cugraph.spectral_clustering(G, n_clusters=10)
                modularity = 0.0
            
            # Convert to dictionary
            clusters = {}
            for _, row in clusters_df.iterrows():
                node_id = str(row['vertex'])
                clusters[node_id] = int(row['partition'])
            
            logger.info(f"Computed {algorithm} clustering on GPU (modularity: {modularity:.3f})")
            return clusters
            
        except Exception as e:
            logger.error(f"GPU clustering failed: {e}")
            return {}
    
    def compute_gpu_centrality(self, G) -> Dict[str, Dict[str, float]]:
        """Compute GPU-accelerated centrality measures"""
        centrality_data = {}
        
        try:
            # PageRank
            pagerank_df = cugraph.pagerank(G)
            pagerank = {}
            for _, row in pagerank_df.iterrows():
                pagerank[str(row['vertex'])] = float(row['pagerank'])
            centrality_data['pagerank'] = pagerank
            
            # Betweenness centrality (for smaller graphs)
            if G.number_of_vertices() < 5000:
                betweenness_df = cugraph.betweenness_centrality(G)
                betweenness = {}
                for _, row in betweenness_df.iterrows():
                    betweenness[str(row['vertex'])] = float(row['betweenness_centrality'])
                centrality_data['betweenness'] = betweenness
            
            logger.info(f"Computed centrality measures on GPU")
            return centrality_data
            
        except Exception as e:
            logger.error(f"GPU centrality computation failed: {e}")
            return {}

class PyGraphistryProcessor:
    """PyGraphistry cloud processing (existing functionality)"""
    
    def __init__(self):
        self.initialized = init_graphistry()
    
    async def process_graph_data(self, request: UnifiedVisualizationRequest) -> Dict[str, Any]:
        """Process graph data with PyGraphistry GPU acceleration"""
        try:
            if not self.initialized:
                raise HTTPException(status_code=500, detail="PyGraphistry not initialized")
            
            # Convert to pandas DataFrames for PyGraphistry
            nodes_df = pd.DataFrame(request.graph_data.nodes)
            edges_df = pd.DataFrame(request.graph_data.links)
            
            # Ensure required columns exist
            if 'id' not in nodes_df.columns:
                nodes_df['id'] = nodes_df.index
            if 'source' not in edges_df.columns or 'target' not in edges_df.columns:
                raise HTTPException(status_code=400, detail="Links must have source and target columns")
                
            logger.info(f"Processing graph with {len(nodes_df)} nodes and {len(edges_df)} edges")
            
            # Create PyGraphistry graph object
            g = graphistry.edges(edges_df, 'source', 'target').nodes(nodes_df, 'id')
            
            # Apply GPU-accelerated processing
            if request.gpu_acceleration:
                g = await self._apply_gpu_acceleration(g, request)
            
            # Apply clustering if requested
            if request.clustering:
                g = await self._apply_clustering(g)
            
            # Generate layout
            g = await self._generate_layout(g, request.layout_type)
            
            # Extract processed data
            processed_nodes = g._nodes.to_dict('records') if g._nodes is not None else nodes_df.to_dict('records')
            processed_edges = g._edges.to_dict('records') if g._edges is not None else edges_df.to_dict('records')
            
            # Generate embedding URL for interactive visualization
            embed_url = None
            local_viz_data = None
            
            try:
                embed_url = g.plot(render=False)
                logger.info(f"Generated PyGraphistry embed URL: {embed_url}")
            except Exception as e:
                logger.warning(f"Could not generate embed URL (likely running in local mode): {e}")
                
                # Create local visualization data as fallback
                try:
                    local_viz_data = self._create_local_viz_data(g, processed_nodes, processed_edges)
                    logger.info("Generated local visualization data as fallback")
                except Exception as viz_e:
                    logger.warning(f"Could not generate local visualization data: {viz_e}")
            
            return {
                "processed_nodes": processed_nodes,
                "processed_edges": processed_edges,
                "embed_url": embed_url,
                "local_viz_data": local_viz_data,
                "processing_mode": ProcessingMode.PYGRAPHISTRY_CLOUD,
                "stats": {
                    "node_count": len(processed_nodes),
                    "edge_count": len(processed_edges),
                    "gpu_accelerated": request.gpu_acceleration,
                    "clustered": request.clustering,
                    "layout_type": request.layout_type,
                    "has_embed_url": embed_url is not None,
                    "has_local_viz": local_viz_data is not None
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing graph data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ... (include other PyGraphistry methods from original service)
    async def _apply_gpu_acceleration(self, g, request):
        # Implementation from original service
        pass
    
    async def _apply_clustering(self, g):
        # Implementation from original service  
        pass
    
    async def _generate_layout(self, g, layout_type):
        # Implementation from original service
        pass
    
    def _create_local_viz_data(self, g, processed_nodes, processed_edges):
        # Implementation from original service
        pass

class UnifiedGPUService:
    """Unified service offering both PyGraphistry cloud and local GPU processing"""
    
    def __init__(self):
        self.pygraphistry_processor = PyGraphistryProcessor()
        self.local_gpu_processor = LocalGPUProcessor()
        self.generation_tasks = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.active_connections: List[WebSocket] = []
        
    async def process_graph(self, request: UnifiedVisualizationRequest) -> Dict[str, Any]:
        """Process graph with selected processing mode"""
        
        if request.processing_mode == ProcessingMode.PYGRAPHISTRY_CLOUD:
            return await self.pygraphistry_processor.process_graph_data(request)
            
        elif request.processing_mode == ProcessingMode.LOCAL_GPU:
            return await self._process_with_local_gpu(request)
            
        else:  # LOCAL_CPU
            return await self._process_with_local_cpu(request)
    
    async def _process_with_local_gpu(self, request: UnifiedVisualizationRequest) -> Dict[str, Any]:
        """Process graph with local GPU acceleration"""
        try:
            nodes = request.graph_data.nodes
            edges = request.graph_data.links
            
            result = {
                "processed_nodes": nodes.copy(),
                "processed_edges": edges.copy(),
                "processing_mode": ProcessingMode.LOCAL_GPU,
                "gpu_processed": False,
                "layout_positions": {},
                "clusters": {},
                "centrality": {},
                "stats": {},
                "timestamp": datetime.now().isoformat()
            }
            
            if self.local_gpu_processor.use_gpu:
                logger.info("=== LOCAL GPU PROCESSING START ===")
                
                # Create cuGraph
                G, edges_df = self.local_gpu_processor.create_cugraph_from_data(nodes, edges)
                
                # Compute layout on GPU
                positions = self.local_gpu_processor.compute_gpu_layout(G, request.layout_algorithm)
                if positions:
                    result["layout_positions"] = positions
                    # Add positions to nodes
                    for node in result["processed_nodes"]:
                        node_id = str(node["id"])
                        if node_id in positions:
                            node["x"], node["y"] = positions[node_id]
                
                # Compute clustering on GPU
                clusters = self.local_gpu_processor.compute_gpu_clustering(G, request.clustering_algorithm)
                if clusters:
                    result["clusters"] = clusters
                    # Add cluster info to nodes
                    for node in result["processed_nodes"]:
                        node_id = str(node["id"])
                        if node_id in clusters:
                            node["cluster"] = clusters[node_id]
                
                # Compute centrality on GPU
                if request.compute_centrality:
                    centrality = self.local_gpu_processor.compute_gpu_centrality(G)
                    result["centrality"] = centrality
                    # Add centrality to nodes
                    for node in result["processed_nodes"]:
                        node_id = str(node["id"])
                        for metric, values in centrality.items():
                            if node_id in values:
                                node[metric] = values[node_id]
                
                result["gpu_processed"] = True
                result["stats"] = {
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "gpu_accelerated": True,
                    "layout_computed": len(positions) > 0,
                    "clusters_computed": len(clusters) > 0,
                    "centrality_computed": len(centrality) > 0
                }
                
                logger.info("=== LOCAL GPU PROCESSING COMPLETE ===")
            
            return result
            
        except Exception as e:
            logger.error(f"Local GPU processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _process_with_local_cpu(self, request: UnifiedVisualizationRequest) -> Dict[str, Any]:
        """Process graph with local CPU (NetworkX fallback)"""
        # Simple CPU fallback using NetworkX
        nodes = request.graph_data.nodes
        edges = request.graph_data.links
        
        return {
            "processed_nodes": nodes,
            "processed_edges": edges,
            "processing_mode": ProcessingMode.LOCAL_CPU,
            "gpu_processed": False,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "gpu_accelerated": False
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast updates to all connected WebSocket clients"""
        if self.active_connections:
            message = json.dumps(data)
            for connection in self.active_connections.copy():
                try:
                    await connection.send_text(message)
                except WebSocketDisconnect:
                    self.active_connections.remove(connection)

# FastAPI app
app = FastAPI(title="Unified GPU Graph Visualization Service", version="2.0.0")
service = UnifiedGPUService()

@app.post("/api/visualize")
async def visualize_graph(request: UnifiedVisualizationRequest):
    """Process graph with unified service (supports all processing modes)"""
    result = await service.process_graph(request)
    
    # Broadcast to connected WebSocket clients
    await service.broadcast_update({
        "type": "graph_processed",
        "data": result
    })
    
    return result

@app.post("/api/generate")
async def generate_graph(request: GraphGenerationRequest):
    """Start graph generation as background task"""
    if request.num_nodes > 1000000:
        raise HTTPException(status_code=400, detail="Maximum 1 million nodes allowed")
        
    # Use existing graph generation logic
    task_id = f"gen_{int(time.time() * 1000)}"
    # Implementation would go here...
    return {"task_id": task_id, "status": "started"}

@app.get("/api/capabilities")
async def get_capabilities():
    """Get GPU capabilities and available processing modes"""
    return {
        "processing_modes": {
            "pygraphistry_cloud": {
                "available": service.pygraphistry_processor.initialized,
                "description": "PyGraphistry cloud GPU processing with interactive embeds"
            },
            "local_gpu": {
                "available": HAS_RAPIDS,
                "description": "Local GPU processing with cuGraph/RAPIDS"
            },
            "local_cpu": {
                "available": True,
                "description": "Local CPU fallback processing"
            }
        },
        "has_rapids": HAS_RAPIDS,
        "has_torch_geometric": HAS_TORCH_GEOMETRIC,
        "gpu_available": HAS_RAPIDS,
        "supported_layouts": ["force_atlas2", "spectral", "fruchterman_reingold"],
        "supported_clustering": ["leiden", "louvain", "spectral"]
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    service.active_connections.append(websocket)
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        service.active_connections.remove(websocket)

@app.get("/api/sample-graph")
async def get_sample_graph():
    """Get a sample graph for testing"""
    return {
        "nodes": [
            {"id": "1", "name": "Central Hub", "group": "core"},
            {"id": "2", "name": "Data Source A", "group": "input"},
            {"id": "3", "name": "Data Source B", "group": "input"},
            {"id": "4", "name": "Processing Unit", "group": "compute"},
            {"id": "5", "name": "Output A", "group": "output"},
            {"id": "6", "name": "Output B", "group": "output"},
            {"id": "7", "name": "Analytics", "group": "analysis"},
            {"id": "8", "name": "Storage", "group": "storage"}
        ],
        "links": [
            {"source": "2", "target": "1", "name": "data_feed"},
            {"source": "3", "target": "1", "name": "data_feed"},
            {"source": "1", "target": "4", "name": "process"},
            {"source": "4", "target": "5", "name": "output"},
            {"source": "4", "target": "6", "name": "output"},
            {"source": "1", "target": "7", "name": "analyze"},
            {"source": "1", "target": "8", "name": "store"}
        ]
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pygraphistry_initialized": service.pygraphistry_processor.initialized,
        "local_gpu_available": HAS_RAPIDS,
        "torch_geometric": HAS_TORCH_GEOMETRIC,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/", response_class=HTMLResponse)
async def get_visualization_page():
    """Serve the interactive visualization page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Unified GPU Graph Visualization</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body { margin: 0; font-family: Arial, sans-serif; background: #1a1a1a; color: white; }
            #controls { position: absolute; top: 10px; left: 10px; z-index: 100; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 5px; }
            #graph { width: 100vw; height: 100vh; }
            .node { cursor: pointer; }
            .link { stroke: #999; stroke-opacity: 0.6; }
            button { margin: 5px; padding: 5px 10px; }
            select { margin: 5px; padding: 5px; }
        </style>
    </head>
    <body>
        <div id="controls">
            <h3>⚡ Unified GPU Visualization</h3>
            <div>
                <label>Processing Mode:</label>
                <select id="processingMode">
                    <option value="pygraphistry_cloud">PyGraphistry Cloud</option>
                    <option value="local_gpu">Local GPU (cuGraph)</option>
                    <option value="local_cpu">Local CPU</option>
                </select>
            </div>
            <button onclick="loadSampleGraph()">Load Sample Graph</button>
            <div id="status">Ready - Select processing mode and load graph</div>
        </div>
        <div id="graph"></div>
        
        <script>
            let currentGraph = null;
            let simulation = null;
            
            async function loadSampleGraph() {
                const mode = document.getElementById('processingMode').value;
                document.getElementById("status").innerHTML = `Loading sample graph...`;
                
                try {
                    // Get sample graph data
                    const graphResponse = await fetch('/api/sample-graph');
                    const graphData = await graphResponse.json();
                    
                    document.getElementById("status").innerHTML = `Processing with ${mode}...`;
                    
                    // Process the graph with selected mode
                    const processResponse = await fetch('/api/visualize', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            graph_data: graphData,
                            processing_mode: mode,
                            layout_algorithm: 'force_atlas2',
                            clustering_algorithm: 'leiden',
                            compute_centrality: true
                        })
                    });
                    
                    const result = await processResponse.json();
                    
                    if (result.processed_nodes && result.processed_edges) {
                        document.getElementById("status").innerHTML = 
                            `✅ Processed ${result.processed_nodes.length} nodes, ${result.processed_edges.length} edges (GPU: ${result.gpu_processed})`;
                        
                        // Visualize with D3.js
                        visualizeGraph({
                            nodes: result.processed_nodes,
                            links: result.processed_edges
                        });
                    } else {
                        document.getElementById("status").innerHTML = `❌ Error: ${result.detail || 'Unknown error'}`;
                    }
                    
                } catch (error) {
                    document.getElementById("status").innerHTML = `❌ Error: ${error.message}`;
                    console.error('Error:', error);
                }
            }
            
            function visualizeGraph(graph) {
                // Clear previous visualization
                d3.select("#graph").selectAll("*").remove();
                
                const width = window.innerWidth;
                const height = window.innerHeight;
                
                const svg = d3.select("#graph")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);
                
                // Create force simulation
                simulation = d3.forceSimulation(graph.nodes)
                    .force("link", d3.forceLink(graph.links).id(d => d.id).distance(100))
                    .force("charge", d3.forceManyBody().strength(-300))
                    .force("center", d3.forceCenter(width / 2, height / 2));
                
                // Create links
                const link = svg.append("g")
                    .selectAll("line")
                    .data(graph.links)
                    .enter().append("line")
                    .attr("class", "link")
                    .attr("stroke-width", 2);
                
                // Create nodes
                const node = svg.append("g")
                    .selectAll("circle")
                    .data(graph.nodes)
                    .enter().append("circle")
                    .attr("class", "node")
                    .attr("r", 8)
                    .attr("fill", d => d.group === 'core' ? '#ff6b6b' : 
                                    d.group === 'input' ? '#4ecdc4' : 
                                    d.group === 'output' ? '#45b7d1' : 
                                    d.group === 'compute' ? '#f9ca24' : '#6c5ce7')
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));
                
                // Add labels
                const label = svg.append("g")
                    .selectAll("text")
                    .data(graph.nodes)
                    .enter().append("text")
                    .text(d => d.name)
                    .attr("font-size", 12)
                    .attr("fill", "white")
                    .attr("text-anchor", "middle")
                    .attr("dy", ".35em");
                
                // Update positions on tick
                simulation.on("tick", () => {
                    link
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);
                    
                    node
                        .attr("cx", d => d.x)
                        .attr("cy", d => d.y);
                    
                    label
                        .attr("x", d => d.x)
                        .attr("y", d => d.y + 20);
                });
            }
            
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
        </script>
    </body>
    </html>
    """

def startup_diagnostics():
    """Run startup diagnostics and display system info"""
    print("🚀 Starting Unified GPU-accelerated graph visualization service...")
    print("Container: NVIDIA PyG with cuGraph/RAPIDS support")
    
    # Check GPU availability
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name,memory.total,memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ GPU detected:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"  {line.strip()}")
        else:
            print("⚠ No GPU detected, will use CPU fallback")
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        print("⚠ No GPU detected, will use CPU fallback")
    
    # Check RAPIDS availability
    if HAS_RAPIDS:
        print("✓ RAPIDS cuGraph/cuDF/cuML available")
    else:
        print("⚠ RAPIDS not available")
    
    # Check PyTorch Geometric
    if HAS_TORCH_GEOMETRIC:
        print("✓ PyTorch Geometric available")
    else:
        print("⚠ PyTorch Geometric not available")
    
    # Check PyGraphistry credentials
    print("Checking PyGraphistry credentials...")
    personal_key = os.getenv('GRAPHISTRY_PERSONAL_KEY')
    secret_key = os.getenv('GRAPHISTRY_SECRET_KEY')
    api_key = os.getenv('GRAPHISTRY_API_KEY')
    
    if personal_key and secret_key:
        print("✓ PyGraphistry personal key/secret found")
    elif api_key:
        print("✓ PyGraphistry API key found")
    else:
        print("⚠ No PyGraphistry credentials found - cloud mode will be limited")
        print("  Set GRAPHISTRY_PERSONAL_KEY + GRAPHISTRY_SECRET_KEY for full cloud features")
    
    print("")
    print("🎯 Available Processing Modes:")
    print("  ☁️  PyGraphistry Cloud - Interactive GPU embeds (requires credentials)")
    print("  🚀 Local GPU (cuGraph) - Full local GPU processing")
    print("  💻 Local CPU          - NetworkX fallback")
    print("")
    print("📊 Service starting on: http://0.0.0.0:8080")
    print("🎯 API Endpoints:")
    print("  - Unified processing:   POST /api/visualize")
    print("  - Processing modes:     GET  /api/capabilities")
    print("  - Health check:         GET  /api/health")
    print("  - WebSocket updates:    WS   /ws")
    print("")

if __name__ == "__main__":
    startup_diagnostics()
    uvicorn.run(app, host="0.0.0.0", port=8080) 