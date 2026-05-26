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
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

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
    import networkx as nx

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

class GraphData(BaseModel):
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]

class VisualizationRequest(BaseModel):
    graph_data: GraphData
    layout_algorithm: str = "force_atlas2"  # force_atlas2, fruchterman_reingold, spectral
    clustering_algorithm: str = "leiden"     # leiden, louvain, spectral
    gpu_acceleration: bool = True
    compute_centrality: bool = True
    
class GPUGraphProcessor:
    """GPU-accelerated graph processing using cuGraph"""
    
    def __init__(self):
        self.use_gpu = HAS_RAPIDS
        logger.info(f"GPU Graph Processor initialized (GPU: {self.use_gpu})")
    
    def create_cugraph_from_data(self, nodes: List[Dict], edges: List[Dict]) -> 'cugraph.Graph':
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
                # Use spectral as fallback since FR might not be available
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

class LocalGPUVisualizationService:
    """Local GPU-powered interactive graph visualization service"""
    
    def __init__(self):
        self.gpu_processor = GPUGraphProcessor()
        self.active_connections: List[WebSocket] = []
        
    async def process_graph(self, request: VisualizationRequest) -> Dict[str, Any]:
        """Process graph with GPU acceleration"""
        try:
            nodes = request.graph_data.nodes
            edges = request.graph_data.links
            
            result = {
                "nodes": nodes.copy(),
                "edges": edges.copy(),
                "gpu_processed": False,
                "layout_positions": {},
                "clusters": {},
                "centrality": {},
                "stats": {},
                "timestamp": datetime.now().isoformat()
            }
            
            if request.gpu_acceleration and self.gpu_processor.use_gpu:
                logger.info("=== GPU PROCESSING START ===")
                
                # Create cuGraph
                G, edges_df = self.gpu_processor.create_cugraph_from_data(nodes, edges)
                
                # Compute layout on GPU
                positions = self.gpu_processor.compute_gpu_layout(G, request.layout_algorithm)
                if positions:
                    result["layout_positions"] = positions
                    # Add positions to nodes
                    for node in result["nodes"]:
                        node_id = str(node["id"])
                        if node_id in positions:
                            node["x"], node["y"] = positions[node_id]
                
                # Compute clustering on GPU
                clusters = self.gpu_processor.compute_gpu_clustering(G, request.clustering_algorithm)
                if clusters:
                    result["clusters"] = clusters
                    # Add cluster info to nodes
                    for node in result["nodes"]:
                        node_id = str(node["id"])
                        if node_id in clusters:
                            node["cluster"] = clusters[node_id]
                
                # Compute centrality on GPU
                if request.compute_centrality:
                    centrality = self.gpu_processor.compute_gpu_centrality(G)
                    result["centrality"] = centrality
                    # Add centrality to nodes
                    for node in result["nodes"]:
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
                
                logger.info("=== GPU PROCESSING COMPLETE ===")
            
            return result
            
        except Exception as e:
            logger.error(f"Graph processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
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
app = FastAPI(title="Local GPU Graph Visualization", version="1.0.0")
service = LocalGPUVisualizationService()

@app.post("/api/process")
async def process_graph(request: VisualizationRequest):
    """Process graph with local GPU acceleration"""
    result = await service.process_graph(request)
    
    # Broadcast to connected WebSocket clients
    await service.broadcast_update({
        "type": "graph_processed",
        "data": result
    })
    
    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    service.active_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        service.active_connections.remove(websocket)

@app.get("/api/capabilities")
async def get_capabilities():
    """Get GPU capabilities"""
    return {
        "has_rapids": HAS_RAPIDS,
        "has_torch_geometric": HAS_TORCH_GEOMETRIC,
        "gpu_available": HAS_RAPIDS,
        "supported_layouts": ["force_atlas2", "spectral", "fruchterman_reingold"],
        "supported_clustering": ["leiden", "louvain", "spectral"],
        "gpu_memory": "N/A"  # Could add GPU memory info here
    }

@app.get("/", response_class=HTMLResponse)
async def get_visualization_page():
    """Serve the interactive visualization page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Local GPU Graph Visualization</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body { margin: 0; font-family: Arial, sans-serif; background: #1a1a1a; color: white; }
            #controls { position: absolute; top: 10px; left: 10px; z-index: 100; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 5px; }
            #graph { width: 100vw; height: 100vh; }
            .node { cursor: pointer; }
            .link { stroke: #999; stroke-opacity: 0.6; }
            button { margin: 5px; padding: 5px 10px; }
        </style>
    </head>
    <body>
        <div id="controls">
            <h3>🚀 Local GPU Visualization</h3>
            <button onclick="loadSampleGraph()">Load Sample Graph</button>
            <div id="status">Ready</div>
        </div>
        <div id="graph"></div>
        
        <script>
            const width = window.innerWidth;
            const height = window.innerHeight;
            
            const svg = d3.select("#graph")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
            
            const g = svg.append("g");
            
            // Add zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on("zoom", (event) => {
                    g.attr("transform", event.transform);
                });
            
            svg.call(zoom);
            
            // WebSocket connection for real-time updates
            const ws = new WebSocket(`ws://localhost:8081/ws`);
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                if (message.type === 'graph_processed') {
                    renderGraph(message.data);
                }
            };
            
            function renderGraph(data) {
                console.log("Rendering graph with", data.nodes.length, "nodes");
                
                // Clear previous graph
                g.selectAll("*").remove();
                
                // Create links
                const links = g.selectAll(".link")
                    .data(data.edges)
                    .enter().append("line")
                    .attr("class", "link")
                    .attr("stroke-width", 1);
                
                // Create nodes
                const nodes = g.selectAll(".node")
                    .data(data.nodes)
                    .enter().append("circle")
                    .attr("class", "node")
                    .attr("r", d => Math.sqrt((d.pagerank || 0.001) * 1000) + 2)
                    .attr("fill", d => d3.schemeCategory10[d.cluster % 10] || "#69b3a2")
                    .attr("stroke", "#fff")
                    .attr("stroke-width", 1.5);
                
                // Add node labels for important nodes
                const labels = g.selectAll(".label")
                    .data(data.nodes.filter(d => (d.pagerank || 0) > 0.01))
                    .enter().append("text")
                    .attr("class", "label")
                    .attr("dy", -3)
                    .attr("text-anchor", "middle")
                    .style("font-size", "10px")
                    .style("fill", "white")
                    .text(d => d.id);
                
                // Position nodes using GPU-computed coordinates
                if (data.layout_positions && Object.keys(data.layout_positions).length > 0) {
                    nodes.attr("cx", d => (data.layout_positions[d.id] && data.layout_positions[d.id][0]) || width/2)
                          .attr("cy", d => (data.layout_positions[d.id] && data.layout_positions[d.id][1]) || height/2);
                    
                    labels.attr("x", d => (data.layout_positions[d.id] && data.layout_positions[d.id][0]) || width/2)
                           .attr("y", d => (data.layout_positions[d.id] && data.layout_positions[d.id][1]) || height/2);
                    
                    links.attr("x1", d => (data.layout_positions[d.source] && data.layout_positions[d.source][0]) || width/2)
                         .attr("y1", d => (data.layout_positions[d.source] && data.layout_positions[d.source][1]) || height/2)
                         .attr("x2", d => (data.layout_positions[d.target] && data.layout_positions[d.target][0]) || width/2)
                         .attr("y2", d => (data.layout_positions[d.target] && data.layout_positions[d.target][1]) || height/2);
                } else {
                    // Fallback to force simulation
                    const simulation = d3.forceSimulation(data.nodes)
                        .force("link", d3.forceLink(data.edges).id(d => d.id))
                        .force("charge", d3.forceManyBody().strength(-30))
                        .force("center", d3.forceCenter(width / 2, height / 2));
                    
                    simulation.on("tick", () => {
                        links.attr("x1", d => d.source.x)
                             .attr("y1", d => d.source.y)
                             .attr("x2", d => d.target.x)
                             .attr("y2", d => d.target.y);
                        
                        nodes.attr("cx", d => d.x)
                             .attr("cy", d => d.y);
                        
                        labels.attr("x", d => d.x)
                              .attr("y", d => d.y);
                    });
                }
                
                // Add tooltips
                nodes.append("title")
                     .text(d => `Node: ${d.id}\\nCluster: ${d.cluster || 'N/A'}\\nPageRank: ${(d.pagerank || 0).toFixed(4)}`);
                
                document.getElementById("status").innerHTML = 
                    `Rendered ${data.nodes.length} nodes, ${data.edges.length} edges (GPU: ${data.gpu_processed})`;
            }
            
            async function loadSampleGraph() {
                // This would load your graph data and send it for processing
                document.getElementById("status").innerHTML = "Loading sample graph...";
                
                // You can integrate this with your existing graph generation
                // For now, this is a placeholder
                alert("Connect this to your graph generation service!");
            }
        </script>
    </body>
    </html>
    """

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_available": HAS_RAPIDS,
        "torch_geometric": HAS_TORCH_GEOMETRIC,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081) 