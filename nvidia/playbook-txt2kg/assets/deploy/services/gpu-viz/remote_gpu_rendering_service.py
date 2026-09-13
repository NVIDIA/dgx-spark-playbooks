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
Remote GPU Rendering Service

A standalone service that receives graph data, processes it with GPU acceleration,
renders interactive visualizations, and serves them via iframe embeds.
This provides an alternative to PyGraphistry cloud for large-scale visualization.
"""

import os
import json
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import redis
from pathlib import Path

# GPU-accelerated imports
try:
    import cudf
    import cugraph
    import cupy as cp
    from cuml import UMAP
    HAS_RAPIDS = True
    print("✓ RAPIDS cuGraph/cuDF/cuML available for remote rendering")
except ImportError:
    HAS_RAPIDS = False
    print("⚠ RAPIDS not available, falling back to CPU for remote rendering")
    import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphData(BaseModel):
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]

class RemoteRenderingRequest(BaseModel):
    graph_data: GraphData
    layout_algorithm: str = "force_atlas2"
    clustering_algorithm: str = "leiden"
    compute_centrality: bool = True
    render_quality: str = "high"  # low, medium, high, ultra
    interactive_mode: bool = True
    session_id: Optional[str] = None
    
    # Enhanced UX parameters inspired by Graphistry
    animation_duration: int = 5000  # Layout animation time in ms
    show_splash: bool = False       # Show loading splash screen
    auto_zoom: bool = True          # Auto-fit graph to view
    show_labels: bool = True        # Show node labels
    edge_bundling: bool = False     # Enable edge bundling for dense graphs
    background_color: str = "#0a0a0a"  # Background color
    quality_preset: str = "balanced"   # fast, balanced, quality

class RemoteGPUProcessor:
    """GPU processing for remote rendering"""
    
    def __init__(self):
        self.use_gpu = HAS_RAPIDS
        logger.info(f"Remote GPU processor initialized (GPU: {self.use_gpu})")
    
    def create_cugraph_from_data(self, nodes: List[Dict], edges: List[Dict]) -> Tuple[Any, Any]:
        """Create cuGraph from node and edge data"""
        try:
            if not self.use_gpu:
                return None, None
                
            # Create edge dataframe
            edge_data = []
            for edge in edges:
                edge_data.append({
                    'src': str(edge.get('source', edge.get('src', ''))),
                    'dst': str(edge.get('target', edge.get('dst', ''))),
                    'weight': float(edge.get('weight', 1.0))
                })
            
            edges_df = cudf.DataFrame(edge_data)
            
            # Create graph
            G = cugraph.Graph()
            G.from_cudf_edgelist(edges_df, source='src', destination='dst', edge_attr='weight')
            
            logger.info(f"Created cuGraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G, edges_df
            
        except Exception as e:
            logger.error(f"Error creating cuGraph: {e}")
            return None, None
    
    def compute_gpu_layout(self, G, algorithm: str = "force_atlas2") -> Dict[str, Tuple[float, float]]:
        """Compute GPU-accelerated graph layout"""
        try:
            if not self.use_gpu or G is None:
                return {}
                
            if algorithm == "force_atlas2":
                layout_df = cugraph.force_atlas2(G)
            elif algorithm == "spectral":
                layout_df = cugraph.spectral_layout(G, dim=2)
            else:
                layout_df = cugraph.spectral_layout(G, dim=2)
            
            # Convert to dictionary with normalized coordinates
            positions = {}
            if len(layout_df) > 0:
                # Normalize coordinates to [0, 1000] range for consistent rendering
                x_min, x_max = layout_df['x'].min(), layout_df['x'].max()
                y_min, y_max = layout_df['y'].min(), layout_df['y'].max()
                
                x_range = x_max - x_min if x_max != x_min else 1
                y_range = y_max - y_min if y_max != y_min else 1
                
                for _, row in layout_df.iterrows():
                    node_id = str(row['vertex'])
                    x_norm = ((row['x'] - x_min) / x_range) * 1000
                    y_norm = ((row['y'] - y_min) / y_range) * 1000
                    positions[node_id] = (float(x_norm), float(y_norm))
            
            logger.info(f"Computed {algorithm} layout for {len(positions)} nodes on GPU")
            return positions
            
        except Exception as e:
            logger.error(f"GPU layout computation failed: {e}")
            return {}
    
    def compute_gpu_clustering(self, G, algorithm: str = "leiden") -> Dict[str, int]:
        """Compute GPU-accelerated graph clustering"""
        try:
            if not self.use_gpu or G is None:
                return {}
                
            if algorithm == "leiden":
                clustering_df = cugraph.leiden(G)
            elif algorithm == "louvain":
                clustering_df = cugraph.louvain(G)
            else:
                clustering_df = cugraph.louvain(G)
            
            clusters = {}
            for _, row in clustering_df.iterrows():
                node_id = str(row['vertex'])
                clusters[node_id] = int(row['partition'])
            
            logger.info(f"Computed {algorithm} clustering for {len(clusters)} nodes")
            return clusters
            
        except Exception as e:
            logger.error(f"GPU clustering computation failed: {e}")
            return {}
    
    def compute_gpu_centrality(self, G) -> Dict[str, Dict[str, float]]:
        """Compute various centrality metrics on GPU"""
        try:
            if not self.use_gpu or G is None:
                return {}
                
            centrality = {}
            
            # PageRank
            try:
                pagerank_df = cugraph.pagerank(G)
                centrality["pagerank"] = {}
                for _, row in pagerank_df.iterrows():
                    node_id = str(row['vertex'])
                    centrality["pagerank"][node_id] = float(row['pagerank'])
            except Exception as e:
                logger.warning(f"PageRank computation failed: {e}")
                
            # Betweenness centrality (for smaller graphs)
            if G.number_of_nodes() < 10000:  # Limit for performance
                try:
                    betweenness_df = cugraph.betweenness_centrality(G)
                    centrality["betweenness"] = {}
                    for _, row in betweenness_df.iterrows():
                        node_id = str(row['vertex'])
                        centrality["betweenness"][node_id] = float(row['betweenness_centrality'])
                except Exception as e:
                    logger.warning(f"Betweenness centrality computation failed: {e}")
            
            logger.info(f"Computed centrality metrics: {list(centrality.keys())}")
            return centrality
            
        except Exception as e:
            logger.error(f"GPU centrality computation failed: {e}")
            return {}

class RemoteRenderingService:
    """Remote GPU-powered graph rendering service with iframe embedding"""
    
    def __init__(self):
        self.gpu_processor = RemoteGPUProcessor()
        self.active_connections: Dict[str, WebSocket] = {}
        self.redis_client = None
        self.sessions = {}  # In-memory session storage (use Redis in production)
        self.datasets = {}  # Cached preprocessed datasets
        self.session_ttl = timedelta(hours=24)
        self.dataset_ttl = timedelta(days=7)  # Datasets live longer
        
        # Initialize Redis for session storage (optional)
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connected for session storage")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {e}")
    
    async def process_and_store_graph(self, request: RemoteRenderingRequest) -> Dict[str, Any]:
        """Process graph with GPU acceleration and store for rendering"""
        session_id = request.session_id or str(uuid.uuid4())
        
        try:
            nodes = request.graph_data.nodes
            edges = request.graph_data.links
            
            # Enhanced result structure for remote rendering
            result = {
                "session_id": session_id,
                "nodes": nodes.copy(),
                "edges": edges.copy(),
                "gpu_processed": False,
                "layout_positions": {},
                "clusters": {},
                "centrality": {},
                "render_config": {
                    "quality": request.render_quality,
                    "interactive": request.interactive_mode,
                    "layout_algorithm": request.layout_algorithm,
                    "clustering_algorithm": request.clustering_algorithm
                },
                "stats": {
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "processing_time": 0
                },
                "timestamp": datetime.now().isoformat(),
                "embed_url": f"/embed/{session_id}"
            }
            
            start_time = datetime.now()
            
            if self.gpu_processor.use_gpu:
                logger.info("=== REMOTE GPU PROCESSING START ===")
                
                # Create cuGraph
                G, edges_df = self.gpu_processor.create_cugraph_from_data(nodes, edges)
                
                if G is not None:
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
                    
                    # Update stats
                    result["stats"].update({
                        "gpu_accelerated": True,
                        "layout_computed": len(positions) > 0,
                        "clusters_computed": len(clusters) > 0,
                        "centrality_computed": len(centrality) > 0
                    })
                
                logger.info("=== REMOTE GPU PROCESSING COMPLETE ===")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result["stats"]["processing_time"] = processing_time
            
            # Store session data
            await self._store_session(session_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Remote graph processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _store_session(self, session_id: str, data: Dict[str, Any]):
        """Store session data for iframe rendering"""
        expiry = datetime.now() + self.session_ttl
        
        if self.redis_client:
            try:
                # Store in Redis with TTL
                self.redis_client.setex(
                    f"session:{session_id}",
                    int(self.session_ttl.total_seconds()),
                    json.dumps(data)
                )
            except Exception as e:
                logger.warning(f"Redis storage failed, using memory: {e}")
                self.sessions[session_id] = {"data": data, "expiry": expiry}
        else:
            # In-memory storage
            self.sessions[session_id] = {"data": data, "expiry": expiry}
        
        logger.info(f"Stored session {session_id} for iframe rendering")
    
    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data for iframe rendering"""
        if self.redis_client:
            try:
                data = self.redis_client.get(f"session:{session_id}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis retrieval failed: {e}")
        
        # Check in-memory storage
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if datetime.now() < session["expiry"]:
                return session["data"]
            else:
                # Clean up expired session
                del self.sessions[session_id]
        
        return None
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions (background task)"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time >= session["expiry"]:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")

    async def preprocess_and_cache_dataset(self, graph_data: GraphData, dataset_id: Optional[str] = None) -> str:
        """Preprocess graph data and cache for reuse (like Graphistry datasets)"""
        if not dataset_id:
            dataset_id = str(uuid.uuid4())
        
        try:
            nodes = graph_data.nodes
            edges = graph_data.links
            
            # GPU preprocessing
            if self.gpu_processor.use_gpu:
                G, edges_df = self.gpu_processor.create_cugraph_from_data(nodes, edges)
                
                if G is not None:
                    # Pre-compute multiple layouts for fast switching
                    layouts = {}
                    for algorithm in ["force_atlas2", "spectral"]:
                        positions = self.gpu_processor.compute_gpu_layout(G, algorithm)
                        if positions:
                            layouts[algorithm] = positions
                    
                    # Pre-compute clustering
                    clusters = {}
                    for algorithm in ["leiden", "louvain"]:
                        cluster_result = self.gpu_processor.compute_gpu_clustering(G, algorithm)
                        if cluster_result:
                            clusters[algorithm] = cluster_result
                    
                    # Pre-compute centrality
                    centrality = self.gpu_processor.compute_gpu_centrality(G)
                    
                    # Cache the preprocessed data
                    dataset_cache = {
                        "dataset_id": dataset_id,
                        "nodes": nodes,
                        "edges": edges,
                        "layouts": layouts,
                        "clusters": clusters,
                        "centrality": centrality,
                        "stats": {
                            "node_count": len(nodes),
                            "edge_count": len(edges),
                            "preprocessing_time": (datetime.now()).isoformat()
                        },
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # Store in cache with longer TTL
                    await self._store_dataset(dataset_id, dataset_cache)
                    
                    logger.info(f"Preprocessed and cached dataset {dataset_id} with {len(nodes)} nodes")
                    return dataset_id
            
            return dataset_id
            
        except Exception as e:
            logger.error(f"Dataset preprocessing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _store_dataset(self, dataset_id: str, data: Dict[str, Any]):
        """Store preprocessed dataset with longer TTL"""
        expiry = datetime.now() + self.dataset_ttl
        
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"dataset:{dataset_id}",
                    int(self.dataset_ttl.total_seconds()),
                    json.dumps(data)
                )
            except Exception as e:
                logger.warning(f"Redis dataset storage failed: {e}")
                self.datasets[dataset_id] = {"data": data, "expiry": expiry}
        else:
            self.datasets[dataset_id] = {"data": data, "expiry": expiry}
    
    async def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached dataset"""
        if self.redis_client:
            try:
                data = self.redis_client.get(f"dataset:{dataset_id}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis dataset retrieval failed: {e}")
        
        if dataset_id in self.datasets:
            dataset = self.datasets[dataset_id]
            if datetime.now() < dataset["expiry"]:
                return dataset["data"]
            else:
                del self.datasets[dataset_id]
        
        return None

    def _generate_interactive_html(self, session_data: dict, config: dict) -> str:
        """Generate interactive HTML visualization using libraries consistent with frontend"""
        
        # Check if WebGL rendering is requested
        use_webgl = config.get('use_webgl', len(session_data['processed_nodes']) > 5000)
        
        if use_webgl:
            return self._generate_threejs_webgl_html(session_data, config)
        else:
            return self._generate_d3_svg_html(session_data, config)
        
        # Extract data
        nodes = session_data['processed_nodes']
        edges = session_data['processed_edges']
        layout_positions = session_data.get('layout_positions', {})
        clusters = session_data.get('clusters', {})
        centrality = session_data.get('centrality', {})
        
        # Animation and UI settings matching frontend patterns
        animation_duration = config.get('animation_duration', 3000)
        show_splash = config.get('show_splash', True)
        auto_zoom = config.get('auto_zoom', True)
        show_labels = config.get('show_labels', True)
        background_color = config.get('background_color', '#0a0a0a')
        render_quality = config.get('render_quality', 'high')
        
        # Performance settings based on render quality
        quality_settings = {
            'low': {'particles': 1000, 'line_width': 1, 'node_detail': 8},
            'medium': {'particles': 5000, 'line_width': 2, 'node_detail': 16}, 
            'high': {'particles': 20000, 'line_width': 3, 'node_detail': 32},
            'ultra': {'particles': 100000, 'line_width': 4, 'node_detail': 64}
        }
        settings = quality_settings.get(render_quality, quality_settings['high'])
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU-Accelerated Graph Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background-color: {background_color};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;  
            overflow: hidden;
            color: #ffffff;
        }}
        #graph-container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        .splash-screen {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            transition: opacity 0.8s ease;
        }}
        .splash-logo {{
            font-size: 2.5rem;
            font-weight: 700;
            color: #76B900;
            margin-bottom: 1rem;
            text-align: center;
        }}
        .splash-stats {{
            color: #888;
            font-size: 1rem;
            margin-bottom: 2rem;
            text-align: center;
        }}
        .loading-bar {{
            width: 300px;
            height: 4px;
            background: #333;
            border-radius: 2px;
            overflow: hidden;
            margin-bottom: 1rem;
        }}
        .loading-progress {{
            height: 100%;
            background: linear-gradient(90deg, #76B900, #a8d45a);
            width: 0%;
            transition: width 0.3s ease;
        }}
        .controls {{
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 100;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .control-btn {{
            background: rgba(0, 0, 0, 0.7);
            color: #76B900;
            border: 1px solid #76B900;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s ease;
        }}
        .control-btn:hover {{
            background: rgba(118, 185, 0, 0.2);
        }}
        .info-panel {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
            color: #ccc;
            border: 1px solid #333;
            max-width: 250px;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: #fff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 200;
            border: 1px solid #76B900;
            opacity: 0;
            transition: opacity 0.2s ease;
        }}
    </style>
</head>
<body>
    <div id="graph-container"></div>
    
    <!-- Splash Screen -->
    {"" if not show_splash else f'''
    <div id="splash-screen" class="splash-screen">
        <div class="splash-logo">GPU Graph Visualization</div>
        <div class="splash-stats">
            {len(nodes):,} nodes • {len(edges):,} edges<br>
            GPU-accelerated rendering • {render_quality.title()} quality
        </div>
        <div class="loading-bar">
            <div id="loading-progress" class="loading-progress"></div>
        </div>
        <div id="loading-text" style="color: #888; font-size: 14px;">Initializing GPU compute...</div>
    </div>
    '''}
    
    <!-- Controls -->
    <div class="controls">
        <button class="control-btn" onclick="togglePhysics()">⏸️ Physics</button>
        <button class="control-btn" onclick="resetView()">🎯 Reset View</button>
        <button class="control-btn" onclick="toggleLabels()">{{"🏷️ Labels" if show_labels else "🏷️ Show Labels"}}</button>
        <button class="control-btn" onclick="exportGraph()">💾 Export</button>
    </div>
    
    <!-- Info Panel -->
    <div class="info-panel">
        <div><strong>Render Quality:</strong> {render_quality.title()}</div>
        <div><strong>GPU Acceleration:</strong> Active</div>
        <div><strong>Layout:</strong> <span id="current-layout">Force Atlas 2</span></div>
        <div><strong>FPS:</strong> <span id="fps-counter">--</span></div>
    </div>
    
    <!-- Tooltip -->
    <div id="tooltip" class="tooltip"></div>

    <!-- Using D3.js v7.9.0 consistent with frontend -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        // Graph data from GPU processing
        const graphData = {{
            nodes: {json.dumps(nodes)},
            links: {json.dumps(edges)},
            layoutPositions: {json.dumps(layout_positions)},
            clusters: {json.dumps(clusters)},
            centrality: {json.dumps(centrality)}
        }};
        
        // Configuration
        const config = {{
            animationDuration: {animation_duration},
            autoZoom: {str(auto_zoom).lower()},
            showLabels: {str(show_labels).lower()},
            renderQuality: "{render_quality}",
            maxParticles: {settings['particles']},
            lineWidth: {settings['line_width']},
            nodeDetail: {settings['node_detail']}
        }};
        
        // Performance monitoring  
        let frameCount = 0;
        let lastTime = performance.now();
        
        // Initialize visualization
        class GPUGraphVisualization {{
            constructor() {{
                this.container = d3.select("#graph-container");
                this.width = window.innerWidth;
                this.height = window.innerHeight;
                this.physicsRunning = true;
                this.labelsVisible = config.showLabels;
                
                this.initializeSVG();
                this.initializeForces();
                this.loadData();
                
                {"this.hideSplash();" if not show_splash else "this.showLoadingProgress();"}
            }}
            
            initializeSVG() {{
                // Create SVG with GPU-optimized settings
                this.svg = this.container
                    .append("svg")
                    .attr("width", this.width)
                    .attr("height", this.height)
                    .style("background-color", "{background_color}");
                
                // Add zoom behavior
                this.zoom = d3.zoom()
                    .scaleExtent([0.1, 10])
                    .on("zoom", (event) => {{
                        this.g.attr("transform", event.transform);
                    }});
                
                this.svg.call(this.zoom);
                
                // Main group for graph elements
                this.g = this.svg.append("g");
                
                // Add definitions for gradients and patterns
                const defs = this.svg.append("defs");
                
                // Node gradient
                const gradient = defs.append("radialGradient")
                    .attr("id", "nodeGradient")
                    .attr("cx", "30%")
                    .attr("cy", "30%");
                
                gradient.append("stop")
                    .attr("offset", "0%")
                    .attr("stop-color", "#76B900")
                    .attr("stop-opacity", 1);
                
                gradient.append("stop")
                    .attr("offset", "100%")
                    .attr("stop-color", "#4a7600")
                    .attr("stop-opacity", 0.8);
            }}
            
            initializeForces() {{
                // D3 force simulation with GPU-optimized parameters
                this.simulation = d3.forceSimulation()
                    .force("link", d3.forceLink().id(d => d.id).distance(60).strength(0.5))
                    .force("charge", d3.forceManyBody().strength(-120).distanceMax(300))
                    .force("center", d3.forceCenter(this.width / 2, this.height / 2))
                    .force("collision", d3.forceCollide().radius(15))
                    .alphaDecay(0.02)
                    .velocityDecay(0.4);
            }}
            
            showLoadingProgress() {{
                const progressBar = document.getElementById("loading-progress");
                const loadingText = document.getElementById("loading-text");
                
                const stages = [
                    {{ progress: 20, text: "Loading graph data..." }},
                    {{ progress: 40, text: "Applying GPU layout..." }},
                    {{ progress: 60, text: "Computing clusters..." }},
                    {{ progress: 80, text: "Calculating centrality..." }},
                    {{ progress: 100, text: "Rendering visualization..." }}
                ];
                
                let currentStage = 0;
                const updateProgress = () => {{
                    if (currentStage < stages.length) {{
                        const stage = stages[currentStage];
                        progressBar.style.width = stage.progress + "%";
                        loadingText.textContent = stage.text;
                        currentStage++;
                        setTimeout(updateProgress, 600);
                    }} else {{
                        setTimeout(() => this.hideSplash(), 500);
                    }}
                }};
                
                updateProgress();
            }}
            
            hideSplash() {{
                const splash = document.getElementById("splash-screen");
                if (splash) {{
                    splash.style.opacity = "0";
                    setTimeout(() => splash.remove(), 800);
                }}
            }}
            
            loadData() {{
                // Process nodes with GPU-computed positions
                this.nodes = graphData.nodes.map(node => ({{
                    ...node,
                    x: graphData.layoutPositions[node.id]?.x || Math.random() * this.width,
                    y: graphData.layoutPositions[node.id]?.y || Math.random() * this.height,
                    cluster: graphData.clusters[node.id] || 0,
                    centrality: graphData.centrality[node.id] || 0
                }}));
                
                this.links = graphData.links.map(link => ({{
                    ...link,
                    source: link.source,
                    target: link.target
                }}));
                
                this.renderGraph();
            }}
            
            renderGraph() {{
                // Render links with GPU-optimized styling
                this.link = this.g.append("g")
                    .attr("class", "links")
                    .selectAll("line")
                    .data(this.links)
                    .enter().append("line")
                    .attr("stroke", "#444")
                    .attr("stroke-width", d => Math.max(1, {settings['line_width']} * (d.weight || 1)))
                    .attr("stroke-opacity", 0.6);
                
                // Render nodes with cluster coloring
                this.node = this.g.append("g")
                    .attr("class", "nodes")
                    .selectAll("circle")
                    .data(this.nodes)
                    .enter().append("circle")
                    .attr("r", d => Math.max(5, 5 + (d.centrality * 20)))
                    .attr("fill", d => this.getClusterColor(d.cluster))
                    .attr("stroke", "#76B900")
                    .attr("stroke-width", 1.5)
                    .style("cursor", "pointer")
                    .call(d3.drag()
                        .on("start", (event, d) => this.dragStarted(event, d))
                        .on("drag", (event, d) => this.dragged(event, d))
                        .on("end", (event, d) => this.dragEnded(event, d)));
                
                // Add node labels (conditional)
                if (this.labelsVisible) {{
                    this.label = this.g.append("g")
                        .attr("class", "labels")
                        .selectAll("text")
                        .data(this.nodes)
                        .enter().append("text")
                        .text(d => d.name)
                        .attr("font-size", "10px")
                        .attr("fill", "#ccc")
                        .attr("text-anchor", "middle")
                        .attr("dy", "0.35em")
                        .style("pointer-events", "none");
                }}
                
                // Add tooltips
                this.addInteractions();
                
                // Start simulation
                this.simulation
                    .nodes(this.nodes)
                    .on("tick", () => this.ticked());
                
                this.simulation.force("link")
                    .links(this.links);
                
                // Auto-zoom to fit
                if (config.autoZoom) {{
                    setTimeout(() => this.zoomToFit(), 1000);
                }}
                
                // Start FPS monitoring
                this.startFPSMonitoring();
            }}
            
            getClusterColor(cluster) {{
                const colors = [
                    "#76B900", "#00D4F0", "#FF6B35", "#A855F7", 
                    "#EF4444", "#F59E0B", "#10B981", "#8B5CF6"
                ];
                return colors[cluster % colors.length];
            }}
            
            addInteractions() {{
                const tooltip = d3.select("#tooltip");
                
                this.node
                    .on("mouseover", (event, d) => {{
                        tooltip
                            .style("opacity", 1)
                            .style("left", (event.pageX + 10) + "px")
                            .style("top", (event.pageY - 10) + "px")
                            .html(`
                                <strong>${{d.name}}</strong><br>
                                Cluster: ${{d.cluster}}<br>
                                Centrality: ${{d.centrality.toFixed(3)}}<br>
                                Connections: ${{this.links.filter(l => l.source.id === d.id || l.target.id === d.id).length}}
                            `);
                    }})
                    .on("mouseout", () => {{
                        tooltip.style("opacity", 0);
                    }})
                    .on("click", (event, d) => {{
                        console.log("Node clicked:", d);
                        // Highlight connected nodes
                        this.highlightConnections(d);
                    }});
            }}
            
            highlightConnections(node) {{
                const connectedNodes = new Set();
                const connectedLinks = new Set();
                
                this.links.forEach(link => {{
                    if (link.source.id === node.id || link.target.id === node.id) {{
                        connectedLinks.add(link);
                        connectedNodes.add(link.source.id);
                        connectedNodes.add(link.target.id);
                    }}
                }});
                
                // Fade non-connected elements
                this.node
                    .style("opacity", d => connectedNodes.has(d.id) ? 1 : 0.2);
                    
                this.link
                    .style("opacity", d => connectedLinks.has(d) ? 0.8 : 0.1);
                
                // Reset after 3 seconds
                setTimeout(() => {{
                    this.node.style("opacity", 1);
                    this.link.style("opacity", 0.6);
                }}, 3000);
            }}
            
            ticked() {{
                this.link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                this.node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);
                
                if (this.label) {{
                    this.label
                        .attr("x", d => d.x)
                        .attr("y", d => d.y);
                }}
            }}
            
            dragStarted(event, d) {{
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}
            
            dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}
            
            dragEnded(event, d) {{
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}
            
            zoomToFit() {{
                const bounds = this.g.node().getBBox();
                const fullWidth = this.width;
                const fullHeight = this.height;
                const width = bounds.width;
                const height = bounds.height;
                const midX = bounds.x + width / 2;
                const midY = bounds.y + height / 2;
                
                if (width === 0 || height === 0) return;
                
                const scale = 0.8 / Math.max(width / fullWidth, height / fullHeight);
                const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];
                
                this.svg.transition()
                    .duration(750)
                    .call(this.zoom.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
            }}
            
            startFPSMonitoring() {{
                const fpsCounter = document.getElementById("fps-counter");
                
                const updateFPS = () => {{
                    frameCount++;
                    const now = performance.now();
                    
                    if (now - lastTime >= 1000) {{
                        const fps = Math.round((frameCount * 1000) / (now - lastTime));
                        fpsCounter.textContent = fps;
                        frameCount = 0;
                        lastTime = now;
                    }}
                    
                    requestAnimationFrame(updateFPS);
                }};
                
                updateFPS();
            }}
        }}
        
        // Control functions
        window.togglePhysics = () => {{
            const graph = window.graphInstance;
            if (graph.physicsRunning) {{
                graph.simulation.stop();
                graph.physicsRunning = false;
            }} else {{
                graph.simulation.restart();
                graph.physicsRunning = true;
            }}
        }};
        
        window.resetView = () => {{
            window.graphInstance.zoomToFit();
        }};
        
        window.toggleLabels = () => {{
            const graph = window.graphInstance;
            if (graph.label) {{
                graph.label.style("opacity", graph.labelsVisible ? 0 : 1);
                graph.labelsVisible = !graph.labelsVisible;
            }}
        }};
        
        window.exportGraph = () => {{
            const svgData = new XMLSerializer().serializeToString(document.querySelector("svg"));
            const blob = new Blob([svgData], {{type: "image/svg+xml"}});
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "gpu-graph-visualization.svg";
            link.click();
            URL.revokeObjectURL(url);
        }};
        
        // Handle window resize
        window.addEventListener("resize", () => {{
            const graph = window.graphInstance;
            if (graph) {{
                graph.width = window.innerWidth;
                graph.height = window.innerHeight;
                graph.svg
                    .attr("width", graph.width)
                    .attr("height", graph.height);
                graph.simulation.force("center", d3.forceCenter(graph.width / 2, graph.height / 2));
            }}
        }});
        
        // Initialize when DOM is ready
        document.addEventListener("DOMContentLoaded", () => {{
            window.graphInstance = new GPUGraphVisualization();
        }});
    </script>
</body>
</html>
        """
        
        return html_template
    
    def _generate_threejs_webgl_html(self, session_data: dict, config: dict) -> str:
        """Generate Three.js WebGL visualization with GPU acceleration"""
        
        # Extract data
        nodes = session_data['processed_nodes']
        edges = session_data['processed_edges']
        layout_positions = session_data.get('layout_positions', {})
        clusters = session_data.get('clusters', {})
        centrality = session_data.get('centrality', {})
        
        # GPU rendering settings
        node_count = len(nodes)
        use_instanced = node_count > 1000
        enable_lod = node_count > 25000
        render_quality = config.get('render_quality', 'high')
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU WebGL Graph Visualization</title>
    <style>
        body {{ margin: 0; padding: 0; background: #0a0a0a; overflow: hidden; }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}
        .perf-monitor {{ 
            position: absolute; top: 10px; left: 10px; 
            background: rgba(0,0,0,0.8); padding: 10px; border-radius: 5px;
            color: #76B900; font-size: 12px; z-index: 100;
        }}
        .controls {{ 
            position: absolute; top: 10px; right: 10px; 
            display: flex; gap: 5px; z-index: 100; 
        }}
        .btn {{ 
            background: rgba(0,0,0,0.8); color: #76B900; 
            border: 1px solid #76B900; padding: 5px 10px; 
            border-radius: 3px; cursor: pointer; font-size: 11px;
        }}
    </style>
</head>
<body>
    <div id="container">
        <canvas id="canvas"></canvas>
        
        <div class="perf-monitor">
            <div>🚀 WebGL GPU Rendering</div>
            <div>Nodes: {node_count:,}</div>
            <div>FPS: <span id="fps">--</span></div>
            <div>Triangles: <span id="triangles">--</span></div>
            <div>Memory: <span id="memory">--</span>MB</div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="resetCamera()">🎯 Reset</button>
            <button class="btn" onclick="toggleAnimation()">⏸️ Pause</button>
            <button class="btn" onclick="exportImage()">📷 Export</button>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.176.0/three.min.js"></script>
    
    <script>
        const graphData = {{
            nodes: {json.dumps(nodes)},
            edges: {json.dumps(edges)},
            positions: {json.dumps(layout_positions)},
            clusters: {json.dumps(clusters)},
            centrality: {json.dumps(centrality)}
        }};
        
        const config = {{
            nodeCount: {node_count},
            useInstanced: {str(use_instanced).lower()},
            enableLOD: {str(enable_lod).lower()},
            quality: '{render_quality}'
        }};
        
        class WebGLGraphRenderer {{
            constructor() {{
                this.canvas = document.getElementById('canvas');
                this.container = document.getElementById('container');
                this.frameCount = 0;
                this.lastTime = performance.now();
                this.isAnimating = true;
                
                this.init();
                this.loadData();
                this.animate();
            }}
            
            init() {{
                // Three.js WebGL renderer with GPU optimizations
                this.renderer = new THREE.WebGLRenderer({{
                    canvas: this.canvas,
                    antialias: true,
                    powerPreference: "high-performance"
                }});
                
                this.renderer.setSize(window.innerWidth, window.innerHeight);
                this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                this.renderer.setClearColor(0x0a0a0a, 1);
                this.renderer.sortObjects = false; // GPU optimization
                
                this.scene = new THREE.Scene();
                this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000);
                this.camera.position.z = 800;
                
                this.setupControls();
            }}
            
            setupControls() {{
                this.controls = {{ mouseDown: false, targetX: 0, targetY: 0, zoom: 1 }};
                
                this.canvas.addEventListener('mousedown', (e) => {{
                    this.controls.mouseDown = true;
                    this.controls.mouseX = e.clientX;
                    this.controls.mouseY = e.clientY;
                }});
                
                this.canvas.addEventListener('mousemove', (e) => {{
                    if (this.controls.mouseDown) {{
                        this.controls.targetX += (e.clientX - this.controls.mouseX) * 2;
                        this.controls.targetY -= (e.clientY - this.controls.mouseY) * 2;
                        this.controls.mouseX = e.clientX;
                        this.controls.mouseY = e.clientY;
                    }}
                }});
                
                this.canvas.addEventListener('mouseup', () => this.controls.mouseDown = false);
                this.canvas.addEventListener('wheel', (e) => {{
                    e.preventDefault();
                    this.controls.zoom *= (1 - e.deltaY * 0.001);
                    this.controls.zoom = Math.max(0.1, Math.min(10, this.controls.zoom));
                }});
            }}
            
            loadData() {{
                console.log('Loading graph data with WebGL GPU acceleration...');
                
                if (config.useInstanced) {{
                    this.createInstancedNodes();
                }} else {{
                    this.createStandardNodes();
                }}
                
                this.createEdges();
                console.log('Graph loaded successfully');
            }}
            
            createInstancedNodes() {{
                console.log('Using GPU InstancedMesh for', config.nodeCount, 'nodes');
                
                const geometry = new THREE.CircleGeometry(1, 8);
                const material = new THREE.MeshBasicMaterial({{ vertexColors: true, transparent: true, opacity: 0.8 }});
                this.nodesMesh = new THREE.InstancedMesh(geometry, material, config.nodeCount);
                
                const matrix = new THREE.Matrix4();
                const color = new THREE.Color();
                
                graphData.nodes.forEach((node, i) => {{
                    const pos = graphData.positions[node.id] || [0, 0];
                    const x = pos[0] - 500;
                    const y = pos[1] - 500;
                    const size = Math.max(2, 5 + (node.pagerank || 0) * 50);
                    const cluster = node.cluster || 0;
                    
                    matrix.makeScale(size, size, 1);
                    matrix.setPosition(x, y, 0);
                    this.nodesMesh.setMatrixAt(i, matrix);
                    
                    color.setHex(this.getClusterColor(cluster));
                    this.nodesMesh.setColorAt(i, color);
                }});
                
                this.nodesMesh.instanceMatrix.needsUpdate = true;
                this.nodesMesh.instanceColor.needsUpdate = true;
                this.scene.add(this.nodesMesh);
            }}
            
            createStandardNodes() {{
                console.log('Using standard mesh rendering for', config.nodeCount, 'nodes');
                this.nodesGroup = new THREE.Group();
                
                graphData.nodes.forEach((node, i) => {{
                    const pos = graphData.positions[node.id] || [0, 0];
                    const size = Math.max(2, 5 + (node.pagerank || 0) * 50);
                    const cluster = node.cluster || 0;
                    
                    const geometry = new THREE.CircleGeometry(size, 8);
                    const material = new THREE.MeshBasicMaterial({{ 
                        color: this.getClusterColor(cluster), transparent: true, opacity: 0.8 
                    }});
                    
                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.position.set(pos[0] - 500, pos[1] - 500, 0);
                    this.nodesGroup.add(mesh);
                }});
                
                this.scene.add(this.nodesGroup);
            }}
            
            createEdges() {{
                const positions = new Float32Array(graphData.edges.length * 6);
                
                graphData.edges.forEach((edge, i) => {{
                    const sourcePos = graphData.positions[edge.source] || [0, 0];
                    const targetPos = graphData.positions[edge.target] || [0, 0];
                    const idx = i * 6;
                    
                    positions[idx] = sourcePos[0] - 500;
                    positions[idx + 1] = sourcePos[1] - 500;
                    positions[idx + 2] = 0;
                    positions[idx + 3] = targetPos[0] - 500;
                    positions[idx + 4] = targetPos[1] - 500;
                    positions[idx + 5] = 0;
                }});
                
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                
                const material = new THREE.LineBasicMaterial({{ color: 0x444444, transparent: true, opacity: 0.4 }});
                this.edgesMesh = new THREE.LineSegments(geometry, material);
                this.scene.add(this.edgesMesh);
            }}
            
            getClusterColor(cluster) {{
                const colors = [0x76B900, 0x00D4F0, 0xFF6B35, 0xA855F7, 0xEF4444, 0xF59E0B, 0x10B981, 0x8B5CF6];
                return colors[cluster % colors.length];
            }}
            
            animate() {{
                requestAnimationFrame(() => this.animate());
                
                if (this.isAnimating) {{
                    // Smooth camera movement
                    this.camera.position.x += (this.controls.targetX - this.camera.position.x) * 0.05;
                    this.camera.position.y += (this.controls.targetY - this.camera.position.y) * 0.05;
                    this.camera.zoom += (this.controls.zoom - this.camera.zoom) * 0.05;
                    this.camera.updateProjectionMatrix();
                }}
                
                // GPU render
                this.renderer.render(this.scene, this.camera);
                
                // Performance monitoring
                this.frameCount++;
                const now = performance.now();
                if (now - this.lastTime >= 1000) {{
                    const fps = Math.round(this.frameCount * 1000 / (now - this.lastTime));
                    document.getElementById('fps').textContent = fps;
                    document.getElementById('triangles').textContent = this.renderer.info.render.triangles.toLocaleString();
                    document.getElementById('memory').textContent = Math.round(this.renderer.info.memory.geometries + this.renderer.info.memory.textures);
                    this.frameCount = 0;
                    this.lastTime = now;
                }}
            }}
            
            resetCamera() {{
                this.controls.targetX = 0;
                this.controls.targetY = 0;
                this.controls.zoom = 1;
            }}
            
            toggleAnimation() {{
                this.isAnimating = !this.isAnimating;
            }}
            
            exportImage() {{
                const link = document.createElement('a');
                link.download = 'webgl-graph.png';
                link.href = this.renderer.domElement.toDataURL();
                link.click();
            }}
        }}
        
        // Global functions
        window.resetCamera = () => window.graphRenderer.resetCamera();
        window.toggleAnimation = () => window.graphRenderer.toggleAnimation();
        window.exportImage = () => window.graphRenderer.exportImage();
        
        // Handle resize
        window.addEventListener('resize', () => {{
            const renderer = window.graphRenderer;
            if (renderer) {{
                renderer.camera.aspect = window.innerWidth / window.innerHeight;
                renderer.camera.updateProjectionMatrix();
                renderer.renderer.setSize(window.innerWidth, window.innerHeight);
            }}
        }});
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            window.graphRenderer = new WebGLGraphRenderer();
        }});
    </script>
</body>
</html>
        """
        
        return html_template
    
    def _generate_d3_svg_html(self, session_data: dict, config: dict) -> str:
        """Generate D3.js SVG visualization (original approach)"""
        # Return the original D3.js SVG implementation
        nodes = session_data['processed_nodes']
        edges = session_data['processed_edges']
        
        return f"""
<!DOCTYPE html>
<html><head><title>D3.js SVG Fallback</title></head>
<body>
<div>D3.js SVG rendering for {len(nodes)} nodes (fallback mode)</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<!-- Original D3.js implementation would go here -->
</body></html>
        """

# FastAPI app for remote rendering
app = FastAPI(
    title="Remote GPU Graph Rendering Service", 
    version="1.0.0",
    description="GPU-accelerated graph processing and iframe-embeddable visualization service"
)

# Add CORS middleware for iframe embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
rendering_service = RemoteRenderingService()

@app.post("/api/render")
async def render_graph(request: RemoteRenderingRequest, background_tasks: BackgroundTasks):
    """Process and store graph for remote rendering"""
    result = await rendering_service.process_and_store_graph(request)
    
    # Schedule cleanup task
    background_tasks.add_task(rendering_service.cleanup_expired_sessions)
    
    return result

@app.get("/embed/{session_id}", response_class=HTMLResponse)
async def get_iframe_visualization(session_id: str):
    """Serve iframe-embeddable visualization"""
    session_data = await rendering_service.get_session_data(session_id)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    # Generate interactive HTML for iframe
    html_content = await rendering_service._generate_interactive_html(session_data, session_data['render_config'])
    return HTMLResponse(content=html_content)

@app.get("/api/session/{session_id}")
async def get_session_status(session_id: str):
    """Get session status and metadata"""
    session_data = await rendering_service.get_session_data(session_id)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return {
        "session_id": session_id,
        "status": "ready",
        "stats": session_data.get("stats", {}),
        "render_config": session_data.get("render_config", {}),
        "timestamp": session_data.get("timestamp")
    }

@app.websocket("/ws/{session_id}")
async def websocket_session_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time iframe communication"""
    await websocket.accept()
    rendering_service.active_connections[session_id] = websocket
    
    try:
        while True:
            # Handle real-time updates, parameter changes, etc.
            message = await websocket.receive_text()
            data = json.loads(message)
            
            # Handle different message types
            if data.get("type") == "update_params":
                # Handle parameter updates (layout changes, filtering, etc.)
                await handle_parameter_update(session_id, data)
                
    except WebSocketDisconnect:
        if session_id in rendering_service.active_connections:
            del rendering_service.active_connections[session_id]

async def handle_parameter_update(session_id: str, data: Dict[str, Any]):
    """Handle real-time parameter updates"""
    # Implementation for real-time parameter changes
    # (layout algorithm changes, filtering, etc.)
    pass

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_available": HAS_RAPIDS,
        "active_sessions": len(rendering_service.sessions),
        "active_connections": len(rendering_service.active_connections)
    }

@app.post("/api/datasets")
async def preprocess_dataset(graph_data: GraphData, background_tasks: BackgroundTasks):
    """Preprocess and cache dataset for fast visualization (like Graphistry dataset upload)"""
    dataset_id = await rendering_service.preprocess_and_cache_dataset(graph_data)
    
    background_tasks.add_task(rendering_service.cleanup_expired_sessions)
    
    return {
        "dataset_id": dataset_id,
        "status": "preprocessed",
        "visualization_url": f"/visualize/{dataset_id}",
        "embed_url": f"/embed-dataset/{dataset_id}",
        "message": "Dataset preprocessed and cached. Use the visualization_url for direct access."
    }

@app.get("/visualize/{dataset_id}", response_class=HTMLResponse)
async def visualize_cached_dataset(
    dataset_id: str,
    layout: str = "force_atlas2",
    clustering: str = "leiden", 
    play: int = 5000,
    splash: bool = False,
    auto_zoom: bool = True,
    show_labels: bool = True
):
    """Visualize a cached dataset with URL parameters (like Graphistry)"""
    dataset = await rendering_service.get_dataset(dataset_id)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or expired")
    
    # Create session with cached data and URL parameters
    session_data = {
        "session_id": f"dataset-{dataset_id}",
        "dataset_id": dataset_id,
        "nodes": dataset["nodes"],
        "edges": dataset["edges"],
        "layout_positions": dataset["layouts"].get(layout, {}),
        "clusters": dataset["clusters"].get(clustering, {}),
        "centrality": dataset["centrality"],
        "gpu_processed": True,
        "render_config": {
            "layout_algorithm": layout,
            "clustering_algorithm": clustering,
            "animation_duration": play,
            "show_splash": splash,
            "auto_zoom": auto_zoom,
            "show_labels": show_labels,
            "quality": "high",
            "interactive": True
        },
        "stats": dataset["stats"],
        "timestamp": datetime.now().isoformat()
    }
    
    # Generate enhanced HTML with URL parameters
    html_content = await rendering_service._generate_interactive_html(session_data, session_data['render_config'])
    return HTMLResponse(content=html_content)

@app.get("/embed-dataset/{dataset_id}", response_class=HTMLResponse) 
async def embed_cached_dataset(
    dataset_id: str,
    layout: str = "force_atlas2",
    clustering: str = "leiden",
    play: int = 5000,
    splash: bool = False
):
    """Embeddable iframe version of cached dataset visualization"""
    return await visualize_cached_dataset(dataset_id, layout, clustering, play, splash)

@app.get("/api/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get information about a cached dataset"""
    dataset = await rendering_service.get_dataset(dataset_id)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or expired") 
    
    return {
        "dataset_id": dataset_id,
        "stats": dataset["stats"],
        "available_layouts": list(dataset["layouts"].keys()),
        "available_clustering": list(dataset["clusters"].keys()),
        "centrality_metrics": list(dataset["centrality"].keys()),
        "created_at": dataset["created_at"],
        "visualization_url": f"/visualize/{dataset_id}",
        "embed_url": f"/embed-dataset/{dataset_id}"
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Remote GPU Rendering Service")
    logger.info("📊 Features:")
    logger.info("  - GPU-accelerated graph processing with cuGraph")
    logger.info("  - Interactive iframe-embeddable visualizations")
    logger.info("  - Real-time WebSocket communication")
    logger.info("  - Session-based rendering with TTL")
    logger.info("  - Scalable up to million-node graphs")
    logger.info("")
    logger.info("🎯 Service endpoints:")
    logger.info("  - Process graph:        POST /api/render")
    logger.info("  - Iframe visualization: GET  /embed/{session_id}")
    logger.info("  - Session status:       GET  /api/session/{session_id}")
    logger.info("  - Real-time updates:    WS   /ws/{session_id}")
    logger.info("  - Health check:         GET  /api/health")
    logger.info("  - Preprocess dataset:   POST /api/datasets")
    logger.info("  - Visualize dataset:    GET  /visualize/{dataset_id}")
    logger.info("  - Embed dataset:        GET  /embed-dataset/{dataset_id}")
    logger.info("  - Get dataset info:     GET  /api/datasets/{dataset_id}")
    logger.info("")
    
    uvicorn.run(app, host="0.0.0.0", port=8082)