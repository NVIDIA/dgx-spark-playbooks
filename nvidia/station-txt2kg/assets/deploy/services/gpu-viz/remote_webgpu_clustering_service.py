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
Remote WebGPU Clustering Service - CuPy Version with Semantic Clustering

Provides GPU-accelerated graph clustering using CuPy instead of cuDF to avoid segfaults.
Uses stable CuPy operations for GPU clustering while maintaining the same API.
Enhanced with semantic clustering based on node names and content similarity.
"""

import os
import json
import uuid
import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import base64
from io import BytesIO

# Import semantic clustering
from semantic_clustering_service import SemanticClusteringEngine, cluster_nodes_by_similarity

# GPU-accelerated imports
try:
    import cupy as cp
    HAS_CUPY = True
    print("✓ CuPy available for stable GPU clustering")
except ImportError:
    HAS_CUPY = False
    print("⚠ CuPy not available, falling back to CPU")

# Optional cuGraph for force simulation (avoid cuDF operations)
try:
    import cugraph
    import cudf
    HAS_CUGRAPH = True
    print("✓ cuGraph available for force simulation")
except ImportError:
    HAS_CUGRAPH = False
    print("⚠ cuGraph not available")
    import networkx as nx

# WebRTC streaming imports
try:
    import cv2
    import PIL.Image as PILImage
    HAS_OPENCV = True
    print("✓ OpenCV available for WebRTC streaming")
except ImportError:
    HAS_OPENCV = False
    print("⚠ OpenCV not available, WebRTC streaming disabled")

# WebGL rendering imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "json"
    HAS_PLOTTING = True
    print("✓ Plotting libraries available for server-side rendering")
except ImportError:
    HAS_PLOTTING = False
    print("⚠ Plotting libraries not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphData(BaseModel):
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]

class ClusteringMode(str):
    HYBRID = "hybrid"
    WEBRTC_STREAM = "webrtc_stream"

class RemoteClusteringRequest(BaseModel):
    graph_data: GraphData
    mode: str = ClusteringMode.HYBRID
    cluster_dimensions: List[int] = [32, 18, 24]
    force_simulation: bool = True
    max_iterations: int = 100
    webrtc_options: Optional[Dict[str, Any]] = None
    
    # Semantic clustering options
    clustering_method: str = "hybrid"  # "spatial", "semantic", "hybrid"
    semantic_algorithm: str = "hierarchical"  # "hierarchical", "kmeans", "dbscan"
    n_clusters: Optional[int] = None
    similarity_threshold: float = 0.7
    
    # Hybrid clustering weights
    name_weight: float = 0.6
    content_weight: float = 0.3
    spatial_weight: float = 0.1

class ClusteringResult(BaseModel):
    clustered_nodes: List[Dict[str, Any]]
    cluster_info: Dict[str, Any]
    processing_time: float
    mode: str
    session_id: Optional[str] = None

class WebRTCSession(BaseModel):
    session_id: str
    client_id: str
    created_at: datetime
    last_frame_time: datetime
    is_active: bool = True

class CuPyClusteringEngine:
    """
    Stable GPU clustering using CuPy arrays instead of cuDF to avoid segfaults
    """
    
    def __init__(self, cluster_dimensions: Tuple[int, int, int] = (32, 18, 24)):
        self.cluster_dimensions = cluster_dimensions
        self.cluster_count = cluster_dimensions[0] * cluster_dimensions[1] * cluster_dimensions[2]
        self.has_gpu = HAS_CUPY
        logger.info(f"CuPy clustering engine initialized with {self.cluster_count} clusters")
        
    def cluster_nodes_gpu(self, nodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform advanced GPU-accelerated clustering using RAPIDS cuML algorithms
        """
        if not self.has_gpu:
            return self._cluster_nodes_cpu(nodes)
            
        try:
            from cuml.cluster import KMeans, DBSCAN, HDBSCAN
            import cupy as cp
            
            start_time = time.time()
            
            # Extract coordinates and prepare feature matrix
            coordinates = []
            for node in nodes:
                x = float(node.get('x', 0))
                y = float(node.get('y', 0)) 
                z = float(node.get('z', 0))
                coordinates.append([x, y, z])
            
            # Create GPU feature matrix
            X = cp.array(coordinates, dtype=cp.float32)
            n_samples = X.shape[0]
            
            print(f"🚀 GPU clustering {n_samples} nodes with RAPIDS cuML...")
            
            # Choose clustering algorithm optimized for performance
            # KMeans is fastest and works well for most graph clustering scenarios
            if n_samples < 5000:
                # Small datasets: moderate cluster count
                n_clusters = min(max(int(np.sqrt(n_samples / 2)), 3), 25)
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, max_iter=100)
                algorithm_name = f"KMeans(k={n_clusters})"
                
            elif n_samples < 25000:
                # Medium datasets: higher cluster count for better granularity
                n_clusters = min(max(int(np.sqrt(n_samples / 1.5)), 10), 50)
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, max_iter=150)
                algorithm_name = f"KMeans(k={n_clusters})"
                
            else:
                # Large datasets: many clusters but capped for performance
                n_clusters = min(max(int(np.sqrt(n_samples)), 20), 100)
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, max_iter=200)
                algorithm_name = f"KMeans(k={n_clusters})"
            
            # Perform GPU clustering
            cluster_labels = clusterer.fit_predict(X)
            
            # Convert results back to CPU
            if hasattr(cluster_labels, 'get'):
                cluster_result = cluster_labels.get()
            else:
                cluster_result = cp.asarray(cluster_labels).get()
            
            # Update nodes with clustering results
            clustered_nodes = []
            for i, node in enumerate(nodes):
                cluster_id = int(cluster_result[i])
                
                clustered_node = {
                    **node,
                    'cluster_index': cluster_id,
                    'node_index': i
                }
                clustered_nodes.append(clustered_node)
            
            # Generate cluster statistics
            unique_clusters = len(np.unique(cluster_result))
            noise_points = 0  # KMeans doesn't produce noise points
            processing_time = time.time() - start_time
            
            print(f"✅ {algorithm_name} completed: {unique_clusters} clusters, {noise_points} noise points in {processing_time:.4f}s")
            
            # Apply intelligent subsampling for large datasets
            if len(nodes) > 10000:
                print(f"🎯 Large dataset detected ({len(nodes)} nodes), applying cluster-based subsampling...")
                clustered_nodes = self._apply_cluster_subsampling(clustered_nodes, cluster_result, target_nodes=5000)
                print(f"✅ Subsampled to {len(clustered_nodes)} representative nodes")
            
            cluster_info = {
                'total_clusters': self.cluster_count,
                'used_clusters': unique_clusters,
                'cluster_dimensions': self.cluster_dimensions,
                'processing_time': processing_time,
                'gpu_accelerated': True,
                'engine': 'RAPIDS cuML',
                'algorithm': algorithm_name,
                'noise_points': int(noise_points),
                'original_node_count': len(nodes),
                'rendered_node_count': len(clustered_nodes),
                'subsampled': len(nodes) > 10000
            }
            
            logger.info(f"CuPy GPU clustering completed in {processing_time:.3f}s for {len(nodes)} nodes -> {unique_clusters} clusters")
            return clustered_nodes, cluster_info
            
        except Exception as e:
            logger.error(f"RAPIDS cuML GPU clustering failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            print(f"❌ GPU clustering error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return self._cluster_nodes_cpu(nodes)
    
    def _apply_cluster_subsampling(self, clustered_nodes: List[Dict[str, Any]], cluster_labels: np.ndarray, target_nodes: int = 5000) -> List[Dict[str, Any]]:
        """
        Apply intelligent cluster-based subsampling to reduce rendering load while preserving cluster structure.
        
        Strategy:
        1. Keep cluster centroids (most representative nodes)
        2. Keep boundary nodes (cluster edges for visual separation)  
        3. Sample remaining nodes proportionally from each cluster
        4. Always keep noise points (outliers are important)
        """
        import cupy as cp
        
        # Separate nodes by cluster
        cluster_groups = {}
        noise_nodes = []
        
        for i, node in enumerate(clustered_nodes):
            cluster_id = cluster_labels[i]
            if cluster_id == -1:  # Noise points
                noise_nodes.append(node)
            else:
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append((i, node))
        
        # Calculate sampling allocation
        total_clusters = len(cluster_groups)
        noise_count = len(noise_nodes)
        
        # Reserve space for noise points and ensure minimum representation
        available_nodes = max(target_nodes - noise_count, total_clusters * 3)  # At least 3 nodes per cluster
        
        selected_nodes = []
        
        # Include noise points if they exist (DBSCAN/HDBSCAN only)
        if noise_nodes:
            selected_nodes.extend(noise_nodes)
            print(f"   📍 Kept {len(noise_nodes)} noise points")
        else:
            print(f"   📍 No noise points (KMeans clustering)")
        
        # Process each cluster
        for cluster_id, cluster_nodes in cluster_groups.items():
            cluster_size = len(cluster_nodes)
            
            if cluster_size == 0:
                continue
                
            # Calculate how many nodes to keep from this cluster
            # Larger clusters get more representation, but with diminishing returns
            cluster_weight = min(cluster_size / len(clustered_nodes), 0.1)  # Cap at 10% weight
            target_from_cluster = max(3, int(available_nodes * cluster_weight))  # Minimum 3 per cluster
            target_from_cluster = min(target_from_cluster, cluster_size)  # Don't exceed cluster size
            
            if target_from_cluster >= cluster_size:
                # Keep all nodes from small clusters
                selected_nodes.extend([node for _, node in cluster_nodes])
            else:
                # Intelligent sampling for large clusters
                cluster_coords = np.array([[float(node.get('x', 0)), float(node.get('y', 0)), float(node.get('z', 0))] for _, node in cluster_nodes])
                
                # Find cluster centroid
                centroid = np.mean(cluster_coords, axis=0)
                
                # Calculate distances from centroid
                distances = np.linalg.norm(cluster_coords - centroid, axis=1)
                
                # Select representative nodes:
                # 1. Centroid node (closest to center)
                centroid_idx = np.argmin(distances)
                selected_indices = {centroid_idx}
                
                # 2. Boundary nodes (furthest from center for cluster separation)
                if target_from_cluster > 1:
                    boundary_count = min(2, target_from_cluster - 1)
                    boundary_indices = np.argsort(distances)[-boundary_count:]
                    selected_indices.update(boundary_indices)
                
                # 3. Random sampling for remaining slots
                remaining_slots = target_from_cluster - len(selected_indices)
                if remaining_slots > 0:
                    available_indices = set(range(len(cluster_nodes))) - selected_indices
                    if available_indices:
                        random_indices = np.random.choice(list(available_indices), 
                                                        size=min(remaining_slots, len(available_indices)), 
                                                        replace=False)
                        selected_indices.update(random_indices)
                
                # Add selected nodes
                for idx in selected_indices:
                    selected_nodes.append(cluster_nodes[idx][1])
        
        print(f"   🎨 Cluster sampling: {len(cluster_groups)} clusters, {len(selected_nodes)} total nodes")
        return selected_nodes
    
    def _cluster_nodes_cpu(self, nodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """CPU fallback clustering implementation"""
        start_time = time.time()
        
        clustered_nodes = []
        for i, node in enumerate(nodes):
            # Apply same clustering logic as GPU version
            x = float(node.get('x', 0))
            y = float(node.get('y', 0))
            z = float(node.get('z', 0))
            
            # Normalize positions
            norm_x = max(0.0, min(0.999, x / 100.0 + 0.5))
            norm_y = max(0.0, min(0.999, y / 100.0 + 0.5))
            norm_z = max(0.001, min(0.999, z / 100.0 + 0.5))
            
            # Apply logarithmic scaling to Z
            log_z = max(0.0, min(0.999, np.log(norm_z) / np.log(0.999)))
            
            # Calculate cluster indices
            cluster_x = min(self.cluster_dimensions[0] - 1, int(norm_x * self.cluster_dimensions[0]))
            cluster_y = min(self.cluster_dimensions[1] - 1, int(norm_y * self.cluster_dimensions[1]))
            cluster_z = min(self.cluster_dimensions[2] - 1, int(log_z * self.cluster_dimensions[2]))
            
            cluster_index = (cluster_x + 
                           cluster_y * self.cluster_dimensions[0] + 
                           cluster_z * self.cluster_dimensions[0] * self.cluster_dimensions[1])
            
            clustered_node = {
                **node,
                'cluster_index': cluster_index,
                'node_index': i
            }
            clustered_nodes.append(clustered_node)
        
        processing_time = time.time() - start_time
        
        cluster_info = {
            'total_clusters': self.cluster_count,
            'cluster_dimensions': self.cluster_dimensions,
            'processing_time': processing_time,
            'gpu_accelerated': False,
            'engine': 'CPU'
        }
        
        logger.info(f"CPU clustering completed in {processing_time:.3f}s for {len(nodes)} nodes")
        return clustered_nodes, cluster_info

class ForceSimulationEngine:
    """
    GPU-accelerated force simulation for graph layout
    """
    
    def __init__(self):
        self.has_gpu = HAS_CUGRAPH
        
    def simulate_forces(self, nodes: List[Dict[str, Any]], links: List[Dict[str, Any]], max_iterations: int = 100) -> List[Dict[str, Any]]:
        """Run force-directed layout simulation"""
        
        if not self.has_gpu or not links:
            return self._simulate_forces_cpu(nodes, links, max_iterations)
            
        try:
            return self._simulate_forces_gpu(nodes, links, max_iterations)
        except Exception as e:
            logger.error(f"GPU force simulation failed: {e}")
            return self._simulate_forces_cpu(nodes, links, max_iterations)
    
    def _simulate_forces_gpu(self, nodes: List[Dict[str, Any]], links: List[Dict[str, Any]], max_iterations: int) -> List[Dict[str, Any]]:
        """GPU-accelerated force simulation using cuGraph (avoid cuDF operations)"""
        
        # Create simple edge list for cuGraph
        edge_list = []
        for link in links:
            source_id = str(link.get('source', ''))
            target_id = str(link.get('target', ''))
            edge_list.append([source_id, target_id])
        
        if not edge_list:
            return nodes
            
        try:
            # Use NetworkX for safer force simulation
            return self._simulate_forces_cpu(nodes, links, max_iterations)
        except Exception as e:
            logger.warning(f"Force simulation failed: {e}")
            return nodes
    
    def _simulate_forces_cpu(self, nodes: List[Dict[str, Any]], links: List[Dict[str, Any]], max_iterations: int) -> List[Dict[str, Any]]:
        """CPU fallback force simulation using NetworkX"""
        
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            G.add_node(str(node.get('id', '')), **node)
            
        # Add edges
        for link in links:
            source = str(link.get('source', ''))
            target = str(link.get('target', ''))
            G.add_edge(source, target)
        
        # Compute spring layout
        pos = nx.spring_layout(G, iterations=max_iterations, k=1.0)
        
        # Update node positions
        updated_nodes = []
        for node in nodes:
            node_id = str(node.get('id', ''))
            if node_id in pos:
                x, y = pos[node_id]
                updated_node = {**node, 'x': float(x * 100), 'y': float(y * 100)}
            else:
                updated_node = node
            updated_nodes.append(updated_node)
            
        return updated_nodes

class WebRTCStreamingEngine:
    """WebRTC streaming engine for real-time graph visualization streaming"""
    
    def __init__(self):
        self.has_rendering = HAS_PLOTTING and HAS_OPENCV
        self.active_sessions: Dict[str, WebRTCSession] = {}
        self.frame_buffer: Dict[str, bytes] = {}
        
    def create_session(self, client_id: str) -> str:
        """Create new WebRTC streaming session"""
        session_id = str(uuid.uuid4())
        session = WebRTCSession(
            session_id=session_id,
            client_id=client_id,
            created_at=datetime.now(),
            last_frame_time=datetime.now()
        )
        self.active_sessions[session_id] = session
        logger.info(f"Created WebRTC session {session_id} for client {client_id}")
        return session_id
    
    def render_graph_frame(self, session_id: str, nodes: List[Dict[str, Any]], links: List[Dict[str, Any]]) -> bool:
        """Render graph to frame buffer for streaming"""
        
        if not self.has_rendering:
            return False
            
        if session_id not in self.active_sessions:
            return False
            
        try:
            # Create 3D plotly visualization
            node_x = [node.get('x', 0) for node in nodes]
            node_y = [node.get('y', 0) for node in nodes] 
            node_z = [node.get('z', 0) for node in nodes]
            node_text = [node.get('name', f"Node {i}") for i, node in enumerate(nodes)]
            node_colors = [node.get('cluster_index', 0) for node in nodes]
            
            # Create node trace
            node_trace = go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers',
                marker=dict(size=8, color=node_colors, colorscale='Viridis', showscale=True),
                text=node_text,
                hovertemplate='%{text}<br>(%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>',
                name='Nodes'
            )
            
            # Create edge traces
            edge_traces = []
            for link in links:
                source_idx = None
                target_idx = None
                
                for i, node in enumerate(nodes):
                    if str(node.get('id', '')) == str(link.get('source', '')):
                        source_idx = i
                    if str(node.get('id', '')) == str(link.get('target', '')):
                        target_idx = i
                        
                if source_idx is not None and target_idx is not None:
                    edge_trace = go.Scatter3d(
                        x=[node_x[source_idx], node_x[target_idx], None],
                        y=[node_y[source_idx], node_y[target_idx], None],
                        z=[node_z[source_idx], node_z[target_idx], None],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False,
                        hoverinfo='none'
                    )
                    edge_traces.append(edge_trace)
            
            # Create figure
            fig = go.Figure(data=[node_trace] + edge_traces)
            fig.update_layout(
                title='GPU-Clustered Knowledge Graph (CuPy)',
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', bgcolor='rgb(10, 10, 10)'),
                showlegend=False,
                paper_bgcolor='rgb(10, 10, 10)',
                plot_bgcolor='rgb(10, 10, 10)',
                font=dict(color='white')
            )
            
            # Convert to image
            img_bytes = pio.to_image(fig, format='png', width=1200, height=800, engine='kaleido')
            
            # Store frame in buffer
            self.frame_buffer[session_id] = img_bytes
            self.active_sessions[session_id].last_frame_time = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Frame rendering failed for session {session_id}: {e}")
            return False
    
    def get_frame(self, session_id: str) -> Optional[bytes]:
        return self.frame_buffer.get(session_id)
    
    def cleanup_session(self, session_id: str):
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        if session_id in self.frame_buffer:
            del self.frame_buffer[session_id]

class RemoteWebGPUService:
    """Main service class with stable CuPy clustering"""
    
    def __init__(self):
        self.clustering_engine = CuPyClusteringEngine()
        self.force_engine = ForceSimulationEngine()
        self.webrtc_engine = WebRTCStreamingEngine()
        self.active_connections: List[WebSocket] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def process_clustering_request(self, request: RemoteClusteringRequest) -> ClusteringResult:
        """Process remote clustering request with semantic clustering support"""
        
        start_time = time.time()
        
        try:
            nodes = request.graph_data.nodes
            links = request.graph_data.links
            
            # Apply force simulation if requested
            if request.force_simulation:
                logger.info("Running force simulation...")
                nodes = self.force_engine.simulate_forces(nodes, links, request.max_iterations)
            
            # Choose clustering method based on request
            if request.clustering_method == "spatial":
                # Use traditional spatial clustering
                logger.info(f"Spatial clustering {len(nodes)} nodes in {request.mode} mode...")
                clustered_nodes, cluster_info = self.clustering_engine.cluster_nodes_gpu(nodes)
                
            elif request.clustering_method == "semantic":
                # Use semantic clustering based on node names/content
                logger.info(f"Semantic clustering {len(nodes)} nodes using {request.semantic_algorithm}...")
                semantic_result = await cluster_nodes_by_similarity(
                    nodes,
                    method="name" if request.semantic_algorithm != "content" else "content",
                    algorithm=request.semantic_algorithm,
                    n_clusters=request.n_clusters,
                    similarity_threshold=request.similarity_threshold
                )
                clustered_nodes = semantic_result.clustered_nodes
                cluster_info = semantic_result.cluster_info
                
            elif request.clustering_method == "hybrid":
                # Use hybrid clustering (semantic + spatial)
                logger.info(f"Hybrid clustering {len(nodes)} nodes...")
                semantic_result = await cluster_nodes_by_similarity(
                    nodes,
                    method="hybrid",
                    algorithm=request.semantic_algorithm,
                    n_clusters=request.n_clusters,
                    name_weight=request.name_weight,
                    content_weight=request.content_weight,
                    spatial_weight=request.spatial_weight
                )
                clustered_nodes = semantic_result.clustered_nodes
                cluster_info = semantic_result.cluster_info
                
            else:
                # Fallback to spatial clustering
                logger.warning(f"Unknown clustering method '{request.clustering_method}', using spatial")
                clustered_nodes, cluster_info = self.clustering_engine.cluster_nodes_gpu(nodes)
            
            processing_time = time.time() - start_time
            
            # Add clustering method info to result
            cluster_info['clustering_method'] = request.clustering_method
            cluster_info['total_processing_time'] = processing_time
            
            result = ClusteringResult(
                clustered_nodes=clustered_nodes,
                cluster_info=cluster_info,
                processing_time=processing_time,
                mode=request.mode
            )
            
            # Handle WebRTC streaming mode
            if request.mode == ClusteringMode.WEBRTC_STREAM:
                session_id = self.webrtc_engine.create_session("remote_client")
                success = self.webrtc_engine.render_graph_frame(session_id, clustered_nodes, links)
                if success:
                    result.session_id = session_id
                    
            return result
            
        except Exception as e:
            logger.error(f"Clustering request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast updates to connected WebSocket clients"""
        if not self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                disconnected.append(connection)
                
        for connection in disconnected:
            self.active_connections.remove(connection)

# FastAPI app setup
app = FastAPI(
    title="Remote WebGPU Clustering Service (CuPy)",
    description="Stable GPU-accelerated graph clustering using CuPy",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = RemoteWebGPUService()

@app.post("/api/cluster", response_model=ClusteringResult)
async def cluster_graph(request: RemoteClusteringRequest):
    """Process graph clustering request"""
    result = await service.process_clustering_request(request)
    
    await service.broadcast_update({
        "type": "clustering_complete",
        "data": result.dict()
    })
    
    return result

@app.get("/api/capabilities")
async def get_capabilities():
    """Get service capabilities"""
    return {
        "modes": {
            "hybrid": {
                "available": True,
                "description": "GPU clustering on server, CPU rendering on client"
            },
            "webrtc_stream": {
                "available": service.webrtc_engine.has_rendering,
                "description": "Full GPU rendering streamed to client browser"
            }
        },
        "clustering_methods": {
            "spatial": {
                "available": True,
                "description": "Traditional spatial/coordinate-based clustering"
            },
            "semantic": {
                "available": True,
                "description": "Semantic clustering based on node names and content similarity"
            },
            "hybrid": {
                "available": True,
                "description": "Combined semantic and spatial clustering with configurable weights"
            }
        },
        "clustering_algorithms": {
            "hierarchical": {
                "available": True,
                "description": "Hierarchical agglomerative clustering"
            },
            "kmeans": {
                "available": True,
                "description": "K-means clustering (GPU accelerated when available)"
            },
            "dbscan": {
                "available": True,
                "description": "Density-based spatial clustering"
            }
        },
        "gpu_acceleration": {
            "cupy_available": HAS_CUPY,
            "cugraph_available": HAS_CUGRAPH,
            "opencv_available": HAS_OPENCV,
            "plotting_available": HAS_PLOTTING,
            "semantic_gpu": HAS_CUPY
        },
        "cluster_dimensions": service.clustering_engine.cluster_dimensions,
        "max_cluster_count": service.clustering_engine.cluster_count
    }

@app.get("/api/stream/{session_id}")
async def stream_frame(session_id: str):
    """Stream rendered frame for WebRTC session"""
    frame_data = service.webrtc_engine.get_frame(session_id)
    if not frame_data:
        raise HTTPException(status_code=404, detail="Frame not found")
        
    return StreamingResponse(
        BytesIO(frame_data),
        media_type="image/png",
        headers={"Cache-Control": "no-cache"}
    )

@app.delete("/api/stream/{session_id}")
async def cleanup_stream(session_id: str):
    """Clean up WebRTC streaming session"""
    service.webrtc_engine.cleanup_session(session_id)
    return {"status": "cleaned up"}

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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_available": HAS_CUPY,
        "webrtc_available": service.webrtc_engine.has_rendering,
        "active_sessions": len(service.webrtc_engine.active_sessions),
        "active_connections": len(service.active_connections),
        "engine": "RAPIDS cuML"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8083))
    logger.info(f"Starting Remote WebGPU Clustering Service (RAPIDS cuML) on port {port}")
    logger.info(f"CuPy GPU acceleration: {'✓' if HAS_CUPY else '✗'}")
    logger.info(f"WebRTC streaming: {'✓' if service.webrtc_engine.has_rendering else '✗'}")
    
    uvicorn.run(
        "remote_webgpu_clustering_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )
