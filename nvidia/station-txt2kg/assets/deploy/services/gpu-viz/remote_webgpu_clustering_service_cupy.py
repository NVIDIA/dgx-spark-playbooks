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
Remote WebGPU Clustering Service - CuPy Version

Provides GPU-accelerated graph clustering using CuPy instead of cuDF to avoid segfaults.
Uses stable CuPy operations for GPU clustering while maintaining the same API.
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
        Perform stable GPU-accelerated clustering using CuPy
        """
        if not self.has_gpu:
            return self._cluster_nodes_cpu(nodes)
            
        try:
            start_time = time.time()
            
            # Extract coordinates using CuPy arrays (stable)
            x_vals = cp.array([float(node.get('x', 0)) for node in nodes])
            y_vals = cp.array([float(node.get('y', 0)) for node in nodes])
            z_vals = cp.array([float(node.get('z', 0)) for node in nodes])
            
            # Apply clustering algorithm (same as WebGPU shader)
            norm_x = cp.clip((x_vals / 100.0 + 0.5), 0.0, 0.999)
            norm_y = cp.clip((y_vals / 100.0 + 0.5), 0.0, 0.999)
            norm_z = cp.clip((z_vals / 100.0 + 0.5), 0.001, 0.999)
            
            # Apply logarithmic scaling to Z dimension
            log_z = cp.clip(cp.log(norm_z) / cp.log(0.999), 0.0, 0.999)
            
            # Calculate cluster indices
            cluster_x = cp.clip((norm_x * self.cluster_dimensions[0]).astype(cp.int32), 0, self.cluster_dimensions[0] - 1)
            cluster_y = cp.clip((norm_y * self.cluster_dimensions[1]).astype(cp.int32), 0, self.cluster_dimensions[1] - 1)
            cluster_z = cp.clip((log_z * self.cluster_dimensions[2]).astype(cp.int32), 0, self.cluster_dimensions[2] - 1)
            
            # Calculate final cluster index
            cluster_indices = (cluster_x + 
                             cluster_y * self.cluster_dimensions[0] + 
                             cluster_z * self.cluster_dimensions[0] * self.cluster_dimensions[1])
            
            # Convert back to CPU for results
            cluster_result = cluster_indices.get()
            
            # Update nodes with clustering results
            clustered_nodes = []
            for i, node in enumerate(nodes):
                clustered_node = {
                    **node,
                    'cluster_index': int(cluster_result[i]),
                    'node_index': i
                }
                clustered_nodes.append(clustered_node)
            
            # Generate cluster statistics
            unique_clusters = len(np.unique(cluster_result))
            processing_time = time.time() - start_time
            
            cluster_info = {
                'total_clusters': self.cluster_count,
                'used_clusters': unique_clusters,
                'cluster_dimensions': self.cluster_dimensions,
                'processing_time': processing_time,
                'gpu_accelerated': True,
                'engine': 'CuPy'
            }
            
            logger.info(f"CuPy GPU clustering completed in {processing_time:.3f}s for {len(nodes)} nodes -> {unique_clusters} clusters")
            return clustered_nodes, cluster_info
            
        except Exception as e:
            logger.error(f"CuPy GPU clustering failed: {e}")
            return self._cluster_nodes_cpu(nodes)
    
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
        """Process remote clustering request"""
        
        start_time = time.time()
        
        try:
            nodes = request.graph_data.nodes
            links = request.graph_data.links
            
            # Apply force simulation if requested
            if request.force_simulation:
                logger.info("Running force simulation...")
                nodes = self.force_engine.simulate_forces(nodes, links, request.max_iterations)
            
            # Perform clustering
            logger.info(f"Clustering {len(nodes)} nodes in {request.mode} mode...")
            clustered_nodes, cluster_info = self.clustering_engine.cluster_nodes_gpu(nodes)
            
            processing_time = time.time() - start_time
            
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
        "gpu_acceleration": {
            "cupy_available": HAS_CUPY,
            "cugraph_available": HAS_CUGRAPH,
            "opencv_available": HAS_OPENCV,
            "plotting_available": HAS_PLOTTING
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
        "engine": "CuPy"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8083))
    logger.info(f"Starting Remote WebGPU Clustering Service (CuPy) on port {port}")
    logger.info(f"CuPy GPU acceleration: {'✓' if HAS_CUPY else '✗'}")
    logger.info(f"WebRTC streaming: {'✓' if service.webrtc_engine.has_rendering else '✗'}")
    
    uvicorn.run(
        "remote_webgpu_clustering_service_cupy:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )
