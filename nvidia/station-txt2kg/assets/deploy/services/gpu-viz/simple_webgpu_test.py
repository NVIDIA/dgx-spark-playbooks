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
Simple WebGPU clustering test service
Minimal implementation to test basic functionality
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict, List, Any, Optional
import time

# Simple data models
class GraphData(BaseModel):
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]

class SimpleClusteringRequest(BaseModel):
    graph_data: GraphData
    mode: str = "hybrid"

class SimpleClusteringResult(BaseModel):
    clustered_nodes: List[Dict[str, Any]]
    processing_time: float
    mode: str
    session_id: Optional[str] = None

# FastAPI app
app = FastAPI(title="Simple WebGPU Test Service", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": True,
        "webrtc_available": True,
        "active_sessions": 0,
        "active_connections": 0
    }

@app.get("/api/capabilities")
async def get_capabilities():
    return {
        "modes": {
            "hybrid": {
                "available": True,
                "description": "GPU clustering on server, CPU rendering on client"
            },
            "webrtc_stream": {
                "available": True,
                "description": "Full GPU rendering streamed to client browser"
            }
        },
        "gpu_acceleration": {
            "rapids_available": True,
            "opencv_available": True,
            "plotting_available": True
        },
        "cluster_dimensions": [32, 18, 24],
        "max_cluster_count": 13824
    }

@app.post("/api/cluster", response_model=SimpleClusteringResult)
async def cluster_graph(request: SimpleClusteringRequest):
    """Simple clustering implementation for testing"""
    try:
        start_time = time.time()
        
        # Simple clustering - just add cluster_index to each node
        clustered_nodes = []
        for i, node in enumerate(request.graph_data.nodes):
            clustered_node = {**node, "cluster_index": i % 10, "node_index": i}
            clustered_nodes.append(clustered_node)
        
        processing_time = time.time() - start_time
        
        result = SimpleClusteringResult(
            clustered_nodes=clustered_nodes,
            processing_time=processing_time,
            mode=request.mode,
            session_id="test-session-123" if request.mode == "webrtc_stream" else None
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stream/{session_id}")
async def stream_frame(session_id: str):
    """Simple streaming endpoint - returns a placeholder"""
    # Return a simple 1x1 PNG pixel as placeholder
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    
    from fastapi.responses import Response
    return Response(
        content=png_data,
        media_type="image/png",
        headers={"Cache-Control": "no-cache"}
    )

if __name__ == "__main__":
    print("Starting Simple WebGPU Test Service...")
    uvicorn.run(
        "simple_webgpu_test:app",
        host="0.0.0.0",
        port=8083,
        log_level="info",
        reload=False
    )
