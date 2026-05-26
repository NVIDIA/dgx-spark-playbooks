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
import graphistry
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import os
import time
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize PyGraphistry
def init_graphistry():
    """Initialize PyGraphistry with GPU acceleration"""
    try:
        # Set up authentication - check for different credential types
        api_key = os.getenv('GRAPHISTRY_API_KEY')
        personal_key = os.getenv('GRAPHISTRY_PERSONAL_KEY')
        secret_key = os.getenv('GRAPHISTRY_SECRET_KEY')
        username = os.getenv('GRAPHISTRY_USERNAME')
        password = os.getenv('GRAPHISTRY_PASSWORD')
        
        if personal_key and secret_key:
            # Configure for cloud API with personal key and secret
            graphistry.register(
                api=3, 
                protocol="https", 
                server="hub.graphistry.com", 
                personal_key_id=personal_key,
                personal_key_secret=secret_key
            )
            logger.info("PyGraphistry initialized with personal key/secret for cloud GPU acceleration")
            return True
        elif api_key:
            # Configure for cloud API with API key
            graphistry.register(api=3, protocol="https", server="hub.graphistry.com", api_key=api_key)
            logger.info("PyGraphistry initialized with API key for cloud GPU acceleration")
            return True
        elif username and password:
            # Configure for cloud API with username/password
            graphistry.register(api=3, protocol="https", server="hub.graphistry.com", 
                              username=username, password=password)
            logger.info("PyGraphistry initialized with username/password for cloud GPU acceleration")
            return True
        else:
            # Configure for local mode
            graphistry.register(api=3)
            logger.info("PyGraphistry initialized in local CPU mode")
            return True
            
    except Exception as e:
        logger.error(f"Failed to initialize PyGraphistry: {e}")
        return False

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
    
class VisualizationRequest(BaseModel):
    graph_data: GraphData
    layout_type: Optional[str] = "force"
    gpu_acceleration: Optional[bool] = True
    clustering: Optional[bool] = False
    node_size_attribute: Optional[str] = None
    node_color_attribute: Optional[str] = None
    edge_weight_attribute: Optional[str] = None

class GraphGenerationStatus(BaseModel):
    task_id: str
    status: str  # "running", "completed", "failed"
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class LargeGraphGenerator:
    """Optimized graph generation using NetworkX and NumPy for performance"""
    
    @staticmethod
    def generate_random_graph(num_nodes: int, avg_degree: int = 5, seed: Optional[int] = None) -> GraphData:
        """Generate random graph using Erdős–Rényi model"""
        if seed:
            np.random.seed(seed)
            
        # Calculate edge probability for desired average degree
        p = avg_degree / (num_nodes - 1)
        
        # Use NetworkX for efficient generation
        G = nx.erdos_renyi_graph(num_nodes, p, seed=seed)
        
        return LargeGraphGenerator._networkx_to_graphdata(G)
    
    @staticmethod
    def generate_scale_free_graph(num_nodes: int, m: int = 3, seed: Optional[int] = None) -> GraphData:
        """Generate scale-free graph using Barabási–Albert model"""
        G = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
        return LargeGraphGenerator._networkx_to_graphdata(G)
    
    @staticmethod
    def generate_small_world_graph(num_nodes: int, k: int = 6, p: float = 0.1, seed: Optional[int] = None) -> GraphData:
        """Generate small-world graph using Watts-Strogatz model"""
        G = nx.watts_strogatz_graph(num_nodes, k, p, seed=seed)
        return LargeGraphGenerator._networkx_to_graphdata(G)
    
    @staticmethod
    def generate_clustered_graph(num_nodes: int, num_clusters: int = 100, seed: Optional[int] = None) -> GraphData:
        """Generate clustered graph with intra and inter-cluster connections"""
        if seed:
            np.random.seed(seed)
            
        cluster_size = num_nodes // num_clusters
        G = nx.Graph()
        
        # Add nodes with cluster information
        for i in range(num_nodes):
            cluster_id = i // cluster_size
            G.add_node(i, cluster=cluster_id)
        
        # Generate intra-cluster edges
        intra_prob = 0.1
        for cluster in range(num_clusters):
            cluster_start = cluster * cluster_size
            cluster_end = min(cluster_start + cluster_size, num_nodes)
            cluster_nodes = list(range(cluster_start, cluster_end))
            
            # Create subgraph for cluster
            cluster_subgraph = nx.erdos_renyi_graph(len(cluster_nodes), intra_prob)
            
            # Add edges to main graph with proper node mapping
            for edge in cluster_subgraph.edges():
                G.add_edge(cluster_nodes[edge[0]], cluster_nodes[edge[1]])
        
        # Generate inter-cluster edges
        inter_prob = 0.001
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if G.nodes[i].get('cluster') != G.nodes[j].get('cluster'):
                    if np.random.random() < inter_prob:
                        G.add_edge(i, j)
        
        return LargeGraphGenerator._networkx_to_graphdata(G)
    
    @staticmethod
    def generate_hierarchical_graph(num_nodes: int, branching_factor: int = 3, seed: Optional[int] = None) -> GraphData:
        """Generate hierarchical (tree-like) graph"""
        G = nx.random_tree(num_nodes, seed=seed)
        
        # Add some cross-links to make it more interesting
        if seed:
            np.random.seed(seed)
        
        # Add 10% additional edges for cross-connections
        num_additional_edges = max(1, num_nodes // 10)
        nodes = list(G.nodes())
        
        for _ in range(num_additional_edges):
            u, v = np.random.choice(nodes, 2, replace=False)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
        
        return LargeGraphGenerator._networkx_to_graphdata(G)
    
    @staticmethod
    def generate_grid_graph(dimensions: List[int], seed: Optional[int] = None) -> GraphData:
        """Generate 2D or 3D grid graph"""
        if len(dimensions) == 2:
            G = nx.grid_2d_graph(dimensions[0], dimensions[1])
        elif len(dimensions) == 3:
            G = nx.grid_graph(dimensions)
        else:
            raise ValueError("Grid dimensions must be 2D or 3D")
        
        # Convert coordinate tuples to integer node IDs
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        
        return LargeGraphGenerator._networkx_to_graphdata(G)
    
    @staticmethod
    def _networkx_to_graphdata(G: nx.Graph) -> GraphData:
        """Convert NetworkX graph to GraphData format"""
        nodes = []
        links = []
        
        # Convert nodes
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            node = {
                "id": f"n{node_id}",
                "name": f"Node {node_id}",
                "val": np.random.randint(1, 11),
                "degree": G.degree(node_id)
            }
            
            # Add cluster information if available
            if 'cluster' in node_data:
                node['group'] = f"cluster_{node_data['cluster']}"
            else:
                node['group'] = f"group_{node_id % 10}"
                
            nodes.append(node)
        
        # Convert edges
        for edge in G.edges():
            link = {
                "source": f"n{edge[0]}",
                "target": f"n{edge[1]}",
                "name": f"link_{edge[0]}_{edge[1]}",
                "weight": np.random.uniform(0.1, 5.0)
            }
            links.append(link)
        
        return GraphData(nodes=nodes, links=links)

class PyGraphistryService:
    def __init__(self):
        self.initialized = init_graphistry()
        self.generation_tasks = {}  # Store background tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def generate_graph_async(self, request: GraphGenerationRequest, task_id: str):
        """Generate graph asynchronously"""
        try:
            self.generation_tasks[task_id] = GraphGenerationStatus(
                task_id=task_id,
                status="running",
                progress=0.0,
                message="Starting graph generation..."
            )
            
            start_time = time.time()
            
            # Update progress
            self.generation_tasks[task_id].progress = 10.0
            self.generation_tasks[task_id].message = f"Generating {request.pattern.value} graph with {request.num_nodes} nodes..."
            
            # Generate graph based on pattern
            if request.pattern == GraphPattern.RANDOM:
                graph_data = LargeGraphGenerator.generate_random_graph(
                    request.num_nodes, request.avg_degree, request.seed
                )
            elif request.pattern == GraphPattern.SCALE_FREE:
                m = min(request.avg_degree, request.num_nodes - 1) if request.avg_degree else 3
                graph_data = LargeGraphGenerator.generate_scale_free_graph(
                    request.num_nodes, m, request.seed
                )
            elif request.pattern == GraphPattern.SMALL_WORLD:
                graph_data = LargeGraphGenerator.generate_small_world_graph(
                    request.num_nodes, 
                    request.small_world_k or 6, 
                    request.small_world_p or 0.1, 
                    request.seed
                )
            elif request.pattern == GraphPattern.CLUSTERED:
                graph_data = LargeGraphGenerator.generate_clustered_graph(
                    request.num_nodes, request.num_clusters or 100, request.seed
                )
            elif request.pattern == GraphPattern.HIERARCHICAL:
                graph_data = LargeGraphGenerator.generate_hierarchical_graph(
                    request.num_nodes, seed=request.seed
                )
            elif request.pattern == GraphPattern.GRID:
                # Calculate grid dimensions for given number of nodes
                if request.grid_dimensions:
                    dimensions = request.grid_dimensions
                else:
                    side_length = int(np.sqrt(request.num_nodes))
                    dimensions = [side_length, side_length]
                graph_data = LargeGraphGenerator.generate_grid_graph(dimensions, request.seed)
            else:
                raise ValueError(f"Unknown graph pattern: {request.pattern}")
            
            # Update progress
            self.generation_tasks[task_id].progress = 80.0
            self.generation_tasks[task_id].message = "Computing graph statistics..."
            
            # Calculate statistics
            generation_time = time.time() - start_time
            stats = {
                "node_count": len(graph_data.nodes),
                "edge_count": len(graph_data.links),
                "generation_time": generation_time,
                "density": len(graph_data.links) / (len(graph_data.nodes) * (len(graph_data.nodes) - 1) / 2) if len(graph_data.nodes) > 1 else 0,
                "avg_degree": 2 * len(graph_data.links) / len(graph_data.nodes) if len(graph_data.nodes) > 0 else 0,
                "pattern": request.pattern.value,
                "parameters": request.model_dump()
            }
            
            # Complete task
            self.generation_tasks[task_id].status = "completed"
            self.generation_tasks[task_id].progress = 100.0
            self.generation_tasks[task_id].message = f"Generated {stats['node_count']} nodes and {stats['edge_count']} edges in {generation_time:.2f}s"
            self.generation_tasks[task_id].result = {
                "graph_data": graph_data.model_dump(),
                "stats": stats
            }
            
            logger.info(f"Graph generation completed for task {task_id}: {stats}")
            
        except Exception as e:
            logger.error(f"Graph generation failed for task {task_id}: {e}")
            self.generation_tasks[task_id].status = "failed"
            self.generation_tasks[task_id].error = str(e)
            self.generation_tasks[task_id].message = f"Generation failed: {e}"
    
    async def start_graph_generation(self, request: GraphGenerationRequest) -> str:
        """Start graph generation as background task"""
        task_id = f"gen_{int(time.time() * 1000)}"
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            self.executor,
            lambda: asyncio.run(self.generate_graph_async(request, task_id))
        )
        
        return task_id
    
    def get_generation_status(self, task_id: str) -> Optional[GraphGenerationStatus]:
        """Get status of graph generation task"""
        return self.generation_tasks.get(task_id)
        
    async def process_graph_data(self, request: VisualizationRequest) -> Dict[str, Any]:
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
            try:
                g = graphistry.edges(edges_df, 'source', 'target').nodes(nodes_df, 'id')
                logger.info(f"Created PyGraphistry graph object")
            except Exception as e:
                logger.error(f"Failed to create PyGraphistry graph: {e}")
                raise HTTPException(status_code=500, detail=f"Graph creation failed: {e}")
            
            # Apply GPU-accelerated processing
            if request.gpu_acceleration:
                g = await self._apply_gpu_acceleration(g, request)
            
            # Apply clustering if requested
            if request.clustering:
                g = await self._apply_clustering(g)
            
            # Generate layout
            g = await self._generate_layout(g, request.layout_type)
            
            # Extract processed data
            try:
                processed_nodes = g._nodes.to_dict('records') if g._nodes is not None else nodes_df.to_dict('records')
                processed_edges = g._edges.to_dict('records') if g._edges is not None else edges_df.to_dict('records')
                logger.info(f"Extracted {len(processed_nodes)} nodes and {len(processed_edges)} edges")
            except Exception as e:
                logger.warning(f"Data extraction failed, using original data: {e}")
                processed_nodes = nodes_df.to_dict('records')
                processed_edges = edges_df.to_dict('records')
            
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
    
    async def _apply_gpu_acceleration(self, g, request: VisualizationRequest):
        """Apply GPU acceleration using PyGraphistry's vector processing"""
        try:
            if not request.gpu_acceleration:
                logger.info("GPU acceleration disabled by request")
                return g
                
            logger.info("=== GPU ACCELERATION ATTEMPT ===")
            logger.info(f"PyGraphistry object type: {type(g)}")
            logger.info(f"Available methods: {[method for method in dir(g) if not method.startswith('_')]}")
            
            has_umap = hasattr(g, 'umap')
            logger.info(f"Has UMAP: {has_umap}")
            
            gpu_operations_successful = 0
            total_gpu_operations = 0
            
            # Apply UMAP for node positioning if available and beneficial
            total_gpu_operations += 1
            try:
                if has_umap and len(g._nodes) > 100 and len(g._nodes) < 10000:
                    logger.info("Attempting UMAP for node positioning...")
                    g = g.umap()
                    logger.info("✓ SUCCESS: Applied UMAP for node positioning")
                    gpu_operations_successful += 1
                else:
                    reason = ("UMAP not available" if not has_umap else 
                             "too few nodes" if len(g._nodes) <= 100 else "too many nodes")
                    logger.warning(f"✗ SKIPPED: UMAP processing ({reason})")
            except Exception as e:
                logger.warning(f"✗ FAILED: UMAP processing failed: {e}")
                
            logger.info(f"=== GPU ACCELERATION SUMMARY ===")
            logger.info(f"GPU operations successful: {gpu_operations_successful}/{total_gpu_operations}")
            logger.info(f"GPU utilization: {(gpu_operations_successful/total_gpu_operations)*100:.1f}%")
            
            return g
        except Exception as e:
            logger.warning(f"GPU acceleration failed completely, falling back to CPU: {e}")
            return g
    
    async def _apply_clustering(self, g):
        """Apply GPU-accelerated clustering"""
        try:
            logger.info("=== CLUSTERING ATTEMPT ===")
            
            # Use PyGraphistry's built-in clustering if available
            if hasattr(g, 'compute_igraph') and len(g._nodes) < 20000:  # Limit for performance
                logger.info("Attempting Leiden community detection...")
                try:
                    g = g.compute_igraph('community_leiden', out_col='cluster')
                    logger.info("✓ SUCCESS: Applied Leiden community detection")
                    return g
                except Exception as e:
                    logger.warning(f"✗ FAILED: Leiden clustering failed: {e}")
                    logger.info("Attempting Louvain community detection as fallback...")
                    try:
                        g = g.compute_igraph('community_louvain', out_col='cluster') 
                        logger.info("✓ SUCCESS: Applied Louvain community detection")
                        return g
                    except Exception as e2:
                        logger.warning(f"✗ FAILED: Louvain clustering also failed: {e2}")
            else:
                reason = "too many nodes" if len(g._nodes) >= 20000 else "compute_igraph not available"
                logger.warning(f"✗ SKIPPED: Clustering ({reason})")
            
            logger.info("=== CLUSTERING SUMMARY: No clustering applied ===")
            return g
        except Exception as e:
            logger.warning(f"Clustering failed completely: {e}")
            return g
    
    async def _generate_layout(self, g, layout_type: str = "force"):
        """Generate layout using PyGraphistry's algorithms"""
        try:
            logger.info(f"Generating {layout_type} layout")
            
            # Only apply layout computation for reasonable graph sizes
            if len(g._nodes) > 50000:
                logger.info("Skipping layout computation for very large graph")
                return g
            
            if hasattr(g, 'compute_igraph'):
                try:
                    if layout_type == "force":
                        g = g.compute_igraph('layout_fruchterman_reingold', out_cols=['x', 'y'])
                        logger.info("Applied Fruchterman-Reingold force layout")
                    elif layout_type == "circular":
                        g = g.compute_igraph('layout_circle', out_cols=['x', 'y'])
                        logger.info("Applied circular layout")
                    elif layout_type == "hierarchical":
                        g = g.compute_igraph('layout_sugiyama', out_cols=['x', 'y'])
                        logger.info("Applied hierarchical layout")
                    else:
                        # Default to force-directed
                        g = g.compute_igraph('layout_fruchterman_reingold', out_cols=['x', 'y'])
                        logger.info("Applied default force layout")
                except Exception as e:
                    logger.warning(f"Layout computation failed: {e}")
            else:
                logger.info("Layout computation not available, using default positioning")
                
            return g
        except Exception as e:
            logger.warning(f"Layout generation failed: {e}")
            return g
    
    def _create_local_viz_data(self, g, processed_nodes: List[Dict], processed_edges: List[Dict]) -> Dict[str, Any]:
        """Create local visualization data when embed URL cannot be generated"""
        try:
            # Extract layout positions if available
            positions = {}
            if g._nodes is not None and 'x' in g._nodes.columns and 'y' in g._nodes.columns:
                for _, row in g._nodes.iterrows():
                    node_id = row.get('id', row.name)
                    positions[str(node_id)] = {
                        'x': float(row['x']) if pd.notna(row['x']) else 0,
                        'y': float(row['y']) if pd.notna(row['y']) else 0
                    }
            
            # Add cluster information if available
            clusters = {}
            if g._nodes is not None and 'cluster' in g._nodes.columns:
                for _, row in g._nodes.iterrows():
                    node_id = row.get('id', row.name)
                    if pd.notna(row['cluster']):
                        clusters[str(node_id)] = int(row['cluster'])
            
            # Create enhanced nodes with layout and cluster info
            enhanced_nodes = []
            for node in processed_nodes:
                enhanced_node = node.copy()
                node_id = str(node.get('id', ''))
                
                if node_id in positions:
                    enhanced_node.update(positions[node_id])
                    
                if node_id in clusters:
                    enhanced_node['cluster'] = clusters[node_id]
                    
                enhanced_nodes.append(enhanced_node)
            
            return {
                "nodes": enhanced_nodes,
                "edges": processed_edges,
                "positions": positions,
                "clusters": clusters,
                "layout_computed": len(positions) > 0,
                "clusters_computed": len(clusters) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to create local visualization data: {e}")
            return {
                "nodes": processed_nodes,
                "edges": processed_edges,
                "positions": {},
                "clusters": {},
                "layout_computed": False,
                "clusters_computed": False
            }
    
    async def get_graph_stats(self, graph_data: GraphData) -> Dict[str, Any]:
        """Get GPU-accelerated graph statistics"""
        try:
            nodes_df = pd.DataFrame(graph_data.nodes)
            edges_df = pd.DataFrame(graph_data.links)
            
            g = graphistry.edges(edges_df, 'source', 'target').nodes(nodes_df, 'id')
            
            # Compute various graph metrics using GPU acceleration
            stats = {
                "node_count": len(nodes_df),
                "edge_count": len(edges_df),
                "density": len(edges_df) / (len(nodes_df) * (len(nodes_df) - 1)) if len(nodes_df) > 1 else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add centrality measures if possible
            try:
                if len(nodes_df) < 10000 and hasattr(g, 'compute_igraph'):  # Only for reasonably sized graphs
                    g_with_metrics = g.compute_igraph('pagerank', out_col='pagerank')
                    
                    if g_with_metrics._nodes is not None and 'pagerank' in g_with_metrics._nodes.columns:
                        pagerank_data = g_with_metrics._nodes['pagerank'].to_list()
                        stats.update({
                            "avg_pagerank": float(np.mean(pagerank_data)),
                            "max_pagerank": float(np.max(pagerank_data))
                        })
                        logger.info("Computed PageRank statistics")
            except Exception as e:
                logger.warning(f"Could not compute centrality measures: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing graph stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# FastAPI app
app = FastAPI(title="PyGraphistry GPU Visualization Service", version="1.0.0")
service = PyGraphistryService()

@app.post("/api/generate")
async def generate_graph(request: GraphGenerationRequest):
    """Start graph generation as background task"""
    if request.num_nodes > 1000000:
        raise HTTPException(status_code=400, detail="Maximum 1 million nodes allowed")
        
    task_id = await service.start_graph_generation(request)
    return {"task_id": task_id, "status": "started"}

@app.get("/api/generate/{task_id}")
async def get_generation_status(task_id: str):
    """Get status of graph generation task"""
    status = service.get_generation_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

@app.post("/api/visualize")
async def visualize_graph(request: VisualizationRequest):
    """Process graph data with PyGraphistry GPU acceleration"""
    return await service.process_graph_data(request)

@app.post("/api/stats")
async def get_graph_statistics(graph_data: GraphData):
    """Get GPU-accelerated graph statistics"""
    return await service.get_graph_stats(graph_data)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pygraphistry_initialized": service.initialized,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/patterns")
async def get_available_patterns():
    """Get list of available graph generation patterns"""
    return {
        "patterns": [
            {
                "name": pattern.value,
                "description": {
                    GraphPattern.RANDOM: "Random graph using Erdős–Rényi model",
                    GraphPattern.SCALE_FREE: "Scale-free graph using Barabási–Albert model",
                    GraphPattern.SMALL_WORLD: "Small-world graph using Watts-Strogatz model",
                    GraphPattern.CLUSTERED: "Clustered graph with community structure",
                    GraphPattern.HIERARCHICAL: "Hierarchical tree-like graph with cross-links",
                    GraphPattern.GRID: "2D or 3D grid graph"
                }[pattern]
            } for pattern in GraphPattern
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 