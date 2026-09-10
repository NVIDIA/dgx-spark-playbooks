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
Semantic Clustering Service for Knowledge Graphs
Groups nodes by semantic similarity of names, types, and content rather than just spatial coordinates
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Try to import GPU libraries
try:
    import cupy as cp
    import cuml
    from cuml.cluster import KMeans as cuKMeans, DBSCAN as cuDBSCAN
    HAS_GPU = True
    print("✅ GPU libraries (CuPy, cuML) available for semantic clustering")
except ImportError:
    HAS_GPU = False
    print("⚠️  GPU libraries not available, using CPU for semantic clustering")

logger = logging.getLogger(__name__)

@dataclass
class SemanticClusterResult:
    """Result of semantic clustering operation"""
    clustered_nodes: List[Dict[str, Any]]
    cluster_info: Dict[str, Any]
    similarity_matrix: Optional[np.ndarray] = None
    cluster_labels: Optional[np.ndarray] = None

class SemanticSimilarityCalculator:
    """Calculate semantic similarity between node names and content"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.fitted = False
    
    def calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two node names using multiple methods"""
        if not name1 or not name2:
            return 0.0
        
        name1_clean = self._clean_name(name1)
        name2_clean = self._clean_name(name2)
        
        # Method 1: Exact match
        if name1_clean == name2_clean:
            return 1.0
        
        # Method 2: Substring match
        if name1_clean in name2_clean or name2_clean in name1_clean:
            return 0.8
        
        # Method 3: Sequence similarity (Levenshtein-based)
        seq_similarity = SequenceMatcher(None, name1_clean, name2_clean).ratio()
        
        # Method 4: Word overlap (Jaccard similarity)
        words1 = set(name1_clean.split())
        words2 = set(name2_clean.split())
        if words1 and words2:
            jaccard_sim = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            jaccard_sim = 0.0
        
        # Method 5: Common prefix/suffix
        prefix_sim = self._prefix_similarity(name1_clean, name2_clean)
        suffix_sim = self._suffix_similarity(name1_clean, name2_clean)
        
        # Combine similarities with weights
        combined_similarity = (
            seq_similarity * 0.3 +
            jaccard_sim * 0.4 +
            prefix_sim * 0.15 +
            suffix_sim * 0.15
        )
        
        return min(combined_similarity, 1.0)
    
    def calculate_content_similarity(self, nodes: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate content similarity matrix using TF-IDF"""
        # Extract text content from nodes
        texts = []
        for node in nodes:
            text_parts = []
            
            # Add node name
            if node.get('name'):
                text_parts.append(str(node['name']))
            
            # Add node type/group
            if node.get('group') or node.get('type'):
                text_parts.append(str(node.get('group', node.get('type', ''))))
            
            # Add any description or content
            for key in ['description', 'content', 'label', 'properties']:
                if node.get(key):
                    text_parts.append(str(node[key]))
            
            # Combine all text
            combined_text = ' '.join(text_parts)
            texts.append(combined_text if combined_text.strip() else node.get('name', 'unnamed'))
        
        # Calculate TF-IDF similarity
        if not self.fitted and texts:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.fitted = True
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    
    def _clean_name(self, name: str) -> str:
        """Clean and normalize node name"""
        if not name:
            return ""
        
        # Convert to lowercase
        cleaned = name.lower().strip()
        
        # Remove special characters but keep spaces and alphanumeric
        cleaned = re.sub(r'[^\w\s-]', ' ', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _prefix_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity based on common prefix"""
        min_len = min(len(name1), len(name2))
        if min_len == 0:
            return 0.0
        
        common_prefix = 0
        for i in range(min_len):
            if name1[i] == name2[i]:
                common_prefix += 1
            else:
                break
        
        return common_prefix / min_len
    
    def _suffix_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity based on common suffix"""
        min_len = min(len(name1), len(name2))
        if min_len == 0:
            return 0.0
        
        common_suffix = 0
        for i in range(1, min_len + 1):
            if name1[-i] == name2[-i]:
                common_suffix += 1
            else:
                break
        
        return common_suffix / min_len

class SemanticClusteringEngine:
    """Main semantic clustering engine"""
    
    def __init__(self, use_gpu: bool = None):
        self.use_gpu = use_gpu if use_gpu is not None else HAS_GPU
        self.similarity_calc = SemanticSimilarityCalculator()
        logger.info(f"Semantic clustering engine initialized (GPU: {self.use_gpu})")
    
    def cluster_by_name_similarity(
        self, 
        nodes: List[Dict[str, Any]], 
        algorithm: str = "hierarchical",
        n_clusters: Optional[int] = None,
        similarity_threshold: float = 0.7
    ) -> SemanticClusterResult:
        """
        Cluster nodes based on name similarity
        
        Args:
            nodes: List of node dictionaries
            algorithm: 'hierarchical', 'kmeans', 'dbscan'
            n_clusters: Number of clusters (for kmeans/hierarchical)
            similarity_threshold: Minimum similarity for clustering (for dbscan)
        """
        start_time = time.time()
        n_nodes = len(nodes)
        
        logger.info(f"🧠 Starting semantic clustering of {n_nodes} nodes using {algorithm}")
        
        if n_nodes < 2:
            return self._create_single_cluster_result(nodes, start_time)
        
        # Calculate name similarity matrix
        similarity_matrix = self._calculate_name_similarity_matrix(nodes)
        
        # Convert similarity to distance matrix
        distance_matrix = 1.0 - similarity_matrix
        
        # Apply clustering algorithm
        if algorithm == "hierarchical":
            cluster_labels = self._hierarchical_clustering(
                distance_matrix, n_clusters or min(10, n_nodes // 2)
            )
        elif algorithm == "kmeans":
            cluster_labels = self._kmeans_clustering(
                similarity_matrix, n_clusters or min(10, n_nodes // 2)
            )
        elif algorithm == "dbscan":
            cluster_labels = self._dbscan_clustering(
                distance_matrix, similarity_threshold
            )
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        # Create clustered nodes
        clustered_nodes = []
        for i, node in enumerate(nodes):
            clustered_node = {
                **node,
                'cluster_id': int(cluster_labels[i]),
                'node_index': i
            }
            clustered_nodes.append(clustered_node)
        
        processing_time = time.time() - start_time
        
        # Calculate cluster statistics
        unique_clusters = len(set(cluster_labels))
        cluster_sizes = defaultdict(int)
        for label in cluster_labels:
            cluster_sizes[label] += 1
        
        cluster_info = {
            'algorithm': f'semantic_{algorithm}',
            'total_clusters': unique_clusters,
            'processing_time': processing_time,
            'gpu_accelerated': self.use_gpu,
            'cluster_sizes': dict(cluster_sizes),
            'average_cluster_size': n_nodes / unique_clusters if unique_clusters > 0 else 0,
            'similarity_threshold': similarity_threshold if algorithm == 'dbscan' else None
        }
        
        logger.info(f"✅ Semantic clustering completed: {unique_clusters} clusters in {processing_time:.3f}s")
        
        return SemanticClusterResult(
            clustered_nodes=clustered_nodes,
            cluster_info=cluster_info,
            similarity_matrix=similarity_matrix,
            cluster_labels=cluster_labels
        )
    
    def cluster_by_content_similarity(
        self,
        nodes: List[Dict[str, Any]],
        algorithm: str = "kmeans",
        n_clusters: Optional[int] = None
    ) -> SemanticClusterResult:
        """Cluster nodes based on content similarity using TF-IDF"""
        start_time = time.time()
        n_nodes = len(nodes)
        
        logger.info(f"📄 Starting content-based clustering of {n_nodes} nodes")
        
        if n_nodes < 2:
            return self._create_single_cluster_result(nodes, start_time)
        
        # Calculate content similarity
        similarity_matrix = self.similarity_calc.calculate_content_similarity(nodes)
        
        # Apply clustering
        if algorithm == "kmeans":
            n_clusters = n_clusters or min(10, n_nodes // 2)
            if self.use_gpu and HAS_GPU:
                cluster_labels = self._gpu_kmeans_clustering(similarity_matrix, n_clusters)
            else:
                cluster_labels = self._kmeans_clustering(similarity_matrix, n_clusters)
        else:
            distance_matrix = 1.0 - similarity_matrix
            cluster_labels = self._hierarchical_clustering(
                distance_matrix, n_clusters or min(10, n_nodes // 2)
            )
        
        # Create result
        clustered_nodes = []
        for i, node in enumerate(nodes):
            clustered_node = {
                **node,
                'cluster_id': int(cluster_labels[i]),
                'node_index': i
            }
            clustered_nodes.append(clustered_node)
        
        processing_time = time.time() - start_time
        unique_clusters = len(set(cluster_labels))
        
        cluster_info = {
            'algorithm': f'content_{algorithm}',
            'total_clusters': unique_clusters,
            'processing_time': processing_time,
            'gpu_accelerated': self.use_gpu and algorithm == 'kmeans',
            'average_cluster_size': n_nodes / unique_clusters if unique_clusters > 0 else 0
        }
        
        logger.info(f"✅ Content clustering completed: {unique_clusters} clusters in {processing_time:.3f}s")
        
        return SemanticClusterResult(
            clustered_nodes=clustered_nodes,
            cluster_info=cluster_info,
            similarity_matrix=similarity_matrix,
            cluster_labels=cluster_labels
        )
    
    def hybrid_clustering(
        self,
        nodes: List[Dict[str, Any]],
        name_weight: float = 0.6,
        content_weight: float = 0.3,
        spatial_weight: float = 0.1,
        algorithm: str = "hierarchical",
        n_clusters: Optional[int] = None
    ) -> SemanticClusterResult:
        """
        Hybrid clustering combining name, content, and spatial similarities
        
        Args:
            name_weight: Weight for name similarity (0.0-1.0)
            content_weight: Weight for content similarity (0.0-1.0) 
            spatial_weight: Weight for spatial similarity (0.0-1.0)
        """
        start_time = time.time()
        n_nodes = len(nodes)
        
        logger.info(f"🔄 Starting hybrid clustering of {n_nodes} nodes")
        logger.info(f"   Weights: name={name_weight}, content={content_weight}, spatial={spatial_weight}")
        
        if n_nodes < 2:
            return self._create_single_cluster_result(nodes, start_time)
        
        # Normalize weights
        total_weight = name_weight + content_weight + spatial_weight
        if total_weight > 0:
            name_weight /= total_weight
            content_weight /= total_weight
            spatial_weight /= total_weight
        
        # Calculate different similarity matrices
        similarities = []
        weights = []
        
        if name_weight > 0:
            name_similarity = self._calculate_name_similarity_matrix(nodes)
            similarities.append(name_similarity)
            weights.append(name_weight)
        
        if content_weight > 0:
            content_similarity = self.similarity_calc.calculate_content_similarity(nodes)
            similarities.append(content_similarity)
            weights.append(content_weight)
        
        if spatial_weight > 0:
            spatial_similarity = self._calculate_spatial_similarity_matrix(nodes)
            similarities.append(spatial_similarity)
            weights.append(spatial_weight)
        
        # Combine similarities
        if not similarities:
            return self._create_single_cluster_result(nodes, start_time)
        
        combined_similarity = np.zeros((n_nodes, n_nodes))
        for similarity, weight in zip(similarities, weights):
            combined_similarity += similarity * weight
        
        # Apply clustering
        distance_matrix = 1.0 - combined_similarity
        
        if algorithm == "hierarchical":
            cluster_labels = self._hierarchical_clustering(
                distance_matrix, n_clusters or min(10, n_nodes // 2)
            )
        elif algorithm == "kmeans":
            cluster_labels = self._kmeans_clustering(
                combined_similarity, n_clusters or min(10, n_nodes // 2)
            )
        else:
            cluster_labels = self._dbscan_clustering(distance_matrix, 0.3)
        
        # Create result
        clustered_nodes = []
        for i, node in enumerate(nodes):
            clustered_node = {
                **node,
                'cluster_id': int(cluster_labels[i]),
                'node_index': i
            }
            clustered_nodes.append(clustered_node)
        
        processing_time = time.time() - start_time
        unique_clusters = len(set(cluster_labels))
        
        cluster_info = {
            'algorithm': f'hybrid_{algorithm}',
            'total_clusters': unique_clusters,
            'processing_time': processing_time,
            'gpu_accelerated': self.use_gpu,
            'weights': {
                'name': name_weight,
                'content': content_weight,
                'spatial': spatial_weight
            },
            'average_cluster_size': n_nodes / unique_clusters if unique_clusters > 0 else 0
        }
        
        logger.info(f"✅ Hybrid clustering completed: {unique_clusters} clusters in {processing_time:.3f}s")
        
        return SemanticClusterResult(
            clustered_nodes=clustered_nodes,
            cluster_info=cluster_info,
            similarity_matrix=combined_similarity,
            cluster_labels=cluster_labels
        )
    
    def _calculate_name_similarity_matrix(self, nodes: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate pairwise name similarity matrix"""
        n_nodes = len(nodes)
        similarity_matrix = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(i, n_nodes):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    name1 = nodes[i].get('name', '')
                    name2 = nodes[j].get('name', '')
                    similarity = self.similarity_calc.calculate_name_similarity(name1, name2)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # Symmetric
        
        return similarity_matrix
    
    def _calculate_spatial_similarity_matrix(self, nodes: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate spatial similarity based on node positions"""
        n_nodes = len(nodes)
        similarity_matrix = np.zeros((n_nodes, n_nodes))
        
        # Extract coordinates
        coords = []
        for node in nodes:
            x = float(node.get('x', 0))
            y = float(node.get('y', 0))
            z = float(node.get('z', 0))
            coords.append([x, y, z])
        
        coords = np.array(coords)
        
        # Calculate pairwise distances
        for i in range(n_nodes):
            for j in range(i, n_nodes):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Euclidean distance
                    dist = np.linalg.norm(coords[i] - coords[j])
                    # Convert distance to similarity (closer = more similar)
                    # Use exponential decay: similarity = exp(-distance/scale)
                    scale = 50.0  # Adjust based on your coordinate system
                    similarity = np.exp(-dist / scale)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def _hierarchical_clustering(self, distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        """Apply hierarchical clustering"""
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        return clusterer.fit_predict(distance_matrix)
    
    def _kmeans_clustering(self, similarity_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        """Apply K-means clustering"""
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return clusterer.fit_predict(similarity_matrix)
    
    def _gpu_kmeans_clustering(self, similarity_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        """Apply GPU-accelerated K-means clustering"""
        try:
            gpu_matrix = cp.array(similarity_matrix, dtype=cp.float32)
            clusterer = cuKMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(gpu_matrix)
            return cp.asnumpy(labels)
        except Exception as e:
            logger.warning(f"GPU K-means failed, falling back to CPU: {e}")
            return self._kmeans_clustering(similarity_matrix, n_clusters)
    
    def _dbscan_clustering(self, distance_matrix: np.ndarray, eps: float) -> np.ndarray:
        """Apply DBSCAN clustering"""
        clusterer = DBSCAN(eps=eps, metric='precomputed', min_samples=2)
        labels = clusterer.fit_predict(distance_matrix)
        
        # DBSCAN uses -1 for noise points, convert to positive integers
        unique_labels = set(labels)
        if -1 in unique_labels:
            # Assign noise points to individual clusters
            max_label = max(labels) if len(unique_labels) > 1 else -1
            noise_cluster = max_label + 1
            labels = np.array([noise_cluster if label == -1 else label for label in labels])
        
        return labels
    
    def _create_single_cluster_result(self, nodes: List[Dict[str, Any]], start_time: float) -> SemanticClusterResult:
        """Create result for single cluster (when too few nodes)"""
        clustered_nodes = []
        for i, node in enumerate(nodes):
            clustered_node = {
                **node,
                'cluster_id': 0,
                'node_index': i
            }
            clustered_nodes.append(clustered_node)
        
        processing_time = time.time() - start_time
        
        cluster_info = {
            'algorithm': 'single_cluster',
            'total_clusters': 1,
            'processing_time': processing_time,
            'gpu_accelerated': False,
            'average_cluster_size': len(nodes)
        }
        
        return SemanticClusterResult(
            clustered_nodes=clustered_nodes,
            cluster_info=cluster_info,
            similarity_matrix=None,
            cluster_labels=np.zeros(len(nodes), dtype=int)
        )

# Convenience functions for easy integration
async def cluster_nodes_by_similarity(
    nodes: List[Dict[str, Any]],
    method: str = "hybrid",
    algorithm: str = "hierarchical",
    n_clusters: Optional[int] = None,
    **kwargs
) -> SemanticClusterResult:
    """
    Main entry point for semantic clustering
    
    Args:
        nodes: List of node dictionaries
        method: 'name', 'content', 'hybrid'
        algorithm: 'hierarchical', 'kmeans', 'dbscan'
        n_clusters: Number of clusters (if applicable)
        **kwargs: Additional parameters for specific methods
    """
    engine = SemanticClusteringEngine()
    
    if method == "name":
        return engine.cluster_by_name_similarity(nodes, algorithm, n_clusters, **kwargs)
    elif method == "content":
        return engine.cluster_by_content_similarity(nodes, algorithm, n_clusters, **kwargs)
    elif method == "hybrid":
        return engine.hybrid_clustering(nodes, algorithm=algorithm, n_clusters=n_clusters, **kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

if __name__ == "__main__":
    # Example usage
    test_nodes = [
        {"name": "Machine Learning", "x": 0, "y": 0, "z": 0, "group": "AI"},
        {"name": "Deep Learning", "x": 10, "y": 5, "z": 2, "group": "AI"},
        {"name": "Neural Networks", "x": 15, "y": 8, "z": 3, "group": "AI"},
        {"name": "Data Science", "x": 20, "y": 10, "z": 5, "group": "Data"},
        {"name": "Statistics", "x": 25, "y": 15, "z": 8, "group": "Math"},
        {"name": "Linear Algebra", "x": 30, "y": 20, "z": 10, "group": "Math"},
    ]
    
    async def test():
        result = await cluster_nodes_by_similarity(test_nodes, method="hybrid")
        print("Cluster Result:", result.cluster_info)
        for node in result.clustered_nodes:
            print(f"  {node['name']} -> Cluster {node['cluster_id']}")
    
    asyncio.run(test())
