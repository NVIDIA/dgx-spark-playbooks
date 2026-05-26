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
# WebGL-Enhanced Remote GPU Rendering Service
# Using Three.js for GPU-accelerated visualization

import json
from typing import Dict, Any, List

class WebGLGPUVisualizationService:
    """Enhanced remote GPU service with Three.js WebGL rendering"""
    
    def _generate_threejs_webgl_html(self, session_data: dict, config: dict) -> str:
        """Generate Three.js WebGL visualization with GPU-accelerated rendering"""
        
        # Extract data
        nodes = session_data['processed_nodes']
        edges = session_data['processed_edges']
        layout_positions = session_data.get('layout_positions', {})
        clusters = session_data.get('clusters', {})
        centrality = session_data.get('centrality', {})
        
        # Configuration
        animation_duration = config.get('animation_duration', 3000)
        show_splash = config.get('show_splash', True)
        auto_zoom = config.get('auto_zoom', True)
        show_labels = config.get('show_labels', True)
        background_color = config.get('background_color', '#0a0a0a')
        render_quality = config.get('render_quality', 'high')
        
        # GPU rendering settings
        gpu_settings = self._get_gpu_rendering_settings(len(nodes), render_quality)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU-Accelerated WebGL Graph Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background-color: {background_color};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            overflow: hidden;
            color: #ffffff;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        #webgl-canvas {{
            display: block;
            width: 100%;
            height: 100%;
        }}
        .performance-monitor {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            color: #76B900;
            z-index: 100;
        }}
        .controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 5px;
            z-index: 100;
        }}
        .control-btn {{
            background: rgba(0, 0, 0, 0.8);
            color: #76B900;
            border: 1px solid #76B900;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        }}
        .control-btn:hover {{
            background: rgba(118, 185, 0, 0.2);
        }}
        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: #fff;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 200;
            border: 1px solid #76B900;
            opacity: 0;
            transition: opacity 0.2s ease;
        }}
        {"" if not show_splash else '''
        .splash-screen {
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
        }
        .splash-logo {
            font-size: 2rem;
            font-weight: 700;
            color: #76B900;
            margin-bottom: 1rem;
        }
        .loading-progress {
            width: 300px;
            height: 4px;
            background: #333;
            border-radius: 2px;
            overflow: hidden;
            margin-bottom: 1rem;
        }
        .loading-bar {
            height: 100%;
            background: linear-gradient(90deg, #76B900, #a8d45a);
            width: 0%;
            transition: width 0.3s ease;
        }
        '''}
    </style>
</head>
<body>
    <div id="container">
        <canvas id="webgl-canvas"></canvas>
        
        <!-- Performance Monitor -->
        <div class="performance-monitor">
            <div>🚀 WebGL GPU Rendering</div>
            <div>FPS: <span id="fps">--</span></div>
            <div>Nodes: {len(nodes):,}</div>
            <div>Triangles: <span id="triangles">--</span></div>
            <div>Memory: <span id="memory">--</span>MB</div>
        </div>
        
        <!-- Controls -->
        <div class="controls">
            <button class="control-btn" onclick="toggleAnimation()">⏸️ Animation</button>
            <button class="control-btn" onclick="resetCamera()">🎯 Reset</button>
            <button class="control-btn" onclick="toggleLabels()">🏷️ Labels</button>
            <button class="control-btn" onclick="toggleClusters()">🎨 Clusters</button>
            <button class="control-btn" onclick="exportImage()">📷 Export</button>
        </div>
        
        <!-- Tooltip -->
        <div id="tooltip" class="tooltip"></div>
        
        {"" if not show_splash else '''
        <div id="splash-screen" class="splash-screen">
            <div class="splash-logo">GPU WebGL Visualization</div>
            <div style="color: #888; margin-bottom: 2rem; text-align: center;">
                Loading {len(nodes):,} nodes with GPU acceleration<br>
                Quality: {render_quality.title()} • WebGL 2.0
            </div>
            <div class="loading-progress">
                <div id="loading-bar" class="loading-bar"></div>
            </div>
            <div id="loading-text" style="color: #888; font-size: 14px;">Initializing WebGL...</div>
        </div>
        '''}
    </div>

    <!-- Three.js Library (matching your package.json version) -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.176.0/three.min.js"></script>
    
    <script>
        // Graph data from GPU processing
        const graphData = {{
            nodes: {json.dumps(nodes)},
            edges: {json.dumps(edges)},
            layoutPositions: {json.dumps(layout_positions)},
            clusters: {json.dumps(clusters)},
            centrality: {json.dumps(centrality)}
        }};
        
        // GPU rendering configuration
        const gpuConfig = {{
            nodeCount: {len(nodes)},
            edgeCount: {len(edges)},
            maxInstancedNodes: {gpu_settings['max_instanced_nodes']},
            useInstancedMesh: {str(gpu_settings['use_instanced_mesh']).lower()},
            enableLOD: {str(gpu_settings['enable_lod']).lower()},
            frustumCulling: {str(gpu_settings['frustum_culling']).lower()},
            textureAtlasSize: {gpu_settings['texture_atlas_size']},
            animationDuration: {animation_duration},
            showLabels: {str(show_labels).lower()},
            autoZoom: {str(auto_zoom).lower()}
        }};
        
        class WebGLGraphVisualization {{
            constructor() {{
                this.container = document.getElementById('container');
                this.canvas = document.getElementById('webgl-canvas');
                
                // Performance monitoring
                this.frameCount = 0;
                this.lastTime = performance.now();
                this.isAnimating = true;
                this.labelsVisible = gpuConfig.showLabels;
                this.clustersVisible = true;
                
                this.init();
                {"this.hideSplash();" if not show_splash else "this.showLoadingProgress();"}
            }}
            
            init() {{
                // Initialize Three.js WebGL renderer with GPU optimizations
                this.renderer = new THREE.WebGLRenderer({{
                    canvas: this.canvas,
                    antialias: true,
                    alpha: true,
                    powerPreference: "high-performance"
                }});
                
                this.renderer.setSize(window.innerWidth, window.innerHeight);
                this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                this.renderer.setClearColor(0x0a0a0a, 1);
                
                // Enable GPU optimizations
                this.renderer.sortObjects = false; // Disable sorting for performance
                
                // Scene setup
                this.scene = new THREE.Scene();
                
                // Camera setup with controls
                this.camera = new THREE.PerspectiveCamera(
                    75, window.innerWidth / window.innerHeight, 0.1, 10000
                );
                this.camera.position.z = 1000;
                
                // Add basic controls
                this.setupControls();
                
                // Load and process graph data
                this.loadGraphData();
                
                // Start render loop
                this.animate();
                
                // Setup interaction
                this.setupInteraction();
                
                // Start performance monitoring
                this.startPerformanceMonitoring();
                
                console.log('WebGL Graph Visualization initialized');
            }}
            
            setupControls() {{
                // Simple camera controls
                this.controls = {{
                    mouseDown: false,
                    mouseX: 0,
                    mouseY: 0,
                    targetX: 0,
                    targetY: 0,
                    zoom: 1
                }};
                
                this.canvas.addEventListener('mousedown', (e) => {{
                    this.controls.mouseDown = true;
                    this.controls.mouseX = e.clientX;
                    this.controls.mouseY = e.clientY;
                }});
                
                this.canvas.addEventListener('mousemove', (e) => {{
                    if (this.controls.mouseDown) {{
                        const deltaX = e.clientX - this.controls.mouseX;
                        const deltaY = e.clientY - this.controls.mouseY;
                        
                        this.controls.targetX += deltaX * 2;
                        this.controls.targetY -= deltaY * 2;
                        
                        this.controls.mouseX = e.clientX;
                        this.controls.mouseY = e.clientY;
                    }}
                }});
                
                this.canvas.addEventListener('mouseup', () => {{
                    this.controls.mouseDown = false;
                }});
                
                this.canvas.addEventListener('wheel', (e) => {{
                    e.preventDefault();
                    this.controls.zoom *= (1 - e.deltaY * 0.001);
                    this.controls.zoom = Math.max(0.1, Math.min(10, this.controls.zoom));
                }});
            }}
            
            loadGraphData() {{
                console.log('Loading graph data with WebGL...');
                
                // Create node geometries and materials
                this.createNodeVisualization();
                this.createEdgeVisualization();
                
                if (this.labelsVisible) {{
                    this.createLabelVisualization();
                }}
                
                console.log('Graph data loaded successfully');
            }}
            
            createNodeVisualization() {{
                const nodeCount = graphData.nodes.length;
                
                if (gpuConfig.useInstancedMesh && nodeCount > 1000) {{
                    // GPU-accelerated instanced rendering for large graphs
                    console.log('Using GPU instanced mesh for', nodeCount, 'nodes');
                    
                    const geometry = new THREE.CircleGeometry(1, 8);
                    const material = new THREE.MeshBasicMaterial({{ 
                        vertexColors: true,
                        transparent: true,
                        opacity: 0.8
                    }});
                    
                    this.nodesMesh = new THREE.InstancedMesh(geometry, material, nodeCount);
                    
                    // Set up instance matrices and colors
                    const matrix = new THREE.Matrix4();
                    const color = new THREE.Color();
                    
                    graphData.nodes.forEach((node, i) => {{
                        // Position from GPU-computed layout
                        const pos = graphData.layoutPositions[node.id] || [0, 0];
                        const x = pos[0] - 500; // Center
                        const y = pos[1] - 500;
                        
                        // Size based on centrality
                        const centrality = node.pagerank || 0.001;
                        const size = Math.max(2, Math.sqrt(centrality * 10000) + 3);
                        
                        // Color based on cluster
                        const cluster = node.cluster || 0;
                        const clusterColor = this.getClusterColor(cluster);
                        
                        // Set instance transform
                        matrix.makeScale(size, size, 1);
                        matrix.setPosition(x, y, 0);
                        this.nodesMesh.setMatrixAt(i, matrix);
                        
                        // Set instance color
                        color.setHex(clusterColor);
                        this.nodesMesh.setColorAt(i, color);
                    }});
                    
                    this.nodesMesh.instanceMatrix.needsUpdate = true;
                    this.nodesMesh.instanceColor.needsUpdate = true;
                    
                    this.scene.add(this.nodesMesh);
                    
                }} else {{
                    // Standard mesh rendering for smaller graphs
                    console.log('Using standard mesh rendering for', nodeCount, 'nodes');
                    
                    this.nodesGroup = new THREE.Group();
                    
                    graphData.nodes.forEach((node, i) => {{
                        const pos = graphData.layoutPositions[node.id] || [0, 0];
                        const x = pos[0] - 500;
                        const y = pos[1] - 500;
                        
                        const centrality = node.pagerank || 0.001;
                        const size = Math.max(2, Math.sqrt(centrality * 10000) + 3);
                        
                        const cluster = node.cluster || 0;
                        const clusterColor = this.getClusterColor(cluster);
                        
                        const geometry = new THREE.CircleGeometry(size, 8);
                        const material = new THREE.MeshBasicMaterial({{ 
                            color: clusterColor,
                            transparent: true,
                            opacity: 0.8
                        }});
                        
                        const nodeMesh = new THREE.Mesh(geometry, material);
                        nodeMesh.position.set(x, y, 0);
                        nodeMesh.userData = {{ nodeData: node, nodeIndex: i }};
                        
                        this.nodesGroup.add(nodeMesh);
                    }});
                    
                    this.scene.add(this.nodesGroup);
                }}
            }}
            
            createEdgeVisualization() {{
                console.log('Creating edge visualization...');
                
                const edgeCount = graphData.edges.length;
                const positions = new Float32Array(edgeCount * 6); // 2 vertices * 3 coordinates
                const colors = new Float32Array(edgeCount * 6); // 2 vertices * 3 colors
                
                graphData.edges.forEach((edge, i) => {{
                    const sourcePos = graphData.layoutPositions[edge.source] || [0, 0];
                    const targetPos = graphData.layoutPositions[edge.target] || [0, 0];
                    
                    const idx = i * 6;
                    
                    // Source vertex
                    positions[idx] = sourcePos[0] - 500;
                    positions[idx + 1] = sourcePos[1] - 500;
                    positions[idx + 2] = 0;
                    
                    // Target vertex  
                    positions[idx + 3] = targetPos[0] - 500;
                    positions[idx + 4] = targetPos[1] - 500;
                    positions[idx + 5] = 0;
                    
                    // Edge color (gray)
                    colors[idx] = colors[idx + 3] = 0.3;
                    colors[idx + 1] = colors[idx + 4] = 0.3;
                    colors[idx + 2] = colors[idx + 5] = 0.3;
                }});
                
                const edgeGeometry = new THREE.BufferGeometry();
                edgeGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                edgeGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                
                const edgeMaterial = new THREE.LineBasicMaterial({{ 
                    vertexColors: true,
                    transparent: true,
                    opacity: 0.4
                }});
                
                this.edgesMesh = new THREE.LineSegments(edgeGeometry, edgeMaterial);
                this.scene.add(this.edgesMesh);
            }}
            
            createLabelVisualization() {{
                // Canvas-based text rendering for labels
                this.labelCanvases = [];
                
                graphData.nodes.forEach((node, i) => {{
                    if (i > 500) return; // Limit labels for performance
                    
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    canvas.width = 256;
                    canvas.height = 64;
                    
                    context.fillStyle = '#ffffff';
                    context.font = '16px Arial';
                    context.textAlign = 'center';
                    context.fillText(node.name || node.id, 128, 32);
                    
                    const texture = new THREE.CanvasTexture(canvas);
                    const material = new THREE.SpriteMaterial({{ map: texture }});
                    const sprite = new THREE.Sprite(material);
                    
                    const pos = graphData.layoutPositions[node.id] || [0, 0];
                    sprite.position.set(pos[0] - 500, pos[1] - 480, 1);
                    sprite.scale.set(50, 12.5, 1);
                    
                    this.scene.add(sprite);
                    this.labelCanvases.push(sprite);
                }});
            }}
            
            getClusterColor(cluster) {{
                // Midnight Tokyo inspired color palette - neon colors in hex format for WebGL
                const colors = [
                    0xFF0080, // Hot pink neon
                    0x00FFFF, // Electric cyan
                    0xFF4081, // Neon pink
                    0x8A2BE2, // Electric purple
                    0x00FF41, // Matrix green
                    0xFF6B35, // Neon orange
                    0x1E90FF, // Electric blue
                    0xFF1493, // Deep pink
                    0x00CED1, // Dark turquoise
                    0x9932CC, // Dark orchid
                    0x32CD32, // Lime green
                    0xFF4500, // Orange red
                    0x4169E1, // Royal blue
                    0xDC143C, // Crimson
                    0x00FA9A, // Medium spring green
                    0xFF69B4, // Hot pink
                    0x1E88E5, // Blue
                    0xE91E63, // Pink
                    0x00E676, // Green
                    0xFF5722, // Deep orange
                    0x673AB7, // Deep purple
                    0x03DAC6, // Teal
                    0xBB86FC, // Light purple
                    0xCF6679  // Light pink
                ];
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
                
                // Render with GPU
                this.renderer.render(this.scene, this.camera);
                
                // Update performance monitor
                this.updatePerformanceMonitor();
            }}
            
            setupInteraction() {{
                const raycaster = new THREE.Raycaster();
                const mouse = new THREE.Vector2();
                const tooltip = document.getElementById('tooltip');
                
                this.canvas.addEventListener('mousemove', (event) => {{
                    if (this.controls.mouseDown) return;
                    
                    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
                    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
                    
                    raycaster.setFromCamera(mouse, this.camera);
                    
                    let intersects = [];
                    if (this.nodesGroup) {{
                        intersects = raycaster.intersectObjects(this.nodesGroup.children);
                    }}
                    
                    if (intersects.length > 0) {{
                        const nodeData = intersects[0].object.userData.nodeData;
                        tooltip.innerHTML = `
                            <strong>${{nodeData.name || nodeData.id}}</strong><br>
                            Cluster: ${{nodeData.cluster || 'N/A'}}<br>
                            PageRank: ${{(nodeData.pagerank || 0).toFixed(4)}}
                        `;
                        tooltip.style.left = (event.clientX + 10) + 'px';
                        tooltip.style.top = (event.clientY - 10) + 'px';
                        tooltip.style.opacity = '1';
                    }} else {{
                        tooltip.style.opacity = '0';
                    }}
                }});
            }}
            
            startPerformanceMonitoring() {{
                setInterval(() => {{
                    const now = performance.now();
                    const fps = Math.round((this.frameCount * 1000) / (now - this.lastTime));
                    
                    document.getElementById('fps').textContent = fps;
                    document.getElementById('triangles').textContent = 
                        (this.renderer.info.render.triangles || 0).toLocaleString();
                    document.getElementById('memory').textContent = 
                        Math.round(this.renderer.info.memory.geometries + this.renderer.info.memory.textures);
                    
                    this.frameCount = 0;
                    this.lastTime = now;
                }}, 1000);
            }}
            
            updatePerformanceMonitor() {{
                this.frameCount++;
            }}
            
            {"showLoadingProgress() { /* Loading animation */ }" if show_splash else ""}
            {"hideSplash() { /* Hide splash */ }" if show_splash else ""}
            
            resetCamera() {{
                this.controls.targetX = 0;
                this.controls.targetY = 0;
                this.controls.zoom = 1;
            }}
            
            toggleAnimation() {{
                this.isAnimating = !this.isAnimating;
            }}
            
            toggleLabels() {{
                this.labelsVisible = !this.labelsVisible;
                this.labelCanvases.forEach(sprite => {{
                    sprite.visible = this.labelsVisible;
                }});
            }}
            
            toggleClusters() {{
                this.clustersVisible = !this.clustersVisible;
                // Toggle cluster coloring
            }}
            
            exportImage() {{
                const link = document.createElement('a');
                link.download = 'webgl-graph.png';
                link.href = this.renderer.domElement.toDataURL();
                link.click();
            }}
        }}
        
        // Global control functions
        window.toggleAnimation = () => window.graphViz.toggleAnimation();
        window.resetCamera = () => window.graphViz.resetCamera();
        window.toggleLabels = () => window.graphViz.toggleLabels();
        window.toggleClusters = () => window.graphViz.toggleClusters();
        window.exportImage = () => window.graphViz.exportImage();
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            if (window.graphViz) {{
                window.graphViz.camera.aspect = window.innerWidth / window.innerHeight;
                window.graphViz.camera.updateProjectionMatrix();
                window.graphViz.renderer.setSize(window.innerWidth, window.innerHeight);
            }}
        }});
        
        // Initialize when DOM is ready
        document.addEventListener('DOMContentLoaded', () => {{
            window.graphViz = new WebGLGraphVisualization();
        }});
    </script>
</body>
</html>
        """
        
        return html_template
    
    def _get_gpu_rendering_settings(self, node_count: int, quality: str) -> Dict[str, Any]:
        """Get GPU rendering settings based on graph size and quality"""
        
        base_settings = {
            'max_instanced_nodes': 100000,
            'use_instanced_mesh': node_count > 1000,
            'enable_lod': node_count > 25000,
            'frustum_culling': node_count > 10000,
            'texture_atlas_size': 1024
        }
        
        quality_multipliers = {
            'low': {'texture_atlas_size': 512, 'max_instanced_nodes': 50000},
            'medium': {'texture_atlas_size': 1024, 'max_instanced_nodes': 75000},
            'high': {'texture_atlas_size': 2048, 'max_instanced_nodes': 100000},
            'ultra': {'texture_atlas_size': 4096, 'max_instanced_nodes': 500000}
        }
        
        settings = base_settings.copy()
        settings.update(quality_multipliers.get(quality, quality_multipliers['high']))
        
        return settings 