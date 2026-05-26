# GPU Graph Visualization Services

## 🚀 Overview

This directory contains optional GPU-accelerated graph visualization services that run separately from the main txt2kg application. These services provide advanced visualization capabilities for large-scale graphs.

**Note**: These services are **optional** and not included in the default docker-compose configurations. They must be run separately.

## 📦 Available Services

### 1. Unified GPU Service (`unified_gpu_service.py`)
Combines **PyGraphistry Cloud** and **Local GPU (cuGraph)** processing into a single FastAPI service.

**Processing Modes:**
| Mode | Description | Requirements |
|------|-------------|--------------|
| **PyGraphistry Cloud** | Interactive GPU embeds in browser | API credentials |
| **Local GPU (cuGraph)** | Full GPU processing on your hardware | NVIDIA GPU + cuGraph |
| **Local CPU** | NetworkX fallback processing | None |

### 2. Remote GPU Rendering Service (`remote_gpu_rendering_service.py`)
Provides GPU-accelerated graph layout and rendering with iframe-embeddable visualizations.

### 3. Local GPU Service (`local_gpu_viz_service.py`)
Local GPU processing service with WebSocket support for real-time updates.

## 🛠️ Setup

### Prerequisites
- NVIDIA GPU with CUDA support (for GPU modes)
- RAPIDS cuGraph (for local GPU processing)
- PyGraphistry account (for cloud mode)

### Installation

```bash
# Install dependencies
pip install -r deploy/services/gpu-viz/requirements.txt

# For remote WebGPU service
pip install -r deploy/services/gpu-viz/requirements-remote-webgpu.txt
```

### Running Services

#### Unified GPU Service
```bash
cd deploy/services/gpu-viz
python unified_gpu_service.py
```

Service runs on: http://localhost:8080

#### Remote GPU Rendering Service
```bash
cd deploy/services/gpu-viz
python remote_gpu_rendering_service.py
```

Service runs on: http://localhost:8082

#### Using Startup Script
```bash
cd deploy/services/gpu-viz
./start_remote_gpu_services.sh
```

## 📡 API Usage

### Process Graph with Mode Selection

```bash
curl -X POST http://localhost:8080/api/visualize \
  -H "Content-Type: application/json" \
  -d '{
    "graph_data": {
      "nodes": [{"id": "1", "name": "Node 1"}, {"id": "2", "name": "Node 2"}],
      "links": [{"source": "1", "target": "2", "name": "edge_1_2"}]
    },
    "processing_mode": "local_gpu",
    "layout_algorithm": "force_atlas2",
    "clustering_algorithm": "leiden",
    "compute_centrality": true
  }'
```

### Check Available Capabilities

```bash
curl http://localhost:8080/api/capabilities
```

Response:
```json
{
  "processing_modes": {
    "pygraphistry_cloud": {"available": true, "description": "..."},
    "local_gpu": {"available": true, "description": "..."},
    "local_cpu": {"available": true, "description": "..."}
  },
  "has_rapids": true,
  "gpu_available": true
}
```

## 🎯 Frontend Integration

The txt2kg frontend includes built-in components for GPU visualization:

- `UnifiedGPUViewer`: Connects to unified GPU service
- `PyGraphistryViewer`: Direct PyGraphistry cloud integration
- `ForceGraphWrapper`: Three.js WebGPU visualization (default)

### Using GPU Services in Frontend

The frontend has API routes that can connect to these services:
- `/api/pygraphistry/*`: PyGraphistry integration
- `/api/unified-gpu/*`: Unified GPU service integration

To use these services, ensure they are running separately and configure the frontend environment variables accordingly.

### Mode-Specific Processing

```javascript
// PyGraphistry Cloud mode
const response = await fetch('/api/unified-gpu/visualize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    graph_data: { nodes, links },
    processing_mode: 'pygraphistry_cloud',
    layout_type: 'force',
    clustering: true,
    gpu_acceleration: true
  })
})

// Local GPU mode  
const response = await fetch('/api/unified-gpu/visualize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    graph_data: { nodes, links },
    processing_mode: 'local_gpu',
    layout_algorithm: 'force_atlas2',
    clustering_algorithm: 'leiden',
    compute_centrality: true
  })
})
```

## 🔧 Configuration Options

### PyGraphistry Cloud Mode
- `layout_type`: "force", "circular", "hierarchical"
- `gpu_acceleration`: true/false
- `clustering`: true/false

### Local GPU Mode  
- `layout_algorithm`: "force_atlas2", "spectral", "fruchterman_reingold"
- `clustering_algorithm`: "leiden", "louvain", "spectral"
- `compute_centrality`: true/false

### Local CPU Mode
- Basic processing with NetworkX fallback
- No additional configuration needed

## 📊 Response Format

```json
{
  "processed_nodes": [...],
  "processed_edges": [...],
  "processing_mode": "local_gpu",
  "embed_url": "https://hub.graphistry.com/...", // Only for cloud mode
  "layout_positions": {...}, // Only for local GPU mode
  "clusters": {...},
  "centrality": {...},
  "stats": {
    "node_count": 1000,
    "edge_count": 5000,
    "gpu_accelerated": true,
    "layout_computed": true,
    "clusters_computed": true
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🚀 Benefits of Unified Approach

### ✅ Advantages
- **Single service** - One port, one deployment
- **Mode switching** - Choose best processing per graph
- **Fallback handling** - Graceful degradation if GPU unavailable  
- **Consistent API** - Same interface for all modes
- **Better testing** - Easy comparison between modes

### 🎯 Use Cases
- **PyGraphistry Cloud**: Sharing visualizations, demos, production embeds
- **Local GPU**: Private data, large-scale processing, custom algorithms
- **Local CPU**: Development, testing, small graphs

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Check RAPIDS installation
python -c "import cudf, cugraph; print('RAPIDS OK')"
```

### PyGraphistry Credentials
```bash
# Verify credentials are set
echo $GRAPHISTRY_PERSONAL_KEY
echo $GRAPHISTRY_SECRET_KEY

# Test connection
python -c "import graphistry; graphistry.register(personal_key_id='$GRAPHISTRY_PERSONAL_KEY', personal_key_secret='$GRAPHISTRY_SECRET_KEY'); print('PyGraphistry OK')"
```

### Service Health
```bash
curl http://localhost:8080/api/health
```

## 📈 Performance Tips

1. **Large graphs (>100k nodes)**: Use `local_gpu` mode
2. **Sharing/demos**: Use `pygraphistry_cloud` mode  
3. **Development**: Use `local_cpu` mode for speed
4. **Mixed workloads**: Switch modes dynamically based on graph size 