//
// SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Remote WebGPU Clustering Client
// Provides fallback clustering when local WebGPU is not available

export interface RemoteClusteringOptions {
  serviceUrl: string;
  mode: 'hybrid' | 'webrtc_stream';
  clusterDimensions: [number, number, number];
  forceSimulation: boolean;
  maxIterations: number;
  webrtcOptions?: {
    autoRefresh: boolean;
    refreshInterval: number;
  };
  
  // Semantic clustering options
  clusteringMethod?: string; // "spatial", "semantic", "hybrid"
  semanticAlgorithm?: string; // "hierarchical", "kmeans", "dbscan"
  numberOfClusters?: number | null;
  similarityThreshold?: number;
  nameWeight?: number;
  contentWeight?: number;
  spatialWeight?: number;
}

export interface ClusteringResult {
  clusteredNodes: any[];
  clusterInfo: {
    totalClusters: number;
    usedClusters: number;
    clusterDimensions: [number, number, number];
    processingTime: number;
    gpuAccelerated: boolean;
    clusterStats?: any;
  };
  processingTime: number;
  mode: string;
  sessionId?: string;
}

export interface ServiceCapabilities {
  modes: {
    hybrid: {
      available: boolean;
      description: string;
    };
    webrtc_stream: {
      available: boolean;
      description: string;
    };
  };
  gpuAcceleration: {
    rapidsAvailable: boolean;
    opencvAvailable: boolean;
    plottingAvailable: boolean;
  };
  clusterDimensions: [number, number, number];
  maxClusterCount: number;
}

/**
 * Remote WebGPU Clustering Client
 * Provides GPU-accelerated clustering for browsers without WebGPU support
 */
export class RemoteWebGPUClusteringClient {
  private serviceUrl: string;
  private useProxy: boolean;
  private websocket: WebSocket | null = null;
  private capabilities: ServiceCapabilities | null = null;
  private eventListeners: Map<string, Function[]> = new Map();

  constructor(serviceUrl: string = 'http://localhost:8083', useProxy: boolean = false) {
    this.serviceUrl = serviceUrl;
    this.useProxy = useProxy;
  }

  private getApiUrl(path: string): string {
    if (this.useProxy) {
      // Use the Next.js API proxy route
      return `/api/remote-webgpu/${path}`;
    } else {
      // Direct connection to service
      return `${this.serviceUrl}/${path}`;
    }
  }

  /**
   * Check if the remote service is available and get its capabilities
   */
  async checkAvailability(): Promise<boolean> {
    try {
      const response = await fetch(this.getApiUrl('api/capabilities'));
      if (response.ok) {
        this.capabilities = await response.json();
        console.log('Remote WebGPU service available:', this.capabilities);
        return true;
      }
      return false;
    } catch (error) {
      console.warn('Remote WebGPU service not available:', error);
      return false;
    }
  }

  /**
   * Get service capabilities
   */
  getCapabilities(): ServiceCapabilities | null {
    return this.capabilities;
  }

  /**
   * Perform remote clustering
   */
  async clusterNodes(
    nodes: any[], 
    links: any[], 
    options: Partial<RemoteClusteringOptions> = {}
  ): Promise<ClusteringResult> {
    const requestOptions: RemoteClusteringOptions = {
      serviceUrl: this.serviceUrl,
      mode: 'hybrid',
      clusterDimensions: [32, 18, 24],
      forceSimulation: true,
      maxIterations: 100,
      ...options
    };

    const requestData = {
      graph_data: {
        nodes,
        links
      },
      mode: requestOptions.mode,
      cluster_dimensions: requestOptions.clusterDimensions,
      force_simulation: requestOptions.forceSimulation,
      max_iterations: requestOptions.maxIterations,
      webrtc_options: requestOptions.webrtcOptions,
      
      // Semantic clustering parameters
      clustering_method: requestOptions.clusteringMethod || "hybrid",
      semantic_algorithm: requestOptions.semanticAlgorithm || "hierarchical",
      n_clusters: requestOptions.numberOfClusters,
      similarity_threshold: requestOptions.similarityThreshold || 0.7,
      name_weight: requestOptions.nameWeight || 0.6,
      content_weight: requestOptions.contentWeight || 0.3,
      spatial_weight: requestOptions.spatialWeight || 0.1
    };

    try {
      const response = await fetch(this.getApiUrl('api/cluster'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw new Error(`Remote clustering failed: ${response.statusText}`);
      }

      const rawResult = await response.json();
      
      // Map snake_case API response to camelCase interface
      const result: ClusteringResult = {
        clusteredNodes: rawResult.clustered_nodes || [],
        clusterInfo: {
          totalClusters: rawResult.total_clusters || 0,
          usedClusters: rawResult.used_clusters || 0,
          clusterDimensions: rawResult.cluster_dimensions || [32, 18, 24],
          processingTime: rawResult.processing_time || 0,
          gpuAccelerated: rawResult.gpu_accelerated || true,
          clusterStats: rawResult.cluster_stats
        },
        processingTime: rawResult.processing_time || 0,
        sessionId: rawResult.session_id,
        mode: rawResult.mode
      };
      
      console.log('🔄 Mapped API response:', {
        originalKeys: Object.keys(rawResult),
        mappedClusteredNodes: result.clusteredNodes?.length,
        processingTime: result.processingTime
      });
      
      // Emit clustering complete event
      this.emit('clusteringComplete', result);
      
      return result;
    } catch (error) {
      console.error('Remote clustering request failed:', error);
      throw error;
    }
  }

  /**
   * Start WebRTC streaming session
   */
  async startWebRTCStreaming(nodes: any[], links: any[]): Promise<string | null> {
    const result = await this.clusterNodes(nodes, links, { mode: 'webrtc_stream' });
    
    if (result.sessionId) {
      console.log(`WebRTC streaming session started: ${result.sessionId}`);
      return result.sessionId;
    }
    
    return null;
  }

  /**
   * Get WebRTC stream frame URL
   */
  getStreamFrameUrl(sessionId: string): string {
    if (this.useProxy) {
      return `/api/remote-webgpu-stream/${sessionId}`;
    } else {
      return `${this.serviceUrl}/api/stream/${sessionId}`;
    }
  }

  /**
   * Cleanup WebRTC streaming session
   */
  async cleanupWebRTCSession(sessionId: string): Promise<void> {
    try {
      await fetch(this.getApiUrl(`api/stream/${sessionId}`), {
        method: 'DELETE'
      });
      console.log(`WebRTC session ${sessionId} cleaned up`);
    } catch (error) {
      console.warn(`Failed to cleanup WebRTC session ${sessionId}:`, error);
    }
  }

  /**
   * Connect to WebSocket for real-time updates
   */
  connectWebSocket(): void {
    if (this.websocket) {
      return;
    }

    if (this.useProxy) {
      // Skip WebSocket connection when using proxy mode
      console.log('WebSocket disabled in proxy mode');
      return;
    }

    try {
      const wsUrl = this.serviceUrl.replace('http', 'ws') + '/ws';
      this.websocket = new WebSocket(wsUrl);

      this.websocket.onopen = () => {
        console.log('Connected to remote WebGPU service WebSocket');
        this.emit('connected');
      };

      this.websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.emit('message', data);
          
          // Handle specific message types
          if (data.type === 'clustering_complete') {
            this.emit('clusteringComplete', data.data);
          }
        } catch (error) {
          console.warn('Failed to parse WebSocket message:', error);
        }
      };

      this.websocket.onclose = () => {
        console.log('Disconnected from remote WebGPU service WebSocket');
        this.websocket = null;
        this.emit('disconnected');
      };

      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  }

  /**
   * Disconnect WebSocket
   */
  disconnectWebSocket(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }

  /**
   * Add event listener
   */
  on(event: string, listener: Function): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event)!.push(listener);
  }

  /**
   * Remove event listener
   */
  off(event: string, listener: Function): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      const index = listeners.indexOf(listener);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }
  }

  /**
   * Emit event
   */
  private emit(event: string, ...args: any[]): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.forEach(listener => {
        try {
          listener(...args);
        } catch (error) {
          console.error(`Event listener error for ${event}:`, error);
        }
      });
    }
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    this.disconnectWebSocket();
    this.eventListeners.clear();
  }
}

/**
 * Enhanced WebGPU Clustering Engine with Remote Fallback
 * Automatically detects WebGPU availability and falls back to remote service
 */
export class EnhancedWebGPUClusteringEngine {
  private localEngine: any | null = null; // WebGPUClusteringEngine
  private remoteClient: RemoteWebGPUClusteringClient;
  private useRemote: boolean = false;
  private isInitialized: boolean = false;
  private lastClusteredData: { nodes: any[], links: any[] } | null = null;
  private clusteringOptions: Partial<RemoteClusteringOptions> = {};

  constructor(
    clusterDimensions: [number, number, number] = [32, 18, 24],
    remoteServiceUrl: string = 'http://localhost:8083'
  ) {
    this.remoteClient = new RemoteWebGPUClusteringClient(remoteServiceUrl, false); // Disable proxy mode for WebSocket
    
    // Try to import local WebGPU engine
    this.tryInitializeLocal(clusterDimensions);
  }

  private async tryInitializeLocal(clusterDimensions: [number, number, number]): Promise<void> {
    // For hybrid mode, skip local WebGPU and go directly to remote
    console.log('Skipping local WebGPU initialization, using remote service for hybrid mode');
    await this.initializeRemote();
  }

  private async initializeRemote(): Promise<void> {
    try {
      console.log('🔄 Checking remote WebGPU service availability...');
      const available = await this.remoteClient.checkAvailability();
      console.log('🎯 Remote service check result:', available);
      
      if (available) {
        this.useRemote = true;
        this.isInitialized = true;
        console.log('✅ Enhanced WebGPU engine initialized with remote cuGraph service');
        
        // Skip WebSocket connection for hybrid mode - we only need HTTP API calls
        console.log('⚙️ Using HTTP API for cuGraph clustering (no WebSocket needed)');
      } else {
        console.error('❌ Remote cuGraph service not available - falling back to CPU');
        this.useRemote = false;
        this.isInitialized = false;
      }
    } catch (error) {
      console.error('❌ Failed to initialize remote cuGraph service:', error);
      this.useRemote = false;
      this.isInitialized = false;
    }
  }

  /**
   * Check if clustering is available (local or remote)
   */
  isAvailable(): boolean {
    return this.isInitialized;
  }

  /**
   * Check if using remote service
   */
  isUsingRemote(): boolean {
    return this.useRemote;
  }

  /**
   * Get service capabilities
   */
  getCapabilities(): ServiceCapabilities | null {
    if (this.useRemote) {
      return this.remoteClient.getCapabilities();
    }
    return null;
  }

  /**
   * Get the last clustered data (for pre-rendering optimization)
   */
  getClusteredData(): { nodes: any[], links: any[] } | null {
    return this.lastClusteredData;
  }

  /**
   * Set clustering options for semantic clustering
   */
  setClusteringOptions(options: Partial<RemoteClusteringOptions>): void {
    this.clusteringOptions = { ...this.clusteringOptions, ...options };
    console.log('🔧 Updated clustering options:', this.clusteringOptions);
  }

  /**
   * Update node positions and compute clusters
   */
  async updateNodePositions(nodes: any[], links: any[] = []): Promise<boolean> {
    console.log('🚀 updateNodePositions called with', nodes.length, 'nodes,', links.length, 'links');
    console.log('🔍 Engine state - initialized:', this.isInitialized, 'useRemote:', this.useRemote);
    
    if (!this.isInitialized) {
      console.warn('❌ Enhanced WebGPU clustering engine not initialized');
      return false;
    }

    try {
      if (this.useRemote) {
        console.log('🌐 Using remote cuGraph clustering service');
        
        // Use remote clustering with semantic options
        const result = await this.remoteClient.clusterNodes(nodes, links, {
          mode: 'hybrid',
          forceSimulation: true,
          ...this.clusteringOptions
        });
        
        console.log('📊 cuGraph clustering result:', result);
        
        // Store the clustered data for potential pre-rendering optimization
        if (result.clusteredNodes) {
          this.lastClusteredData = {
            nodes: result.clusteredNodes.map(node => ({
              ...node,
              cluster_index: node.cluster_index, // Keep original
              clusterIndex: node.cluster_index   // Add camelCase for frontend
            })),
            links: links
          };
        }
        
        // Update nodes with clustering results
        if (result.clusteredNodes && result.clusteredNodes.length === nodes.length) {
          let clustersFound = new Set();
          result.clusteredNodes.forEach((clusteredNode, i) => {
            if (nodes[i]) {
              nodes[i].clusterIndex = clusteredNode.cluster_index;
              nodes[i].nodeIndex = clusteredNode.node_index;
              clustersFound.add(clusteredNode.cluster_index);
              // Update positions if force simulation was applied
              if (clusteredNode.x !== undefined) nodes[i].x = clusteredNode.x;
              if (clusteredNode.y !== undefined) nodes[i].y = clusteredNode.y;
              if (clusteredNode.z !== undefined) nodes[i].z = clusteredNode.z;
            }
          });
          console.log(`🎯 cuGraph found ${clustersFound.size} clusters:`, Array.from(clustersFound));
        } else {
          console.warn('⚠️ cuGraph result mismatch - expected', nodes.length, 'got', result.clusteredNodes?.length);
        }
        
        return true;
      } else {
        console.log('💻 Using local WebGPU engine (fallback)');
        // Use local WebGPU engine
        return this.localEngine?.updateNodePositions(nodes) || false;
      }
    } catch (error) {
      console.error('❌ Failed to update node positions:', error);
      return false;
    }
  }

  /**
   * Start WebRTC streaming mode (remote only)
   */
  async startWebRTCStreaming(nodes: any[], links: any[]): Promise<string | null> {
    if (!this.useRemote) {
      console.warn('WebRTC streaming only available with remote service');
      return null;
    }

    return await this.remoteClient.startWebRTCStreaming(nodes, links);
  }

  /**
   * Get WebRTC stream frame URL
   */
  getStreamFrameUrl(sessionId: string): string | null {
    if (!this.useRemote) {
      return null;
    }
    return this.remoteClient.getStreamFrameUrl(sessionId);
  }

  /**
   * Add event listener for remote events
   */
  on(event: string, listener: Function): void {
    if (this.useRemote) {
      this.remoteClient.on(event, listener);
    }
  }

  /**
   * Remove event listener
   */
  off(event: string, listener: Function): void {
    if (this.useRemote) {
      this.remoteClient.off(event, listener);
    }
  }

  /**
   * Read clustered data (local only)
   */
  async readClusteredData(): Promise<any[] | null> {
    if (this.useRemote) {
      console.warn('readClusteredData not available with remote service');
      return null;
    }
    
    return this.localEngine?.readClusteredData() || null;
  }

  /**
   * Dispose of resources
   */
  dispose(): void {
    if (this.localEngine) {
      this.localEngine.dispose();
      this.localEngine = null;
    }
    
    this.remoteClient.dispose();
    this.isInitialized = false;
    this.useRemote = false;
  }
}
