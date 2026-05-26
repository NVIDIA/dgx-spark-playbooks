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
// WebGPU Clustering utilities for NVIDIA GPU acceleration
// This implements clustered rendering for knowledge graphs

// Define WebGPU types for TypeScript
declare global {
  interface Navigator {
    gpu?: {
      requestAdapter: (options?: GPURequestAdapterOptions) => Promise<GPUAdapter | null>;
    };
  }
  
  interface GPURequestAdapterOptions {
    powerPreference?: 'high-performance' | 'low-power';
  }
  
  interface GPUAdapter {
    name?: string;
    requestDevice: (options?: GPUDeviceDescriptor) => Promise<GPUDevice | null>;
  }
  
  interface GPUDeviceDescriptor {
    requiredFeatures?: string[];
  }
  
  interface GPUDevice {
    createBuffer: (descriptor: GPUBufferDescriptor) => GPUBuffer;
    createShaderModule: (descriptor: GPUShaderModuleDescriptor) => GPUShaderModule;
    createComputePipeline: (descriptor: GPUComputePipelineDescriptor) => GPUComputePipeline;
    createBindGroup: (descriptor: GPUBindGroupDescriptor) => GPUBindGroup;
    createCommandEncoder: () => GPUCommandEncoder;
    queue: GPUQueue;
  }
  
  interface GPUQueue {
    writeBuffer: (buffer: GPUBuffer, offset: number, data: BufferSource) => void;
    submit: (commandBuffers: GPUCommandBuffer[]) => void;
  }
  
  interface GPUBufferDescriptor {
    size: number;
    usage: number;
  }
  
  interface GPUBuffer {
    size: number;
    mapAsync: (mode: number, offset?: number, size?: number) => Promise<void>;
    getMappedRange: (offset?: number, size?: number) => ArrayBuffer;
    unmap: () => void;
    destroy: () => void;
  }
  
  interface GPUShaderModuleDescriptor {
    code: string;
  }
  
  interface GPUShaderModule {}
  
  interface GPUComputePipelineDescriptor {
    layout: 'auto' | GPUPipelineLayout;
    compute: {
      module: GPUShaderModule;
      entryPoint: string;
    };
  }
  
  interface GPUPipelineLayout {}
  
  interface GPUComputePipeline {
    getBindGroupLayout: (index: number) => GPUBindGroupLayout;
  }
  
  interface GPUBindGroupLayout {}
  
  interface GPUBindGroupDescriptor {
    layout: GPUBindGroupLayout;
    entries: Array<{
      binding: number;
      resource: { buffer: GPUBuffer } | { sampler: GPUSampler } | { texture: GPUTexture };
    }>;
  }
  
  interface GPUBindGroup {}
  
  interface GPUCommandEncoder {
    beginComputePass: () => GPUComputePassEncoder;
    copyBufferToBuffer: (
      source: GPUBuffer,
      sourceOffset: number,
      destination: GPUBuffer,
      destinationOffset: number,
      size: number
    ) => void;
    finish: () => GPUCommandBuffer;
  }
  
  interface GPUComputePassEncoder {
    setPipeline: (pipeline: GPUComputePipeline) => void;
    setBindGroup: (index: number, bindGroup: GPUBindGroup) => void;
    dispatchWorkgroups: (x: number, y: number, z: number) => void;
    end: () => void;
  }
  
  interface GPUCommandBuffer {}
  
  interface GPUSampler {}
  
  interface GPUTexture {}
}

// WebGPU buffer usage flags - use explicit values instead of enums
const GPU_BUFFER_USAGE = {
  COPY_SRC: 0x0001,
  COPY_DST: 0x0002,
  MAP_READ: 0x0004,
  MAP_WRITE: 0x0008,
  STORAGE: 0x0080,
  UNIFORM: 0x0040
};

// WebGPU map mode flags - use explicit values instead of enums
const GPU_MAP_MODE = {
  READ: 0x0001,
  WRITE: 0x0002
};

/**
 * Represents a 3D cluster in space
 */
interface Cluster {
  minBounds: [number, number, number];
  maxBounds: [number, number, number];
  nodeIndices: Uint32Array;
  count: number;
  capacity: number;
}

/**
 * Builds and manages clustered rendering for large graphs on WebGPU
 * Optimized for NVIDIA GPUs through specialized workgroup sizes and memory access patterns
 */
export class WebGPUClusteringEngine {
  private device: GPUDevice | null = null;
  private clusterDimensions: [number, number, number];
  private clusterCount: number;
  private clustersBuffer: GPUBuffer | null = null;
  private nodeBuffer: GPUBuffer | null = null;
  private computePipeline: GPUComputePipeline | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private isInitialized = false;
  private isNvidiaGPU = false;
  private forceBuffer: GPUBuffer | null = null;
  private forceComputePipeline: GPUComputePipeline | null = null;
  private forceBindGroup: GPUBindGroup | null = null;
  
  /**
   * Creates a new WebGPU clustering engine
   * @param clusterDimensions X, Y, Z dimensions of the cluster grid
   */
  constructor(clusterDimensions: [number, number, number] = [32, 18, 24]) {
    this.clusterDimensions = clusterDimensions;
    this.clusterCount = clusterDimensions[0] * clusterDimensions[1] * clusterDimensions[2];
    console.log(`Creating WebGPU clustering engine with ${this.clusterCount} clusters`);
  }
  
  /**
   * Initializes the WebGPU device and resources
   */
  async initialize(): Promise<boolean> {
    try {
      if (!navigator.gpu) {
        console.warn("WebGPU not supported in this browser");
        return false;
      }
      
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });
      
      if (!adapter) {
        console.warn("No suitable GPU adapter found");
        return false;
      }
      
      // Log adapter info - helpful for debugging NVIDIA support
      if (adapter.name) {
        console.log(`GPU detected: ${adapter.name}`);
        // Check if we're running on an NVIDIA GPU
        this.isNvidiaGPU = adapter.name.toLowerCase().includes('nvidia');
        if (this.isNvidiaGPU) {
          console.log("NVIDIA GPU detected - using optimized settings");
        }
      }
      
      this.device = await adapter.requestDevice({
        requiredFeatures: ['timestamp-query', 'bgra8unorm-storage']
      });
      
      if (!this.device) {
        console.warn("Failed to get GPU device");
        return false;
      }
      
      this.isInitialized = true;
      console.log("WebGPU clustering engine initialized successfully");
      return true;
    } catch (error) {
      console.error("Failed to initialize WebGPU:", error);
      return false;
    }
  }
  
  /**
   * Creates compute resources for clustering on the GPU
   * @param nodeCount Number of nodes in the graph
   */
  createComputeResources(nodeCount: number): boolean {
    if (!this.isInitialized || !this.device) {
      console.warn("WebGPU clustering engine not initialized");
      return false;
    }
    
    try {
      // Create buffer for clusters
      const clusterBufferSize = this.clusterCount * 64; // Size for cluster data
      this.clustersBuffer = this.device.createBuffer({
        size: clusterBufferSize,
        usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST
      });
      
      // Create buffer for nodes
      const nodeBufferSize = nodeCount * 32; // Size for node data (position, size, etc.)
      this.nodeBuffer = this.device.createBuffer({
        size: nodeBufferSize,
        usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.COPY_SRC
      });
      
      // Optimize shader based on GPU vendor - NVIDIA GPUs work better with 
      // specific workgroup sizes and memory access patterns
      const workgroupSize = this.isNvidiaGPU ? 128 : 64; // NVIDIA GPUs prefer larger workgroups
      
      // Create compute shader module for clustering
      const shaderModule = this.device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read_write> clusters: array<Cluster>;
          @group(0) @binding(1) var<storage, read_write> nodes: array<Node>;
          
          struct Cluster {
            minBounds: vec3f,
            padding1: f32,
            maxBounds: vec3f,
            padding2: f32,
            count: u32,
            capacity: u32,
            padding3: u32,
            padding4: u32,
          };
          
          struct Node {
            position: vec3f,
            size: f32,
            clusterIndex: u32,
            nodeIndex: u32,
            padding1: u32,
            padding2: u32,
          };
          
          // Improved clustering for WebGPU
          @compute @workgroup_size(${workgroupSize}, 1, 1)
          fn main(@builtin(global_invocation_id) global_id: vec3u) {
            let nodeIndex = global_id.x;
            if (nodeIndex >= arrayLength(&nodes)) {
              return;
            }
            
            // Optimized clustering algorithm for NVIDIA GPUs
            let node = nodes[nodeIndex];
            
            // Use log-scaled clusters in Z dimension for better distribution
            // This works better for graph visualization where nodes tend to cluster
            // at certain depths
            let clusterX = u32(clamp(node.position.x / 100.0 + 0.5, 0.0, 0.999) * ${this.clusterDimensions[0]}.0);
            let clusterY = u32(clamp(node.position.y / 100.0 + 0.5, 0.0, 0.999) * ${this.clusterDimensions[1]}.0);
            
            // For Z-dimension, use logarithmic scaling for better distribution
            let normalizedZ = clamp(node.position.z / 100.0 + 0.5, 0.001, 0.999);
            // Map using log scale (compressed at the edges, more detail in the center)
            let logZ = log(normalizedZ) / log(0.999);
            let clusterZ = u32(clamp(logZ, 0.0, 0.999) * ${this.clusterDimensions[2]}.0);
            
            // Calculate final cluster index
            let clusterIndex = clusterX + 
                              clusterY * ${this.clusterDimensions[0]}u + 
                              clusterZ * ${this.clusterDimensions[0]}u * ${this.clusterDimensions[1]}u;
                              
            // Store the cluster assignment
            nodes[nodeIndex].clusterIndex = clusterIndex;
          }
        `
      });
      
      // Create compute pipeline
      this.computePipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        }
      });
      
      // Create bind group
      this.bindGroup = this.device.createBindGroup({
        layout: this.computePipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: this.clustersBuffer
            }
          },
          {
            binding: 1,
            resource: {
              buffer: this.nodeBuffer
            }
          }
        ]
      });
      
      console.log("WebGPU compute resources created successfully");
      return true;
    } catch (error) {
      console.error("Failed to create compute resources:", error);
      return false;
    }
  }
  
  /**
   * Updates node positions and computes clusters
   * @param nodes Array of node data with positions
   */
  updateNodePositions(nodes: any[]): boolean {
    if (!this.isInitialized || !this.device || !this.computePipeline || !this.bindGroup) {
      console.warn("WebGPU clustering engine not fully initialized");
      return false;
    }
    
    try {
      // Update node buffer with latest positions
      const nodeData = new Float32Array(nodes.length * 8); // 8 floats per node
      
      nodes.forEach((node, i) => {
        // Convert node data to format expected by shader
        const baseIndex = i * 8;
        nodeData[baseIndex] = node.x || 0;     // position.x
        nodeData[baseIndex + 1] = node.y || 0; // position.y
        nodeData[baseIndex + 2] = node.z || 0; // position.z
        nodeData[baseIndex + 3] = node.val || 1; // size
        nodeData[baseIndex + 4] = 0;           // clusterIndex (will be set by compute shader)
        nodeData[baseIndex + 5] = i;           // nodeIndex
        nodeData[baseIndex + 6] = 0;           // padding
        nodeData[baseIndex + 7] = 0;           // padding
      });
      
      // Write node data to GPU
      this.device.queue.writeBuffer(this.nodeBuffer!, 0, nodeData);
      
      // Set up command encoder
      const commandEncoder = this.device.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();
      
      computePass.setPipeline(this.computePipeline);
      computePass.setBindGroup(0, this.bindGroup);
      
      // Dispatch workgroups - optimized for NVIDIA GPUs
      // NVIDIA GPUs work better with fewer, larger workgroups
      const workgroupSize = this.isNvidiaGPU ? 128 : 64;
      const workgroupCount = Math.ceil(nodes.length / workgroupSize);
      computePass.dispatchWorkgroups(workgroupCount, 1, 1);
      computePass.end();
      
      // Submit commands
      this.device.queue.submit([commandEncoder.finish()]);
      
      return true;
    } catch (error) {
      console.error("Failed to update node positions:", error);
      return false;
    }
  }
  
  /**
   * Reads back the clustered node data
   * @returns Clustered node data or null if failed
   */
  async readClusteredData(): Promise<any[] | null> {
    if (!this.isInitialized || !this.device || !this.nodeBuffer) {
      console.warn("WebGPU clustering engine not fully initialized");
      return null;
    }
    
    try {
      // Create a buffer for reading back the results
      const readBuffer = this.device.createBuffer({
        size: this.nodeBuffer.size,
        usage: GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.MAP_READ
      });
      
      // Copy results to the readable buffer
      const commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(
        this.nodeBuffer, 0,
        readBuffer, 0,
        this.nodeBuffer.size
      );
      
      // Submit copy commands
      this.device.queue.submit([commandEncoder.finish()]);
      
      // Map the buffer for reading
      await readBuffer.mapAsync(GPU_MAP_MODE.READ);
      const data = new Float32Array(readBuffer.getMappedRange());
      
      // Process the results
      const nodeCount = data.length / 8;
      const results: any[] = [];
      
      for (let i = 0; i < nodeCount; i++) {
        const baseIndex = i * 8;
        results.push({
          index: i,
          position: {
            x: data[baseIndex],
            y: data[baseIndex + 1],
            z: data[baseIndex + 2]
          },
          size: data[baseIndex + 3],
          clusterIndex: data[baseIndex + 4],
          nodeIndex: data[baseIndex + 5]
        });
      }
      
      // Clean up
      readBuffer.unmap();
      
      return results;
    } catch (error) {
      console.error("Failed to read clustered data:", error);
      return null;
    }
  }
  
  /**
   * Creates a GPU-accelerated force calculation pipeline for graph layout
   * Optimized for large graphs to offload physics calculations to the GPU
   * @param nodeCount Number of nodes in the graph
   * @param linkCount Number of links in the graph
   */
  async createClusteredForce(nodeCount: number, linkCount: number): Promise<boolean> {
    if (!this.isInitialized || !this.device) {
      console.warn("WebGPU clustering engine not initialized");
      return false;
    }
    
    try {
      // Create buffer for forces
      const forceBufferSize = nodeCount * 16; // 4 floats (x,y,z forces + padding) per node
      this.forceBuffer = this.device.createBuffer({
        size: forceBufferSize,
        usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.COPY_SRC
      });
      
      // Create link buffer if we have links
      let linkBuffer = null;
      if (linkCount > 0) {
        const linkBufferSize = linkCount * 16; // 4 integers (source, target, strength, padding) per link
        linkBuffer = this.device.createBuffer({
          size: linkBufferSize,
          usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST
        });
      }
      
      // Optimize workgroup size for the current GPU
      const workgroupSize = this.isNvidiaGPU ? 256 : 64; // NVIDIA GPUs benefit from larger workgroups
      
      // Create compute shader module for force calculation
      const forceShaderModule = this.device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
          @group(0) @binding(1) var<storage, read_write> forces: array<Force>;
          @group(0) @binding(2) var<storage, read> links: array<Link>;
          
          struct Node {
            position: vec3f,
            size: f32,
            clusterIndex: u32,
            nodeIndex: u32,
            padding1: u32,
            padding2: u32,
          };
          
          struct Force {
            force: vec3f,
            padding: f32,
          };
          
          struct Link {
            source: u32,
            target: u32,
            strength: f32,
            padding: u32,
          };
          
          struct SimParams {
            repulsionStrength: f32,
            attractionStrength: f32,
            maxDistance: f32,
            deltaTime: f32,
            numNodes: u32,
            numLinks: u32,
          };
          
          @group(0) @binding(3) var<uniform> params: SimParams;
          
          // NVIDIA-optimized force calculation
          @compute @workgroup_size(${workgroupSize}, 1, 1)
          fn calculateForces(@builtin(global_invocation_id) global_id: vec3u) {
            let nodeIndex = global_id.x;
            if (nodeIndex >= params.numNodes) {
              return;
            }
            
            let node = nodes[nodeIndex];
            var totalForce = vec3f(0.0, 0.0, 0.0);
            
            // Calculate repulsive forces (node-node)
            for (var i = 0u; i < params.numNodes; i++) {
              if (i == nodeIndex) {
                continue; // Skip self
              }
              
              let otherNode = nodes[i];
              let dx = node.position.x - otherNode.position.x;
              let dy = node.position.y - otherNode.position.y;
              let dz = node.position.z - otherNode.position.z;
              
              let distSq = dx*dx + dy*dy + dz*dz;
              if (distSq < 0.01) { // Avoid division by zero
                // Add small random jitter if nodes are too close
                totalForce += vec3f(
                  (fract(sin(f32(nodeIndex) * 78.233)) - 0.5) * 0.1,
                  (fract(sin(f32(nodeIndex) * 43.191)) - 0.5) * 0.1,
                  (fract(sin(f32(nodeIndex) * 28.976)) - 0.5) * 0.1
                );
                continue;
              }
              
              // Inverse square law for repulsion with distance limiting
              let dist = sqrt(distSq);
              if (dist > params.maxDistance) {
                continue; // Skip if too far away
              }
              
              let repulsionFactor = params.repulsionStrength / max(distSq, 0.1);
              let forceX = dx * repulsionFactor;
              let forceY = dy * repulsionFactor;
              let forceZ = dz * repulsionFactor;
              
              totalForce += vec3f(forceX, forceY, forceZ);
            }
            
            // Calculate attractive forces (links)
            for (var i = 0u; i < params.numLinks; i++) {
              let link = links[i];
              
              // Check if this node is part of the link
              if (link.source == nodeIndex || link.target == nodeIndex) {
                let otherNodeIndex = select(link.source, link.target, link.target == nodeIndex);
                let otherNode = nodes[otherNodeIndex];
                
                let dx = otherNode.position.x - node.position.x;
                let dy = otherNode.position.y - node.position.y;
                let dz = otherNode.position.z - node.position.z;
                
                let dist = sqrt(dx*dx + dy*dy + dz*dz);
                if (dist < 0.01) continue; // Avoid division by zero
                
                // Hooke's law for attraction
                let attractionFactor = params.attractionStrength * link.strength * dist;
                let dirX = dx / dist;
                let dirY = dy / dist;
                let dirZ = dz / dist;
                
                totalForce += vec3f(
                  dirX * attractionFactor,
                  dirY * attractionFactor,
                  dirZ * attractionFactor
                );
              }
            }
            
            // Store the calculated force
            forces[nodeIndex].force = totalForce;
          }
          
          // Apply calculated forces to update positions
          @compute @workgroup_size(${workgroupSize}, 1, 1)
          fn applyForces(@builtin(global_invocation_id) global_id: vec3u) {
            let nodeIndex = global_id.x;
            if (nodeIndex >= params.numNodes) {
              return;
            }
            
            let force = forces[nodeIndex].force;
            
            // Apply force to position with damping
            nodes[nodeIndex].position += force * params.deltaTime;
          }
        `
      });
      
      // Create compute pipeline
      this.forceComputePipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: forceShaderModule,
          entryPoint: 'calculateForces'
        }
      });
      
      // Create a separate pipeline for applying forces
      const applyForcesPipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: forceShaderModule,
          entryPoint: 'applyForces'
        }
      });
      
      // Create simulation parameters buffer
      const paramsBuffer = this.device.createBuffer({
        size: 32, // 6 params, 32 bytes total
        usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST
      });
      
      // Set default simulation parameters
      const defaultParams = new Float32Array([
        0.5,    // repulsionStrength
        0.01,   // attractionStrength
        200.0,  // maxDistance
        0.05,   // deltaTime
        nodeCount, // numNodes
        linkCount  // numLinks
      ]);
      
      this.device.queue.writeBuffer(paramsBuffer, 0, defaultParams);
      
      // Create bind group entries
      const bindGroupEntries = [
        {
          binding: 0,
          resource: { buffer: this.nodeBuffer! }
        },
        {
          binding: 1,
          resource: { buffer: this.forceBuffer }
        },
        {
          binding: 3,
          resource: { buffer: paramsBuffer }
        }
      ];
      
      // Add link buffer if it exists
      if (linkBuffer) {
        bindGroupEntries.push({
          binding: 2,
          resource: { buffer: linkBuffer }
        });
      }
      
      // Create bind group
      this.forceBindGroup = this.device.createBindGroup({
        layout: this.forceComputePipeline.getBindGroupLayout(0),
        entries: bindGroupEntries
      });
      
      console.log("GPU-accelerated force calculation pipeline created successfully");
      return true;
    } catch (error) {
      console.error("Failed to create force calculation pipeline:", error);
      return false;
    }
  }
  
  /**
   * Disposes of WebGPU resources
   */
  dispose(): void {
    this.clustersBuffer?.destroy();
    this.nodeBuffer?.destroy();
    this.forceBuffer?.destroy();
    this.clustersBuffer = null;
    this.nodeBuffer = null;
    this.forceBuffer = null;
    this.computePipeline = null;
    this.bindGroup = null;
    this.forceComputePipeline = null;
    this.forceBindGroup = null;
    this.device = null;
    this.isInitialized = false;
  }
} 