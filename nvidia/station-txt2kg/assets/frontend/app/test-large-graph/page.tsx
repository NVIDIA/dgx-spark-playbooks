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
"use client"

import React, { useState, useEffect } from "react"
import { ForceGraphWrapper } from "@/components/force-graph-wrapper"

// Define interfaces matching those in ForceGraphWrapper (or import if shared)
interface NodeObject {
  id: string
  name: string
  val?: number
  group?: string
}

interface LinkObject {
  source: string
  target: string
  name: string
}

interface GraphData {
  nodes: NodeObject[]
  links: LinkObject[]
}

// Function to generate mock data
const generateMockData = (numNodes: number, numLinks: number): GraphData => {
  console.log(`Generating mock data: ${numNodes} nodes, ${numLinks} links...`);
  const nodes: NodeObject[] = [];
  const nodeIds = new Set<string>();

  // Generate Nodes
  for (let i = 0; i < numNodes; i++) {
    const id = `node_${i}`;
    nodes.push({ id, name: `Node ${i}`, group: `group_${i % 10}` });
    nodeIds.add(id);
  }
  console.log(`Generated ${nodes.length} nodes.`);

  // Generate Links
  const links: LinkObject[] = [];
  const linkIds = new Set<string>(); // To avoid duplicate links

  while (links.length < numLinks) {
    if (nodes.length < 2) break; // Need at least 2 nodes for a link

    const sourceIndex = Math.floor(Math.random() * nodes.length);
    let targetIndex = Math.floor(Math.random() * nodes.length);

    // Ensure source and target are different
    while (sourceIndex === targetIndex) {
      targetIndex = Math.floor(Math.random() * nodes.length);
    }

    const sourceId = nodes[sourceIndex].id;
    const targetId = nodes[targetIndex].id;

    // Create a unique ID for the link pair (order doesn't matter for uniqueness check)
    const linkId = [sourceId, targetId].sort().join('-');

    if (!linkIds.has(linkId)) {
      links.push({
        source: sourceId,
        target: targetId,
        name: `link_${sourceId}_${targetId}`,
      });
      linkIds.add(linkId);
    }

    // Log progress occasionally to prevent freezing perception
    if (links.length % 100000 === 0 && links.length > 0) {
      console.log(`Generated ${links.length}/${numLinks} links...`);
    }
  }
  console.log(`Generated ${links.length} links.`);

  return { nodes, links };
};

export default function TestLargeGraphPage() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    console.log("Starting data generation effect...");
    setIsLoading(true);
    setError(null);
    try {
      // Use setTimeout to allow the loading state to render before blocking the main thread
      setTimeout(() => {
        const startTime = performance.now();
        const data = generateMockData(10000, 50000); // 10k nodes, 50k links
        const endTime = performance.now();
        console.log(`Data generation took ${(endTime - startTime) / 1000} seconds.`);
        setGraphData(data);
        setIsLoading(false);
      }, 50); // Small delay
    } catch (err: any) {
      console.error("Error generating mock data:", err);
      setError(`Failed to generate mock data: ${err.message}`);
      setIsLoading(false);
    }
  }, []);

  return (
    <div style={{ height: "100vh", width: "100vw", position: "relative" }}>
      {isLoading && (
        <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", color: "white", zIndex: 10 }}>
          Generating large graph data (10k nodes, 50k links)... Please wait.
        </div>
      )}
      {error && (
        <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", color: "red", background: "rgba(0,0,0,0.8)", padding: "20px", borderRadius: "8px", zIndex: 10 }}>
          Error: {error}
        </div>
      )}
      {!isLoading && !error && graphData && (
        <ForceGraphWrapper
          jsonData={graphData}
          fullscreen={true} // Use fullscreen prop if available, otherwise style div
          onError={(err) => {
            console.error("Graph visualization error:", err);
            setError(`Graph rendering error: ${err.message}`);
          }}
        />
      )}
    </div>
  );
} 