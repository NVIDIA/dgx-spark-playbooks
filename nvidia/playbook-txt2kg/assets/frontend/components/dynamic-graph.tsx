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

import dynamic from "next/dynamic"
import type { Triple } from "@/utils/text-processing"

// Dynamically import the GraphVisualization component with no SSR
// This allows for GPU-accelerated WebGL rendering
const DynamicGraphVisualization = dynamic(
  () => import("./graph-visualization").then((mod) => ({ default: mod.GraphVisualization })),
  {
    ssr: false,
    loading: () => (
      <div className="h-[400px] bg-gray-900 rounded-lg flex items-center justify-center">
        <div className="text-nvidia-green">Loading GPU-accelerated graph visualization...</div>
      </div>
    ),
  },
)

interface DynamicGraphProps {
  triples: Triple[]
}

export function DynamicGraph({ triples }: DynamicGraphProps) {
  return <DynamicGraphVisualization triples={triples} />
}

