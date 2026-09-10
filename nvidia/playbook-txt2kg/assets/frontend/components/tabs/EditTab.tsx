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
import { TripleViewer } from "@/components/triple-viewer"
import { Network } from "lucide-react"

export function EditTab() {
  return (
    <div className="nvidia-build-card p-0 overflow-hidden">
      <div className="p-6 border-b border-border/10">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-8 h-8 rounded-lg bg-nvidia-green/15 flex items-center justify-center">
            <Network className="h-4 w-4 text-nvidia-green" />
          </div>
          <h2 className="text-lg font-semibold text-foreground">Knowledge Triples</h2>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Review and edit extracted knowledge triples from your documents
        </p>
      </div>
      
      <div className="min-h-[500px]">
        <TripleViewer />
      </div>
    </div>
  )
} 