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

import { Network, Zap } from "lucide-react"
import { useDocuments } from "@/contexts/document-context"
import { Loader2 } from "lucide-react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

export function GraphActions() {
  const { documents, processDocuments, isProcessing, openGraphVisualization } = useDocuments()

  const hasNewDocuments = documents.some((doc) => doc.status === "New")
  const hasProcessedDocuments = documents.some(
    (doc) => doc.status === "Processed" && doc.triples && doc.triples.length > 0,
  )

  const handleProcessDocuments = async () => {
    try {
      // Get IDs of documents with "New" status
      const newDocumentIds = documents
        .filter(doc => doc.status === "New")
        .map(doc => doc.id);
        
      if (newDocumentIds.length === 0) {
        console.log("No new documents to process");
        return;
      }
      
      await processDocuments(newDocumentIds, {
        useLangChain: false,
        useGraphTransformer: false,
        promptConfigs: undefined
      });
    } catch (error) {
      console.error('Error processing documents:', error);
    }
  }

  // Helper to get tooltip content for disabled Process button
  const getProcessTooltip = () => {
    if (isProcessing) return "Processing in progress..."
    if (!hasNewDocuments && documents.length === 0) return "Upload documents first to extract knowledge triples"
    if (!hasNewDocuments) return "All documents have been processed"
    return "Extract knowledge triples from uploaded documents"
  }

  // Helper to get tooltip content for disabled View Graph button
  const getViewGraphTooltip = () => {
    if (isProcessing) return "Wait for processing to complete"
    if (!hasProcessedDocuments && documents.length === 0) return "Upload and process documents first"
    if (!hasProcessedDocuments) return "Process documents first to generate knowledge triples"
    return "Visualize the knowledge graph from extracted triples"
  }

  return (
    <TooltipProvider>
      <div className="flex gap-3 items-center">
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              className={`btn-primary ${!hasNewDocuments || isProcessing ? "opacity-60 cursor-not-allowed" : ""}`}
              disabled={!hasNewDocuments || isProcessing}
              onClick={handleProcessDocuments}
            >
              {isProcessing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Zap className="h-4 w-4" />
                  Process Documents
                </>
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent>
            <p>{getProcessTooltip()}</p>
          </TooltipContent>
        </Tooltip>
        
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              className={`btn-primary ${!hasProcessedDocuments || isProcessing ? "opacity-60 cursor-not-allowed" : ""}`}
              disabled={!hasProcessedDocuments || isProcessing}
              onClick={() => openGraphVisualization()}
            >
              <Network className="h-4 w-4" />
              View Knowledge Graph
            </button>
          </TooltipTrigger>
          <TooltipContent>
            <p>{getViewGraphTooltip()}</p>
          </TooltipContent>
        </Tooltip>
      </div>
    </TooltipProvider>
  )
}

