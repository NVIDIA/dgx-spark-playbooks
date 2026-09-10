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

import { FC, useState, useEffect } from 'react'
import { useDocuments } from '@/contexts/document-context'
import { Zap, Loader2 } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface DocumentActionsProps {
  documentId: string
}

export const DocumentActions: FC<DocumentActionsProps> = ({ documentId }) => {
  const { generateEmbeddings, isGeneratingEmbeddings, documents } = useDocuments()
  const [processingEmbeddingsId, setProcessingEmbeddingsId] = useState<string | null>(null)
  const [embeddingsProvider, setEmbeddingsProvider] = useState<string>("local")
  
  // Get the document status to determine if we can generate embeddings
  const document = documents.find(doc => doc.id === documentId)
  const isProcessed = document?.status === 'Processed'
  
  // Load embeddings provider setting from localStorage when component mounts
  useEffect(() => {
    const storedProvider = localStorage.getItem("embeddings_provider") || "local"
    setEmbeddingsProvider(storedProvider)
    
    // Create a custom event listener for embeddings settings changes
    const handleEmbeddingsSettingsChanged = () => {
      const updatedProvider = localStorage.getItem("embeddings_provider") || "local"
      setEmbeddingsProvider(updatedProvider)
      console.log("Document actions detected embeddings settings change:", updatedProvider)
    }
    
    // Listen for a custom event that will be dispatched when settings are saved
    window.addEventListener('embeddings-settings-changed', handleEmbeddingsSettingsChanged)
    
    return () => {
      window.removeEventListener('embeddings-settings-changed', handleEmbeddingsSettingsChanged)
    }
  }, [])

  // Handle embeddings generation
  const handleGenerateEmbeddings = async () => {
    setProcessingEmbeddingsId(documentId)
    try {
      await generateEmbeddings(documentId)
    } finally {
      setProcessingEmbeddingsId(null)
    }
  }
  
  const getProviderLabel = () => {
    return embeddingsProvider === "nvidia" ? "NVIDIA API" : "Local Transformer"
  }
  
  return (
    <div className="flex gap-2">
      {isProcessed && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={handleGenerateEmbeddings}
                disabled={isGeneratingEmbeddings || processingEmbeddingsId === documentId}
                className="btn-outline flex items-center justify-center gap-2 py-2 px-3 rounded-md text-sm text-primary border-primary hover:bg-primary/10 disabled:opacity-50 disabled:cursor-not-allowed"
                title="Generate vector embeddings from this document"
              >
                {processingEmbeddingsId === documentId ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Zap className="h-4 w-4" />
                )}
                <span>
                  {isGeneratingEmbeddings || processingEmbeddingsId === documentId ? 
                    'Generating...' : 
                    'Generate Embeddings'}
                </span>
              </button>
            </TooltipTrigger>
            <TooltipContent side="top">
              Using {getProviderLabel()} for embeddings generation
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  )
} 