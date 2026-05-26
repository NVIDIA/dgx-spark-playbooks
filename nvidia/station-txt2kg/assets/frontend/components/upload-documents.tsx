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

import type React from "react"

import { useState } from "react"
import { Upload, AlertCircle, FileText } from "lucide-react"
import { useDocuments } from "@/contexts/document-context"

export function UploadDocuments() {
  const { addDocuments } = useDocuments()
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const validateFiles = (files: File[]): File[] => {
    setError(null)
    const validFiles = Array.from(files).filter((file) => {
      const isValidType = file.name.endsWith(".md") || file.name.endsWith(".csv") || file.name.endsWith(".txt") || file.name.endsWith(".json")
      if (!isValidType) {
        setError("Only markdown (.md), CSV (.csv), text (.txt), and JSON (.json) files are supported.")
        return false
      }
      return true
    })
    return validFiles
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files)
    const validFiles = validateFiles(files)

    if (validFiles.length > 0) {
      addDocuments(validFiles)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files)
      const validFiles = validateFiles(files)

      if (validFiles.length > 0) {
        addDocuments(validFiles)
      }

      // Reset the input to allow uploading the same file again
      e.target.value = ""
    }
  }

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="h-4 w-4 text-destructive mt-0.5 flex-shrink-0" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      <div
        className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 hover:border-nvidia-green/50 hover:bg-nvidia-green/5
                   ${isDragging ? "border-nvidia-green bg-nvidia-green/10 scale-[1.02]" : "border-border/40 hover:border-nvidia-green/40"}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => document.getElementById("file-upload")?.click()}
      >
        <input id="file-upload" type="file" multiple className="hidden" accept=".md,.csv,.txt,.json" onChange={handleFileSelect} />
        <div className="flex flex-col items-center">
          <div className="w-16 h-16 rounded-2xl bg-nvidia-green/10 flex items-center justify-center mb-4 border border-nvidia-green/20">
            <Upload className="h-8 w-8 text-nvidia-green" />
          </div>
          <h3 className="text-lg font-semibold text-foreground mb-2">Drag & Drop Files</h3>
          <p className="text-sm text-muted-foreground mb-4">
            or <button className="font-medium text-nvidia-green hover:text-nvidia-green/80 underline underline-offset-2">browse files</button>
          </p>
          <div className="inline-flex items-center gap-2 text-xs text-muted-foreground bg-muted/40 px-3 py-1.5 rounded-full border border-border/30">
            <FileText className="h-3 w-3" />
            <span>.md, .csv, .txt, .json supported</span>
          </div>
        </div>
      </div>
    </div>
  )
}

