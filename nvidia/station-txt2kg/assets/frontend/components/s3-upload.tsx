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

import { useState } from "react"
import { Upload, AlertCircle, FileText } from "lucide-react"
import { uploadFileToS3 } from "@/utils/s3-storage"
import { useDocuments } from "@/contexts/document-context"
import { toast } from "@/hooks/use-toast"

export function S3Upload() {
  const { addDocuments } = useDocuments()
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
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

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files)
    const validFiles = validateFiles(files)

    if (validFiles.length > 0) {
      await uploadFiles(validFiles)
    }
  }

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files)
      const validFiles = validateFiles(files)

      if (validFiles.length > 0) {
        await uploadFiles(validFiles)
      }

      // Reset the input to allow uploading the same file again
      e.target.value = ""
    }
  }

  const uploadFiles = async (files: File[]) => {
    setIsUploading(true)
    setError(null)
    
    try {
      // First add to document context so user sees them immediately
      addDocuments(files)
      
      // Then upload to S3
      for (const file of files) {
        try {
          // Upload to S3
          const key = await uploadFileToS3(file)
          
          toast({
            title: "File Uploaded",
            description: `${file.name} uploaded to S3 storage successfully`,
            duration: 3000,
          })
        } catch (err) {
          console.error(`Error uploading ${file.name}:`, err)
          setError(`Failed to upload ${file.name} to S3: ${err instanceof Error ? err.message : String(err)}`)
          
          toast({
            title: "Upload Failed",
            description: `Failed to upload ${file.name} to S3`,
            variant: "destructive",
            duration: 5000,
          })
        }
      }
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="space-y-4">
      {error && (
        <div className="bg-destructive/10 border border-destructive/30 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-destructive mt-0.5 flex-shrink-0" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      <div
        className={`glass-card rounded-xl p-5 text-center cursor-pointer transition-all hover-lift
                   ${isDragging ? "border-primary animate-glow" : "border-border/50"}
                   ${isUploading ? "opacity-70 cursor-not-allowed" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isUploading && document.getElementById("s3-file-upload")?.click()}
      >
        <input 
          id="s3-file-upload" 
          type="file" 
          multiple 
          className="hidden" 
          accept=".md,.csv,.txt" 
          onChange={handleFileSelect}
          disabled={isUploading}
        />
          <div className="flex flex-col items-center">
          <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center mb-3">
            <Upload className={`h-7 w-7 text-primary ${isUploading ? "animate-pulse" : ""}`} />
          </div>
          <h3 className="text-sm font-semibold mb-1.5">
            {isUploading ? "Uploading to S3..." : "Upload to S3"}
          </h3>
          <p className="text-xs text-muted-foreground mb-2">
            {isUploading ? (
              "Please wait while files are being uploaded"
            ) : (
              <>
                or <span className="font-medium text-primary">browse files</span>
              </>
            )}
          </p>
          <div className="flex items-center gap-2 text-[11px] text-muted-foreground bg-secondary/50 px-2.5 py-1 rounded-full">
            <FileText className="h-3 w-3" />
            <span>Markdown (.md), CSV (.csv), and text (.txt) files</span>
          </div>
        </div>
      </div>
    </div>
  )
} 