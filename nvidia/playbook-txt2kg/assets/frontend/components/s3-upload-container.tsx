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

import { useState, useEffect } from "react"
import { S3Upload } from "@/components/s3-upload"

export function S3UploadContainer() {
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    // Check if we have env variables set in localStorage
    const checkS3Connection = () => {
      const savedEndpoint = localStorage.getItem("S3_ENDPOINT")
      const savedBucket = localStorage.getItem("S3_BUCKET")
      const savedAccessKey = localStorage.getItem("S3_ACCESS_KEY")
      const savedSecretKey = localStorage.getItem("S3_SECRET_KEY")
      const isConnectedStatus = localStorage.getItem("S3_CONNECTED")

      // Consider connected if all values are set and there's a successful connection flag
      setIsConnected(
        !!savedEndpoint && 
        !!savedBucket && 
        !!savedAccessKey && 
        !!savedSecretKey && 
        isConnectedStatus === "true"
      )
    }

    // Check connection status on mount
    checkS3Connection()

    // Also listen for S3 connection changes
    const handleS3ConnectionChange = () => {
      checkS3Connection()
    }

    window.addEventListener('s3ConnectionChanged', handleS3ConnectionChange)
    
    return () => {
      window.removeEventListener('s3ConnectionChanged', handleS3ConnectionChange)
    }
  }, [])

  if (!isConnected) {
    return null
  }

  return (
    <div className="mt-8">
      <div className="border-t border-border/20 pt-8">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-6 h-6 rounded-md bg-nvidia-green/15 flex items-center justify-center">
            <svg className="h-3 w-3 text-nvidia-green" viewBox="0 0 24 24" fill="currentColor">
              <path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/>
            </svg>
          </div>
          <h3 className="text-base font-semibold text-foreground">S3 Storage</h3>
        </div>
        <p className="text-sm text-muted-foreground mb-6 leading-relaxed">Upload documents directly from S3 bucket</p>
        <S3Upload />
      </div>
    </div>
  )
} 