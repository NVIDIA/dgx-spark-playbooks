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
import { Database } from "lucide-react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { listFilesInS3 } from "@/utils/s3-storage"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/hooks/use-toast"

export function S3Connection() {
  const [isConnected, setIsConnected] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [endpoint, setEndpoint] = useState("")
  const [bucket, setBucket] = useState("")
  const [accessKey, setAccessKey] = useState("")
  const [secretKey, setSecretKey] = useState("")
  const [fileCount, setFileCount] = useState(0)

  useEffect(() => {
    // Check if we have env variables set in localStorage
    const savedEndpoint = localStorage.getItem("S3_ENDPOINT")
    const savedBucket = localStorage.getItem("S3_BUCKET")
    const savedAccessKey = localStorage.getItem("S3_ACCESS_KEY")
    const savedSecretKey = localStorage.getItem("S3_SECRET_KEY")

    if (savedEndpoint) setEndpoint(savedEndpoint)
    if (savedBucket) setBucket(savedBucket)
    if (savedAccessKey) setAccessKey(savedAccessKey)
    if (savedSecretKey) setSecretKey(savedSecretKey)

    // If all values are set, check connection
    if (savedEndpoint && savedBucket && savedAccessKey && savedSecretKey) {
      checkConnection()
    }
  }, [])

  const saveSettings = () => {
    localStorage.setItem("S3_ENDPOINT", endpoint)
    localStorage.setItem("S3_BUCKET", bucket)
    localStorage.setItem("S3_ACCESS_KEY", accessKey)
    localStorage.setItem("S3_SECRET_KEY", secretKey)

    // Set these in window for runtime access
    window.process = window.process || {}
    window.process.env = window.process.env || {}
    window.process.env.S3_ENDPOINT = endpoint
    window.process.env.S3_BUCKET = bucket
    window.process.env.S3_ACCESS_KEY = accessKey
    window.process.env.S3_SECRET_KEY = secretKey
  }

  const checkConnection = async () => {
    setIsLoading(true)
    
    try {
      saveSettings()
      
      // Try to list files to verify connection
      const files = await listFilesInS3()
      setFileCount(files.length)
      setIsConnected(true)
      
      // Save connection status to localStorage
      localStorage.setItem("S3_CONNECTED", "true")
      
      // Dispatch event to notify other components
      window.dispatchEvent(new CustomEvent('s3ConnectionChanged', { 
        detail: { isConnected: true } 
      }))
      
      toast({
        title: "Connected to S3",
        description: `Successfully connected to ${bucket} bucket. Found ${files.length} files.`,
        variant: "default",
      })
    } catch (error) {
      console.error("Failed to connect to S3:", error)
      setIsConnected(false)
      
      // Save connection status to localStorage
      localStorage.setItem("S3_CONNECTED", "false")
      
      // Dispatch event to notify other components
      window.dispatchEvent(new CustomEvent('s3ConnectionChanged', { 
        detail: { isConnected: false } 
      }))
      
      toast({
        title: "Connection Failed",
        description: error instanceof Error ? error.message : "Could not connect to S3 storage",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleConnect = (e: React.FormEvent) => {
    e.preventDefault()
    checkConnection()
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Database className="h-5 w-5 text-primary" />
          S3 Storage Connection
        </CardTitle>
        <CardDescription>
          Connect to an S3-compatible storage service
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleConnect} className="space-y-4">
          <div className="grid gap-2">
            <Label htmlFor="endpoint">Endpoint URL</Label>
            <Input
              id="endpoint"
              placeholder="http://localhost:9000"
              value={endpoint}
              onChange={(e) => setEndpoint(e.target.value)}
              required
            />
          </div>
          
          <div className="grid gap-2">
            <Label htmlFor="bucket">Bucket Name</Label>
            <Input
              id="bucket"
              placeholder="txt2kg"
              value={bucket}
              onChange={(e) => setBucket(e.target.value)}
              required
            />
          </div>
          
          <div className="grid gap-2">
            <Label htmlFor="access-key">Access Key</Label>
            <Input
              id="access-key"
              placeholder="Access Key ID"
              value={accessKey}
              onChange={(e) => setAccessKey(e.target.value)}
              required
            />
          </div>
          
          <div className="grid gap-2">
            <Label htmlFor="secret-key">Secret Key</Label>
            <Input
              id="secret-key"
              type="password"
              placeholder="Secret Access Key"
              value={secretKey}
              onChange={(e) => setSecretKey(e.target.value)}
              required
            />
          </div>
          
          <Button type="submit" disabled={isLoading} className="w-full">
            {isLoading ? (
              <>
                <Skeleton className="h-4 w-4 mr-2 rounded-full" />
                Connecting...
              </>
            ) : isConnected ? (
              "Reconnect"
            ) : (
              "Connect"
            )}
          </Button>
        </form>
      </CardContent>
      {isConnected && (
        <CardFooter className="border-t px-6 py-4 bg-muted/50">
          <div className="flex items-center text-sm">
            <div className="flex items-center mr-2">
              <div className="h-2.5 w-2.5 rounded-full bg-green-500 mr-1.5"></div>
              <span className="font-medium">Connected</span>
            </div>
            <span className="text-muted-foreground">
              {fileCount} files in bucket
            </span>
          </div>
        </CardFooter>
      )}
    </Card>
  )
} 