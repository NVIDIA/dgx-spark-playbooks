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
import { Network, Database, Zap, AlertCircle, RefreshCw } from "lucide-react"

interface Neo4jConnectionProps {
  className?: string
}

export function Neo4jConnection({ className }: Neo4jConnectionProps) {
  const [connectionStatus, setConnectionStatus] = useState<"connected" | "disconnected" | "checking">("disconnected")
  const [error, setError] = useState<string | null>(null)
  const [nodeCount, setNodeCount] = useState<number | null>(null)
  const [relationshipCount, setRelationshipCount] = useState<number | null>(null)
  const [connectionUrl, setConnectionUrl] = useState<string>("")

  // Check Neo4j connection status
  const checkConnection = async () => {
    setConnectionStatus("checking")
    setError(null)
    
    try {
      // Get credentials from localStorage
      const dbUrl = localStorage.getItem("NEO4J_URL")
      const dbUsername = localStorage.getItem("NEO4J_USERNAME")
      const dbPassword = localStorage.getItem("NEO4J_PASSWORD")
      
      // Add query parameters if credentials exist
      const queryParams = new URLSearchParams()
      if (dbUrl) queryParams.append("url", dbUrl)
      if (dbUsername) queryParams.append("username", dbUsername)
      if (dbPassword) queryParams.append("password", dbPassword)
      
      const queryString = queryParams.toString()
      const endpoint = queryString ? `/api/neo4j?${queryString}` : '/api/neo4j'
      
      const response = await fetch(endpoint)
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to connect to Neo4j')
      }
      
      const data = await response.json()
      setNodeCount(data.nodes?.length || 0)
      setRelationshipCount(data.links?.length || 0)
      // Use the connection URL from the API response
      if (data.connectionUrl) {
        setConnectionUrl(data.connectionUrl)
      } else if (dbUrl) {
        setConnectionUrl(dbUrl)
      }
      setConnectionStatus("connected")
    } catch (err) {
      console.error('Neo4j connection error:', err)
      setConnectionStatus("disconnected")
      setError(err instanceof Error ? err.message : 'Unknown error connecting to Neo4j')
    }
  }

  // Disconnect from Neo4j
  const disconnect = async () => {
    try {
      const response = await fetch('/api/neo4j/disconnect', {
        method: 'POST',
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to disconnect from Neo4j')
      }
      
      setConnectionStatus("disconnected")
      setNodeCount(null)
      setRelationshipCount(null)
    } catch (err) {
      console.error('Neo4j disconnect error:', err)
      setError(err instanceof Error ? err.message : 'Unknown error disconnecting from Neo4j')
    }
  }

  // Check connection on component mount
  // useEffect(() => {
  //   checkConnection()
  // }, [])

  return (
    <div className={`glass-card rounded-xl overflow-hidden ${className}`}>
      <div className="p-5 border-b border-border/50">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Network className="h-5 w-5 text-primary" />
          Graph DB
        </h2>
      </div>
      
      <div className="p-5 space-y-4">
        {connectionStatus === "checking" && (
          <div className="flex items-center gap-2 text-sm">
            <RefreshCw className="h-4 w-4 animate-spin" />
            <span>Checking connection...</span>
          </div>
        )}
        
        {connectionStatus === "connected" && (
          <>
            <div className="flex items-center gap-2 text-sm">
              <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></span>
              <span className="text-foreground font-mono text-xs bg-secondary px-2 py-1 rounded">
                {connectionUrl}
              </span>
            </div>
            
            {(nodeCount !== null || relationshipCount !== null) && (
              <div className="text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4" />
                  <span>{nodeCount} nodes, {relationshipCount} relationships</span>
                </div>
              </div>
            )}
          </>
        )}
        
        {connectionStatus === "disconnected" && (
          <>
            <div className="flex items-center gap-2 text-sm">
              <span className="h-2 w-2 rounded-full bg-destructive"></span>
              <span className="text-foreground font-mono text-xs bg-secondary px-2 py-1 rounded">
                Not connected
              </span>
            </div>
          </>
        )}
      </div>
      
      <div className="flex p-4 gap-2 border-t border-border/50 bg-card">
        <button 
          className="btn-outline flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-md text-sm"
          onClick={checkConnection}
        >
          <RefreshCw className="h-4 w-4" />
          <span>Refresh</span>
        </button>
        
        {connectionStatus === "connected" ? (
          <button 
            className="btn-outline flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-md text-sm text-destructive border-destructive hover:bg-destructive/10"
            onClick={disconnect}
          >
            <Zap className="h-4 w-4" />
            <span>Disconnect</span>
          </button>
        ) : (
          <button 
            className="btn-outline flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-md text-sm text-primary border-primary hover:bg-primary/10"
            onClick={checkConnection}
          >
            <Zap className="h-4 w-4" />
            <span>Connect</span>
          </button>
        )}
      </div>
    </div>
  )
}