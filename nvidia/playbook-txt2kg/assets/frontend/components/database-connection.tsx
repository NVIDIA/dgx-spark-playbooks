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
import { Network, Database, Zap, AlertCircle, RefreshCw, ChevronDown, ChevronUp, InfoIcon, Trash2, LogOut } from "lucide-react"
import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { Button } from "@/components/ui/button"
import { VectorDBStats } from '@/types/graph'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger, DialogClose } from "@/components/ui/dialog"

interface DatabaseConnectionProps {
  className?: string
}

export function DatabaseConnection({ className }: DatabaseConnectionProps) {
  // Neo4j/Graph DB state
  const [graphConnectionStatus, setGraphConnectionStatus] = useState<"connected" | "disconnected" | "checking">("disconnected")
  const [graphError, setGraphError] = useState<string | null>(null)
  const [nodeCount, setNodeCount] = useState<number | null>(null)
  const [relationshipCount, setRelationshipCount] = useState<number | null>(null)
  const [connectionUrl, setConnectionUrl] = useState<string>("")
  const [dbType, setDbType] = useState<string>("")
  const [isClearingDB, setIsClearingDB] = useState<boolean>(false)
  const [showClearDialog, setShowClearDialog] = useState<boolean>(false)

  // Vector DB state
  const [vectorConnectionStatus, setVectorConnectionStatus] = useState<"connected" | "disconnected" | "checking">("disconnected")
  const [vectorError, setVectorError] = useState<string | null>(null)
  const [vectorStats, setVectorStats] = useState<VectorDBStats>({ nodes: 0, relationships: 0, source: 'none' })
  const [isClearingVectorDB, setIsClearingVectorDB] = useState<boolean>(false)
  const [showClearVectorDialog, setShowClearVectorDialog] = useState<boolean>(false)

  // UI state
  const [expandedSection, setExpandedSection] = useState<"graph" | "vector" | null>("graph")
  
  // Check graph database connection status (Neo4j or ArangoDB)
  const checkGraphConnection = async () => {
    setGraphConnectionStatus("checking")
    setGraphError(null)
    
    try {
      // Get database type from localStorage, fall back to fetching from server
      let graphDbType = localStorage.getItem("graph_db_type")
      if (!graphDbType) {
        // Fetch server's default (from GRAPH_DB_TYPE env var)
        try {
          const settingsRes = await fetch('/api/settings')
          const settingsData = await settingsRes.json()
          graphDbType = settingsData.settings?.graph_db_type || 'neo4j'
        } catch {
          graphDbType = 'neo4j'
        }
      }
      setDbType(graphDbType === "arangodb" ? "ArangoDB" : "Neo4j")
      
      if (graphDbType === "neo4j") {
        // Neo4j connection logic - use the unified graph-db endpoint
        const dbUrl = localStorage.getItem("NEO4J_URL")
        const dbUsername = localStorage.getItem("NEO4J_USERNAME")
        const dbPassword = localStorage.getItem("NEO4J_PASSWORD")
        
        // Add query parameters with type=neo4j
        const queryParams = new URLSearchParams()
        queryParams.append("type", "neo4j")
        if (dbUrl) queryParams.append("url", dbUrl)
        if (dbUsername) queryParams.append("username", dbUsername)
        if (dbPassword) queryParams.append("password", dbPassword)
        
        const endpoint = `/api/graph-db?${queryParams.toString()}`
        
        const response = await fetch(endpoint)
        
        if (!response.ok) {
          const errorData = await response.json()
          const errorMessage = errorData.error || 'Failed to connect to Neo4j'
          console.error('Neo4j connection failed:', errorMessage)
          setGraphConnectionStatus("disconnected")
          setGraphError(errorMessage)
          return
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
      } else {
        // ArangoDB connection logic - use the unified graph-db endpoint with type=arangodb
        const arangoUrl = localStorage.getItem("arango_url") || "http://localhost:8529"
        const arangoDb = localStorage.getItem("arango_db") || "txt2kg"
        const arangoUser = localStorage.getItem("arango_user") || ""
        const arangoPassword = localStorage.getItem("arango_password") || ""
        
        // Add query parameters with type=arangodb
        const queryParams = new URLSearchParams()
        queryParams.append("type", "arangodb")
        if (arangoUrl) queryParams.append("url", arangoUrl)
        if (arangoDb) queryParams.append("dbName", arangoDb)
        if (arangoUser) queryParams.append("username", arangoUser)
        if (arangoPassword) queryParams.append("password", arangoPassword)
        
        const endpoint = `/api/graph-db?${queryParams.toString()}`
        
        const response = await fetch(endpoint)
        
        if (!response.ok) {
          const errorData = await response.json()
          const errorMessage = errorData.error || 'Failed to connect to ArangoDB'
          console.error('ArangoDB connection failed:', errorMessage)
          setGraphConnectionStatus("disconnected")
          setGraphError(errorMessage)
          return
        }
        
        const data = await response.json()
        setNodeCount(data.nodes?.length || 0)
        setRelationshipCount(data.links?.length || 0)
        
        // Set ArangoDB connection URL
        setConnectionUrl(`${arangoUrl}/_db/${arangoDb}`)
      }
      
      setGraphConnectionStatus("connected")
    } catch (err) {
      console.error('Graph database connection error:', err)
      setGraphConnectionStatus("disconnected")
      setGraphError(err instanceof Error ? err.message : 'Unknown error connecting to database')
    }
  }

  // Disconnect from graph database
  const disconnectGraph = async () => {
    try {
      // Use current dbType state which was already determined from server/localStorage
      const graphDbType = dbType === "Neo4j" ? "neo4j" : "arangodb"
      const endpoint = graphDbType === "neo4j" ? '/api/neo4j/disconnect' : '/api/graph-db/disconnect'
      
      const response = await fetch(endpoint, {
        method: 'POST',
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        const errorMessage = errorData.error || `Failed to disconnect from ${graphDbType}`
        console.error('Graph database disconnect failed:', errorMessage)
        setGraphError(errorMessage)
        return
      }
      
      setGraphConnectionStatus("disconnected")
      setNodeCount(null)
      setRelationshipCount(null)
    } catch (err) {
      console.error('Graph database disconnect error:', err)
      setGraphError(err instanceof Error ? err.message : 'Unknown error disconnecting from database')
    }
  }

  // Fetch vector DB stats
  const fetchVectorStats = async () => {
    try {
      const response = await fetch('/api/vector-db/stats');
      const data = await response.json();

      if (response.ok) {
        setVectorStats({
          nodes: typeof data.totalVectorCount === 'number' ? data.totalVectorCount : 0,
          relationships: 0, // Vector DB doesn't store relationships
          source: data.source || 'unknown',
          httpHealthy: data.httpHealthy,
          // Store additional Qdrant stats
          ...(data.status && { status: data.status }),
          ...(data.vectorSize && { vectorSize: data.vectorSize }),
          ...(data.distance && { distance: data.distance }),
          ...(data.url && { url: data.url }),
        } as any);

        // If we have a healthy HTTP connection, we're connected
        if (data.httpHealthy) {
          setVectorConnectionStatus("connected");
          setVectorError(null);
        } else {
          setVectorConnectionStatus("disconnected");
          setVectorError(data.error || 'Connection failed');
        }
      } else {
        console.error('Failed to fetch vector DB stats:', data);
        setVectorConnectionStatus("disconnected");
        setVectorError(data.error || 'Failed to connect to vector database');
      }
    } catch (error) {
      console.error('Error fetching vector DB stats:', error);
      setVectorConnectionStatus("disconnected");
      setVectorError(error instanceof Error ? error.message : 'Error connecting to vector database');
    }
  };

  // Check vector connection
  const checkVectorConnection = async () => {
    setVectorConnectionStatus("checking")
    setVectorError(null)
    
    try {
      await fetchVectorStats();
    } catch (error) {
      console.error('Error connecting to Vector DB:', error)
      setVectorConnectionStatus("disconnected")
      setVectorError(error instanceof Error ? error.message : 'Unknown error connecting to Vector DB')
    }
  }

  // Reset vector connection state
  const disconnectVector = async () => {
    setVectorConnectionStatus("disconnected")
    setVectorStats({ nodes: 0, relationships: 0, source: 'none' })
  }

  // Clear the graph database
  const clearGraphDatabase = async () => {
    if (graphConnectionStatus !== "connected") {
      return
    }
    
    setIsClearingDB(true)
    setGraphError(null)
    
    try {
      // Call API to clear the database
      const response = await fetch('/api/graph-db/clear', {
        method: 'POST',
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        const errorMessage = errorData.error || 'Failed to clear database'
        console.error('Graph database clear failed:', errorMessage)
        setGraphError(errorMessage)
        return
      }
      
      // Refresh graph connection to update stats
      await checkGraphConnection()
      
      setShowClearDialog(false)
    } catch (err) {
      console.error('Graph database clear error:', err)
      setGraphError(err instanceof Error ? err.message : 'Unknown error clearing database')
    } finally {
      setIsClearingDB(false)
    }
  }

  // Clear the vector database
  const clearVectorDatabase = async () => {
    if (vectorConnectionStatus !== "connected") {
      return
    }
    
    setIsClearingVectorDB(true)
    setVectorError(null)
    
    try {
      // Call API to clear the database
      const response = await fetch('/api/vector-db/clear', {
        method: 'POST',
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        const errorMessage = errorData.error || 'Failed to clear vector database'
        console.error('Vector database clear failed:', errorMessage)
        setVectorError(errorMessage)
        return
      }
      
      // Refresh vector connection to update stats
      await checkVectorConnection()
      
      setShowClearVectorDialog(false)
    } catch (err) {
      console.error('Vector database clear error:', err)
      setVectorError(err instanceof Error ? err.message : 'Unknown error clearing vector database')
    } finally {
      setIsClearingVectorDB(false)
    }
  }

  // Check both connections on mount
  useEffect(() => {
    checkGraphConnection()
    checkVectorConnection()
  }, [])

  const toggleSection = (section: "graph" | "vector") => {
    if (expandedSection === section) {
      setExpandedSection(null)
    } else {
      setExpandedSection(section)
    }
  }

  return (
    <div className={`rounded-xl overflow-hidden border border-border/50 bg-card/30 backdrop-blur-sm ${className}`}>
      {/* Graph DB Section */}
      <Collapsible open={expandedSection === "graph"} onOpenChange={() => toggleSection("graph")}>
        <div className="p-3 border-b border-border/50 cursor-pointer flex justify-between items-center" onClick={() => toggleSection("graph")}>
          <div className="flex items-center gap-1.5">
            <Network className="h-3.5 w-3.5 text-primary" />
            <h3 className="text-xs md:text-sm font-medium">Graph DB</h3>
          </div>
          
          <div className="flex items-center gap-2">
            {graphConnectionStatus === "checking" && (
              <RefreshCw className="h-3 w-3 animate-spin text-yellow-500" />
            )}
            {graphConnectionStatus === "connected" && (
              <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse"></span>
            )}
            {graphConnectionStatus === "disconnected" && (
              <span className="h-1.5 w-1.5 rounded-full bg-destructive"></span>
            )}
            <CollapsibleTrigger asChild>
              <button className="p-1 rounded-full hover:bg-secondary">
                {expandedSection === "graph" ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>
            </CollapsibleTrigger>
          </div>
        </div>
        
        <CollapsibleContent>
          <div className="p-3 space-y-2 bg-card/50">
            {graphConnectionStatus === "checking" && (
              <div className="flex items-center gap-2 text-xs">
                <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                <span>Checking connection...</span>
              </div>
            )}
            
            {graphConnectionStatus === "connected" && (
              <>
                <div className="flex items-center gap-2 text-xs md:text-sm">
                  <span className="text-foreground font-medium">
                    {dbType}
                  </span>
                  <span className="text-foreground font-mono text-[11px] bg-secondary/50 px-2 py-0.5 rounded truncate max-w-full">
                    {connectionUrl}
                  </span>
                </div>
                
                {(nodeCount !== null || relationshipCount !== null) && (
                  <div className="text-xs md:text-sm text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <Database className="h-3.5 w-3.5" />
                      <span>{nodeCount?.toLocaleString()} nodes, {relationshipCount?.toLocaleString()} relationships</span>
                    </div>
                  </div>
                )}
              </>
            )}
            
            {graphConnectionStatus === "disconnected" && (
              <div className="flex items-center gap-2 text-xs">
                <span className="text-foreground font-mono text-[11px] bg-secondary/50 px-2 py-0.5 rounded">
                  Not connected
                </span>
              </div>
            )}
            
            {graphError && (
              <div className="text-xs md:text-sm text-red-600 bg-red-50 dark:bg-red-950/20 dark:text-red-400 p-2 rounded">
                <p className="whitespace-normal break-words">Error: {graphError}</p>
              </div>
            )}
            
            <div className="flex gap-2">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={checkGraphConnection}
                      disabled={graphConnectionStatus === "checking"}
                      className="flex-1 text-xs h-7 px-2"
                    >
                      <RefreshCw className={`h-3 w-3 ${graphConnectionStatus === "checking" ? "animate-spin" : ""}`} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{graphConnectionStatus === "checking" ? "Checking..." : "Refresh connection"}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              
              {graphConnectionStatus === "connected" ? (
                <>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button 
                          variant="outline" 
                          size="sm" 
                          onClick={disconnectGraph}
                          className="flex-1 text-xs h-7 px-2"
                        >
                          <LogOut className="h-3 w-3" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Disconnect</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  
                  <Dialog open={showClearDialog} onOpenChange={setShowClearDialog}>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <DialogTrigger asChild>
                            <Button 
                              variant="destructive" 
                              size="sm" 
                              className="flex-1 text-xs h-7 px-2"
                              disabled={isClearingDB}
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </DialogTrigger>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Clear database</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle className="text-destructive">Clear Database</DialogTitle>
                        <DialogDescription>
                          Are you sure you want to clear all data from the {dbType} database? This action cannot be undone.
                        </DialogDescription>
                      </DialogHeader>
                      <Alert variant="destructive" className="mt-2">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Warning</AlertTitle>
                        <AlertDescription>
                          This will permanently delete all nodes and relationships from the database.
                        </AlertDescription>
                      </Alert>
                      <DialogFooter className="gap-2 mt-4">
                        <DialogClose asChild>
                          <Button variant="outline" size="sm">Cancel</Button>
                        </DialogClose>
                        <Button 
                          variant="destructive" 
                          size="sm"
                          onClick={clearGraphDatabase}
                          disabled={isClearingDB}
                        >
                          {isClearingDB ? "Clearing..." : "Clear Database"}
                        </Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                </>
              ) : (
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={() => {
                    // Open Graph DB settings
                    const event = new CustomEvent('open-settings', { 
                      detail: { tab: 'graph' } 
                    });
                    window.dispatchEvent(event);
                  }}
                  className="flex-1 text-xs h-7"
                >
                  Configure
                </Button>
              )}
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
      
      {/* Vector DB Section */}
      <Collapsible open={expandedSection === "vector"} onOpenChange={() => toggleSection("vector")}>
        <div className="p-3 cursor-pointer flex justify-between items-center" onClick={() => toggleSection("vector")}>
          <div className="flex items-center gap-1.5">
            <Database className="h-3.5 w-3.5 text-primary" />
            <h3 className="text-xs md:text-sm font-medium">Vector DB</h3>
          </div>
          
          <div className="flex items-center gap-2">
            {vectorConnectionStatus === "checking" && (
              <RefreshCw className="h-3 w-3 animate-spin text-yellow-500" />
            )}
            {vectorConnectionStatus === "connected" && (
              <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse"></span>
            )}
            {vectorConnectionStatus === "disconnected" && (
              <span className="h-1.5 w-1.5 rounded-full bg-destructive"></span>
            )}
            <CollapsibleTrigger asChild>
              <button className="p-1 rounded-full hover:bg-secondary">
                {expandedSection === "vector" ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>
            </CollapsibleTrigger>
          </div>
        </div>
        
        <CollapsibleContent>
          <div className="p-3 space-y-2 bg-card/50">
            {vectorConnectionStatus === "checking" && (
              <div className="flex items-center gap-2 text-xs">
                <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                <span>Checking connection...</span>
              </div>
            )}
            
            {vectorConnectionStatus === "connected" && (
              <>
                <div className="flex items-center gap-2 text-xs md:text-sm">
                  <span className="text-foreground font-medium">
                    Qdrant
                  </span>
                  <span className="text-foreground font-mono text-[11px] bg-secondary/50 px-2 py-0.5 rounded truncate max-w-full">
                    {(vectorStats as any).url || 'http://qdrant:6333'}
                  </span>
                </div>

                <div className="text-xs md:text-sm text-muted-foreground space-y-1">
                  <div className="flex items-center gap-2">
                    <Database className="h-3.5 w-3.5" />
                    <span>{vectorStats.nodes.toLocaleString()} vectors indexed</span>
                  </div>

                  {(vectorStats as any).status && (
                    <div className="flex items-center gap-2">
                      <Zap className="h-3.5 w-3.5" />
                      <span>Status: <span className="capitalize">{(vectorStats as any).status}</span></span>
                    </div>
                  )}

                  {(vectorStats as any).vectorSize && (
                    <div className="flex items-center gap-2">
                      <InfoIcon className="h-3.5 w-3.5" />
                      <span>{(vectorStats as any).vectorSize}d ({(vectorStats as any).distance})</span>
                    </div>
                  )}
                </div>
              </>
            )}
            
            {vectorConnectionStatus === "disconnected" && (
              <div className="flex items-center gap-2 text-xs">
                <span className="text-foreground font-mono text-[11px] bg-secondary/50 px-2 py-0.5 rounded">
                  Not connected
                </span>
              </div>
            )}
            
            {vectorError && (
              <div className="text-xs md:text-sm text-red-600 bg-red-50 dark:bg-red-950/20 dark:text-red-400 p-2 rounded">
                <p className="whitespace-normal break-words">Error: {vectorError}</p>
              </div>
            )}
            
            <div className="flex gap-2">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={checkVectorConnection}
                      disabled={vectorConnectionStatus === "checking"}
                      className="flex-1 text-xs h-7 px-2"
                    >
                      <RefreshCw className={`h-3 w-3 ${vectorConnectionStatus === "checking" ? "animate-spin" : ""}`} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{vectorConnectionStatus === "checking" ? "Checking..." : "Refresh connection"}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>

              {vectorConnectionStatus === "connected" ? (
                <>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={disconnectVector}
                          className="flex-1 text-xs h-7 px-2"
                        >
                          <LogOut className="h-3 w-3" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Disconnect</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>

                  <Dialog open={showClearVectorDialog} onOpenChange={setShowClearVectorDialog}>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <DialogTrigger asChild>
                            <Button
                              variant="destructive"
                              size="sm"
                              className="flex-1 text-xs h-7 px-2"
                              disabled={isClearingVectorDB}
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </DialogTrigger>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Clear database</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle className="text-destructive">Clear Qdrant Database</DialogTitle>
                        <DialogDescription>
                          Are you sure you want to clear all data from the Qdrant vector database? This action cannot be undone.
                        </DialogDescription>
                      </DialogHeader>
                      <Alert variant="destructive" className="mt-2">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Warning</AlertTitle>
                        <AlertDescription>
                          This will permanently delete all vectors from the Qdrant database.
                        </AlertDescription>
                      </Alert>
                      <DialogFooter className="gap-2 mt-4">
                        <DialogClose asChild>
                          <Button variant="outline" size="sm">Cancel</Button>
                        </DialogClose>
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={clearVectorDatabase}
                          disabled={isClearingVectorDB}
                        >
                          {isClearingVectorDB ? "Clearing..." : "Clear Database"}
                        </Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                </>
              ) : (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    // Open Vector DB settings
                    const event = new CustomEvent('open-settings', {
                      detail: { tab: 'vectordb' }
                    });
                    window.dispatchEvent(event);
                  }}
                  className="flex-1 text-xs h-7"
                >
                  Configure
                </Button>
              )}
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
} 