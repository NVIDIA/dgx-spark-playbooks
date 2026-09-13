"use client"

import { useState, useEffect } from "react"
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { InfoIcon } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { VectorDBStats } from '@/types/graph'

interface PineconeConnectionProps {
  className?: string
}

export function PineconeConnection({ className }: PineconeConnectionProps) {
  const [connectionStatus, setConnectionStatus] = useState<"connected" | "disconnected" | "checking">("disconnected")
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState<VectorDBStats>({ nodes: 0, relationships: 0, source: 'none' })

  // Fetch vector DB stats
  const fetchStats = async () => {
    try {
      const response = await fetch('/api/pinecone-diag/stats');
      const data = await response.json();
      
      if (response.ok) {
        setStats({
          nodes: typeof data.totalVectorCount === 'number' ? data.totalVectorCount : 0,
          relationships: 0, // Vector DB doesn't store relationships
          source: data.source || 'unknown',
          httpHealthy: data.httpHealthy
        });
        
        // If we have a healthy HTTP connection, we're connected
        if (data.httpHealthy) {
          setConnectionStatus("connected");
          setError(null);
        } else {
          setConnectionStatus("disconnected");
          setError(data.error || 'Connection failed');
        }
        
        console.log('Vector DB stats:', data);
      } else {
        console.error('Failed to fetch vector DB stats:', data);
        setConnectionStatus("disconnected");
        setError(data.error || 'Failed to connect to vector database');
      }
    } catch (error) {
      console.error('Error fetching vector DB stats:', error);
      setConnectionStatus("disconnected");
      setError(error instanceof Error ? error.message : 'Error connecting to vector database');
    }
  };

  // Check connection status and stats
  const checkConnection = async () => {
    setConnectionStatus("checking")
    setError(null)
    
    try {
      await fetchStats(); // Fetch stats directly - our status is based on having embeddings
    } catch (error) {
      console.error('Error connecting to Vector DB:', error)
      setConnectionStatus("disconnected")
      setError(error instanceof Error ? error.message : 'Unknown error connecting to Vector DB')
    }
  }

  // Reset connection state
  const disconnect = async () => {
    setConnectionStatus("disconnected")
    setStats({ nodes: 0, relationships: 0, source: 'none' })
  }

  // Initial connection check
  useEffect(() => {
    checkConnection()
  }, [])

  return (
    <div className={`flex flex-col items-start space-y-4 p-4 border rounded-md ${className}`}>
      <div className="flex justify-between w-full">
        <h2 className="text-lg font-medium">Vector DB</h2>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger>
              <InfoIcon className="h-5 w-5 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent>
              <p>Local Pinecone stores vector embeddings in memory for semantic search</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      
      <div className="flex items-center space-x-2">
        <span className="text-sm">Status:</span>
        {connectionStatus === "connected" ? (
          <Badge variant="outline" className="bg-green-50 text-green-700 hover:bg-green-50 border-green-200">Connected</Badge>
        ) : connectionStatus === "checking" ? (
          <Badge variant="outline" className="bg-yellow-50 text-yellow-700 hover:bg-yellow-50 border-yellow-200">Checking...</Badge>
        ) : (
          <Badge variant="outline" className="bg-red-50 text-red-700 hover:bg-red-50 border-red-200">Disconnected</Badge>
        )}
      </div>
      
      {error && (
        <div className="text-sm text-red-600 bg-red-50 p-2 rounded w-full overflow-auto max-h-20">
          <p className="whitespace-normal break-words">Error: {error}</p>
          {error.includes('404') && (
            <p className="mt-1 text-xs">
              The Pinecone server is running but the index doesn't exist yet. 
              <button 
                onClick={async () => {
                  setConnectionStatus("checking");
                  setError(null);
                  try {
                    const response = await fetch('/api/pinecone-diag/create-index', { method: 'POST' });
                    if (response.ok) {
                      // Wait a bit for the index to be created
                      await new Promise(resolve => setTimeout(resolve, 2000));
                      checkConnection();
                    } else {
                      const data = await response.json();
                      setError(data.error || 'Failed to create index');
                      setConnectionStatus("disconnected");
                    }
                  } catch (err) {
                    setError(err instanceof Error ? err.message : 'Error creating index');
                    setConnectionStatus("disconnected");
                  }
                }}
                className="ml-1 text-blue-600 hover:text-blue-800 underline"
              >
                Click here to create the index
              </button>
              <br />
              <span className="text-xs text-gray-600">Or using Docker Compose: </span>
              <code className="mx-1 px-1 bg-gray-100 rounded">docker-compose restart pinecone</code>
            </p>
          )}
        </div>
      )}
      
      <div className="text-sm space-y-1 w-full">
        <div className="flex justify-between">
          <span className="text-muted-foreground">Vectors:</span>
          <span>{stats.nodes}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Source:</span>
          <span>{stats.source} local</span>
        </div>
      </div>
      
      <div className="flex space-x-2">
        <Button 
          variant="outline" 
          size="sm" 
          onClick={checkConnection}
          disabled={connectionStatus === "checking"}
        >
          {connectionStatus === "checking" ? "Checking..." : "Check Connection"}
        </Button>
        
        {connectionStatus === "connected" && (
          <Button 
            variant="outline" 
            size="sm" 
            onClick={disconnect}
          >
            Disconnect
          </Button>
        )}
      </div>
    </div>
  )
} 