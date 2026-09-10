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
"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowLeft, BarChart2, Database, Clock, Target, AlertTriangle } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { NvidiaIcon } from "@/components/nvidia-icon";

interface MetricsData {
  totalTriples: number;
  totalEntities: number;
  avgQueryTime: number;
  avgRelevance: number;
  precision: number;
  recall: number;
  f1Score: number;
  topQueries: { query: string; count: number }[];
  queryTimesByMode?: Record<string, number>;
  queryLogStats?: {
    totalQueryLogs: number;
    totalExecutions: number;
    lastQueriedAt?: string;
  };
}

export default function MetricsPage() {
  const { toast } = useToast();
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      setIsLoading(true);
      setError(null);
      try {
        // Fetch real metrics from the API endpoint
        const response = await fetch('/api/metrics');
        
        if (!response.ok) {
          throw new Error(`Failed to fetch metrics: ${response.statusText}`);
        }
        
        const data = await response.json();
        setMetrics(data);
      } catch (error) {
        console.error("Error fetching metrics:", error);
        const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
        setError(errorMessage);
        toast({
          title: "Failed to load metrics",
          description: "Could not retrieve performance data.",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchMetrics();
  }, [toast]);

  // Function to refresh metrics data
  const refreshMetrics = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/metrics');
      if (!response.ok) {
        throw new Error(`Failed to refresh metrics: ${response.statusText}`);
      }
      const data = await response.json();
      setMetrics(data);
      toast({
        title: "Metrics refreshed",
        description: "Performance metrics have been updated",
      });
    } catch (error) {
      console.error("Error refreshing metrics:", error);
      const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
      setError(errorMessage);
      toast({
        title: "Failed to refresh metrics",
        description: "Could not update performance data",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Modern Gradient Header */}
      <main className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            <Link href="/rag" className="btn-outline">
              <ArrowLeft className="h-4 w-4" />
              Back to RAG Query
            </Link>
            <h1 className="text-xl font-bold ml-4">RAG Performance Metrics</h1>
          </div>
          
          <Button 
            variant="outline" 
            size="sm" 
            onClick={refreshMetrics} 
            disabled={isLoading}
            className="flex items-center gap-2"
          >
            {isLoading ? (
              <div className="h-4 w-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                width="16" 
                height="16" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                className="h-4 w-4"
              >
                <path d="M21 2v6h-6"></path>
                <path d="M3 12a9 9 0 0 1 15-6.7L21 8"></path>
                <path d="M3 12a9 9 0 0 0 15 6.7L21 16"></path>
                <path d="M21 22v-6h-6"></path>
              </svg>
            )}
            Refresh
          </Button>
          
          {/* Fix Logs Button */}
          <Button 
            variant="outline" 
            size="sm" 
            onClick={async () => {
              try {
                setIsLoading(true);
                const response = await fetch('/api/fix-query-logs');
                if (!response.ok) {
                  throw new Error('Failed to fix logs');
                }
                const data = await response.json();
                toast({
                  title: "Logs fixed",
                  description: `Fixed ${data.results.fixed} query logs. ${data.results.data.length} logs total.`
                });
                // Refresh metrics after fixing
                refreshMetrics();
              } catch (error) {
                console.error('Error fixing logs:', error);
                toast({
                  title: "Error",
                  description: "Failed to fix query logs",
                  variant: "destructive"
                });
              } finally {
                setIsLoading(false);
              }
            }}
            disabled={isLoading}
            className="flex items-center gap-2 ml-2"
          >
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              width="16" 
              height="16" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              className="h-4 w-4"
            >
              <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
              <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
            </svg>
            Fix Logs
          </Button>
        </div>
        
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center p-8 bg-destructive/10 border border-destructive/30 rounded-lg">
            <AlertTriangle className="h-12 w-12 text-destructive mb-4" />
            <h3 className="text-lg font-semibold mb-2">Failed to load metrics</h3>
            <p className="text-muted-foreground text-center max-w-md">{error}</p>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={refreshMetrics}
              className="mt-4"
            >
              Try Again
            </Button>
          </div>
        ) : metrics ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Knowledge Base Stats */}
            <Card className="glass-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Database className="h-4 w-4 text-primary" />
                  Knowledge Base
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div>
                    <div className="text-xs text-muted-foreground">Total Triples</div>
                    <div className="text-2xl font-bold">{metrics.totalTriples.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Unique Entities</div>
                    <div className="text-2xl font-bold">{metrics.totalEntities.toLocaleString()}</div>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            {/* Performance Stats */}
            <Card className="glass-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Clock className="h-4 w-4 text-primary" />
                  Query Performance
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {metrics.avgQueryTime > 0 ? (
                    <div>
                      <div className="text-xs text-muted-foreground">Avg. Query Time</div>
                      <div className="text-2xl font-bold">{Math.round(metrics.avgQueryTime)} ms</div>
                    </div>
                  ) : (
                    <div>
                      <div className="text-xs text-muted-foreground">Avg. Query Time</div>
                      <div className="text-2xl font-bold">No data</div>
                    </div>
                  )}
                  {metrics.avgRelevance > 0 ? (
                    <div>
                      <div className="text-xs text-muted-foreground">Avg. Relevance Score</div>
                      <div className="text-2xl font-bold">{(metrics.avgRelevance * 100).toFixed(1)}%</div>
                    </div>
                  ) : (
                    <div>
                      <div className="text-xs text-muted-foreground">Avg. Relevance Score</div>
                      <div className="text-2xl font-bold">No data</div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
            
            {/* Precision Metrics - Only show if we have real data */}
            {(metrics.precision > 0 || metrics.recall > 0) && (
              <Card className="glass-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Target className="h-4 w-4 text-primary" />
                    Retrieval Metrics
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <div className="text-xs text-muted-foreground">Precision</div>
                        <div className="text-xl font-bold">{metrics.precision > 0 ? `${(metrics.precision * 100).toFixed(1)}%` : "No data"}</div>
                      </div>
                      <div>
                        <div className="text-xs text-muted-foreground">Recall</div>
                        <div className="text-xl font-bold">{metrics.recall > 0 ? `${(metrics.recall * 100).toFixed(1)}%` : "No data"}</div>
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">F1 Score</div>
                      <div className="text-xl font-bold">{metrics.f1Score > 0 ? `${(metrics.f1Score * 100).toFixed(1)}%` : "No data"}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
            
            {/* Top Queries */}
            <Card className="glass-card md:col-span-2 lg:col-span-1">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <BarChart2 className="h-4 w-4 text-primary" />
                  Top Queries
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {metrics.topQueries.length === 0 ? (
                    <div className="text-center text-muted-foreground text-sm p-4">
                      No queries have been logged yet. Try running some queries!
                    </div>
                  ) : metrics.topQueries.every(item => item.count === 0) ? (
                    <div className="text-center text-muted-foreground text-sm p-4">
                      Query logs exist but have 0 counts. Try refreshing or making new queries.
                    </div>
                  ) : (
                    metrics.topQueries.map((item, i) => (
                      <div key={i} className="flex justify-between items-center text-sm">
                        <div className="truncate flex-1 pr-2">{item.query}</div>
                        <div className="text-muted-foreground font-mono text-xs bg-muted rounded-full px-2 py-0.5">
                          {item.count}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>
            
            {/* Query Log Stats - For Debugging */}
            {metrics.queryLogStats && (
              <Card className="glass-card md:col-span-2 lg:col-span-4 mt-6">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Database className="h-4 w-4 text-primary" />
                    Query Log Stats (Debug)
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-3 bg-muted/20 rounded-md">
                      <div className="text-xs text-muted-foreground">Total Query Logs</div>
                      <div className="text-lg font-bold">{metrics.queryLogStats.totalQueryLogs}</div>
                    </div>
                    <div className="p-3 bg-muted/20 rounded-md">
                      <div className="text-xs text-muted-foreground">Total Query Executions</div>
                      <div className="text-lg font-bold">{metrics.queryLogStats.totalExecutions}</div>
                    </div>
                    <div className="p-3 bg-muted/20 rounded-md">
                      <div className="text-xs text-muted-foreground">Last Queried At</div>
                      <div className="text-lg font-bold">
                        {metrics.queryLogStats.lastQueriedAt ? 
                          new Date(metrics.queryLogStats.lastQueriedAt).toLocaleString() : 
                          'Never'
                        }
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-4 text-xs text-muted-foreground">
                    <p>If you see query logs but 0 counts, try these steps:</p>
                    <ol className="list-decimal pl-5 mt-2 space-y-1">
                      <li>Visit <code className="bg-muted px-1 rounded">/api/query-log/test?query=Test%20query</code> to manually add a test query</li>
                      <li>Click the refresh button above to update the metrics</li>
                      <li>Make new queries using the RAG Query page</li>
                    </ol>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        ) : (
          <div className="text-center p-8 bg-muted/20 rounded-lg">
            <p>No metrics data available</p>
          </div>
        )}
      </main>
    </div>
  );
} 