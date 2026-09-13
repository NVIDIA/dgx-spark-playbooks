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
import { RagQuery, RagParams } from "@/components/rag-query";
import type { Triple } from "@/types/graph";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { DatabaseConnection } from "@/components/database-connection";
import { NvidiaIcon } from "@/components/nvidia-icon";
import { ArrowLeft, BarChart2, Search as SearchIcon } from "lucide-react";

export default function RagPage() {
  const router = useRouter();
  const [results, setResults] = useState<Triple[] | null>(null);
  const [llmAnswer, setLlmAnswer] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [vectorEnabled, setVectorEnabled] = useState(false);
  const [metrics, setMetrics] = useState<{
    avgQueryTime: number;
    avgRelevance: number;
    precision: number;
    recall: number;
    queryTimesByMode?: Record<string, number>;
  } | null>(null);
  const [currentParams, setCurrentParams] = useState<RagParams>({
    kNeighbors: 4096,
    fanout: 400,
    numHops: 2,
    topK: 5,
    useVectorSearch: false,
    usePureRag: false,
    queryMode: 'traditional'
  });

  // Initialize backend when the page loads
  useEffect(() => {
    // Initialize the backend services
    const initializeBackend = async () => {
      try {
        // Check graph database connection (ArangoDB by default)
        const graphResponse = await fetch('/api/graph-db', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        
        if (!graphResponse.ok) {
          const errorData = await graphResponse.json();
          console.warn('Graph database connection warning:', errorData.error);
        }
        
        // Check if vector search is available
        const vectorResponse = await fetch('/api/vector-db/stats');
        if (vectorResponse.ok) {
          const data = await vectorResponse.json();
          setVectorEnabled(data.totalVectorCount > 0);
        }
        
        // Fetch basic metrics
        const metricsResponse = await fetch('/api/metrics');
        if (metricsResponse.ok) {
          const data = await metricsResponse.json();
          setMetrics({
            avgQueryTime: data.avgQueryTime,
            avgRelevance: data.avgRelevance,
            precision: data.precision,
            recall: data.recall,
            queryTimesByMode: data.queryTimesByMode
          });
        }
      } catch (error) {
        console.warn('Error initializing backends:', error);
      }
    };

    initializeBackend();
  }, []);

  const handleQuerySubmit = async (query: string, params: RagParams) => {
    setIsLoading(true);
    setErrorMessage(null);
    setCurrentParams(params); // Store current params for UI rendering
    const startTime = Date.now();
    let queryMode: 'pure-rag' | 'vector-search' | 'traditional' = 'traditional';
    let resultCount = 0;
    let relevanceScore = 0;
    
    // Debug logging
    console.log('🔍 Query params:', {
      usePureRag: params.usePureRag,
      useVectorSearch: params.useVectorSearch,
      vectorEnabled,
      queryMode: params.queryMode
    });
    
    try {
      // If using pure RAG (Qdrant + LangChain) without graph search
      if (params.usePureRag) {
        queryMode = 'pure-rag';
        try {
          console.log('Using pure RAG with Qdrant and NVIDIA LLM for query:', query);
          const ragResponse = await fetch('/api/rag-query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              query,
              topK: params.topK
            })
          });
          
          if (ragResponse.ok) {
            const data = await ragResponse.json();
            console.log('📥 RAG Response data:', { 
              hasAnswer: !!data.answer, 
              answerLength: data.answer?.length,
              documentCount: data.documentCount 
            });
            // Handle the answer - we might need to display differently than triples
            if (data.answer) {
              console.log('✅ Setting answer in results:', data.answer.substring(0, 100) + '...');

              // Set the LLM answer for display (same as traditional mode)
              setLlmAnswer(data.answer);

              // Set empty results array since Pure RAG doesn't return triples
              setResults([]);

              resultCount = data.documentCount || 0;
              relevanceScore = data.relevanceScore || 0;
              
              // Log the query with performance metrics
              logQuery(query, queryMode, {
                executionTimeMs: Date.now() - startTime,
                relevanceScore,
                resultCount
              });
              
              console.log(`✅ Pure RAG query completed. Retrieved ${resultCount} document chunks`);
              setIsLoading(false);
              return;
            }
          } else {
            // If the RAG query fails, log but continue to try other methods
            const errorData = await ragResponse.json();
            throw new Error(errorData.error || 'Failed to execute pure RAG query');
          }
        } catch (ragError) {
          console.warn('Pure RAG query error (falling back to other methods):', ragError);
          // Continue to other query methods as fallback
        }
      }
      
      // If we have vector embeddings AND explicitly selected vector search, use enhanced query with metadata
      if (vectorEnabled && params.useVectorSearch && !params.usePureRag) {
        queryMode = 'vector-search';
        try {
          console.log('Using enhanced RAG with LangChain for query:', query);
          const enhancedResponse = await fetch('/api/enhanced-query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              query,
              kNeighbors: params.kNeighbors,
              fanout: params.fanout,
              numHops: params.numHops,
              topK: params.topK
            })
          });
          
          if (enhancedResponse.ok) {
            const data = await enhancedResponse.json();
            // Update the results
            setResults(data.relevantTriples || []);
            resultCount = data.count || 0;
            relevanceScore = data.relevanceScore || 0;
            
            // Log the query with performance metrics
            logQuery(query, queryMode, {
              executionTimeMs: Date.now() - startTime,
              relevanceScore,
              resultCount,
              precision: data.precision || 0,
              recall: data.recall || 0,
            });
            
            // Log to console instead of showing alert
            let message = `Enhanced query completed. Found ${resultCount} relevant triples`;
            if (data.metadata?.entityMatches) {
              message += ` from ${data.metadata.entityMatches} matched entities`;
            }
            console.log(message);
            setIsLoading(false);
            return;
          }
        } catch (enhancedError) {
          console.warn('Enhanced query error (falling back to traditional query):', enhancedError);
          // Continue to traditional query as fallback
        }
      }
      
      // Call the LLM-enhanced graph query API
      console.log('✅ Using Graph Search + LLM approach');
      queryMode = 'traditional';
      
      // Get selected LLM model from localStorage
      let llmModel = undefined;
      let llmProvider = undefined;
      try {
        const savedModel = localStorage.getItem("selectedModelForRAG");
        if (savedModel) {
          const modelData = JSON.parse(savedModel);
          llmModel = modelData.model;
          llmProvider = modelData.provider;
          console.log(`Using LLM: ${llmModel} (${llmProvider})`);
        }
      } catch (e) {
        console.warn("Could not load selected LLM model, using default");
      }
      
      const response = await fetch(`/api/graph-query-llm`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          topK: params.topK || 5,
          useTraditional: true,
          llmModel,
          llmProvider
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to query with LLM');
      }
      
      const data = await response.json();

      // Log sample of retrieved triples for debugging
      console.log('📊 Retrieved Triples (sample):', data.triples.slice(0, 3));
      console.log('🤖 LLM-Generated Answer (preview):', data.answer?.substring(0, 200) + '...');
      console.log('📈 Triple Count:', data.count);

      // DEBUG: Check if depth/pathLength are present in received data
      if (data.triples && data.triples.length > 0) {
        console.log('🔍 First triple structure:', JSON.stringify(data.triples[0], null, 2));
        console.log('🔍 Has depth?', 'depth' in data.triples[0]);
        console.log('🔍 Has pathLength?', 'pathLength' in data.triples[0]);
      }
      
      // Update the results with the triples (for display)
      setResults(data.triples || []);
      resultCount = data.count || 0;
      relevanceScore = 0; // No relevance score for traditional search
      
      // Store the LLM answer for display
      if (data.answer) {
        console.log('✅ Setting llmAnswer state (length:', data.answer.length, 'chars)');
        setLlmAnswer(data.answer);
      } else {
        console.log('⚠️ No answer in response');
        setLlmAnswer(null);
      }
      
      // Log the query with performance metrics
      logQuery(query, queryMode, {
        executionTimeMs: Date.now() - startTime,
        relevanceScore,
        resultCount,
        precision: data.precision || 0,
        recall: data.recall || 0,
      });
      
      // Log to console instead of showing alert
      let message = `Query completed. Found ${resultCount} relevant triples`;
      if (vectorEnabled && params.useVectorSearch) {
        message += ` (using standard vector search)`;
      }
      console.log(message);
    } catch (error) {
      console.error("RAG query error:", error);
      setErrorMessage(error instanceof Error ? error.message : "An unknown error occurred");
      setResults([]);
      
      // Log failed query
      logQuery(query, queryMode, {
        executionTimeMs: Date.now() - startTime,
        resultCount: 0,
        error: error instanceof Error ? error.message : "Unknown error"
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Helper function to log queries
  const logQuery = async (
    query: string, 
    queryMode: 'pure-rag' | 'vector-search' | 'traditional',
    metrics: {
      executionTimeMs: number;
      relevanceScore?: number;
      precision?: number;
      recall?: number;
      resultCount: number;
      error?: string;
    }
  ) => {
    try {
      await fetch('/api/query-log', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          queryMode,
          metrics
        })
      });
      console.log('Query logged successfully');
    } catch (error) {
      // Non-blocking error, just log it
      console.warn('Failed to log query:', error);
    }
  };

  const clearResults = () => {
    setResults(null);
    setLlmAnswer(null);
    setErrorMessage(null);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Main Content */}
      <main className="container mx-auto px-6 py-12">
        {/* Header Section */}
        <div className="flex items-center justify-between mb-8">
          <Link href="/" className="inline-flex items-center gap-3 px-4 py-2 text-sm font-medium border border-border/40 hover:border-border/60 bg-background hover:bg-muted/30 rounded-lg transition-colors">
            <ArrowLeft className="h-4 w-4" />
            Back to Documents
          </Link>
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Left Column - Database Connections */}
          <div className="lg:col-span-1 space-y-6">
            <div className="nvidia-build-card">
              <DatabaseConnection />
            </div>
            
            {/* Performance Metrics Card */}
            {metrics && (
              <div className="nvidia-build-card">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-md bg-nvidia-green/15 flex items-center justify-center">
                      <BarChart2 className="h-3 w-3 text-nvidia-green" />
                    </div>
                    <h3 className="text-base font-semibold text-foreground">Performance Metrics</h3>
                  </div>
                  <Link href="/rag/metrics" className="text-xs text-nvidia-green hover:text-nvidia-green/80 font-medium underline underline-offset-2">
                    View All
                  </Link>
                </div>
                
                <div className="space-y-3 text-sm">
                  {/* Query times by mode */}
                  {metrics.queryTimesByMode && Object.keys(metrics.queryTimesByMode).length > 0 ? (
                    <>
                      {metrics.queryTimesByMode['pure-rag'] !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Pure RAG:</span>
                          <span className="font-medium">{(metrics.queryTimesByMode['pure-rag'] / 1000).toFixed(2)}s</span>
                        </div>
                      )}
                      {metrics.queryTimesByMode['traditional'] !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Graph Search:</span>
                          <span className="font-medium">{(metrics.queryTimesByMode['traditional'] / 1000).toFixed(2)}s</span>
                        </div>
                      )}
                      {metrics.queryTimesByMode['vector-search'] !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">GraphRAG:</span>
                          <span className="font-medium">{(metrics.queryTimesByMode['vector-search'] / 1000).toFixed(2)}s</span>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Avg. Query Time:</span>
                      <span className="font-medium">{metrics.avgQueryTime > 0 ? `${metrics.avgQueryTime.toFixed(2)}ms` : "No data"}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
          
          {/* Right Column - RAG Query Interface */}
          <div className="lg:col-span-3">
            <RagQuery
              onQuerySubmit={handleQuerySubmit}
              clearResults={clearResults}
              isLoading={isLoading}
              error={errorMessage}
              vectorEnabled={vectorEnabled}
            />
            
            {/* LLM Answer Section */}
            {llmAnswer && (
              <div className="mt-8 nvidia-build-card">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-6 h-6 rounded-md bg-nvidia-green/15 flex items-center justify-center">
                    <SearchIcon className="h-3 w-3 text-nvidia-green" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground">Answer</h3>
                  {currentParams.queryMode && (
                    <span className="text-xs px-2.5 py-1 rounded-full font-medium bg-nvidia-green/10 text-nvidia-green border border-nvidia-green/20">
                      {currentParams.queryMode === 'pure-rag' ? 'Pure RAG' :
                       currentParams.queryMode === 'vector-search' ? 'GraphRAG' :
                       'Graph Search'}
                    </span>
                  )}
                </div>
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  {(() => {
                    // Parse <think> tags
                    const thinkMatch = llmAnswer.match(/<think>([\s\S]*?)<\/think>/);
                    const thinkContent = thinkMatch ? thinkMatch[1].trim() : null;
                    const mainAnswer = thinkContent
                      ? llmAnswer.replace(/<think>[\s\S]*?<\/think>/, '').trim()
                      : llmAnswer;

                    return (
                      <>
                        {thinkContent && (
                          <details className="mb-4 bg-muted/10 border border-border/20 rounded-xl overflow-hidden group">
                            <summary className="cursor-pointer p-4 hover:bg-muted/20 transition-colors flex items-center gap-2">
                              <svg className="w-4 h-4 transform transition-transform group-open:rotate-90" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                              </svg>
                              <span className="text-sm font-medium text-muted-foreground">Reasoning Process</span>
                            </summary>
                            <div className="p-4 pt-0 text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap border-t border-border/10">
                              {thinkContent}
                            </div>
                          </details>
                        )}
                        <div className="bg-muted/20 border border-border/20 p-6 rounded-xl">
                          <div
                            className="text-foreground leading-relaxed whitespace-pre-wrap"
                            dangerouslySetInnerHTML={{
                              __html: mainAnswer
                                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                            }}
                          />
                        </div>
                      </>
                    );
                  })()}
                </div>
              </div>
            )}
            
            {/* Results Section */}
            {results && results.length > 0 && !currentParams.usePureRag && (
              <div className="mt-8 nvidia-build-card">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-6 h-6 rounded-md bg-nvidia-green/15 flex items-center justify-center">
                    <SearchIcon className="h-3 w-3 text-nvidia-green" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground">
                    {llmAnswer ? `Retrieved Knowledge (${results.length})` : `Results (${results.length})`}
                  </h3>
                  {results.some((r: any) => r.pathLength && r.pathLength > 1) && (
                    <span className="text-xs px-2.5 py-1 rounded-full font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20 flex items-center gap-1.5">
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      Multi-hop enabled
                    </span>
                  )}
                </div>
                <div className="space-y-4">
                  {results.map((triple, index) => (
                    <div key={index} className="bg-muted/20 border border-border/20 p-4 rounded-xl">
                      {currentParams.usePureRag ? (
                        // Pure RAG display format (no subject/predicate/object columns)
                        <div className="p-2 rounded">
                          {triple.usedFallback && (
                            <div className="mb-2 text-sm px-3 py-1 bg-amber-500/20 text-amber-700 dark:text-amber-400 rounded-md inline-block">
                              Using general knowledge (no documents found)
                            </div>
                          )}
                          <p className="font-medium break-words">{triple.object}</p>
                        </div>
                      ) : (
                        // Standard triple display for other modes
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          <div className="bg-background/60 border border-border/30 p-3 rounded-lg">
                            <p className="text-xs font-medium text-nvidia-green uppercase tracking-wider mb-1">Subject</p>
                            <p className="font-medium break-words text-foreground">{triple.subject}</p>
                          </div>
                          <div className="bg-background/60 border border-border/30 p-3 rounded-lg">
                            <p className="text-xs font-medium text-nvidia-green uppercase tracking-wider mb-1">Predicate</p>
                            <p className="font-medium break-words text-foreground">{triple.predicate}</p>
                          </div>
                          <div className="bg-background/60 border border-border/30 p-3 rounded-lg">
                            <p className="text-xs font-medium text-nvidia-green uppercase tracking-wider mb-1">Object</p>
                            <p className="font-medium break-words text-foreground">{triple.object}</p>
                          </div>
                        </div>
                      )}
                      {triple.confidence && !currentParams.usePureRag && (
                        <div className="mt-3 flex items-center gap-4 text-xs">
                          <div className="flex items-center gap-1.5">
                            <div className="w-2 h-2 rounded-full bg-nvidia-green/60"></div>
                            <span className="text-muted-foreground">
                              Confidence: <span className="font-medium text-foreground">{(triple.confidence * 100).toFixed(1)}%</span>
                            </span>
                          </div>
                          {triple.depth !== undefined && (
                            <div className="flex items-center gap-1.5">
                              <svg className="w-3 h-3 text-nvidia-green/60" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                              </svg>
                              <span className="text-muted-foreground">
                                Hop: <span className="font-medium text-foreground">{triple.depth + 1}</span>
                              </span>
                            </div>
                          )}
                          {triple.pathLength !== undefined && triple.pathLength > 1 && (
                            <div className="flex items-center gap-1.5">
                              <svg className="w-3 h-3 text-amber-500/60" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                              </svg>
                              <span className="text-amber-600/80 dark:text-amber-400/80">
                                Multi-hop path (length: <span className="font-medium">{triple.pathLength}</span>)
                              </span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {results && results.length === 0 && !isLoading && !currentParams.usePureRag && (
              <div className="mt-8 nvidia-build-card border-dashed">
                <div className="text-center py-8">
                  <div className="w-12 h-12 rounded-xl bg-muted/30 flex items-center justify-center mx-auto mb-4">
                    <SearchIcon className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <p className="text-foreground font-medium mb-2">No results found for your query</p>
                  <p className="text-sm text-muted-foreground">Try adjusting your query or parameters</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
} 